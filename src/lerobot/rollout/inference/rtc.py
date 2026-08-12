# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-Time Chunking inference engine.

A background thread produces action chunks asynchronously via
:meth:`policy.predict_action_chunk`.  The main control loop polls
``get_action`` for the next ready action; observations flow the other
way via ``notify_observation``.
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from collections import deque
from threading import Event, Lock, Thread
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc import ActionQueue, LatencyTracker, reanchor_relative_rtc_prefix
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import (
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
)
from lerobot.utils.feature_utils import build_dataset_frame

from ..robot_wrapper import ThreadSafeRobot
from .base import InferenceEngine

logger = logging.getLogger(__name__)


class _RolloutViz:
    """Throttled background saver of the policy's-eye head view + pixel-head
    marker. Runs entirely off the control loop: images are copied in the RTC
    thread (~1 ms, once per VIZ_PERIOD_S) and encoded on a daemon thread.
    Enabled via env ROLLOUT_VIZ=1; never blocks (queue drops when full)."""

    VIZ_PERIOD_S = 2.0
    MAX_FILES = 60

    def __init__(self):
        import os
        import queue as _q
        self.enabled = os.environ.get("ROLLOUT_VIZ", "1") == "1"
        self.q = _q.Queue(maxsize=4)
        self.last_t = 0.0
        self.n = 0
        self.dir = None
        self.thread = None

    def maybe_submit(self, head_img, uv_norm, uv_stats, progress):
        if not self.enabled or head_img is None or self.n >= self.MAX_FILES:
            return
        now = time.perf_counter()
        if now - self.last_t < self.VIZ_PERIOD_S:
            return
        self.last_t = now
        try:
            import numpy as _np
            img = head_img.detach().cpu().numpy() if hasattr(head_img, "detach") else _np.array(head_img)
            self.q.put_nowait((img.copy(), uv_norm, uv_stats, progress, self.n))
            self.n += 1
        except Exception:  # noqa: BLE001 — viz must never break inference
            pass
        if self.thread is None:
            import threading
            from pathlib import Path
            self.dir = Path("outputs/rollout_logs") / time.strftime("pixelviz-%Y%m%d-%H%M%S")
            self.dir.mkdir(parents=True, exist_ok=True)
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
            logger.info("rollout viz -> %s", self.dir)

    def _worker(self):
        import cv2
        import numpy as np
        while True:
            img, uv, stats, prog, idx = self.q.get()
            try:
                if img.ndim == 3 and img.shape[0] in (1, 3):        # CHW -> HWC
                    img = np.transpose(img, (1, 2, 0))
                if img.dtype != np.uint8:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                h, w = img.shape[:2]
                if uv is not None:
                    u, v = uv
                    if stats is not None:
                        lo, hi = stats
                        u = ((u + 1) / 2) * (hi[0] - lo[0]) + lo[0]
                        v = ((v + 1) / 2) * (hi[1] - lo[1]) + lo[1]
                    px, py = int(u * w), int(v * h)
                    cv2.drawMarker(img, (px, py), (255, 0, 255), cv2.MARKER_CROSS, 30, 2)
                    cv2.circle(img, (px, py), 12, (255, 0, 255), 2)
                    cv2.putText(img, f"puck ({u:.3f},{v:.3f})", (10, h - 12),
                                0, 0.55, (255, 0, 255), 2)
                if prog is not None:
                    cv2.putText(img, f"progress {prog:+.3f}", (10, 24),
                                0, 0.7, (0, 255, 255), 2)
                cv2.imwrite(str(self.dir / f"viz_{idx:03d}.jpg"), img)
            except Exception:  # noqa: BLE001
                pass

# How long the RTC loop sleeps when paused, idle, or backpressured by a full queue.
_RTC_IDLE_SLEEP_S: float = 0.01
# Backoff between transient inference errors (per consecutive failure).
_RTC_ERROR_RETRY_DELAY_S: float = 0.5
# Consecutive transient errors tolerated before giving up and propagating shutdown.
_RTC_MAX_CONSECUTIVE_ERRORS: int = 10
# Hard timeout for joining the RTC thread on stop().
_RTC_JOIN_TIMEOUT_S: float = 3.0


# ---------------------------------------------------------------------------
# RTC helpers
# ---------------------------------------------------------------------------


def _normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Pad or truncate RTC prefix actions to a fixed length for stable compiled inference."""
    if prev_actions.ndim != 2:
        raise ValueError(f"Expected 2D [T, A] tensor, got shape={tuple(prev_actions.shape)}")
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded


# ---------------------------------------------------------------------------
# RTCInferenceEngine
# ---------------------------------------------------------------------------


class RTCInferenceEngine(InferenceEngine):
    """Async RTC inference: a background thread produces action chunks.

    ``get_action`` pops the next action from the shared queue (or
    returns ``None`` if the queue is empty).  The main loop should call
    ``notify_observation`` every tick and ``pause``/``resume`` around
    human-intervention phases.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        robot_wrapper: ThreadSafeRobot,
        rtc_config: RTCConfig,
        hw_features: dict,
        task: str,
        fps: float,
        device: str | None,
        use_torch_compile: bool = False,
        compile_warmup_inferences: int = 2,
        rtc_queue_threshold: int = 30,
        shutdown_event: Event | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._robot = robot_wrapper
        self._rtc_config = rtc_config
        self._hw_features = hw_features
        self._task = task
        self._fps = fps
        self._device = device or "cpu"
        self._use_torch_compile = use_torch_compile
        self._compile_warmup_inferences = compile_warmup_inferences
        self._rtc_queue_threshold = rtc_queue_threshold
        self._viz = _RolloutViz()
        self._uv_stats = None            # filled after _normalizer_step below

        self._action_queue: ActionQueue | None = None
        self._obs_holder: dict[str, Any] = {}
        # NOTE: self._task is re-read at every replan, so set_task takes
        # effect on the next inference without an engine rebuild. Same for
        # self._obs_constants (set_obs_constants).
        self._obs_constants: dict[str, Any] = {}
        # Latest progress-head readout in raw units ([-1, 0], 0 = subtask
        # complete); None when the policy has no progress head. Updated once
        # per replan; consumed by stage-transition logic.
        self.last_progress: float | None = None
        # Policies with n_obs_steps > 1 (e.g. diffusion) were trained on
        # consecutive-tick observation windows; keep the last n raw
        # observations (notify_observation ticks at the control rate) and
        # window them at generation time.
        self._n_obs_steps = int(getattr(policy.config, "n_obs_steps", 1) or 1)
        self._obs_history: deque = deque(maxlen=self._n_obs_steps)
        self._obs_lock = Lock()
        self._policy_active = Event()
        self._compile_warmup_done = Event()
        self._shutdown_event = Event()
        self._rtc_error = Event()
        self._global_shutdown_event = shutdown_event
        self._rtc_thread: Thread | None = None

        if not self._use_torch_compile:
            self._compile_warmup_done.set()
            logger.info("RTCInferenceEngine initialized (torch.compile disabled, no warmup needed)")
        else:
            logger.info(
                "RTCInferenceEngine initialized (torch.compile enabled, %d warmup inferences)",
                compile_warmup_inferences,
            )

        # Processor introspection for relative-action re-anchoring.
        self._action_adapter = None
        self._relative_step = next(
            (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
            None,
        )
        self._normalizer_step = next(
            (s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep)),
            None,
        )
        try:
            st = getattr(self._normalizer_step, "stats", None) or {}
            uv = st.get("observation.puck_uv")
            if uv is not None:
                import numpy as _np
                self._uv_stats = (_np.asarray(uv["min"], dtype=float),
                                  _np.asarray(uv["max"], dtype=float))
        except Exception:  # noqa: BLE001
            self._uv_stats = None
        if self._relative_step is not None:
            if self._relative_step.action_names is None:
                cfg_names = getattr(policy.config, "action_feature_names", None)
                if cfg_names:
                    self._relative_step.action_names = list(cfg_names)
                else:
                    self._relative_step.action_names = [
                        k for k in robot_wrapper.action_features if k.endswith(".pos")
                    ]
            logger.info("Relative actions enabled: RTC prefix will be re-anchored")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True once torch.compile warmup is complete (or immediately if compile is disabled)."""
        return self._compile_warmup_done.is_set()

    @property
    def failed(self) -> bool:
        """True if the RTC background thread exited due to an unrecoverable error."""
        return self._rtc_error.is_set()

    @property
    def action_queue(self) -> ActionQueue | None:
        """The shared action queue between the RTC thread and the main loop."""
        return self._action_queue

    def start(self) -> None:
        """Launch the RTC background thread."""
        self._action_queue = ActionQueue(self._rtc_config)
        self._obs_holder = {
            "obs": None,
            "robot_type": self._robot.robot_type,
        }
        self._shutdown_event.clear()
        self._rtc_thread = Thread(
            target=self._rtc_loop,
            daemon=True,
            name="RTCInference",
        )
        self._rtc_thread.start()
        logger.info("RTC inference thread started")

    def stop(self) -> None:
        """Signal the RTC thread to stop and wait for it."""
        logger.info("Stopping RTC inference thread...")
        self._shutdown_event.set()
        self._policy_active.clear()
        if self._rtc_thread is not None and self._rtc_thread.is_alive():
            self._rtc_thread.join(timeout=_RTC_JOIN_TIMEOUT_S)
            if self._rtc_thread.is_alive():
                logger.warning("RTC thread did not join within %.1fs", _RTC_JOIN_TIMEOUT_S)
            else:
                logger.info("RTC inference thread stopped")
            self._rtc_thread = None

    def pause(self) -> None:
        """Pause the RTC background thread."""
        logger.info("Pausing RTC inference thread")
        self._policy_active.clear()

    def resume(self) -> None:
        """Resume the RTC background thread."""
        logger.info("Resuming RTC inference thread")
        if self._action_queue is not None and self._rtc_config.splice_blend_steps > 0:
            # Anchor the first chunk's cross-fade at the robot's current pose
            # so inference ramps out of the held position instead of jumping.
            try:
                pos = self._robot.get_pos_observation()
                seed = torch.tensor([float(pos[k]) for k in self._robot.action_features])
                self._action_queue.set_seed(seed)
            except Exception:
                logger.exception("Could not seed first-chunk blend (continuing without)")
        self._policy_active.set()

    def set_action_adapter(self, adapter) -> None:
        """Optional action-space adapter (see SyncInferenceConfig.action_adapter):
        adapt_observation maps each robot-space obs frame into the policy's
        space; adapt_chunk maps a postprocessed action chunk back to
        robot-space rows before it enters the queue. When set, the prev-chunk
        reanchoring path is skipped (converted leftovers are no longer in
        policy space; adapter-based policies must ignore prev_chunk_left_over).
        """
        self._action_adapter = adapter

    def set_task(self, task: str) -> None:
        """Swap the language prompt; consumed at the next replan (staged
        prompt-only stages share one engine and switch the task string)."""
        if task != self._task:
            logger.info("task -> %r", task)
            self._task = task

    def set_obs_constants(self, constants: dict | None) -> None:
        """Constant observation features injected into every frame at replan
        time — e.g. a per-stage task one-hot ({'observation.task_onehot':
        np.array([1,0,0], dtype=np.float32)}) for task-conditioned policies.
        The key must be a declared policy input feature so the normalizer
        has stats for it. Consumed at the next replan; pass None/{} to clear."""
        self._obs_constants = dict(constants or {})
        if self._obs_constants:
            logger.info("obs constants -> %s",
                        {k: getattr(v, "tolist", lambda: v)()
                         for k, v in self._obs_constants.items()})

    def reset(self) -> None:
        self.last_progress = None  # stale readouts must not trigger auto-advance
        """Reset the policy, processors, and action queue."""
        logger.info("Resetting RTC inference state (policy + processors + queue)")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()
        if self._action_adapter is not None:
            self._action_adapter.reset()
        if self._action_queue is not None:
            self._action_queue.clear()
        with self._obs_lock:
            self._obs_history.clear()

    # ------------------------------------------------------------------
    # Action production (called from main thread)
    # ------------------------------------------------------------------

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """Pop the next action from the RTC queue (ignores ``obs_frame``)."""
        if self._action_queue is None:
            return None
        return self._action_queue.get()

    def notify_observation(self, obs: dict) -> None:
        """Publish the latest observation for the RTC thread to consume."""
        with self._obs_lock:
            self._obs_holder["obs"] = obs
            self._obs_history.append(obs)

    # ------------------------------------------------------------------
    # RTC: background inference thread
    # ------------------------------------------------------------------

    def _rtc_loop(self) -> None:
        """Background thread that generates action chunks via RTC."""
        try:
            latency_tracker = LatencyTracker()
            time_per_chunk = 1.0 / self._fps
            policy_device = torch.device(self._device)

            # Even without torch.compile the FIRST inference pays one-off CUDA
            # autotuning (observed 0.6-1.5s vs 0.25s steady for pi05); it must
            # not seed the latency tracker or delay compensation runs ~2x-6x
            # pessimistic for the rest of the session (merges land after the
            # old chunk is exhausted -> lurch at every chunk boundary).
            warmup_required = max(1, self._compile_warmup_inferences) if self._use_torch_compile else 1
            inference_count = 0
            consecutive_errors = 0

            while not self._shutdown_event.is_set():
                if not self._policy_active.is_set():
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                queue = self._action_queue
                with self._obs_lock:
                    obs = self._obs_holder.get("obs")
                if queue is None or obs is None:
                    time.sleep(_RTC_IDLE_SLEEP_S)
                    continue

                if queue.qsize() <= self._rtc_queue_threshold:
                    try:
                        current_time = time.perf_counter()
                        idx_before = queue.get_action_index()
                        prev_actions = queue.get_left_over()

                        latency = latency_tracker.max()
                        delay = math.ceil(latency / time_per_chunk) if latency else 0

                        if self._n_obs_steps > 1:
                            # Preprocess the last n consecutive-tick raw
                            # observations individually (the image steps are
                            # 4-D only), then stack into (B, n_obs_steps, ...)
                            # as the policy was trained.
                            with self._obs_lock:
                                window = list(self._obs_history)
                            while len(window) < self._n_obs_steps:
                                window.insert(0, window[0])
                            frames = []
                            viz_head = None
                            for o in window:
                                fb = build_dataset_frame(self._hw_features, o, prefix="observation")
                                if self._action_adapter is not None:
                                    fb = self._action_adapter.adapt_observation(fb)
                                viz_head = fb.get("observation.images.head", viz_head)
                                if self._obs_constants:
                                    fb.update(self._obs_constants)
                                fb = prepare_observation_for_inference(
                                    fb, policy_device, self._task, self._robot.robot_type
                                )
                                fb["task"] = [self._task]
                                frames.append(self._preprocessor(fb))
                            preprocessed = {
                                k: torch.stack([f[k] for f in frames], dim=1)
                                for k, v in frames[-1].items()
                                if isinstance(v, torch.Tensor)
                            }
                            preprocessed["task"] = [self._task]
                        else:
                            obs_batch = build_dataset_frame(self._hw_features, obs, prefix="observation")
                            if self._action_adapter is not None:
                                obs_batch = self._action_adapter.adapt_observation(obs_batch)
                            if self._obs_constants:
                                obs_batch.update(self._obs_constants)
                            obs_batch = prepare_observation_for_inference(
                                obs_batch, policy_device, self._task, self._robot.robot_type
                            )
                            obs_batch["task"] = [self._task]
                            preprocessed = self._preprocessor(obs_batch)

                        if prev_actions is not None and self._relative_step is not None:
                            # Rebase against the raw cached state so the leftover tail stays in
                            # the training-time coordinate frame. With an action adapter the
                            # served queue is robot-space, so the reanchorable absolutes come
                            # from the queue's pre-adapter policy stream instead.
                            raw_state = self._relative_step.get_cached_state()
                            if raw_state is not None:
                                prev_abs = (queue.get_policy_left_over()
                                            if self._action_adapter is not None
                                            else queue.get_processed_left_over())
                                if prev_abs is not None and prev_abs.numel() > 0:
                                    prev_actions = reanchor_relative_rtc_prefix(
                                        prev_actions_absolute=prev_abs,
                                        current_state=raw_state,
                                        relative_step=self._relative_step,
                                        normalizer_step=self._normalizer_step,
                                        policy_device=policy_device,
                                    )

                        if prev_actions is not None:
                            prev_actions = _normalize_prev_actions_length(
                                prev_actions, target_steps=self._rtc_config.execution_horizon
                            )

                        actions = self._policy.predict_action_chunk(
                            preprocessed, inference_delay=delay, prev_chunk_left_over=prev_actions
                        )

                        # Progress readout (policies with a progress head).
                        # The head trains on MIN_MAX-normalized targets whose
                        # stats are exactly min=-1/max=0 by construction, so
                        # norm = 2x + 1 and the inverse is fixed: x=(norm-1)/2.
                        p_norm = getattr(
                            getattr(self._policy, "diffusion", self._policy),
                            "_last_progress_norm", None)
                        if p_norm is not None:
                            self.last_progress = (p_norm - 1.0) / 2.0
                            # console readout throttled to ~1/s (replans can
                            # be every ~0.5 s; don't spam the terminal)
                            now_p = time.perf_counter()
                            if now_p - getattr(self, "_progress_printed_t", 0.0) >= 1.0:
                                self._progress_printed_t = now_p
                                print(f"[progress] {self.last_progress:+.3f}  "
                                      "(-1 = start, 0 = task complete)", flush=True)

                        original = actions.squeeze(0).clone()
                        processed = self._postprocessor(actions).squeeze(0)
                        policy_abs = None
                        if self._action_adapter is not None:
                            # keep the absolute policy-space chunk: the queue
                            # stores it index-aligned so the next replan can
                            # reanchor the leftover tail (adapter output is
                            # robot-space and can't be rebased)
                            policy_abs = processed.clone()
                            processed = self._action_adapter.adapt_chunk(processed)
                        new_latency = time.perf_counter() - current_time
                        new_delay = math.ceil(new_latency / time_per_chunk)

                        inference_count += 1
                        consecutive_errors = 0
                        is_warmup = inference_count <= warmup_required
                        if is_warmup:
                            latency_tracker.reset()
                        else:
                            latency_tracker.add(new_latency)

                        queue.merge(original, processed, new_delay, idx_before,
                                    policy_actions=policy_abs)

                        # throttled background viz (no control-loop impact)
                        uv_n = getattr(
                            getattr(self._policy, "diffusion", self._policy),
                            "_last_pixel_norm", None)
                        self._viz.maybe_submit(viz_head if self._n_obs_steps > 1 else None,
                                               uv_n, self._uv_stats, self.last_progress)

                        if (
                            is_warmup
                            and inference_count >= warmup_required
                            and not self._compile_warmup_done.is_set()
                        ):
                            self._compile_warmup_done.set()
                            logger.info("Compile warmup complete (%d inferences)", inference_count)

                        logger.debug("RTC inference latency=%.2fs, queue=%d", new_latency, queue.qsize())

                    except Exception as e:
                        consecutive_errors += 1
                        logger.error(
                            "RTC inference error (%d/%d): %s",
                            consecutive_errors,
                            _RTC_MAX_CONSECUTIVE_ERRORS,
                            e,
                        )
                        logger.debug(traceback.format_exc())
                        if consecutive_errors >= _RTC_MAX_CONSECUTIVE_ERRORS:
                            # Persistent failure: stop retrying and propagate shutdown.
                            raise
                        time.sleep(_RTC_ERROR_RETRY_DELAY_S)
                else:
                    time.sleep(_RTC_IDLE_SLEEP_S)

        except Exception as e:
            logger.error("Fatal error in RTC thread: %s", e)
            logger.error(traceback.format_exc())
            self._rtc_error.set()
            # Unblock any warmup waiters so the main loop doesn't spin forever
            self._compile_warmup_done.set()
            # Signal the top-level shutdown so strategies exit their control loops
            if self._global_shutdown_event is not None:
                self._global_shutdown_event.set()
