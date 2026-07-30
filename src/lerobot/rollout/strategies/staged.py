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

"""Staged rollout strategy: interactive operation over multiple policies.

Letter keys q/w/e/r/t select the active model (each with its own checkpoint,
inference config, task string and duration cap), 'g' runs it; everything
else behaves like the interactive strategy (number keys 1..0 = poses,
's' = stop, 'o' = open gripper while idle). ESC quits — 'q' is a model key.

All stage policies are loaded onto the GPU at setup, so switching is
instant. Stage 1 arrives pre-built through the normal rollout context
(repo-side config preparation copies it to the top-level policy/inference/
task); the remaining stages are built here against the same robot wrapper,
features and action keys.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import ImageCropResizeProcessorStep

from lerobot.policies.rtc.configuration_rtc import RTCConfig

from ..context import RolloutContext
from ..inference import (
    InferenceEngine,
    RTCInferenceConfig,
    SyncInferenceConfig,
    create_inference_engine,
)
from .interactive import InteractiveStrategy

logger = logging.getLogger(__name__)

MODEL_KEYS = ("q", "w", "e", "r", "t")


# Default: progress must sit above the stage threshold this long
# (continuously) before the stage counts as complete, so one replan-noise
# spike can't advance. Stages whose progress OSCILLATES semantically
# (scrape regresses at every retry stroke and may never hold above the
# threshold) should set complete_sustain_s: 0 — first crossing advances.
DEFAULT_COMPLETE_SUSTAIN_S = 1.0


@dataclass
class _Stage:
    key: str
    name: str
    task: str
    duration: float                    # per-stage timeout: cap = assumed done
    engine: InferenceEngine
    complete_threshold: float | None = None   # progress >= thr -> complete
    complete_sustain_s: float = DEFAULT_COMPLETE_SUSTAIN_S


class StagedStrategy(InteractiveStrategy):
    """Interactive strategy with q/w/e/r/t model selection."""

    QUIT_KEYS = ("\x1b",)  # ESC only: 'q' selects model 1

    def setup(self, ctx: RolloutContext) -> None:
        super().setup(ctx)
        stages = self.config.stages
        if not stages:
            raise ValueError("staged strategy needs strategy.stages (did "
                             "config preparation expand the card references?)")
        if len(stages) > len(MODEL_KEYS):
            raise ValueError(f"at most {len(MODEL_KEYS)} stages")

        cfg = ctx.runtime.cfg
        self._stages: list[_Stage] = []
        for i, st in enumerate(stages):
            if i == 0:
                # pre-built by the normal context path (top-level policy)
                engine = ctx.policy.inference
            else:
                logger.info("loading stage %d/%d: %s ...", i + 1, len(stages),
                            st.get("name", st.get("key", "?")))
                engine = _build_stage_engine(ctx, st)
                engine.reset()
                engine.start()
            self._stages.append(_Stage(
                key=st.get("key", MODEL_KEYS[i]),
                name=st.get("name", MODEL_KEYS[i]),
                task=st.get("task", cfg.task),
                duration=float(st.get("duration", cfg.duration)),
                engine=engine,
                complete_threshold=(None if st.get("complete_threshold") is None
                                    else float(st["complete_threshold"])),
                complete_sustain_s=float(st.get("complete_sustain_s",
                                                DEFAULT_COMPLETE_SUSTAIN_S)),
            ))
        self._active = 0
        self._complete_since: float | None = None
        if self.config.auto_advance:
            missing = [s_.name for s_, st in zip(self._stages, stages, strict=True)
                       if st.get("complete_threshold") is None]
            if missing:
                logger.warning("auto_advance: stages %s have no "
                               "complete_threshold — they advance on timeout only",
                               missing)
        keymap = " | ".join(f"{s.key}={s.name}" for s in self._stages)
        logger.info("Staged strategy ready — models: %s | 1..0=poses | "
                    "g=run active model | s=stop | o=open gripper %.0f%% | "
                    "ESC=quit", keymap, cfg.strategy.open_gripper_fraction * 100)
        self._announce()

    def _announce(self) -> None:
        s = self._stages[self._active]
        logger.info(">>> active model: %s [%s] — task %r, duration %.0fs — "
                    "g to run", s.name, s.key, s.task, s.duration)

    def _select(self, idx: int) -> None:
        self._active = idx
        self._engine = self._stages[idx].engine
        self._cached_obs_processed = None
        self._announce()

    def _inference_duration(self, cfg) -> float:
        return self._stages[self._active].duration

    def _window_complete(self) -> bool:
        """Auto-advance completion: the active stage's progress readout has
        sat at/above its threshold for COMPLETE_SUSTAIN_S."""
        import time

        if not self.config.auto_advance:
            return False
        stage = self._stages[self._active]
        if stage.complete_threshold is None:
            return False
        p = getattr(stage.engine, "last_progress", None)
        if p is None or p < stage.complete_threshold:
            self._complete_since = None
            return False
        if stage.complete_sustain_s <= 0:
            logger.info("stage complete: progress %+.3f >= %+.3f", p,
                        stage.complete_threshold)
            return True
        if self._complete_since is None:
            self._complete_since = time.perf_counter()
            return False
        if time.perf_counter() - self._complete_since >= stage.complete_sustain_s:
            logger.info("stage complete: progress %+.3f >= %+.3f held %.1fs", p,
                        stage.complete_threshold, stage.complete_sustain_s)
            return True
        return False

    def _run_chain(self, ctx: RolloutContext) -> None:
        """Auto-advance: run from the selected stage to the last one. A
        stage ends on its sustained progress threshold or its duration cap
        (timeout = assumed complete); either advances. 's' stops the whole
        chain, ESC quits the session."""
        import time

        while not ctx.runtime.shutdown_event.is_set():
            stage = self._stages[self._active]
            self._complete_since = None
            why = self._run_inference(ctx)
            progress = getattr(stage.engine, "last_progress", None)
            logger.info("<<< %s [%s] ended (%s)%s", stage.name, stage.key, why,
                        f" — progress {progress:+.3f}" if progress is not None else "")
            if why in ("stop key", "quit key", "shutdown"):
                break
            if self._active + 1 >= len(self._stages):
                logger.info("=== chain complete (all %d stages) ===",
                            len(self._stages))
                end_pose = self.config.chain_end_pose
                if end_pose:
                    if end_pose in self._poses:
                        self._goto_pose(ctx.hardware.robot_wrapper, end_pose)
                    else:
                        logger.warning("chain_end_pose %r not in poses file",
                                       end_pose)
                self._select(0)   # next chain starts from stage 1
                break
            time.sleep(self.config.advance_settle_s)
            self._select(self._active + 1)
            logger.info("auto-advancing...")

    def run(self, ctx: RolloutContext) -> None:
        import time

        robot = ctx.hardware.robot_wrapper
        while not ctx.runtime.shutdown_event.is_set():
            k = self._keys.popleft() if self._keys else None
            if k in self.QUIT_KEYS:
                logger.info("quit requested")
                break
            elif k and k.isdigit() and 1 <= (10 if k == "0" else int(k)) <= len(self._pose_names):
                name = self._pose_names[(10 if k == "0" else int(k)) - 1]
                self._goto_pose(robot, name)
                self._announce()
            elif k is not None and k in MODEL_KEYS and MODEL_KEYS.index(k) < len(self._stages):
                self._select(MODEL_KEYS.index(k))
            elif k == "g":
                if self.config.auto_advance:
                    self._run_chain(ctx)
                else:
                    self._run_inference(ctx)
                    s = self._stages[self._active]
                    progress = getattr(s.engine, "last_progress", None)
                    logger.info("<<< %s [%s] window ended%s", s.name, s.key,
                                f" — progress {progress:+.3f}" if progress is not None else "")
            elif k == "o":
                self._open_gripper(robot)
            else:
                time.sleep(0.05)


def _build_stage_engine(ctx: RolloutContext, stage: dict) -> InferenceEngine:
    """Build (policy -> processors -> engine) for a non-primary stage,
    sharing the robot wrapper / features / action keys of the rollout
    context. Mirrors steps 1, 6 and 7 of context.build_rollout_context —
    keep in sync with it (single-policy source of truth)."""
    cfg = ctx.runtime.cfg

    pol = dict(stage.get("policy") or {})
    path = pol.pop("path", None) or pol.pop("pretrained_path", None)
    if not path:
        raise ValueError(f"stage {stage.get('name')}: policy.path required")
    pcfg = PreTrainedConfig.from_pretrained(path)
    for k, v in pol.items():
        if not hasattr(pcfg, k):
            raise ValueError(f"stage {stage.get('name')}: unknown policy key {k!r}")
        setattr(pcfg, k, v)
    if hasattr(pcfg, "compile_model"):
        pcfg.compile_model = cfg.use_torch_compile

    inf = dict(stage.get("inference") or {})
    inf_type = inf.pop("type", "rtc")
    if inf_type == "rtc":
        icfg = RTCInferenceConfig(
            rtc=RTCConfig(**(inf.pop("rtc", None) or {})),
            queue_threshold=int(inf.pop("queue_threshold", 30)),
            action_adapter=inf.pop("action_adapter", None),
        )
    elif inf_type == "sync":
        icfg = SyncInferenceConfig(action_adapter=inf.pop("action_adapter", None))
    else:
        raise ValueError(f"stage {stage.get('name')}: unknown inference type {inf_type!r}")
    if inf:
        raise ValueError(f"stage {stage.get('name')}: unknown inference keys {sorted(inf)}")

    policy_class = get_policy_class(pcfg.type)
    policy = policy_class.from_pretrained(path, config=pcfg)
    if isinstance(icfg, RTCInferenceConfig):
        policy.config.rtc_config = icfg.rtc
        if hasattr(policy, "init_rtc_processor"):
            policy.init_rtc_processor()
    policy = policy.to(cfg.device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=pcfg,
        pretrained_path=path,
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    # camera crop/resize to the policy's input resolution (same rule as the
    # primary path in context.py)
    visual_hw = {k: tuple(ft.shape[-2:])
                 for k, ft in (pcfg.input_features or {}).items()
                 if ft.type is FeatureType.VISUAL}
    if visual_hw and cfg.robot is not None and getattr(cfg.robot, "cameras", None):
        camera_hw = {f"observation.images.{n}": (c.height, c.width)
                     for n, c in cfg.robot.cameras.items()}
        mismatched = {k: camera_hw[k] for k, hw in visual_hw.items()
                      if k in camera_hw and camera_hw[k] != hw}
        crop_params = {f"observation.images.{n}": tuple(b)
                       for n, b in (getattr(cfg, "image_crops", None) or {}).items()}
        if mismatched or crop_params:
            sizes = set(visual_hw.values())
            if len(sizes) != 1:
                raise ValueError(f"stage {stage.get('name')}: mixed policy "
                                 f"input sizes {sizes}")
            preprocessor.steps = [
                ImageCropResizeProcessorStep(crop_params_dict=crop_params or None,
                                             resize_size=next(iter(sizes))),
                *preprocessor.steps,
            ]

    return create_inference_engine(
        icfg,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=ctx.hardware.robot_wrapper,
        hw_features=ctx.data.hw_features,
        dataset_features=ctx.data.dataset_features,
        ordered_action_keys=ctx.data.ordered_action_keys,
        task=stage.get("task", cfg.task),
        fps=cfg.fps,
        device=cfg.device,
        use_torch_compile=cfg.use_torch_compile,
        compile_warmup_inferences=cfg.compile_warmup_inferences,
        shutdown_event=ctx.runtime.shutdown_event,
    )
