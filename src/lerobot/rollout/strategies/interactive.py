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

"""Interactive rollout strategy: keyed pose moves + gated inference.

Number keys 1..N smoothly move the follower to named poses (from a poses
yaml), 'g' starts policy inference, 's' stops it (robot holds position),
'q'/ESC ends the session. ``duration`` acts as a per-inference-run cap
(0 = until 's'). No data recording.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from lerobot.common.control_utils import move_robot_to_named_pose
from lerobot.utils.robot_utils import precise_sleep

from ..context import RolloutContext
from .base import BaseStrategy
from .core import send_next_action

logger = logging.getLogger(__name__)

MAX_POSES = 5


class InteractiveStrategy(BaseStrategy):
    """Keyed pose moves + operator-gated inference (no recording)."""

    def setup(self, ctx: RolloutContext) -> None:
        super().setup(ctx)
        cfg = self.config
        self._pose_names = [p.strip() for p in cfg.poses.split(",") if p.strip()]
        if len(self._pose_names) > MAX_POSES:
            raise ValueError(f"at most {MAX_POSES} poses (got {self._pose_names})")
        self._poses = yaml.safe_load(open(cfg.poses_file))
        missing = [n for n in self._pose_names if n not in self._poses]
        if missing:
            raise ValueError(f"poses {missing} not found in {cfg.poses_file} "
                             f"(available: {sorted(self._poses)})")
        self._side = getattr(ctx.runtime.cfg.robot, "side", None)

        from pynput import keyboard

        self._keys: deque[str] = deque()

        def on_press(key):
            ch = getattr(key, "char", None)
            if key == keyboard.Key.esc:
                ch = "q"
            if ch:
                self._keys.append(ch.lower())

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()
        self._run_idx = 0
        self._last_pose = "startup"
        self._session_dir = None
        if cfg.log_dir:
            self._session_dir = Path(cfg.log_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
            self._session_dir.mkdir(parents=True, exist_ok=True)
            logger.info("debug logs -> %s", self._session_dir)
        keymap = " | ".join(f"{i + 1}={n}" for i, n in enumerate(self._pose_names))
        logger.info("Interactive strategy ready — %s | g=start inference | "
                    "s=stop | q/ESC=quit", keymap)

    def run(self, ctx: RolloutContext) -> None:
        robot = ctx.hardware.robot_wrapper
        while not ctx.runtime.shutdown_event.is_set():
            k = self._keys.popleft() if self._keys else None
            if k == "q":
                logger.info("quit requested")
                break
            elif k and k.isdigit() and 1 <= int(k) <= len(self._pose_names):
                name = self._pose_names[int(k) - 1]
                logger.info("moving to pose '%s'...", name)
                move_robot_to_named_pose(robot, self._poses, name,
                                         duration_s=self.config.move_duration_s,
                                         side=self._side)
                self._last_pose = name
                logger.info("at pose '%s' — g to start inference", name)
            elif k == "g":
                self._run_inference(ctx)
            else:
                time.sleep(0.05)

    def _run_inference(self, ctx: RolloutContext) -> None:
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator
        control_interval = interpolator.get_control_interval(cfg.fps)

        self._keys.clear()
        engine.reset()
        engine.resume()
        self._cached_obs_processed = None
        log = None
        if self._session_dir is not None:
            self._run_idx += 1
            log = {"t": [], "state": [], "action": [], "dt": [],
                   "frames": [], "state_keys": None, "action_keys": None,
                   "dir": self._session_dir / f"run{self._run_idx:02d}"}
        logger.info("inference STARTED — 's' to stop%s",
                    f" (auto-stop after {cfg.duration:.0f}s)" if cfg.duration > 0 else "")
        start = time.perf_counter()
        why = "shutdown"
        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()
            if self._keys:
                k = self._keys.popleft()
                if k == "s":
                    why = "stop key"
                    break
                if k == "q":
                    why = "quit key"
                    ctx.runtime.shutdown_event.set()
                    break
            if cfg.duration > 0 and (time.perf_counter() - start) >= cfg.duration:
                why = f"duration cap {cfg.duration:.0f}s"
                break

            obs = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)
            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if log is not None and action_dict:
                self._log_step(log, loop_start - start, obs, action_dict, dt)
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
        engine.pause()
        if log is not None:
            self._flush_log(log, why)
        logger.info("inference STOPPED (%s) — number keys to reposition, g to rerun", why)

    def _log_step(self, log, t, obs, action_dict, dt):
        if log["state_keys"] is None:
            log["state_keys"] = sorted(k for k in obs if k.endswith(".pos"))
            log["action_keys"] = sorted(action_dict) if action_dict else []
        log["t"].append(t)
        log["dt"].append(dt)
        log["state"].append([float(obs[k]) for k in log["state_keys"]])
        log["action"].append([float(action_dict.get(k, np.nan)) for k in log["action_keys"]])
        if len(log["t"]) % self.config.log_frame_stride == 0:
            import cv2

            for k, v in obs.items():
                if isinstance(v, np.ndarray) and v.ndim == 3:
                    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(v, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ok:
                        log["frames"].append((len(log["t"]) - 1, k, buf.tobytes()))

    def _flush_log(self, log, why):
        d = log["dir"]
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "trace.npz",
                 t=np.array(log["t"]), dt=np.array(log["dt"]),
                 state=np.array(log["state"]), action=np.array(log["action"]))
        for step, cam, buf in log["frames"]:
            (d / f"{cam}_{step:04d}.jpg").write_bytes(buf)
        (d / "meta.json").write_text(json.dumps({
            "start_pose": self._last_pose, "stopped": why,
            "steps": len(log["t"]),
            "state_keys": log["state_keys"], "action_keys": log["action_keys"],
            "frame_stride": self.config.log_frame_stride,
        }, indent=2))
        logger.info("debug log written: %s (%d steps, %d frames)",
                    d, len(log["t"]), len(log["frames"]))

    def teardown(self, ctx: RolloutContext) -> None:
        try:
            self._listener.stop()
        except Exception:
            pass
        super().teardown(ctx)
