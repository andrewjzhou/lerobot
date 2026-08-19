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
'o' while idle partially opens the gripper (to free a piece held at stop
time), 'q'/ESC ends the session. ``duration`` acts as a per-inference-run
cap (0 = until 's'). No data recording.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from lerobot.common.control_utils import follower_smooth_move_to, move_robot_to_named_pose
from lerobot.utils.robot_utils import precise_sleep

from ..context import RolloutContext
from .base import BaseStrategy
from .core import send_next_action

logger = logging.getLogger(__name__)

MAX_POSES = 10  # keys 1..9 and 0 (=10th)


class InteractiveStrategy(BaseStrategy):
    """Keyed pose moves + operator-gated inference (no recording)."""

    # Keys that end the session; ESC arrives as the "\x1b" sentinel.
    # StagedStrategy narrows this to ESC-only so 'q' can select a model.
    QUIT_KEYS = ("q", "\x1b")

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

        # Sampled-start bank: one extra number key after the named poses;
        # each press draws a FRESH pre-IK'd joint pose from the bank.
        self._bank = None
        self._bank_name = ""
        if getattr(cfg, "sample_pose_bank", ""):
            bank = json.loads(Path(cfg.sample_pose_bank).read_text())
            if not bank.get("entries"):
                raise ValueError(f"{cfg.sample_pose_bank}: no entries")
            if len(self._pose_names) + 1 > MAX_POSES:
                raise ValueError("no number key left for the sample bank "
                                 f"({len(self._pose_names)} poses already)")
            self._bank = bank["entries"]
            self._bank_name = bank.get("name", Path(cfg.sample_pose_bank).stem)
            self._bank_rng = np.random.default_rng()   # fresh draw each press
            logger.info("sampled-start bank '%s': %d poses on key %d",
                        self._bank_name, len(self._bank), len(self._pose_names) + 1)

        from pynput import keyboard

        self._keys: deque[str] = deque()

        def on_press(key):
            ch = getattr(key, "char", None)
            if key == keyboard.Key.esc:
                ch = "\x1b"
            if ch:
                self._keys.append(ch.lower())

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()

        # pynput taps key events at the window system, but every keystroke
        # ALSO lands in the terminal's stdin buffer, which nothing reads:
        # keys echo into the log lines during the session and the shell
        # receives the whole backlog when the program exits. Take the tty
        # out of echo/canonical mode for the session; teardown flushes the
        # buffer and restores the original attributes.
        self._tty_attrs = None
        try:
            import sys as _sys
            import termios as _termios
            if _sys.stdin.isatty():
                fd = _sys.stdin.fileno()
                self._tty_attrs = _termios.tcgetattr(fd)
                quiet = _termios.tcgetattr(fd)
                quiet[3] &= ~(_termios.ECHO | _termios.ICANON)
                _termios.tcsetattr(fd, _termios.TCSADRAIN, quiet)
        except Exception:
            self._tty_attrs = None
        self._run_idx = 0
        self._last_pose = "startup"
        self._session_dir = None
        if cfg.log_dir:
            self._session_dir = Path(cfg.log_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
            self._session_dir.mkdir(parents=True, exist_ok=True)
            logger.info("debug logs -> %s", self._session_dir)
        keymap = " | ".join(f"{i + 1}={n}" for i, n in enumerate(self._pose_names))
        if self._bank is not None:
            keymap += f" | {len(self._pose_names) + 1}={self._bank_name}(random)"
        quit_label = "q/ESC" if "q" in self.QUIT_KEYS else "ESC"
        extras = ""
        if getattr(cfg, "progress_stop_threshold", 0.0) > 0:
            extras = (f" | auto-stop at progress >= "
                      f"{cfg.progress_stop_threshold:.0%}")
        logger.info("Interactive strategy ready — %s | g=start inference | "
                    "s=stop | o=open gripper %.0f%% | %s=quit%s",
                    keymap, cfg.open_gripper_fraction * 100, quit_label, extras)

    def run(self, ctx: RolloutContext) -> None:
        robot = ctx.hardware.robot_wrapper
        while not ctx.runtime.shutdown_event.is_set():
            k = self._keys.popleft() if self._keys else None
            if k in self.QUIT_KEYS:
                logger.info("quit requested")
                break
            elif k and k.isdigit() and 1 <= (10 if k == "0" else int(k)) <= len(self._pose_names):
                name = self._pose_names[(10 if k == "0" else int(k)) - 1]
                self._goto_pose(robot, name)
            elif (k and k.isdigit() and self._bank is not None
                  and (10 if k == "0" else int(k)) == len(self._pose_names) + 1):
                self._goto_sampled_pose(robot)
            elif k == "g":
                self._run_inference(ctx)
            elif k == "o":
                self._open_gripper(robot)
            else:
                time.sleep(0.05)

    def _open_gripper(self, robot) -> None:
        """'o' while idle: partially open the gripper so a piece gripped at
        stop time can be pried out. Bimanual robots open the LEFT gripper
        (the tool that actually grips). Opens only — if the jaw is already
        past the target it stays put. All other joints hold position."""
        if hasattr(robot, "get_pos_observation"):
            obs = dict(robot.get_pos_observation())
        else:
            obs = {k: v for k, v in robot.get_observation().items()
                   if k.endswith(".pos")}
        grips = sorted(k for k in obs if "gripper" in k and k.endswith(".pos"))
        key = next((k for k in grips if k.startswith("left_")),
                   grips[0] if grips else None)
        if key is None:
            logger.warning("'o': no gripper joint in observation")
            return
        # OpenArm v2 mirrored grippers: left opens toward +65, right -65.
        open_deg = -65.0 if key.startswith("right_") else 65.0
        frac = self.config.open_gripper_fraction
        target_deg = frac * open_deg
        cur = float(obs[key])
        if (target_deg - cur) * (1.0 if open_deg > 0 else -1.0) <= 0:
            logger.info("'o': %s already at %+.1f deg (>= %.0f%% open) — "
                        "leaving it", key, cur, frac * 100)
            return
        logger.info("opening %s to %.0f%% (%+.1f deg)...", key, frac * 100,
                    target_deg)
        target = dict(obs)
        target[key] = target_deg
        follower_smooth_move_to(robot, obs, target, duration_s=1.0, fps=30)
        logger.info("gripper opened — number keys to repose, g to rerun")

    # Worst-case EE lever arm for translating pose_speed_m_s into a joint
    # sweep duration (arm at full reach; shorter postures move slower than
    # the configured EE speed, which errs safe).
    POSE_LEVER_ARM_M = 0.6

    def _pose_duration(self, current: dict, target: dict) -> float:
        cfg = self.config
        if cfg.pose_speed_m_s <= 0:
            return cfg.move_duration_s
        deltas = [abs(target[k] - current[k]) for k in target
                  if k in current and not k.endswith("gripper.pos")]
        max_rad = math.radians(max(deltas, default=0.0))
        dur = max_rad * self.POSE_LEVER_ARM_M / cfg.pose_speed_m_s
        return min(max(dur, cfg.pose_min_duration_s), cfg.move_duration_s)

    def _spline_abort_check(self) -> bool:
        """Polled during pose moves: 's' aborts (robot holds); quit keys
        abort AND stay queued so the outer loop exits. Other keys typed
        mid-move are discarded — they were not meant for the new pose."""
        while self._keys:
            k = self._keys.popleft()
            if k == "s":
                return True
            if k in self.QUIT_KEYS:
                self._keys.appendleft(k)
                return True
        return False

    def _goto_pose(self, robot, name: str) -> bool:
        """Speed-scaled, operator-stoppable move to a named pose.
        Returns True if it was aborted."""
        logger.info("moving to pose '%s'...", name)
        aborted = move_robot_to_named_pose(
            robot, self._poses, name, fps=30, side=self._side,
            abort_check=self._spline_abort_check,
            duration_fn=self._pose_duration)
        if aborted:
            logger.info("pose move to '%s' STOPPED — holding position", name)
        else:
            self._last_pose = name
            logger.info("at pose '%s'", name)
        return aborted

    def _goto_sampled_pose(self, robot) -> None:
        """Draw a fresh pose from the sample bank and move there. The drawn
        index goes into _last_pose so the run's meta.json records exactly
        which start was used (good/bad starts stay attributable)."""
        i = int(self._bank_rng.integers(len(self._bank)))
        entry = self._bank[i]
        tag = f"{self._bank_name}#{i}"
        self._poses[tag] = {self._side or "right": entry["joints"]}
        logger.info("sampled start %s: %s", tag,
                    {k: round(v, 1) for k, v in entry["joints"].items()})
        self._goto_pose(robot, tag)

    def _inference_duration(self, cfg) -> float:
        """Per-run cap in seconds; staged mode returns the active stage's."""
        return cfg.duration

    def _window_complete(self) -> bool:
        """Hook polled every control tick: True ends the window (reason from
        _complete_reason). Staged auto-advance overrides this wholesale;
        here it implements the progress-head auto-stop when
        progress_stop_threshold is set."""
        thr = getattr(self.config, "progress_stop_threshold", 0.0)
        if thr <= 0:
            return False
        p = getattr(self._engine, "last_progress", None)
        if p is None:                      # no progress head / no replan yet
            return False
        fraction = 1.0 + p                 # last_progress in [-1, 0]
        if fraction >= thr:
            self._complete_reason = (
                f"progress {fraction:.0%} >= {thr:.0%} (readout {p:+.3f})")
            return True
        return False

    def _run_inference(self, ctx: RolloutContext) -> str:
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator
        control_interval = interpolator.get_control_interval(cfg.fps)
        # A stale _prev from the previous inference window would make the
        # first ramp interpolate from wherever the arm USED to be ->
        # violent lunge at window start (observed 23-44 deg, 2026-08-14).
        interpolator.reset()  # stale _prev guard


        self._keys.clear()
        self._complete_reason = None
        engine.reset()
        engine.resume()
        self._cached_obs_processed = None
        log = None
        if self._session_dir is not None:
            self._run_idx += 1
            log = {"t": [], "state": [], "action": [], "dt": [],
                   "frames": [], "state_keys": None, "action_keys": None,
                   "dir": self._session_dir / f"run{self._run_idx:02d}"}
        duration = self._inference_duration(cfg)
        logger.info("inference STARTED — 's' to stop%s",
                    f" (auto-stop after {duration:.0f}s)" if duration > 0 else "")
        start = time.perf_counter()
        why = "shutdown"
        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()
            if self._keys:
                k = self._keys.popleft()
                if k == "s":
                    why = "stop key"
                    break
                if k in self.QUIT_KEYS:
                    why = "quit key"
                    ctx.runtime.shutdown_event.set()
                    break
            if duration > 0 and (time.perf_counter() - start) >= duration:
                why = f"duration cap {duration:.0f}s"
                break
            if self._window_complete():
                why = getattr(self, "_complete_reason", None) or "stage complete"
                break

            obs = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator,
                                           engine=self._engine)
            self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            dt = time.perf_counter() - loop_start
            if log is not None and action_dict:
                self._log_step(log, loop_start - start, obs, action_dict, dt, loop_start)
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
        engine.pause()
        if log is not None:
            self._flush_log(log, why)
        logger.info("inference STOPPED (%s) — number keys to reposition, g to rerun", why)
        return why

    def _log_step(self, log, t, obs, action_dict, dt, t_abs=float('nan')):
        if log["state_keys"] is None:
            log["state_keys"] = sorted(k for k in obs if k.endswith((".pos", ".torque")))
            log["action_keys"] = sorted(action_dict) if action_dict else []
        log["t"].append(t)
        log["dt"].append(dt)
        info = getattr(self._engine, "last_action_info", None)
        log.setdefault("plan", []).append(
            (info[0], info[1], info[2]) if info else (-1, -1, float("nan")))
        log.setdefault("t_abs", []).append(t_abs)
        ph = getattr(self._interpolator, "phase", (0, 1))
        log.setdefault("interp", []).append(ph)
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
                 state=np.array(log["state"]), action=np.array(log["action"]),
                 plan=np.array(log.get("plan", []), dtype=np.float64),
                 t_abs=np.array(log.get("t_abs", []), dtype=np.float64),
                 interp=np.array(log.get("interp", []), dtype=np.int64))
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
        try:
            if getattr(self, "_tty_attrs", None) is not None:
                import sys as _sys
                import termios as _termios
                fd = _sys.stdin.fileno()
                _termios.tcflush(fd, _termios.TCIFLUSH)   # drop typed backlog
                _termios.tcsetattr(fd, _termios.TCSADRAIN, self._tty_attrs)
        except Exception:
            pass
        super().teardown(ctx)
