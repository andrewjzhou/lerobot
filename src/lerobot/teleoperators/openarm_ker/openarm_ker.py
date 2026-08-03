#!/usr/bin/env python

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

import logging
import time
from typing import Any

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_openarm_ker import OpenArmKerConfig

logger = logging.getLogger(__name__)

# KERStream wire commands (openarm_ker SDK exposes send_command only).
_CMD_STANDBY = b"\x01"
_CMD_STREAM = b"\x02"

# angles[16] channel layout (firmware ENCODER_CONFIG order).
_JOINT_SLICE = {"right": slice(0, 7), "left": slice(8, 15)}
_TRIGGER_IDX = {"right": 7, "left": 15}
_JOINT_NAMES = [f"joint_{i}" for i in range(1, 8)]


def _map_range(x: float, in_a: float, in_b: float, out_a: float, out_b: float) -> float:
    """Linear map with output clipped to [min(out), max(out)] — trigger
    overtravel must never command the gripper past its end stop."""
    if in_b == in_a:
        return out_a
    y = out_a + (x - in_a) * (out_b - out_a) / (in_b - in_a)
    lo, hi = (out_a, out_b) if out_a <= out_b else (out_b, out_a)
    return min(max(y, lo), hi)


class OpenArmKer(Teleoperator):
    """OpenArm KER bimanual leader (magnetic encoders over one USB stream).

    Unlike the Mini (one serial bus per arm + Bi wrapper), the KER is a
    single device carrying both arms, so this one class serves both the
    bimanual and the single-arm (``side:``) configurations. Joints pass
    through 1:1 in degrees (firmware already applies signs/offsets);
    only the trigger is rescaled to the follower's gripper range.

    Zero calibration is on-device (jig + touchscreen, stored in ESP32
    NVS): ``calibrate()`` is a documented no-op.
    """

    config_class = OpenArmKerConfig
    name = "openarm_ker"

    def __init__(self, config: OpenArmKerConfig):
        super().__init__(config)
        self.config = config
        if config.side is not None and config.side not in ("left", "right"):
            raise ValueError(f"Invalid side '{config.side}'; expected 'left', 'right', or None.")
        self._sides = [config.side] if config.side else ["left", "right"]
        self._prefix = (lambda s, k: k) if config.side else (lambda s, k: f"{s}_{k}")
        self._trigger_range = {
            "right": tuple(config.right_trigger_range),
            "left": tuple(config.left_trigger_range),
        }
        self._gripper_range = {
            "right": tuple(config.right_gripper_range),
            "left": tuple(config.left_gripper_range),
        }
        self._stream = None
        # Last good raw angle per channel: channels flagged in the packet's
        # errors[] hold their previous value instead of passing garbage on.
        self._last_good: list[float] | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {
            self._prefix(side, f"{name}.pos"): float
            for side in self._sides
            for name in (*_JOINT_NAMES, "gripper")
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._stream is not None and self._stream.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            from openarm_ker import KERStream
        except ImportError as e:
            raise ImportError(
                "openarm_ker SDK is required for the KER teleoperator: "
                'pip install "openarm-ker @ git+https://github.com/enactic/openarm_ker"'
            ) from e

        logger.info(
            "Connecting OpenArm KER (transport=%s%s)...",
            self.config.transport,
            "" if self.config.transport == "usb" else f", port={self.config.port}",
        )
        self._stream = KERStream(
            transport=self.config.transport,
            port=self.config.port,
            baud=self.config.baud,
            vid=self.config.usb_vid,
            pid=self.config.usb_pid,
        )
        self._stream.connect()
        # The ping/schema handshake leaves the device in STANDBY.
        self._stream.send_command(_CMD_STREAM)

        deadline = time.perf_counter() + self.config.connect_timeout_s
        packet = None
        while time.perf_counter() < deadline:
            packet = self._stream.latest()
            if packet is not None and "angles" in packet:
                break
            time.sleep(0.05)
        if packet is None or "angles" not in packet:
            self._stream.close()
            self._stream = None
            raise RuntimeError(
                "OpenArm KER connected but no stream packets arrived within "
                f"{self.config.connect_timeout_s:.0f}s. Press START on the KER "
                "touchscreen; if the device is missing entirely, check the udev "
                "rule (idVendor 303a) / libusb, or try transport: serial."
            )
        self._last_good = list(packet["angles"])
        logger.info("%s connected (streaming, %d channels).", self, len(self._last_good))

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info(
            "OpenArm KER zero calibration is ON-DEVICE (calibration jig + "
            "'Zero Reset' on the M5 touchscreen; stored in device NVS). "
            "Nothing to do host-side."
        )

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        packet = self._stream.latest()
        if packet is not None and "angles" in packet:
            angles = packet["angles"]
            errors = packet.get("errors")
            for i, val in enumerate(angles):
                if errors is not None and i < len(errors) and errors[i]:
                    continue  # hold last good value for faulted channel
                self._last_good[i] = float(val)
        # else: stream hiccup — serve the previous values (latest() may
        # briefly return None right after a firmware jump-detect stop).

        action: RobotAction = {}
        for side in self._sides:
            joints = self._last_good[_JOINT_SLICE[side]]
            for name, val in zip(_JOINT_NAMES, joints, strict=True):
                action[self._prefix(side, f"{name}.pos")] = val
            trig = self._last_good[_TRIGGER_IDX[side]]
            action[self._prefix(side, "gripper.pos")] = _map_range(
                trig, *self._trigger_range[side], *self._gripper_range[side]
            )
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # passive device — no motors to drive

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self._stream.send_command(_CMD_STANDBY)
        except Exception:
            pass  # best-effort; device may already be unplugged
        self._stream.close()
        self._stream = None
        logger.info("%s disconnected.", self)
