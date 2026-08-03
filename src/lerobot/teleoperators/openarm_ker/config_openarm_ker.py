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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("openarm_ker")
@dataclass
class OpenArmKerConfig(TeleoperatorConfig):
    """OpenArm KER (enactic) — motorless bimanual leader arm.

    ONE USB device streams both arms: 16 channels of magnetic-encoder
    angles in DEGREES, already in the OpenArm joint frame (signs and
    offsets applied in firmware; zero calibration lives on the device —
    jig + touchscreen — so there are no host-side calibration files).
    Channel layout: 0-6 right joint_1..7, 7 right trigger, 8-14 left
    joint_1..7, 15 left trigger.
    """

    # KERStream transport: "usb" = vendor mode via libusb (needs the
    # idVendor 303a udev rule); "serial" = USB-CDC.
    transport: str = "usb"
    # Serial transport only.
    port: str = "/dev/ttyACM0"
    baud: int = 2_000_000
    usb_vid: int = 0x303A
    usb_pid: int = 0x4002

    # None -> bimanual: emits left_/right_-prefixed action keys.
    # "left"/"right" -> that arm only, unprefixed keys (the --side path).
    side: str | None = None

    # Trigger raw travel (released, squeezed) in degrees — the dora
    # reference node's values; tune per unit if the physical travel differs.
    right_trigger_range: tuple[float, float] = (0.0, -60.0)
    left_trigger_range: tuple[float, float] = (0.0, 60.0)
    # Follower gripper command at (released, squeezed), degrees. MUST stay
    # inside the follower's limits (right (-65, 0), left (0, +65)) — a
    # gripper commanded past its end stop stalls and sags the 24 V rail.
    # Output is clipped to this range regardless of trigger overtravel.
    right_gripper_range: tuple[float, float] = (-65.0, 0.0)
    left_gripper_range: tuple[float, float] = (65.0, 0.0)

    # Seconds to wait for the first stream packet at connect (the operator
    # must press START on the KER touchscreen).
    connect_timeout_s: float = 10.0
