"""Command-frame anchoring (RTCConfig.anchor_on_command, 2026-08-22).

The engine swaps the last COMMANDED joint targets into the observation it
feeds the policy, so plans are anchored in the same frame the queue splices
in. These tests pin the swap helper's contract; the config default must stay
False (opt-in per task card — the tradeoff is documented there).
"""

import numpy as np

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout.inference.rtc import _swap_command_frame


def test_swap_replaces_only_matching_keys():
    img = np.zeros((4, 4, 3), np.uint8)
    obs = {"joint_1.pos": 10.0, "joint_2.pos": 20.0, "joint_1.vel": 5.0,
           "head": img, "task": "t"}
    cmd = {"joint_1.pos": 11.5, "joint_2.pos": 21.5, "gripper.pos": 3.0}
    out = _swap_command_frame(obs, cmd)
    assert out["joint_1.pos"] == 11.5 and out["joint_2.pos"] == 21.5
    assert out["joint_1.vel"] == 5.0, "non-command keys untouched"
    assert "gripper.pos" not in out, "command keys absent from obs are not added"
    assert out["head"] is img, "camera arrays shared by reference, not copied"
    assert obs["joint_1.pos"] == 10.0, "original observation dict untouched"


def test_swap_is_noop_without_command():
    obs = {"joint_1.pos": 10.0}
    assert _swap_command_frame(obs, None) is obs
    assert _swap_command_frame(obs, {}) is obs


def test_anchor_on_command_defaults_off():
    assert RTCConfig().anchor_on_command is False
