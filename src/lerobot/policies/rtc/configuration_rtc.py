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

"""
Real Time Chunking (RTC) and Bidirectional Decoding (BID) configuration classes.

Based on:
- Real Time Chunking: https://www.physicalintelligence.company/research/real_time_chunking
"""

from dataclasses import dataclass

from lerobot.configs import RTCAttentionSchedule


@dataclass
class RTCConfig:
    """Configuration for Real Time Chunking (RTC) inference.

    RTC improves real-time inference by treating chunk generation as an inpainting problem,
    strategically handling overlapping timesteps between action chunks using prefix attention.
    """

    # Infrastructure
    enabled: bool = True

    # Actuation-latency matching: skip this many EXTRA rows at every chunk
    # merge so the action issued at wall time t targets the pose intended
    # for t + N ticks — the plant (PD lag, measured ~86 ms on the OpenArm)
    # then lands ON the demo trajectory instead of trailing it. Demos have
    # zero tracking lag (the human hand is the plant), so this is the one
    # train/deploy timing term not matched by construction. 0 = off.
    execution_latency_ticks: int = 0

    # Anchor inference on the COMMANDED pose instead of the measured one:
    # the engine swaps the last sent joint targets into the observation it
    # feeds the policy (2026-08-22). WHY: plans are se3-anchored on the obs
    # state but the queue splices in command space; during motion the two
    # frames differ by speed x ~1 tick (measured 10-20 mm at 166-308 mm/s),
    # injected as a backward step at every splice. Training (hand-held UMI
    # rig) had cmd == meas, so the command frame IS the training-time
    # meaning of the state channel. DELIBERATE TRADEOFF (Andrew, 2026-08-22):
    # the policy becomes blind to cmd/meas divergence — a stalled or
    # obstructed arm looks like perfect tracking. Revisit for contact tasks.
    anchor_on_command: bool = False

    # Core RTC settings
    # EXP = the RTC paper's schedule (weights sag exponentially right after
    # the frozen region); was LINEAR until 2026-08-22.
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.EXP
    # 5.0 = the reference implementation's eval default
    # (real-time-chunking-kinetix RealtimeMethodConfig); was 10.0 until
    # 2026-08-22.
    max_guidance_weight: float = 5.0
    execution_horizon: int = 10
    # Cross-fade this many steps from the old chunk's remaining actions into
    # each replacement chunk (0 = hard switch). For policies without
    # prefix-inpainting support (e.g. diffusion), this bounds the command
    # discontinuity when consecutive chunks pick different trajectory modes.
    splice_blend_steps: int = 0
    # HOW consecutive chunks are joined at a merge (read per merge -> live):
    #   "blend"   PLAN-TO-PLAN cross-fade: the first splice_blend_steps served
    #             rows mix the old chunk's remaining rows with the new chunk's
    #             (ACT-temporal-ensemble-flavoured). On a mode flip the mix is
    #             a trajectory belonging to NEITHER plan.
    #   "replace" UMI-style (Chi et al. eval_real): the old plan's future is
    #             DISCARDED; the first splice_blend_steps rows ramp from the
    #             LAST SERVED COMMAND (a point, not a plan) onto the new plan.
    #             Commits to the new mode immediately; worst case is a bounded
    #             ramp, never an average of two modes.
    #   "none"    hard switch: new chunk served as-is (raw receding horizon;
    #             also the right setting when the POLICY already guarantees
    #             continuity, e.g. RTC prefix guidance).
    splice_mode: str = "blend"

    # Debug settings
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self):
        """Validate RTC configuration parameters."""
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")
        if self.splice_mode not in ("blend", "replace", "none"):
            raise ValueError(f"splice_mode must be blend|replace|none, got {self.splice_mode!r}")
