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

"""Action queue management for Real-Time Chunking (RTC).

This module provides ActionQueue, a thread-safe queue for managing action chunks
in real-time control scenarios. It supports both RTC-enabled and non-RTC modes,
handling action merging and leftover tracking.
"""

import logging
from threading import Lock

import torch
from torch import Tensor

from .configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control.

    This queue handles two types of action sequences:
    - Original actions: Used for RTC to compute leftovers from previous chunks
    - Processed actions: Post-processed actions ready for robot execution

    The queue operates in two modes:
    1. RTC-enabled: Replaces the entire queue with new actions, accounting for inference delay
    2. RTC-disabled: Appends new actions to the queue, maintaining continuity

    Args:
        cfg (RTCConfig): Configuration for Real-Time Chunking behavior.

    Attributes:
        queue (Tensor | None): Processed actions for robot rollout (time_steps, action_dim).
        original_queue (Tensor | None): Original actions for RTC computation (time_steps, action_dim).
        last_index (int): Current consumption index in the queue.
    """

    def __init__(self, cfg: RTCConfig):
        """Initialize the action queue.

        Args:
            cfg: RTC configuration controlling queue behavior.
        """
        self.queue = None  # Processed actions for robot rollout
        self.original_queue = None  # Original actions for RTC
        # Pre-adapter processed actions (absolute, policy feature space).
        # Only populated when an action adapter converts chunks to robot
        # space: the adapter output in `queue` can't be re-anchored for the
        # RTC prev-chunk prefix, this stream can.
        self.policy_queue = None
        self.lock = Lock()
        self.last_index = 0
        # provenance for timing audits: which plan, which row, born when
        self.chunk_seq = 0
        self._skip = 0
        self._birth_t = None
        self.last_info = None      # (chunk_seq, plan_row, birth_t) of last get()
        self.cfg = cfg
        # Optional blend anchor for the FIRST chunk of a run: the robot's
        # pose at inference start (processed/robot action space). Later
        # chunks blend from the previous queue instead.
        self.seed_action: Tensor | None = None

    def get(self) -> Tensor | None:
        """Get the next action from the queue.

        Returns:
            Tensor | None: The next action (action_dim,) or None if queue is empty.
                          Returns a clone to prevent external modifications.
        """
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            action = self.queue[self.last_index]
            self.last_info = (self.chunk_seq, self._skip + self.last_index, self._birth_t)
            self.last_index += 1
            return action.clone()

    def clear(self) -> None:
        """Clear queued actions and reset consumption index."""
        with self.lock:
            self.queue = None
            self.original_queue = None
            self.policy_queue = None
            self.last_index = 0
            self.seed_action = None

    def set_seed(self, action: Tensor) -> None:
        """Anchor the first chunk's cross-fade at this action (robot pose)."""
        with self.lock:
            self.seed_action = action

    def qsize(self) -> int:
        """Get the number of remaining actions in the queue.

        Returns:
            int: Number of unconsumed actions.
        """
        with self.lock:
            if self.queue is None:
                return 0
            return len(self.queue) - self.last_index

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if no actions remain, False otherwise.
        """
        with self.lock:
            if self.queue is None:
                return True
            return len(self.queue) - self.last_index <= 0

    def get_action_index(self) -> int:
        """Get the current action consumption index.

        Returns:
            int: Index of the next action to be consumed.
        """
        with self.lock:
            return self.last_index

    def get_left_over(self) -> Tensor | None:
        """Get leftover original actions for RTC prev_chunk_left_over.

        These are the unconsumed actions from the current chunk, which will be
        used by RTC to compute corrections for the next chunk.

        Returns:
            Tensor | None: Remaining original actions (remaining_steps, action_dim),
                          or None if no original queue exists.
        """
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :].clone()

    def get_processed_left_over(self) -> Tensor | None:
        """Get leftover processed actions (the actions currently executed by the robot).

        Returns:
            Tensor | None: Remaining processed actions (remaining_steps, action_dim),
                or None if no processed queue exists.
        """
        with self.lock:
            if self.queue is None:
                return None
            return self.queue[self.last_index :].clone()

    def get_policy_left_over(self) -> Tensor | None:
        """Get leftover pre-adapter processed actions (absolute, policy
        feature space) — the reanchorable equivalent of
        ``get_processed_left_over`` when an action adapter is in use.

        Returns:
            Tensor | None: Remaining pre-adapter actions
                (remaining_steps, action_dim), or None if not tracked.
        """
        with self.lock:
            if self.policy_queue is None:
                return None
            return self.policy_queue[self.last_index :].clone()

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = None,
        policy_actions: Tensor | None = None,
        birth_t: float | None = None,
    ):
        """Merge new actions into the queue.

        This method operates differently based on RTC mode:
        - RTC enabled: Replaces the queue, accounting for inference delay
        - RTC disabled: Appends to the queue, maintaining continuity

        Args:
            original_actions: Unprocessed actions from policy (time_steps, action_dim).
            processed_actions: Post-processed actions for robot (time_steps, action_dim).
            real_delay: Number of time steps of inference delay.
            action_index_before_inference: Index before inference started, for validation.
            policy_actions: Pre-adapter processed actions (absolute, policy
                feature space); pass when an action adapter converted
                ``processed_actions`` to robot space, so RTC reanchoring has
                a policy-space tail to work from.
        """
        with self.lock:
            if self.cfg.enabled:
                delay = self._check_and_resolve_delays(real_delay, action_index_before_inference)
                if self.queue is None:
                    # FIRST chunk of a window: the robot was HOLDING while
                    # inference ran, so no plan time actually elapsed — the
                    # plan starts at the held pose and row 0 is the correct
                    # first action. Skipping rows here jumps the arm to a
                    # pose the trajectory only reaches delay*tick later
                    # (the "first inference is always the worst" lunge).
                    delay = 0
                self.chunk_seq += 1
                self._birth_t = birth_t
                self._skip = max(0, min(delay, len(original_actions), len(processed_actions)))
                self._replace_actions_queue(
                    original_actions, processed_actions, delay, policy_actions
                )
                return

            # Append mode: chunks queue back-to-back; the delay bookkeeping
            # (and its mismatch warning) only applies to replace mode.
            self._append_actions_queue(original_actions, processed_actions, policy_actions)

    def _replace_actions_queue(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        policy_actions: Tensor | None = None,
    ):
        """Replace the queue with new actions (RTC mode).

        Discards the first `real_delay` actions since they correspond to the time
        spent during inference, when the robot was executing previous actions.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
            real_delay: Number of time steps to skip due to inference delay.
        """
        clamped_delay = max(0, min(real_delay, len(original_actions), len(processed_actions)))
        new_original = original_actions[clamped_delay:].clone()
        new_processed = processed_actions[clamped_delay:].clone()
        # Keep the policy-space stream index-aligned with the served queue;
        # it stays UNBLENDED — it feeds reanchoring, which needs the policy's
        # actual plan, not the served cross-fade.
        self.policy_queue = (policy_actions[clamped_delay:].clone()
                             if policy_actions is not None else None)

        blend = self.cfg.splice_blend_steps
        if blend > 0 and self.queue is None and self.seed_action is not None:
            # First chunk of a run: blend out of the robot's held pose (the
            # chunk may start noticeably away from it — warmup latency drops
            # its first actions). Processed space only.
            n = min(blend, len(new_processed))
            seed = self.seed_action.to(new_processed.dtype)
            for i in range(n):
                a = (i + 1) / (n + 1)
                new_processed[i] = (1 - a) * seed + a * new_processed[i]
            self.seed_action = None
        elif blend > 0 and self.queue is not None:
            # Cross-fade from the old chunk's remaining (time-aligned) actions
            # into the new chunk so a trajectory-mode switch cannot produce a
            # step discontinuity in the served command stream.
            #
            # The fade must NOT shrink with the old-queue remainder: by merge
            # time the replan has usually consumed the queue to ~0 rows, so a
            # remainder-limited fade silently collapses to nothing and every
            # splice becomes a step discontinuity (observed 12-19 deg/tick).
            # Where the old plan has no row left, anchor on its last row —
            # falling back to the last SERVED command (queue[last_index - 1])
            # when the old queue is fully consumed.
            old_processed = self.queue[self.last_index:]
            old_original = (self.original_queue[self.last_index:]
                            if self.original_queue is not None else old_processed)
            if len(old_processed) == 0 and self.last_index > 0:
                old_processed = self.queue[self.last_index - 1:self.last_index]
            n = min(blend, len(new_processed)) if len(old_processed) > 0 else 0
            if n > 0:
                alphas = torch.linspace(1.0 / (n + 1), n / (n + 1), n,
                                        dtype=new_processed.dtype)
                for i in range(n):
                    a = alphas[i]
                    src = old_processed[min(i, len(old_processed) - 1)]
                    new_processed[i] = (1 - a) * src.to(new_processed.dtype) + a * new_processed[i]
                    if i < len(old_original):
                        new_original[i] = (1 - a) * old_original[i].to(new_original.dtype) + a * new_original[i]

        self.original_queue = new_original
        self.queue = new_processed

        logger.debug(f"original_actions shape: {self.original_queue.shape}")
        logger.debug(f"processed_actions shape: {self.queue.shape}")
        logger.debug(f"real_delay: {real_delay}, clamped_delay: {clamped_delay}")

        self.last_index = 0

    def _append_actions_queue(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        policy_actions: Tensor | None = None,
    ):
        """Append new actions to the queue (non-RTC mode).

        Removes already-consumed actions and appends new ones, maintaining
        queue continuity without replacement.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
            policy_actions: Pre-adapter processed actions (see ``merge``).
        """
        if self.queue is None:
            self.original_queue = original_actions.clone()
            self.queue = processed_actions.clone()
            self.policy_queue = (policy_actions.clone()
                                 if policy_actions is not None else None)
            return

        self.original_queue = torch.cat([self.original_queue, original_actions.clone()])
        self.original_queue = self.original_queue[self.last_index :]

        self.queue = torch.cat([self.queue, processed_actions.clone()])
        self.queue = self.queue[self.last_index :]

        if self.policy_queue is not None and policy_actions is not None:
            self.policy_queue = torch.cat([self.policy_queue, policy_actions.clone()])
            self.policy_queue = self.policy_queue[self.last_index :]
        else:
            # An adapter either feeds this stream every merge or never;
            # a mixed sequence can't stay index-aligned, so drop it.
            self.policy_queue = None

        self.last_index = 0

    def _check_and_resolve_delays(
        self, real_delay: int, action_index_before_inference: int | None = None
    ) -> int:
        """Validate that computed delays match expectations.

        Compares the delay computed from inference latency with the actual
        number of actions consumed during inference.

        Args:
            real_delay: Delay computed from inference latency.
            action_index_before_inference: Action index when inference started.

        Returns:
            int: Delay to use.
        """
        effective_delay = max(0, real_delay)

        if action_index_before_inference is not None:
            indexes_diff = max(0, self.last_index - action_index_before_inference)
            if indexes_diff != real_delay:
                logger.debug(
                    "Indexes diff is not equal to real delay. indexes_diff=%d, real_delay=%d",
                    indexes_diff,
                    real_delay,
                )
                return real_delay

        return effective_delay
