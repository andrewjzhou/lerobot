"""Splice-mode semantics at chunk merges (blend / replace / none).

The property under test is WHOSE trajectory the served rows belong to:
  blend   -> a mix of old plan and new plan (plan-to-plan cross-fade)
  replace -> a ramp from the last SERVED command onto the new plan only
             (UMI semantics: the old plan's future is never mixed in)
  none    -> the new plan exactly (hard receding-horizon switch)
"""

import pytest
import torch

from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig


def _queue(mode: str, blend: int = 3) -> ActionQueue:
    return ActionQueue(RTCConfig(enabled=True, splice_blend_steps=blend,
                                 splice_mode=mode))


def _merge_two_chunks(q: ActionQueue, old_val: float, new_val: float,
                      serve: int = 4, rows: int = 10, dim: int = 2):
    """First chunk constant old_val; serve some rows; merge constant new_val."""
    old = torch.full((rows, dim), old_val)
    q.merge(old.clone(), old.clone(), real_delay=0)
    for _ in range(serve):
        q.get()
    new = torch.full((rows, dim), new_val)
    q.merge(new.clone(), new.clone(), real_delay=0)
    return q


def test_none_serves_new_plan_exactly():
    q = _merge_two_chunks(_queue("none"), old_val=0.0, new_val=10.0)
    assert torch.all(q.queue == 10.0), "hard switch must not modify the new chunk"


def test_replace_ramps_from_last_served_command_only():
    q = _merge_two_chunks(_queue("replace", blend=3), old_val=0.0, new_val=10.0)
    # ramp rows: (1-a)*anchor + a*new with a = 1/4, 2/4, 3/4; anchor = 0.0
    expected = [10.0 * 0.25, 10.0 * 0.5, 10.0 * 0.75]
    got = q.queue[:3, 0].tolist()
    assert got == pytest.approx(expected), got
    assert torch.all(q.queue[3:] == 10.0)
    # the ORIGINAL stream must remain the policy's actual plan (unblended)
    assert torch.all(q.original_queue == 10.0)


def test_replace_never_mixes_old_plan_rows():
    """Old plan has REMAINING future rows with a distinctive slope; replace
    mode must ignore them entirely (blend mode would mix them in)."""
    q = _queue("replace", blend=3)
    old = torch.stack([torch.arange(10.0), torch.arange(10.0)], dim=1)  # slope 1
    q.merge(old.clone(), old.clone(), real_delay=0)
    q.get()                                     # served row 0 -> anchor = old[0] = 0
    new = torch.full((10, 2), 100.0)
    q.merge(new.clone(), new.clone(), real_delay=0)
    # blend mode would use old rows 1,2,3 (values 1,2,3) as sources; replace
    # must use ONLY the last served command (0.0)
    expected = [100.0 * 0.25, 100.0 * 0.5, 100.0 * 0.75]
    assert q.queue[:3, 0].tolist() == pytest.approx(expected)


def test_blend_mixes_old_plan_rows():
    """Regression guard for the existing behavior: blend uses the old plan's
    time-aligned REMAINING rows, not just the last served command."""
    q = _queue("blend", blend=3)
    old = torch.stack([torch.arange(10.0)] * 2, dim=1)
    q.merge(old.clone(), old.clone(), real_delay=0)
    q.get()                                     # last_index=1; old remaining = rows 1..9
    new = torch.full((10, 2), 100.0)
    q.merge(new.clone(), new.clone(), real_delay=0)
    # alphas = 1/4, 2/4, 3/4 against old rows 1, 2, 3
    expected = [0.75 * 1 + 0.25 * 100, 0.5 * 2 + 0.5 * 100, 0.25 * 3 + 0.75 * 100]
    assert q.queue[:3, 0].tolist() == pytest.approx(expected)


def test_replace_first_chunk_uses_seed():
    """First chunk of a run has no previous plan: every mode blends out of
    the seeded held pose."""
    q = _queue("replace", blend=2)
    q.set_seed(torch.zeros(2))
    new = torch.full((10, 2), 9.0)
    q.merge(new.clone(), new.clone(), real_delay=0)
    assert q.queue[:2, 0].tolist() == pytest.approx([3.0, 6.0])
    assert torch.all(q.queue[2:] == 9.0)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        RTCConfig(splice_mode="ensemble")
