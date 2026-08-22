"""x0-space RTC guidance (guide_x0) — the DDIM port of prefix guidance.

Synthetic tests, no model: the contract is that the frozen prefix converges
to the target, the ramp region moves partially, zero-weight rows (including
the past-rows offset) never move, and s is clamped so guidance can never
overshoot the target.
"""

import pytest
import torch

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


def _proc(**kw) -> RTCProcessor:
    return RTCProcessor(RTCConfig(execution_horizon=8, **kw))


def test_frozen_prefix_projects_to_target_at_low_tau():
    """At low tau (early denoising) the schedule saturates -> s = 1 -> the
    frozen rows equal the target exactly after one application."""
    p = _proc()
    x0 = torch.zeros(1, 16, 9)
    target = torch.full((1, 16, 9), 5.0)
    out = p.guide_x0(x0, target, inference_delay=4, tau=0.05)
    assert torch.allclose(out[:, :4], target[:, :4]), "frozen rows must project to target"


def test_ramp_partial_and_tail_untouched():
    p = _proc()
    x0 = torch.zeros(1, 16, 9)
    target = torch.full((1, 16, 9), 10.0)
    out = p.guide_x0(x0, target, inference_delay=3, tau=0.05, execution_horizon=8)
    # ramp region: strictly between 0 and target, monotonically decreasing pull
    ramp = out[0, 3:8, 0]
    assert torch.all(ramp > 0) and torch.all(ramp < 10.0)
    assert torch.all(ramp[:-1] >= ramp[1:]), "prefix weights must decay over the ramp"
    # beyond the execution horizon: zero weight -> untouched
    assert torch.all(out[0, 8:] == 0.0)


def test_row_offset_protects_past_rows():
    """The diffusion horizon starts n_obs-1 rows in the past; those rows must
    never be guided (there is no committed prefix for the past)."""
    p = _proc()
    x0 = torch.zeros(1, 16, 9)
    target = torch.full((1, 16, 9), 7.0)
    out = p.guide_x0(x0, target, inference_delay=4, tau=0.05, row_offset=1)
    assert torch.all(out[0, 0] == 0.0), "past row must be untouched"
    assert torch.allclose(out[0, 1:5], target[0, 1:5]), \
        "frozen region must start AFTER the offset"


def test_no_overshoot_at_any_tau():
    """s is clamped to 1: guided x0 must lie on the segment [x0, target]."""
    p = _proc(max_guidance_weight=10.0)
    x0 = torch.zeros(1, 16, 9)
    target = torch.ones(1, 16, 9)
    for tau in (0.01, 0.3, 0.5, 0.7, 0.99):
        out = p.guide_x0(x0, target, inference_delay=6, tau=tau)
        assert torch.all(out >= -1e-6) and torch.all(out <= 1.0 + 1e-6), \
            f"overshoot at tau={tau}"


def test_none_prefix_is_identity():
    p = _proc()
    x0 = torch.randn(1, 16, 9)
    assert torch.equal(p.guide_x0(x0, None, 4, tau=0.5), x0)


def test_short_prefix_is_right_padded_semantics():
    """A leftover shorter than the horizon only constrains the rows it covers
    (caller pads with zeros; weights beyond execution_horizon are zero)."""
    p = _proc()
    x0 = torch.zeros(1, 16, 9)
    target = torch.zeros(1, 16, 9)
    target[:, :5] = 3.0
    out = p.guide_x0(x0, target, inference_delay=5, tau=0.05, execution_horizon=5)
    assert torch.allclose(out[0, :5, 0], torch.full((5,), 3.0))
    assert torch.all(out[0, 5:] == 0.0)
