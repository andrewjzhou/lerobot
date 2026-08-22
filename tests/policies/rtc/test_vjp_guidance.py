"""True-VJP RTC guidance (the reference implementation's formulation).

The conditional_sample VJP branch computes corr = J^T (W * (y - x0)) where
J is the Jacobian of the denoiser map x_t -> x0_hat. These tests replicate
the exact autograd pattern with an analytic linear denoiser and check the
guidance-weight schedule (min(c * inv_r2, max_w), NOT clamped to 1).
"""

import torch

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor


def test_vjp_correction_matches_analytic_jacobian():
    """Linear 'unet' eps = x @ M: x0 = (x - b*(x@M))/a has Jacobian
    J = (I - b*M)/a, so the VJP of cotangent e must be (e - b*(e@M^T))/a."""
    torch.manual_seed(0)
    B, T, D = 1, 6, 4
    a, b = 0.8, 0.6
    M = torch.randn(D, D)
    sample = torch.randn(B, T, D)
    e = torch.randn(B, T, D)

    with torch.enable_grad():
        x_in = sample.detach().clone().requires_grad_(True)
        eps = x_in @ M
        x0 = (x_in - b * eps) / a
        corr = torch.autograd.grad(x0, x_in, grad_outputs=e)[0]

    expected = (e - b * (e @ M.T)) / a
    assert torch.allclose(corr, expected, atol=1e-5)


def test_vjp_pattern_works_under_no_grad():
    """The engine calls predict under @torch.no_grad(); enable_grad inside
    must still produce the correction."""
    with torch.no_grad():
        with torch.enable_grad():
            x = torch.ones(1, 3, 2).requires_grad_(True)
            x0 = 2.0 * x
            corr = torch.autograd.grad(x0, x, grad_outputs=torch.ones_like(x0))[0]
    assert torch.allclose(corr, torch.full_like(corr, 2.0))


def test_guidance_weight_schedule_values():
    p = RTCProcessor(RTCConfig())          # max_guidance_weight default = 5
    kw = {"device": "cpu", "dtype": torch.float32}
    w_mid = p.pinv_guidance_weight(0.5, **kw)
    assert torch.allclose(w_mid, torch.tensor(2.0), atol=1e-5), \
        "tau=0.5: c=1, inv_r2=2 -> w=2"
    assert float(p.pinv_guidance_weight(1e-6, **kw)) == 5.0, "tau->0 caps at max_w"
    assert float(p.pinv_guidance_weight(1 - 1e-6, **kw)) == 5.0, "tau->1 caps at max_w"
    # NOT clamped to 1 (that clamp belongs only to the identity-x0 form)
    assert float(w_mid) > 1.0


def test_prefix_weight_vector_alignment():
    """Row offset zero-weights the past rows; frozen/ramp clamped to eff."""
    p = RTCProcessor(RTCConfig(execution_horizon=10))
    w = p.prefix_weight_vector(4, 10, 16, row_offset=1,
                               device="cpu", dtype=torch.float32)
    assert w.shape == (1, 16, 1)
    assert w[0, 0, 0] == 0.0, "past row must be zero-weight"
    assert torch.all(w[0, 1:5, 0] == 1.0), "frozen span after the offset"
    assert torch.all(w[0, 11:, 0] == 0.0), "beyond ramp end"
