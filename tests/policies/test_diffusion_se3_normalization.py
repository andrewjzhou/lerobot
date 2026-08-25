"""End-to-end pose round trip through the SE(3) diffusion processor pipeline.

REGRESSION GUARD for the 2026-08-18 rot6d corruption.

The bug: `make_diffusion_pre_post_processors` dropped "action" from the
normalizer's `features` dict in SE(3) mode, believing that disabled action
normalization. It does not — `NormalizerProcessorStep` normalizes observations
from `features` but runs the ACTION path unconditionally
(`_normalize_action`), which never consults `features`. So the ABSOLUTE action
was still MIN_MAX-scaled before the model relativized it. Per-dim affine
scaling is fine for translation but destroys a rotation: rot6d components
reached ~8 where a rotation column caps at 1, and the Gram-Schmidt inside
`pose9_to_mat` then silently projected that back onto SO(3) — a non-invertible
repair. Measured cost: position exact, rotation 2.89 deg median / 8.76 deg max
error in the training targets.

Why it went unnoticed: every check done at the time passed with the bug
present. The SE(3) helpers round-tripped at 1e-7 *in isolation* (they never saw
the processor), and "predicted chunks are orthonormal" is guaranteed by
Gram-Schmidt whether or not the rotation is correct. Only an end-to-end round
trip through the real pipeline catches it — which is what this test is.
"""

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors
from lerobot.policies.diffusion.se3 import mat_to_pose9, pose9_to_mat, relativize_window


def _random_pose9(rng, n):
    """n valid absolute poses: xyz in metres + the first two columns of R."""
    out = np.zeros((n, 9))
    out[:, :3] = rng.normal(0, 0.15, (n, 3))
    for i in range(n):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        out[i, 3:9] = np.concatenate([q[:, 0], q[:, 1]])
    return out


def _config(pos_only: bool) -> DiffusionConfig:
    cfg = DiffusionConfig(
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        use_se3_relative=True,
        use_se3_normalize=True,
        se3_normalize_pos_only=pos_only,
        se3_rel_stats={
            "state_min": [-0.03] * 3 + [-1.0] * 6,
            "state_max": [0.03] * 3 + [1.0] * 6,
            "action_min": [-0.2] * 3 + [-1.0] * 6,
            "action_max": [0.2] * 3 + [1.0] * 6,
        },
    )
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(9,)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(9,))}
    cfg.device = "cpu"
    return cfg


def _stats():
    # Deliberately ASYMMETRIC per-dim ranges — that asymmetry is what shears a
    # rotation. Equal ranges would hide the bug.
    return {
        "action": {
            "min": torch.tensor([-0.3, -0.1, -0.25, -1.0, -0.4, -1.0, -0.6, -1.0, -0.2]),
            "max": torch.tensor([0.2, 0.5, 0.15, 1.0, 0.9, 1.0, 0.4, 1.0, 0.8]),
        },
        "observation.state": {
            "min": torch.tensor([-0.3] * 3 + [-1.0] * 6),
            "max": torch.tensor([0.3] * 3 + [1.0] * 6),
        },
    }


def _round_trip(pos_only: bool):
    """raw action -> pre() -> relativize -> derelativize -> post() -> raw?"""
    cfg = _config(pos_only)
    pre, post = make_diffusion_pre_post_processors(cfg, dataset_stats=_stats())
    rng = np.random.default_rng(0)
    pos_err, rot_err = [], []
    for _ in range(20):
        state = _random_pose9(rng, 2)
        action = _random_pose9(rng, 16)
        batch = {
            "observation.state": torch.tensor(state).float().unsqueeze(0),
            "action": torch.tensor(action).float().unsqueeze(0),
            "task": ["t"],
        }
        with torch.inference_mode():
            out = pre(batch)
        anchor = pose9_to_mat(out["observation.state"][:, -1].double())
        rel = relativize_window(out["action"].double(), anchor)
        back = mat_to_pose9(anchor.unsqueeze(1) @ pose9_to_mat(rel))
        with torch.inference_mode():
            final = post(back.float())
        final = (final if isinstance(final, torch.Tensor) else final["action"]).double().numpy()[0]
        pos_err.append(np.abs(final[:, :3] - action[:, :3]).max())
        u1 = final[:, 3:6] / np.linalg.norm(final[:, 3:6], axis=-1, keepdims=True)
        u2 = action[:, 3:6] / np.linalg.norm(action[:, 3:6], axis=-1, keepdims=True)
        rot_err.append(np.degrees(np.arccos(np.clip((u1 * u2).sum(-1), -1, 1))).max())
    return max(pos_err), max(rot_err)


def test_pos_only_round_trip_is_exact():
    """With se3_normalize_pos_only, poses survive the pipeline untouched."""
    pos, rot = _round_trip(pos_only=True)
    assert pos < 1e-5, f"position corrupted by {pos * 1000:.4f} mm"
    assert rot < 1e-3, f"rotation corrupted by {rot:.4f} deg"


def test_pos_only_disables_absolute_action_normalization():
    """The ACTION path must be IDENTITY in BOTH directions — dropping 'action'
    from `features` alone does not stop it (that was the bug)."""
    pre, post = make_diffusion_pre_post_processors(_config(True), dataset_stats=_stats())
    for pipeline, name in ((pre, "pre"), (post, "post")):
        maps = [s.norm_map for s in pipeline.steps if hasattr(s, "norm_map")]
        assert maps, f"{name}: no normalizer step found"
        for m in maps:
            assert m.get(FeatureType.ACTION) is NormalizationMode.IDENTITY, (
                f"{name}: ACTION is {m.get(FeatureType.ACTION)}, expected IDENTITY"
            )


def test_action_is_untouched_by_the_processor():
    """A raw absolute action must come out of pre() bit-identical."""
    pre, _ = make_diffusion_pre_post_processors(_config(True), dataset_stats=_stats())
    rng = np.random.default_rng(1)
    action = torch.tensor(_random_pose9(rng, 16)).float().unsqueeze(0)
    batch = {
        "observation.state": torch.tensor(_random_pose9(rng, 2)).float().unsqueeze(0),
        "action": action.clone(),
        "task": ["t"],
    }
    with torch.inference_mode():
        out = pre(batch)
    assert torch.allclose(out["action"], action, atol=1e-7), "processor still scales the action"


@pytest.mark.parametrize("pos_only", [True, False])
def test_position_always_survives(pos_only):
    """Translation round-trips exactly either way — the damage was rotation-only,
    so a position-only check would NOT have caught the original bug."""
    pos, _ = _round_trip(pos_only)
    assert pos < 1e-5


# --- 10-dim poses: dim 9 = gripper (stack_cup onward) ------------------------
#
# The gripper is an ABSOLUTE closure fraction. It must pass through the SE(3)
# relativization untouched (UMI never re-expresses the width in the anchor
# frame) and be MIN_MAX-normalized alongside position when
# se3_normalize_pos_only. Before the pass-through existed, pose9_to_mat
# silently DROPPED dim 9 inside relativize_window and the window came back
# 9-dim — a shape error downstream at best, a silently vanished gripper at
# worst. These tests pin both properties.


def _random_pose10(rng, n):
    out = np.zeros((n, 10))
    out[:, :9] = _random_pose9(rng, n)
    out[:, 9] = rng.uniform(0.2, 0.85, n)
    return out


def _grip_config() -> DiffusionConfig:
    cfg = DiffusionConfig(
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        device="cpu",
        use_se3_relative=True,
        use_se3_normalize=True,
        se3_normalize_pos_only=True,
        crop_shape=None,
        se3_rel_stats={
            "state_min": [-0.03] * 3 + [-1.0] * 6 + [0.2],
            "state_max": [0.03] * 3 + [1.0] * 6 + [0.85],
            "action_min": [-0.2] * 3 + [-1.0] * 6 + [0.2],
            "action_max": [0.2] * 3 + [1.0] * 6 + [0.85],
        },
    )
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(10,)),
        # never forwarded — the policy just refuses to instantiate imageless
        "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(10,))}
    return cfg


def test_gripper_dim_passes_through_relativization():
    """relativize/derelativize keep dims 9: bit-identical and untouched by
    the anchor transform; the pose part matches the 9-dim path exactly."""
    from lerobot.policies.diffusion.se3 import derelativize_window

    rng = np.random.default_rng(2)
    poses = torch.tensor(_random_pose10(rng, 16)).double().unsqueeze(0)
    anchor = pose9_to_mat(torch.tensor(_random_pose9(rng, 1)).double())
    rel = relativize_window(poses, anchor)
    assert rel.shape == poses.shape
    assert torch.equal(rel[..., 9], poses[..., 9]), "gripper altered by relativization"
    rel9 = relativize_window(poses[..., :9], anchor)
    assert torch.allclose(rel[..., :9], rel9, atol=1e-12), "pose part diverged from 9-dim path"
    back = derelativize_window(rel, anchor)
    assert torch.allclose(back, poses, atol=1e-9), "10-dim round trip broke"


def test_gripper_dim_full_policy_round_trip():
    """Model-level: _se3_relativize_batch -> _se3_derelativize_actions is the
    identity on absolute 10-dim actions, and the network-space gripper is
    MIN_MAX-mapped to [-1, 1] (UMI range-normalizes position + gripper)."""
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    cfg = _grip_config()
    policy = DiffusionPolicy(cfg)
    rng = np.random.default_rng(3)
    state = torch.tensor(_random_pose10(rng, 2)).float().unsqueeze(0)
    action = torch.tensor(_random_pose10(rng, 16)).float().unsqueeze(0)
    batch = policy._se3_relativize_batch({"observation.state": state, "action": action})
    lo, hi = 0.2, 0.85
    expect = 2 * (action[..., 9] - lo) / (hi - lo) - 1
    assert torch.allclose(batch["action"][..., 9], expect, atol=1e-6), (
        "network-space gripper is not MIN_MAX-normalized"
    )
    back = policy._se3_derelativize_actions(batch["action"])
    assert torch.allclose(back, action, atol=1e-5), "10-dim policy round trip broke"
