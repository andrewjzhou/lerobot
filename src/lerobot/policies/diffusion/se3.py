"""SE(3) helpers for UMI-style relative pose windows (torch, batched).

Pose layout everywhere: 9D = [x, y, z, rot6d], where rot6d is the FIRST TWO
COLUMNS of the rotation matrix stacked: [r11, r21, r31, r12, r22, r32].
This matches the push-box dataset builder (controller/vive_frame.R_to_rot6d)
and the diffusion_policy fork's trace_dataset convention.

Poses may carry EXTRA dims appended after the 9 (e.g. dim 9 = gripper jaw,
stack_cup onward). relativize/derelativize transform only the leading 9 and
pass the rest through untouched — UMI-faithful: the gripper channel is an
ABSOLUTE width/closure, never re-expressed in the anchor frame.
"""

from __future__ import annotations

import torch
from torch import Tensor


def rot6d_to_mat(r6: Tensor) -> Tensor:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt (columns convention)."""
    a, b = r6[..., :3], r6[..., 3:6]
    c0 = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b = b - (c0 * b).sum(dim=-1, keepdim=True) * c0
    c1 = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    c2 = torch.cross(c0, c1, dim=-1)
    return torch.stack([c0, c1, c2], dim=-1)


def mat_to_rot6d(rot: Tensor) -> Tensor:
    """(..., 3, 3) -> (..., 6): first two columns stacked."""
    return torch.cat([rot[..., :, 0], rot[..., :, 1]], dim=-1)


def pose9_to_mat(pose: Tensor) -> Tensor:
    """(..., 9) -> (..., 4, 4) homogeneous transform."""
    mat = torch.zeros(*pose.shape[:-1], 4, 4, dtype=pose.dtype, device=pose.device)
    mat[..., :3, :3] = rot6d_to_mat(pose[..., 3:9])
    mat[..., :3, 3] = pose[..., :3]
    mat[..., 3, 3] = 1.0
    return mat


def mat_to_pose9(mat: Tensor) -> Tensor:
    """(..., 4, 4) -> (..., 9)."""
    return torch.cat([mat[..., :3, 3], mat_to_rot6d(mat[..., :3, :3])], dim=-1)


def invert(mat: Tensor) -> Tensor:
    """Inverse of (..., 4, 4) rigid transforms (exact, no linalg.inv)."""
    rot_t = mat[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(mat)
    out[..., :3, :3] = rot_t
    out[..., :3, 3] = (-rot_t @ mat[..., :3, 3:4]).squeeze(-1)
    out[..., 3, 3] = 1.0
    return out


def relativize_window(poses: Tensor, anchor: Tensor) -> Tensor:
    """Re-express a pose window in the anchor's frame.

    poses:  (B, T, 9+G) absolute poses (G >= 0 extra pass-through dims).
    anchor: (B, 4, 4) absolute pose of the anchor step.
    Returns (B, T, 9+G) with the anchor step's pose mapped to identity
    ([0,0,0, 1,0,0, 0,1,0]) and dims 9: passed through untouched.
    """
    rel = mat_to_pose9(invert(anchor).unsqueeze(1) @ pose9_to_mat(poses))
    if poses.shape[-1] > 9:
        rel = torch.cat([rel, poses[..., 9:]], dim=-1)
    return rel


def derelativize_window(rel_poses: Tensor, anchor: Tensor) -> Tensor:
    """Inverse of relativize_window: rel (B, T, 9+G) + anchor (B, 4, 4) -> abs."""
    out = mat_to_pose9(anchor.unsqueeze(1) @ pose9_to_mat(rel_poses))
    if rel_poses.shape[-1] > 9:
        out = torch.cat([out, rel_poses[..., 9:]], dim=-1)
    return out
