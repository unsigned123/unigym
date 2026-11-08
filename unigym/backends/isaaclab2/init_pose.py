"""yaml ``init_pose`` section parsing + random initial-pose sampling.

Torch-only and unit-testable.  Same style as :mod:`unigym.env.episode`: the
parsing/sampling logic is pure
torch functions, consumed by isaaclab2/env.py on the server side inside the reset
chain, unit-testable locally without an isaaclab environment.

yaml form::

    init_pose:
      prob: 1.0               # fraction of envs randomized per reset
      roll_range: [60, 300]   # deg, world frame about X
      pitch_range: [-30, 30]  # deg, world frame about Y
      yaw_range: [-180, 180]  # deg, world frame about Z
      height: 0.42            # spawn base-z (m); the low clearance lets the
                              # fall settle naturally toward the target pose
"""
from __future__ import annotations

import torch


def _euler_xyz_to_quat(roll, pitch, yaw):
    """ZYX (roll-pitch-yaw, world axes) -> (w, x, y, z) quaternion, batched.

    Pure torch; convention matches Isaac Lab ``quat_from_euler_xyz``.
    """
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def sample_init_pose(n, roll_range, pitch_range, yaw_range, height=0.42,
                     device="cpu"):
    """Sample per-env initial root poses, optionally lying/fallen.

    Args:
        n: number of envs to sample.
        roll_range: ``[lo, hi]`` roll range in DEGREES.
        pitch_range: ``[lo, hi]`` pitch range in DEGREES.
        yaw_range: ``[lo, hi]`` yaw range in DEGREES.
        height: spawn base-z (m); near-ground so the body settles into the
            sampled orientation on first contact.
        device: torch device for the returned tensors.

    Returns:
        ``(pos, quat)``: shape ``(n, 3)`` base positions at height and
        ``(n, 4)`` (w, x, y, z) quaternions drawn uniformly from the ranges.
    """
    lo = torch.tensor([r[0] for r in (roll_range, pitch_range, yaw_range)],
                      device=device, dtype=torch.float32)
    hi = torch.tensor([r[1] for r in (roll_range, pitch_range, yaw_range)],
                      device=device, dtype=torch.float32)
    deg = lo + (hi - lo) * torch.rand(n, 3, device=device)
    rad = torch.deg2rad(deg)
    pos = torch.zeros(n, 3, device=device)
    pos[:, 2] = float(height)
    quat = _euler_xyz_to_quat(rad[:, 0], rad[:, 1], rad[:, 2])
    return pos, quat


def build_init_pose_cfg(cfg):
    """Validate the yaml ``init_pose`` section -> normalized dict or None.

    Returns None when the section is absent/empty (default: upright spawn).
    """
    if not cfg:
        return None
    normalized = dict(cfg)
    prob = float(normalized.get("prob", 1.0))
    if not (0.0 <= prob <= 1.0):
        raise ValueError(f"init_pose.prob must be in [0, 1], got {prob}")
    normalized["prob"] = prob
    for key in ("roll_range", "pitch_range", "yaw_range"):
        rng = list(normalized.get(key, [0.0, 0.0]))
        if len(rng) != 2:
            raise ValueError(f"init_pose.{key} must be [lo, hi], got {rng!r}")
        lo, hi = float(rng[0]), float(rng[1])
        if not (lo <= hi):
            raise ValueError(f"init_pose.{key} requires lo <= hi, got ({lo}, {hi})")
        normalized[key] = (lo, hi)
    normalized.setdefault("height", 0.42)
    return normalized
