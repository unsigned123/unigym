"""The single quaternion conversion layer (backend-traits).

Repo-internal convention is **wxyz** and never changes: obs record keys
(``base_quat_w``), adapters, criteria, npz artifacts all stay wxyz.  External
conventions are explicit at the boundary only:

* ``XYZW`` — scipy ``R.from_quat`` / IsaacLab 3.x data views / Warp / PhysX;
* ``from_isaaclab_data`` / ``to_isaaclab_cfg`` — the ONLY version-aware
  conversion points (IsaacLab 2.3 data/cfg are wxyz-native: identity;
  IsaacLab 3.x is xyzw: reorder).  The isaaclab variant is probed through
  the central backend registry — per-variant backends convert at their own
  boundary and conventions never mix (design §R.2).

Pure torch+numpy; zero engine imports (``core/`` is in the agnostic strong
guard).  Math functions are dual-dtype: numpy in / numpy out, torch in /
torch out — every formula is byte-preserved from the modules that
previously owned it (cameras/rigs, backends/mujoco/cameras, props/base,
env/backends/mujoco/env.py, isaaclab2/env.py, robots/g1), which now delegate
here.  Semantic camera frames (ISAAC_CAM +X fwd / MUJOCO_CAM -Z fwd /
ROS / OPENCV) are orthogonal to component order; the isaac->mujoco camera
bridge lives here as :func:`camera_isaac_to_mujoco`.

History: camera_contract §P8 black-frame incident and the HA 0% round-2
frame mismatch were both hand-written quaternion math — this module is the
mechanization that bans it (new quaternion arithmetic outside core/quat.py
is a review-flagged smell; see test_arch quat soft-gate).
"""
from __future__ import annotations

import math
from typing import cast

import numpy as np
import torch

__all__ = [
    "WXYZ", "XYZW",
    "to_wxyz", "to_xyzw", "normalize",
    "mul", "conj", "rotate", "rotate_inv", "to_rotmat",
    "yaw_from_quat", "about_x", "about_z",
    "yaw_pitch", "yaw_pitch_batch", "lookat", "lookat_batch",
    "az_el", "to_euler_rpy",
    "camera_bridge_quat", "camera_isaac_to_mujoco",
    "from_isaaclab_data", "to_isaaclab_cfg",
]

# component-order conventions (external world)
WXYZ = "wxyz"   # repo-internal / mujoco / IsaacLab 2.x data / IsaacLab 2.x cfg
XYZW = "xyzw"   # scipy(R.from_quat) / IsaacLab 3.x data / Warp / PhysX

_REORDER = [1, 2, 3, 0]      # xyzw -> wxyz (and the inverse permutation)
_FWD_REORDER = [3, 0, 1, 2]  # wxyz -> xyzw


# --------------------------------------------------------------------------- #
# component order (zero-cost views on numpy; copies on torch fancy-index)
# --------------------------------------------------------------------------- #
def to_wxyz(q, src: str = XYZW):
    """Reorder *q* to wxyz.  ``src=WXYZ`` -> unchanged reference."""
    if src == WXYZ:
        return q
    if src != XYZW:
        raise ValueError(f"unknown quaternion convention {src!r} (wxyz|xyzw)")
    if isinstance(q, torch.Tensor):
        return q[..., _FWD_REORDER]
    return np.asarray(q)[..., _FWD_REORDER]


def to_xyzw(q, dst: str = XYZW):
    """Reorder *q* to xyzw.  ``dst=XYZW`` semantics: input is wxyz."""
    if dst != XYZW:
        raise ValueError(f"unknown quaternion convention {dst!r} (wxyz|xyzw)")
    if isinstance(q, torch.Tensor):
        return q[..., _REORDER]
    return np.asarray(q)[..., _REORDER]


def normalize(q):
    """Unit quaternion (last-axis norm); float32-preserving for numpy."""
    if isinstance(q, torch.Tensor):
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return (q / np.clip(norm, 1e-12, None)).astype(np.float32)


# --------------------------------------------------------------------------- #
# math (all wxyz in/out; dispatch on dtype)
# --------------------------------------------------------------------------- #
def mul(a, b):
    """Hamilton product a (x) b, wxyz (broadcasting both sides)."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        aw, ax, ay, az = a.unbind(-1)
        bw, bx, by, bz = b.unbind(-1)
        return torch.stack(
            [aw * bw - ax * bx - ay * by - az * bz,
             aw * bx + ax * bw + ay * bz - az * by,
             aw * by - ax * bz + ay * bw + az * bx,
             aw * bz + ax * by - ay * bx + az * bw], dim=-1)
    a = np.asarray(a)
    b = np.asarray(b)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
         w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2],
        axis=-1)


def conj(q):
    """Conjugate (w, -x, -y, -z); the inverse for unit quats."""
    if isinstance(q, torch.Tensor):
        out = q.clone()
        out[..., 1:] = -out[..., 1:]
        return out
    q = np.asarray(q)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def rotate(q, v):
    """R(q) . v for q=(...,4) wxyz, v=(...,3).  Body -> world."""
    u = q[..., 1:]
    w = q[..., 0:1]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def rotate_inv(q, v):
    """R(q)^T . v for q=(...,4) wxyz, v=(...,3).  World -> body."""
    u = q[..., 1:]
    w = q[..., 0:1]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v - w * t + torch.cross(u, t, dim=-1)


def to_rotmat(q):
    """(...,4) wxyz quat -> (...,3,3) rotation matrix (world <- local rows).

    numpy path: raw formula, no normalization (byte-equal to the cameras /
    mujoco_env originals).  torch path: input normalized with a 1e-12
    clamp first (byte-equal to the sonic_g1 original).
    """
    if isinstance(q, torch.Tensor):
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        w, x, y, z = q.unbind(-1)
        return torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ], dim=-1).unflatten(-1, (3, 3))
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def yaw_from_quat(q):
    """Yaw (rad) about +Z from (N, 4) wxyz quaternions (torch batch)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def about_x(angle) -> torch.Tensor:
    """Rotation about +X by *angle* rad, wxyz float32 tensor (4,)."""
    return torch.tensor(
        [math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0],
        dtype=torch.float32)


def about_z(theta):
    """Rotation about +Z by *theta* rad -> (...,4) wxyz.

    torch batch in / torch batch out; python scalar in / float64 numpy
    4-vector out (the spawn-rotation shape robots/g1 consumes).
    """
    if isinstance(theta, torch.Tensor):
        half = 0.5 * theta
        return torch.stack([torch.cos(half), torch.zeros_like(half),
                            torch.zeros_like(half), torch.sin(half)], dim=-1)
    half = 0.5 * float(theta)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)],
                    dtype=np.float64)


# --------------------------------------------------------------------------- #
# semantic construction (camera_contract convention chain, mechanized)
# --------------------------------------------------------------------------- #
def yaw_pitch(yaw_rad, pitch_rad) -> np.ndarray:
    """Mount-frame orientation offset (w,x,y,z): yaw about +Z then pitch
    about +Y (positive pitch tilts forward DOWN).  (0, 0) -> identity."""
    y, p = float(yaw_rad) / 2.0, float(pitch_rad) / 2.0
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    # q_yaw (x=y=0) (x) q_pitch (x=z=0), Hamilton product, wxyz
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp])


def yaw_pitch_batch(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Batched :func:`yaw_pitch`: (k,) rad -> (k, 4) wxyz."""
    cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    cp, sp = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
    return torch.stack([cy * cp, -sy * sp, cy * sp, sy * cp], dim=-1)


def yaw_pitch_roll(yaw_rad, pitch_rad, roll_rad) -> np.ndarray:
    """Mount-frame orientation offset (w,x,y,z): yaw about +Z, pitch about
    +Y, then roll about the VIEW axis (+X after yaw/pitch).

    Roll spins the image around the camera's own forward axis, so it is
    independent of where the camera points; a zero roll reproduces
    :func:`yaw_pitch` exactly and (0, 0, 0) is the identity.
    """
    r = 0.5 * float(roll_rad)
    return cast(np.ndarray, mul(yaw_pitch(yaw_rad, pitch_rad),
                                np.array([math.cos(r), math.sin(r), 0.0, 0.0])))


def yaw_pitch_roll_batch(yaw: torch.Tensor, pitch: torch.Tensor,
                         roll: torch.Tensor) -> torch.Tensor:
    """Batched :func:`yaw_pitch_roll`: (k,) rad x3 -> (k, 4) wxyz."""
    cr, sr = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
    zero = torch.zeros_like(sr)
    return cast(torch.Tensor, mul(yaw_pitch_batch(yaw, pitch),
                                  torch.stack([cr, sr, zero, zero], dim=-1)))


def lookat(pos, lookat) -> np.ndarray:
    """Single-pose world-convention camera quaternion (w,x,y,z): forward
    (+X) from ``pos`` toward ``lookat`` (forward=+X, up=+Z).

    Reference implementation — :func:`lookat_batch` is checked against it.
    """
    d = np.asarray(lookat, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    dist = np.linalg.norm(d)
    if dist < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    d /= dist
    yaw = np.arctan2(d[1], d[0])
    elev = float(np.arcsin(np.clip(d[2], -1.0, 1.0)))
    q_yaw = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
    q_pitch = np.array([np.cos(0.5 * elev), 0.0, -np.sin(0.5 * elev), 0.0])
    q = cast(np.ndarray, mul(q_yaw, q_pitch))
    return q / np.linalg.norm(q)


def lookat_batch(pos: torch.Tensor, lookat: torch.Tensor) -> torch.Tensor:
    """Batched world-convention look-at quaternion (w,x,y,z); (N, 3) in."""
    d = lookat - pos
    d = d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-8)
    yaw = torch.atan2(d[:, 1], d[:, 0])
    elev = torch.asin(torch.clamp(d[:, 2], -1.0, 1.0))
    q_yaw = torch.stack([torch.cos(0.5 * yaw), torch.zeros_like(yaw),
                         torch.zeros_like(yaw), torch.sin(0.5 * yaw)], dim=-1)
    q_pitch = torch.stack([torch.cos(0.5 * elev), torch.zeros_like(elev),
                           -torch.sin(0.5 * elev), torch.zeros_like(elev)],
                          dim=-1)
    return cast(torch.Tensor, mul(q_yaw, q_pitch))


def az_el(az_deg: float, el_deg: float) -> tuple:
    """Azimuth/elevation (degrees) -> wxyz quaternion orienting a
    DistantLight so its rays arrive from that sun direction (local +Z
    toward the sun; USDLux DistantLight emits along local -Z)."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    sx = math.cos(el) * math.cos(az)
    sy = math.cos(el) * math.sin(az)
    sz = math.sin(el)
    # rotation axis = (0,0,1) x sun_dir, angle = acos(sz)
    ax, ay = -sy, sx
    n = math.hypot(ax, ay)
    if n < 1e-9:
        # sun along +/-Z: identity or 180 deg about X
        return (1.0, 0.0, 0.0, 0.0) if sz >= 0 else (0.0, 1.0, 0.0, 0.0)
    ax, ay = ax / n, ay / n
    angle = math.acos(max(-1.0, min(1.0, sz)))
    ha = angle / 2.0
    s, c = math.sin(ha), math.cos(ha)
    return (c, ax * s, ay * s, 0.0)


def to_euler_rpy(quat) -> np.ndarray:
    """wxyz quat -> roll-pitch-yaw (xyz convention)."""
    euler_vec = np.zeros(3)
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    euler_vec[0] = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        euler_vec[1] = np.copysign(np.pi / 2, sinp)
    else:
        euler_vec[1] = np.arcsin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    euler_vec[2] = np.arctan2(siny_cosp, cosy_cosp)
    return euler_vec


# --------------------------------------------------------------------------- #
# camera frame bridge (ISAAC_CAM +X fwd +Z up -> MUJOCO_CAM -Z fwd +Y up)
# --------------------------------------------------------------------------- #
def camera_bridge_quat() -> np.ndarray:
    """The fixed Isaac->MuJoCo camera orientation bridge, (w, x, y, z)."""
    return np.array([0.5, 0.5, -0.5, -0.5])


def camera_isaac_to_mujoco(q_isaac_wxyz) -> np.ndarray:
    """Isaac-convention camera quat (w,x,y,z; forward +X up +Z) ->
    MuJoCo camera quat (forward -Z up +Y): ``q_isaac (x) q_bridge``."""
    return cast(np.ndarray, mul(np.asarray(q_isaac_wxyz, dtype=np.float64),
                                camera_bridge_quat()))


# --------------------------------------------------------------------------- #
# version-aware isaaclab boundary (the ONLY conversion points)
# --------------------------------------------------------------------------- #
def _isaaclab_is_xyzw() -> bool:
    """True when the probed isaaclab major version is >= 3 (xyzw data/cfg)."""
    from ..backends.registry import isaaclab_version

    ver = isaaclab_version()
    return ver is not None and ver[0] >= 3


def from_isaaclab_data(q):
    """isaaclab data view -> repo wxyz.  il2.3: identity (wxyz-native);
    il3: xyzw -> wxyz.  Called at every ``asset.data.*_quat*`` read."""
    if _isaaclab_is_xyzw():
        return to_wxyz(q, XYZW)
    return q


def to_isaaclab_cfg(q):
    """repo wxyz -> isaaclab cfg ``rot=`` value.  il2.3: identity;
    il3: wxyz -> xyzw.  Called at every cfg-construction rot= site."""
    if _isaaclab_is_xyzw():
        return to_xyzw(q)
    return q
