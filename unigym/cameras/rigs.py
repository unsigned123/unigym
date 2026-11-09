"""Camera rig vocabulary: neutral pose helpers + the backend factory face.

The engine-neutral parts of the old rig module stay here: the recorder
parity defaults, the quaternion pose helpers and the backend-resolving
factory.  The IsaacLab rig classes (``BodyRig`` / ``WorldRig`` /
``ChaseRig``) now live in ``unigym/backends/isaaclab2/cameras/rigs.py``
and the mujoco/mjwarp rigs already lived in their backends.

The module is **torch-only at import time** and imports no backend at
module level: both :func:`build_rig` and :func:`build_neutral_rig` resolve
the backend factory lazily through :attr:`BackendInfo.camera_backend`.
"""
from __future__ import annotations

from typing import Callable, cast

import numpy as np
import torch

from ..core import quat
from .base import CameraRig, CameraSpec

# recorder parity defaults for chase/world look-at
DEFAULT_CHASE_OFFSET: dict = {
    "pos": [-1.2, -1.8, 1.2],     # trail behind + side, elevated
    "lookat": [1.0, 0.0, 0.25],   # ahead of robot, body height
}


def yaw_pitch_quat(yaw_rad, pitch_rad) -> np.ndarray:
    """Mount-frame orientation-offset quaternion (w, x, y, z): yaw about +Z
    then pitch about +Y (positive pitch tilts forward DOWN — the repo's
    ``pitch_down_deg`` convention).  (0, 0) -> identity.

    Live-property math : ``apply_properties`` yaw/pitch compose this
    OUTSIDE the rig's computed orientation, in the mount frame (body rigs:
    base frame; world/chase: world frame).

    Delegates to :func:`unigym.core.quat.yaw_pitch` (the single quaternion
    layer; M1 convergence).
    """
    return quat.yaw_pitch(yaw_rad, pitch_rad)


def _yaw_pitch_quat_batch(yaw: torch.Tensor,
                          pitch: torch.Tensor) -> torch.Tensor:
    """Batched :func:`yaw_pitch_quat`: (k,) rad -> (k, 4) wxyz."""
    return quat.yaw_pitch_batch(yaw, pitch)


def yaw_pitch_roll_quat(yaw_rad, pitch_rad, roll_rad) -> np.ndarray:
    """Mount-frame offset with a VIEW-AXIS roll (see
    :func:`unigym.core.quat.yaw_pitch_roll`).  Composed outside the rig's
    computed orientation like :func:`yaw_pitch_quat`."""
    return quat.yaw_pitch_roll(yaw_rad, pitch_rad, roll_rad)


def _yaw_pitch_roll_quat_batch(yaw: torch.Tensor, pitch: torch.Tensor,
                               roll: torch.Tensor) -> torch.Tensor:
    """Batched :func:`yaw_pitch_roll_quat`: (k,) rad x3 -> (k, 4) wxyz."""
    return quat.yaw_pitch_roll_batch(yaw, pitch, roll)


def _quat_mul_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched Hamilton product of (k, 4) wxyz quaternions."""
    return cast(torch.Tensor, quat.mul(a, b))


def build_neutral_rig(spec: CameraSpec, render_groups=None):
    """Engine-neutral compatibility name for the IsaacLab rig builder.

    Kept neutral: the rig classes moved into
    ``backends/isaaclab2/cameras`` and this name resolves that backend
    factory through the registry, exactly like :func:`build_rig`.
    """
    return build_rig(spec, backend="isaaclab", render_groups=render_groups)


def build_rig(spec: CameraSpec, backend: str = "isaaclab",
              render_groups=None) -> CameraRig:
    """Factory: rig instance from the spec's ``rig.type``.

    ``backend`` selects the sensor stack through the registry's lazy
    ``camera_backend`` face: ``isaaclab`` (default; the backend
    CameraCfg/TiledCameraCfg rigs), ``mujoco`` (M2b: MjSpec-injected
    ``<camera>`` + ``mujoco.Renderer``) or ``mjwarp`` (C3: N=1 delegates
    to that same mirror renderer, N>1 uses the native warp batch renderer).
    ``render_groups``: optional per-config geom-class override for the
    capture flags face (camera_contract §14; mujoco-only in v1).
    """
    from ..backends.registry import backend_info, build_name

    factory = backend_info(build_name(backend)).camera_backend
    if factory is None:
        raise ValueError(
            f"camera {spec.name!r}: backend {backend!r} has no camera "
            f"rig factory (hint: use mujoco/mjwarp/isaaclab)")
    rig_factory = cast(Callable[..., object], factory())
    return cast(CameraRig, rig_factory(spec, render_groups=render_groups))


def lookat_quat(pos, lookat) -> np.ndarray:
    """Single-pose world-convention camera quaternion (w, x, y, z):
    forward (+X) from ``pos`` toward ``lookat``.

    World convention axes: forward=+X, up=+Z (Isaac Lab camera offset).
    Composed as yaw about +Z then pitch about +Y; a positive rotation about
    +Y maps +X to a *downward* tilt (R_y(a)·X̂ has elevation −a), so the
    pitch angle is −elevation of the view direction.  This is the reference
    implementation :func:`_lookat_quat_batch` is checked against.
    """
    return quat.lookat(pos, lookat)


def _lookat_quat_batch(pos: torch.Tensor, lookat: torch.Tensor) -> torch.Tensor:
    """Batched world-convention look-at quaternion (w,x,y,z).

    forward=+X, up=+Z; yaw about +Z then pitch about +Y (positive pitch
    maps +X downward — same convention as :func:`lookat_quat`).
    ``pos``/``lookat``: (N, 3).
    """
    return quat.lookat_batch(pos, lookat)


def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Yaw (rad) about +Z from (N, 4) (w,x,y,z) quaternions."""
    return quat.yaw_from_quat(q)
