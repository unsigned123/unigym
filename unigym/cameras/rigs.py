"""Camera rig vocabulary: neutral pose helpers + the backend factory face.

The engine-neutral parts of the old rig module stay here: the recorder
parity defaults, the quaternion pose helpers and the backend-resolving
factory.  The IsaacLab rig classes (``BodyRig`` / ``WorldRig`` /
``ChaseRig``) now live in ``unigym/backends/isaaclab2/cameras/rigs.py``
and the mujoco/mjwarp rigs already lived in their backends.

The module is **torch-only at import time** and imports no backend at
module level: both :func:`build_rig` and :func:`build_neutral_rig` resolve
the backend factory lazily through :attr:`BackendBindings.camera_backend`.
"""

from __future__ import annotations

from typing import Callable, cast

import numpy as np
import torch

from ..core import quat
from .base import CameraRig, CameraSpec

# recorder parity defaults for chase/world look-at
DEFAULT_CHASE_OFFSET: dict = {
    "pos": [-1.2, -1.8, 1.2],  # trail behind + side, elevated
    "lookat": [1.0, 0.0, 0.25],  # ahead of robot, body height
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


def _yaw_pitch_quat_batch(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Batched :func:`yaw_pitch_quat`: (k,) rad -> (k, 4) wxyz."""
    return quat.yaw_pitch_batch(yaw, pitch)


def yaw_pitch_roll_quat(yaw_rad, pitch_rad, roll_rad) -> np.ndarray:
    """Mount-frame offset with a VIEW-AXIS roll (see
    :func:`unigym.core.quat.yaw_pitch_roll`).  Composed outside the rig's
    computed orientation like :func:`yaw_pitch_quat`."""
    return quat.yaw_pitch_roll(yaw_rad, pitch_rad, roll_rad)


def _yaw_pitch_roll_quat_batch(
    yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor
) -> torch.Tensor:
    """Batched :func:`yaw_pitch_roll_quat`: (k,) rad x3 -> (k, 4) wxyz."""
    return quat.yaw_pitch_roll_batch(yaw, pitch, roll)


def _quat_mul_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched Hamilton product of (k, 4) wxyz quaternions."""
    return cast(torch.Tensor, quat.mul(a, b))


def build_neutral_rig(
    spec: CameraSpec,
    backend: str,
    render_groups=None,
    *,
    rig_builder: Callable[..., object] | None = None,
):
    """Engine-neutral compatibility name for the rig builders.

    Kept neutral in the sense that it names no engine: the rig classes live
    in the backend packages and this name uses the *injected* rig builder,
    exactly like :func:`build_rig`.  The backend is explicit (canonical; no
    alias and no default).
    """
    return build_rig(spec, backend=backend, render_groups=render_groups, rig_builder=rig_builder)


def build_rig(
    spec: CameraSpec,
    backend: str,
    render_groups=None,
    *,
    rig_builder: Callable[..., object] | None = None,
) -> CameraRig:
    """Factory: rig instance from the spec's ``rig.type``.

    ``backend`` (required canonical env backend) names the sensor stack:
    ``mujoco`` (M2b: MjSpec-injected ``<camera>`` + ``mujoco.Renderer``),
    ``mjwarp`` (C3: N=1 delegates to that same mirror renderer, N>1 uses
    the native warp batch renderer) or the IsaacLab identities
    (``isaaclab2`` / ``isaaclab3_physx`` / ``isaaclab3_newton``; the
    backend CameraCfg / TiledCameraCfg rigs).  ``rig_builder`` is the
    backend rig factory injected by composition (no registry lookup).
    ``render_groups``: optional per-config geom-class override for the
    capture flags face (camera_contract §14; mujoco-only in v1).
    """
    from ..capabilities.traits import CANONICAL_BACKENDS, build_name

    if backend is None or not str(backend).strip():
        raise ValueError(
            f"camera rig: env backend is required; choose one of: "
            f"{', '.join(CANONICAL_BACKENDS)}"
        )
    canonical = build_name(backend)
    if rig_builder is None:
        raise ValueError(
            f"camera {spec.name!r}: backend {canonical!r} has no camera "
            f"rig factory (available: {', '.join(CANONICAL_BACKENDS)})"
        )
    return cast(CameraRig, rig_builder(spec, render_groups=render_groups))


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
