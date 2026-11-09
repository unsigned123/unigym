"""Shared MuJoCo-family camera helpers (v100 G6).

Pure convention/geometry helpers shared by the mujoco ``Renderer`` rigs and
the mjwarp warp camera rigs: the ``<camera>`` MjSpec injection, the
Isaac->MuJoCo orientation bridge, and the fov/offset resolution.  The
concrete ``mujoco.Renderer`` rig implementation stays in
``backends/mujoco/cameras.py``; this module only holds the engine-light
pieces both backends need.
"""
from __future__ import annotations

import math

import numpy as np

from unigym.core import quat
from unigym.cameras.base import CameraSpec, rig_type
from unigym.cameras.rigs import lookat_quat


MJ_CAMERA_PREFIX = "dc_"

# BodyRig default mount body (the G1 freejoint root body).
DEFAULT_BODY_MOUNT = "pelvis"


# --------------------------------------------------------------------------- #
# convention bridge (pure math, offline-testable)
# --------------------------------------------------------------------------- #
def bridge_quat() -> np.ndarray:
    """The fixed Isaac->MuJoCo camera orientation bridge, (w, x, y, z)."""
    return quat.camera_bridge_quat()


def _quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (4,) wxyz quaternions."""
    return np.asarray(quat.mul(a, b))


def mj_camera_quat(q_isaac_wxyz) -> np.ndarray:
    """Isaac-convention camera quaternion -> MuJoCo camera quaternion.

    Isaac form: (w, x, y, z), forward +X, up +Z.  MuJoCo form: forward
    -Z, up +Y.  Bridge: ``q_isaac ⊗ q_bridge``.
    """
    return quat.camera_isaac_to_mujoco(q_isaac_wxyz)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate of a (4,) wxyz quaternion (= inverse for unit quats)."""
    return np.asarray(quat.conj(q))


def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """(4,) wxyz unit quat -> (3,3) rotation matrix (world <- local)."""
    return np.asarray(quat.to_rotmat(q_wxyz))


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """(3,3) rotation matrix (world <- local) -> (4,) wxyz quat.

    Shepperd's method; the positive-w branch.
    """
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def fovy_from_spec(spec: CameraSpec) -> float:
    """The MuJoCo ``fovy`` (vertical FOV, deg) reproducing the spec's pinhole.

    The spec solves the focal length from the HORIZONTAL fov
    (``focal = aperture / (2 tan(hfov/2))``); MJ cameras are parameterized
    by the vertical fov, so ``tan(vfov/2) = tan(hfov/2) * (h / w)``.
    """
    tan_h = math.tan(math.radians(float(spec.hfov_deg)) / 2.0)
    tan_v = tan_h * float(spec.height) / float(spec.width)
    return math.degrees(2.0 * math.atan(tan_v))


def _rig_offset_quat(spec: CameraSpec) -> np.ndarray:
    """Isaac-convention mount quaternion (wxyz), standard resolution order.

    Same order as ``isaaclab_backend.build_sensor_cfg``: explicit
    ``rig.rot`` wins,
    ``body`` defaults to the pitch-down quat, world/chase to pos->lookat.
    """
    rig = spec.rig or {}
    rtype = rig_type(spec)
    if rig.get("rot") is not None:
        return np.asarray([float(x) for x in rig["rot"]], dtype=np.float64)
    if rtype == "body":
        pitch = math.radians(float(rig.get("pitch_down_deg", 0.0)))
        return np.array([math.cos(pitch / 2.0), 0.0,
                         math.sin(pitch / 2.0), 0.0])
    pos = tuple(float(x) for x in rig.get("pos", (0.0, 0.0, 0.0)))
    lookat = rig.get("lookat", (1.0, 0.0, 0.25))
    return np.asarray(lookat_quat(pos, lookat), dtype=np.float64)


# --------------------------------------------------------------------------- #
# MjSpec injection (pre-compile)
# --------------------------------------------------------------------------- #
def inject_spec_camera(spec: CameraSpec, mjspec) -> None:
    """Add the rig's ``<camera>`` element to an ``mujoco.MjSpec``.

    Mounting (M2b ruling; ``attach`` = the BodyRig link mount):
      * ``body``  -> attached to the mount body (``rig.attach``; default
        ``pelvis``) with the mount-frame ``rig.pos`` offset — the body
        attach makes the camera follow the robot for free (no per-step
        re-posing, unlike the isaaclab scene-level sensor prim).
      * ``world`` -> worldbody at the world-frame ``rig.pos``.
      * ``chase`` -> worldbody placeholder; ``sync_pose`` overwrites
        ``model.cam_pos/cam_quat`` every step tail.
    """
    rtype = rig_type(spec)
    name = MJ_CAMERA_PREFIX + spec.name
    fovy = fovy_from_spec(spec)
    rig = spec.rig or {}
    try:
        if rtype == "body":
            body_name = str(rig.get("attach") or DEFAULT_BODY_MOUNT)
            body = mjspec.body(body_name)
            if body is None:
                # same fail-fast-with-candidates pattern as the contacts
                # package (contacts/discovery.py): name-addressed mounts
                # list the addressable bodies at the failure site
                bodies = [b.name for b in mjspec.bodies if b.name]
                raise ValueError(
                    f"body {body_name!r} not found -- available mount "
                    f"bodies: {bodies}")
            pos = [float(x) for x in rig.get("pos", (0.0, 0.0, 0.0))]
            quat = mj_camera_quat(_rig_offset_quat(spec))
        elif rtype == "world":
            body = mjspec.worldbody
            pos = [float(x) for x in rig.get("pos", (0.0, 0.0, 0.0))]
            quat = mj_camera_quat(_rig_offset_quat(spec))
        elif rtype == "chase":
            body = mjspec.worldbody
            pos = [0.0, 0.0, 1.0]           # placeholder; overwritten per step
            quat = bridge_quat()
        elif rtype == "fixed":
            raise ValueError(
                f"camera {spec.name!r}: rig type 'fixed' was renamed "
                f"'world' (camera unification) -- update the config; "
                f"there is no alias")
        else:
            raise ValueError(f"camera {spec.name!r}: unknown rig type "
                             f"{rtype!r} (body|world|chase)")
    except (KeyError, ValueError) as e:
        if rtype == "body":
            raise ValueError(
                f"camera {spec.name!r}: mount body "
                f"{rig.get('body') or DEFAULT_BODY_MOUNT!r} not found in the "
                f"MJ model ({e})") from e
        raise
    body.add_camera(name=name, pos=pos, quat=[float(q) for q in quat],
                    fovy=fovy)
