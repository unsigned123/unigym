"""Private MuJoCo candidate probes for explicit collision geom bounds."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - capability is queried at runtime
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc

__all__ = [
    "CollisionPair",
    "ProbeCapabilityError",
    "ProbeError",
    "ProbeResult",
    "probe_freejoint_candidate",
]


class ProbeError(ValueError):
    """Invalid model references or candidate input."""


class ProbeCapabilityError(RuntimeError):
    """MuJoCo or the declared collision query cannot provide the requested fact."""


@dataclass(frozen=True)
class CollisionPair:
    """One explicitly checked candidate geom and obstacle geom pair."""

    candidate_geom: str
    obstacle_geom: str
    allow_support_contact: bool = False
    penetration_tolerance_m: float = 1e-7


@dataclass(frozen=True)
class ProbeResult:
    """Probe verdict and measured signed distances, in metres."""

    accepted: bool
    reason: str | None
    distances_m: tuple[float, ...]


def _id(model, kind, name: str) -> int:
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ProbeError(f"MuJoCo {kind.name} {name!r} was not found")
    return value


def _pose(position, quaternion_wxyz):
    try:
        pos = np.asarray(position, dtype=np.float64)
        quat = np.asarray(quaternion_wxyz, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ProbeError("position and quaternion must be numeric vectors") from exc
    if (pos.shape != (3,) or quat.shape != (4,)
            or not np.isfinite(pos).all() or not np.isfinite(quat).all()):
        raise ProbeError("position must be finite XYZ and quaternion finite WXYZ")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ProbeError("quaternion must be non-zero")
    return pos, quat / norm


def probe_freejoint_candidate(
    model,
    live_data,
    joint_name: str,
    position,
    quaternion_wxyz,
    pairs: Iterable[CollisionPair],
    min_clearance_m: float = 0.0,
) -> ProbeResult:
    """Evaluate a freejoint pose without modifying live MjData.

    Collision and positive clearance are evaluated only for caller-declared
    geom pairs using MuJoCo's signed narrow-phase geom distance. The caller
    supplies simple collision geoms in the model; this does not infer a region
    transform, collision envelope, unplaced-object parking pose, or all pairs.
    """
    if mujoco is None:  # pragma: no cover
        raise ProbeCapabilityError(
            f"MuJoCo placement probes are unavailable: {_MUJOCO_IMPORT_ERROR}")
    try:
        clearance = float(min_clearance_m)
    except (TypeError, ValueError) as exc:
        raise ProbeError("min_clearance_m must be a finite non-negative number") from exc
    if not math.isfinite(clearance) or clearance < 0:
        raise ProbeError("min_clearance_m must be a finite non-negative number")
    pos, quat = _pose(position, quaternion_wxyz)
    pair_list = tuple(pairs)
    if any(not isinstance(pair, CollisionPair) for pair in pair_list):
        raise ProbeError("pairs must contain CollisionPair values")
    jid = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ProbeError(f"joint {joint_name!r} is not a freejoint")
    qadr = int(model.jnt_qposadr[jid])

    # All forward and distance queries run on a fresh private data object.
    probe_data = mujoco.MjData(model)
    mujoco.mj_copyData(probe_data, model, live_data)
    probe_data.qpos[qadr:qadr + 3] = pos
    probe_data.qpos[qadr + 3:qadr + 7] = quat
    mujoco.mj_forward(model, probe_data)

    distances = []
    for pair in pair_list:
        geom1 = _id(model, mujoco.mjtObj.mjOBJ_GEOM, pair.candidate_geom)
        geom2 = _id(model, mujoco.mjtObj.mjOBJ_GEOM, pair.obstacle_geom)
        try:
            tolerance = float(pair.penetration_tolerance_m)
        except (TypeError, ValueError) as exc:
            raise ProbeError("penetration_tolerance_m must be finite and non-negative") from exc
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ProbeError("penetration_tolerance_m must be finite and non-negative")
        distance = float(mujoco.mj_geomDistance(
            model, probe_data, geom1, geom2, max(1.0, clearance + tolerance), None))
        distances.append(distance)
        if distance < -tolerance:
            return ProbeResult(False, "penetration", tuple(distances))
        if clearance > 0 and distance < clearance:
            # A support exception is only for near-zero contact, never overlap.
            if not (pair.allow_support_contact and distance >= -tolerance
                    and distance <= tolerance):
                return ProbeResult(False, "insufficient_clearance", tuple(distances))
    return ProbeResult(True, None, tuple(distances))
