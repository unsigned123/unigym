"""Small engine/device helpers for the mjwarp backend (v094 P3/P4).

Split out of ``env.py`` so that module keeps its maintainability index; the
helpers are stateless (gain tables, action-feed scatter, device resolution,
option-enum resolution).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from unigym.core import quat
from unigym.robots.g1_constants import (
    G1_KD_ARMS,
    G1_KD_FEET,
    G1_KD_LEGS,
    G1_KD_WAIST,
    G1_KP_ARMS,
    G1_KP_FEET,
    G1_KP_LEGS,
    G1_KP_WAIST,
)


def _gather_gain(joint: str, table) -> Optional[float]:
    """First regex (fullmatch) hit in a g1_constants gain group table."""
    import re
    for expr, val in table.items():
        if re.fullmatch(expr, joint):
            return float(val)
    return None


def _policy_gains(joint_names, kp_override=None, kd_override=None):
    """Per-policy-joint (kp, kd) from the g1_constants playground tables.

    Groups tried in order: legs / feet / waist / arms (the isaaclab profile's
    implicit-PD groups, same numbers).  yaml ``kp``/``kd`` overrides (joint
    name or regex keys, applied via the same merge semantics as the profile)
        take precedence per joint.
    """
    groups_kp = (G1_KP_LEGS, G1_KP_FEET, G1_KP_WAIST, G1_KP_ARMS)
    groups_kd = (G1_KD_LEGS, G1_KD_FEET, G1_KD_WAIST, G1_KD_ARMS)
    kp, kd = [], []
    for name in joint_names:
        kv = _override_hit(name, kp_override)
        dv = _override_hit(name, kd_override)
        if kv is None:
            for tab in groups_kp:
                kv = _gather_gain(name, tab)
                if kv is not None:
                    break
        if dv is None:
            for tab in groups_kd:
                dv = _gather_gain(name, tab)
                if dv is not None:
                    break
        if kv is None or dv is None:
            raise ValueError(
                f"mujoco backend: no PD gain group matches policy joint "
                f"{name!r} (kp={kv}, kd={dv}); extend the gain tables or "
                f"set env.params.kp/kd")
        kp.append(kv)
        kd.append(dv)
    return np.asarray(kp, dtype=np.float64), np.asarray(kd, dtype=np.float64)


def _override_hit(name: str, override) -> Optional[float]:
    if not override:
        return None
    import re
    if name in override:
        return float(override[name])
    for expr, val in override.items():
        if re.fullmatch(str(expr), name):
            return float(val)
    return None


def _quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """(4,) wxyz unit quat -> (3,3) rotation matrix (world<-body rows).

    (DELEGATED to unigym.core.quat.to_rotmat, M1 convergence)
    """
    return np.asarray(quat.to_rotmat(quat_wxyz))


def _resolve_warp_device(device) -> str:
    """Normalize/set the warp device ("cpu" or "cuda:N"; default cuda:0).

    The warp stack is initialized once here (idempotent).  A CUDA request
    on a host without a visible GPU fails fast with the warp error -- never
    a silent CPU fallback.
    """
    import warp as _wp

    _wp.init()  # idempotent across warp versions (no is_initialized in 1.19)
    name = str(device or "").strip().lower()
    if name in ("", "auto", "none", "default"):
        devices = _wp.get_devices()
        name = "cuda:0" if any(d.is_cuda for d in devices) else "cpu"
    elif name == "cuda":
        name = "cuda:0"
    if name != "cpu" and not name.startswith("cuda:"):
        raise ValueError(
            f"MjwarpEnv device must be 'cpu' or 'cuda:N', got {device!r}")
    _wp.set_device(name)
    print(f"[mjwarp_env] warp device: {name}", flush=True)
    return name


def _mj_enum(val):
    """Resolve a mujoco option enum by member name or int.

    E.g. 'mjINT_IMPLICITFAST' -> the plain int value (setattr target for
    model.opt).
    """
    if isinstance(val, (int, float)):
        return int(val)
    import mujoco
    for enum_t in (mujoco.mjtIntegrator, mujoco.mjtCone, mujoco.mjtJoint,
                   mujoco.mjtSolver, mujoco.mjtEnableBit):
        member = getattr(enum_t, str(val), None)
        if member is not None:
            return int(member)
    raise ValueError(f"mujoco backend: unknown enum member {val!r} in mj_opt")


def _put_warp_model(model, camera_manager, num_envs: int):
    """``put_model`` with per-world camera poses batched at N>1.

    v095 C3: a camera manager at N>1 needs ``cam_pos``/``cam_quat`` as batch
    fields (one pose column per world).  N=1 -- or no cameras -- keeps the
    exact historical unbatched ``put_model`` call (bit-identical path).
    """
    import mujoco_warp as mjw

    n = int(num_envs)
    if camera_manager is not None and n > 1:
        return mjw.put_model(model, batch_sizes={"cam_pos": n,
                                                 "cam_quat": n})
    return mjw.put_model(model)
