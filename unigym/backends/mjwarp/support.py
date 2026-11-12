"""Small engine/device helpers for the mjwarp backend.

Split out of ``env.py`` so that module keeps its maintainability index; the
helpers are stateless (device resolution and option-enum resolution).
"""

from __future__ import annotations

import numpy as np

from unigym.core import quat


def _quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    """(4,) wxyz unit quat -> (3,3) rotation matrix (world<-body rows).

    (DELEGATED to unigym.core.quat.to_rotmat.)
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
        raise ValueError(f"MjwarpEnv device must be 'cpu' or 'cuda:N', got {device!r}")
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

    for enum_t in (
        mujoco.mjtIntegrator,
        mujoco.mjtCone,
        mujoco.mjtJoint,
        mujoco.mjtSolver,
        mujoco.mjtEnableBit,
    ):
        member = getattr(enum_t, str(val), None)
        if member is not None:
            return int(member)
    raise ValueError(f"mujoco backend: unknown enum member {val!r} in mj_opt")


def _put_warp_model(model, camera_manager, num_envs: int):
    """``put_model`` with per-world camera poses batched at N>1.

    A camera manager at N>1 needs ``cam_pos``/``cam_quat`` as batch
    fields (one pose column per world).  N=1 -- or no cameras -- keeps the
    exact historical unbatched ``put_model`` call (bit-identical path).
    """
    import mujoco_warp as mjw

    n = int(num_envs)
    if camera_manager is not None and n > 1:
        return mjw.put_model(model, batch_sizes={"cam_pos": n, "cam_quat": n})
    return mjw.put_model(model)
