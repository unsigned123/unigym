"""Engine profile for the IsaacLab 3 PhysX backend identity.

Module import is engine-free: the profile only imports
:class:`EngineProfile` and ``math``; every isaaclab import lives inside a
function body and runs when the shared env asks the profile for a value.
"""

from __future__ import annotations

import math

from unigym.backends.shared.isaaclab3_common.engines import EngineProfile

__all__ = ["PHYSX_PROFILE"]


def _build_physics_cfg():
    """Build the PhysX ``SimulationCfg.physics`` value."""
    from isaaclab_physx.physics import PhysxCfg

    return PhysxCfg()


def _renderer_cfg():
    """Build the PhysX (RTX) camera renderer cfg."""
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    return IsaacRtxRendererCfg()


def _get_body_com(asset: object):
    """Read body CoM positions as a float32 ``(N, B, 3)`` tensor."""
    import torch

    data = getattr(asset, "data", None)
    if data is None:
        raise TypeError("body-CoM asset exposes no .data surface")
    value = getattr(data, "body_com_pos_b", None)
    if value is None:
        raise TypeError("body-CoM data exposes no body_com_pos_b surface")
    payload = getattr(value, "torch", value)
    return torch.as_tensor(payload, dtype=torch.float32)


def _set_body_com(asset: object, coms: object, env_ids: object) -> None:
    """Write body CoM positions via ``set_coms_index`` (identity xyzw quat)."""
    import torch

    setter = getattr(asset, "set_coms_index", None)
    if not callable(setter):
        raise TypeError("body-CoM asset exposes no set_coms_index method")
    pos = torch.as_tensor(coms, dtype=torch.float32)
    data = getattr(asset, "data", None)
    pose_value = getattr(data, "body_com_pose_b", None) if data is not None else None
    ids = env_ids
    if pose_value is not None:
        payload = getattr(pose_value, "torch", pose_value)
        current = torch.as_tensor(payload, dtype=torch.float32)
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=current.device)
        rows = current[ids].clone()
        rows[..., :3] = pos
    else:
        quat = torch.zeros(*pos.shape[:-1], 4, dtype=pos.dtype, device=pos.device)
        quat[..., 3] = 1.0  # identity in il3 xyzw
        rows = torch.cat((pos, quat), dim=-1)
    setter(coms=rows, env_ids=ids)


PHYSX_PROFILE = EngineProfile(
    name="physx",
    build_physics_cfg=_build_physics_cfg,
    renderer_cfg=_renderer_cfg,
    depth_background=math.inf,
    default_joint_ordering="physx",
    get_body_com=_get_body_com,
    set_body_com=_set_body_com,
)
