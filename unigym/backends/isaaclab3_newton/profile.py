"""Engine profile for the IsaacLab 3 Newton backend identity.

Module import is engine-free: the profile only imports
:class:`EngineProfile`; every isaaclab import lives inside a function body and
runs when the shared env asks the profile for a value.
"""

from __future__ import annotations

from unigym.backends.shared.isaaclab3_common.engines import EngineProfile

__all__ = ["NEWTON_PROFILE"]


def _build_physics_cfg():
    """Build the Newton ``SimulationCfg.physics`` value (MJWarp solver)."""
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    return NewtonCfg(solver_cfg=MJWarpSolverCfg())


def _renderer_cfg():
    """Build the Newton (Warp) camera renderer cfg."""
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    return NewtonWarpRendererCfg()


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
    """Write body CoM positions via ``set_coms_index`` (positions only)."""
    import torch

    setter = getattr(asset, "set_coms_index", None)
    if not callable(setter):
        raise TypeError("body-CoM asset exposes no set_coms_index method")
    setter(coms=torch.as_tensor(coms, dtype=torch.float32), env_ids=env_ids)


NEWTON_PROFILE = EngineProfile(
    name="newton",
    build_physics_cfg=_build_physics_cfg,
    renderer_cfg=_renderer_cfg,
    depth_background=0.0,
    default_joint_ordering="mjwarp",
    get_body_com=_get_body_com,
    set_body_com=_set_body_com,
)
