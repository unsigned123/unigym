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


def _set_body_com(asset: object, com: object, env_ids: object) -> None:
    """Write body CoM positions for the DR event (lands in C2)."""
    raise NotImplementedError("IsaacLab 3 Newton set_body_com lands in C2 (set_coms_index)")


NEWTON_PROFILE = EngineProfile(
    name="newton",
    build_physics_cfg=_build_physics_cfg,
    renderer_cfg=_renderer_cfg,
    depth_background=0.0,
    default_joint_ordering="mjwarp",
    set_body_com=_set_body_com,
)
