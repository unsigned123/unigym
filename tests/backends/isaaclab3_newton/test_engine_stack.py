"""P0 stack probes for the IsaacLab 3 Newton lane."""

from __future__ import annotations

from unigym.backends.isaaclab3_newton.env import Isaaclab3NewtonEnv
from unigym.backends.isaaclab3_newton.profile import NEWTON_PROFILE


def test_env_shell_wiring() -> None:
    assert Isaaclab3NewtonEnv.backend_name == "isaaclab3_newton"
    assert Isaaclab3NewtonEnv.engine_profile is NEWTON_PROFILE
    assert Isaaclab3NewtonEnv.asset_formats == ("usd", "urdf")


def test_registry_selects_the_newton_identity() -> None:
    from unigym.backends import registry

    assert registry.select_backend("isaaclab3_newton") == "isaaclab3_newton"
    assert registry.newton_version() is not None


def test_newton_physics_cfg_constructs() -> None:
    from isaaclab.sim.simulation_cfg import SimulationCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    cfg = SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
    assert "newton" in cfg.physics.class_type.lower()
    assert isinstance(cfg.physics.solver_cfg, MJWarpSolverCfg)


def test_factory_backend_without_context_is_newton() -> None:
    """Regression lock for the rc1 construction-order trap."""
    from isaaclab.sim.simulation_context import SimulationContext
    from isaaclab.utils.backend_utils import FactoryBase

    SimulationContext.clear_instance()
    assert FactoryBase._get_backend() == "newton"


def test_newton_warp_renderer_constructs() -> None:
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    assert NewtonWarpRendererCfg() is not None


def test_camera_default_renderer_is_rtx() -> None:
    """The default renderer is Kit/RTX; Newton must opt in explicitly."""
    from isaaclab.utils.backend_utils import get_default_renderer_cfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    assert isinstance(get_default_renderer_cfg(), IsaacRtxRendererCfg)
