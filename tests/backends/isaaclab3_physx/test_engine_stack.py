"""P0 stack probes for the IsaacLab 3 PhysX lane."""

from __future__ import annotations

import argparse

import pytest

from unigym.backends import registry
from unigym.backends.isaaclab3_physx.env import Isaaclab3PhysxEnv
from unigym.backends.isaaclab3_physx.profile import PHYSX_PROFILE


def test_env_shell_wiring() -> None:
    assert Isaaclab3PhysxEnv.backend_name == "isaaclab3_physx"
    assert Isaaclab3PhysxEnv.engine_profile is PHYSX_PROFILE
    assert Isaaclab3PhysxEnv.asset_formats == ("usd", "urdf")


def test_registry_probes_isaaclab_3() -> None:
    version = registry.isaaclab_version()
    assert version is not None
    assert version[0] >= 3


def test_registry_selects_the_physx_identity() -> None:
    assert registry.select_backend("isaaclab3_physx") == "isaaclab3_physx"
    assert registry.build_name("isaaclab3_physx") == "isaaclab3_physx"
    assert "isaaclab3" not in registry.CANONICAL_BACKENDS
    assert "isaaclab" not in registry.CANONICAL_BACKENDS

    for alias in ("isaaclab3", "isaaclab"):
        with pytest.raises(ValueError, match=r"aliases were removed"):
            registry.build_name(alias)
        with pytest.raises(registry.BackendSelectionError):
            registry.select_backend(alias)


def test_physx_physics_cfg_constructs() -> None:
    from isaaclab.sim.simulation_cfg import SimulationCfg
    from isaaclab_physx.physics import PhysxCfg

    cfg = SimulationCfg(physics=PhysxCfg())
    assert "physx" in cfg.physics.class_type.lower()


def test_launcher_args_drop_headless_and_enable_cameras() -> None:
    """rc1 moves headless/camera handling behind the visualizer config."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    assert not hasattr(args, "headless")
    assert not hasattr(args, "enable_cameras")
