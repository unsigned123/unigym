"""P0 stack probes for the IsaacLab 3 PhysX lane."""

from __future__ import annotations

import argparse

from unigym.backends import registry


def test_registry_probes_isaaclab_3() -> None:
    version = registry.isaaclab_version()
    assert version is not None
    assert version[0] >= 3


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
