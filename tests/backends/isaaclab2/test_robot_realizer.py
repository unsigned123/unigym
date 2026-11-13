"""The isaaclab realizer selects a declared asset format or fails explicitly."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.sim, pytest.mark.usefixtures("simulation_app")]


def test_isaaclab_realizer_rejects_a_profile_without_an_effective_asset():
    """A profile without any asset format is rejected by asset, not by name."""
    from unigym.assets.asset_ref import AssetRefError
    from unigym.backends.isaaclab2.robots import build_robot_realization
    from unigym.scenes.articulation import ArticulationSpec
    from tests._support.profiles import fixture_profile

    profile = fixture_profile(name="arm")

    with pytest.raises(AssetRefError, match="no asset reference is declared"):
        build_robot_realization(profile, spec=ArticulationSpec(name="arm"), params={})


def test_isaaclab_realizer_does_not_fall_back_from_a_model_without_a_format():
    """An explicit spec.model without USD/URDF never falls back to the profile."""
    from unigym.assets.asset_ref import AssetRef, AssetRefError
    from unigym.backends.isaaclab2.robots import build_robot_realization
    from unigym.scenes.articulation import ArticulationSpec
    from tests._support.profiles import fixture_profile

    profile = fixture_profile(name="arm", usd_path="/profile/arm.usd")
    spec = ArticulationSpec(name="arm", model=AssetRef(mjcf="model.xml"))

    with pytest.raises(AssetRefError, match="no compatible format"):
        build_robot_realization(profile, spec=spec, params={})


def test_isaaclab_realizer_builds_a_urdf_articulation_cfg(tiny_urdf):
    """A URDF-only effective asset builds the real UrdfFileCfg spawn."""
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners import UrdfFileCfg

    from unigym.actuators import ActuationGroup
    from unigym.assets.asset_ref import AssetRef
    from unigym.backends.isaaclab2.robots import build_robot_realization
    from unigym.scenes.articulation import ArticulationSpec
    from tests._support.profiles import fixture_profile

    profile = fixture_profile(name="arm", usd_path="/profile/arm.usd")
    spec = ArticulationSpec(
        name="arm",
        model=AssetRef(urdf=str(tiny_urdf)),
        root="fixed",
        spawn={"pos": [0.0, 0.0, 0.5]},
        actuation=(
            ActuationGroup(
                name="body",
                law="standard_pd",
                joints=("shoulder", "elbow"),
                gains={"kp": 1.0, "kd": 0.1},
            ),
        ),
    )

    cfg = build_robot_realization(profile, spec=spec, params={})

    assert isinstance(cfg, ArticulationCfg)
    assert isinstance(cfg.spawn, UrdfFileCfg)
    assert cfg.spawn.asset_path == str(tiny_urdf)
    assert cfg.spawn.fix_base is True
    assert cfg.spawn.merge_fixed_joints is False
    assert cfg.spawn.collision_from_visuals is False
    assert cfg.init_state.pos == (0.0, 0.0, 0.5)
    assert set(cfg.actuators) == {"body"}
