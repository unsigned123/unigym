"""The isaaclab robot realizer builds the expected G1 ArticulationCfg."""

from __future__ import annotations

import math

import pytest


def test_g1_realizer_builds_the_expected_articulation_cfg():
    """Spawn, joints and actuator groups come from the profile/spec."""
    from isaaclab.assets import ArticulationCfg

    from unigym.backends.isaaclab2.robots.g1 import realize_g1
    from unigym.robots import build_robot_profile
    from unigym.robots.g1_constants import G1_EFFORT_LEGS
    from unigym.scenes.articulation import ArticulationSpec

    profile = build_robot_profile({"name": "g1"})
    spec = ArticulationSpec(name="g1", spawn={"pos": [1.0, 2.0, 0.91], "yaw": math.pi / 2})

    cfg = realize_g1(profile, spec=spec, params={})

    assert isinstance(cfg, ArticulationCfg)
    assert cfg.prim_path == "{ENV_REGEX_NS}/Robot"
    assert cfg.spawn.activate_contact_sensors is True
    assert cfg.init_state.pos == (1.0, 2.0, 0.91)
    assert cfg.init_state.rot == pytest.approx(
        (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    )
    assert cfg.init_state.joint_pos == dict(profile.default_joint_pos)
    assert set(cfg.actuators) == {"legs", "feet", "waist", "arms"}
    assert cfg.actuators["legs"].effort_limit_sim == G1_EFFORT_LEGS


def test_g1_realizer_absent_spawn_uses_the_backend_default_height():
    """Absent spec.spawn keeps the G1 backend default height and zero yaw."""
    from unigym.backends.isaaclab2.robots.g1 import realize_g1
    from unigym.robots import build_robot_profile
    from unigym.robots.g1_constants import G1_SPAWN_HEIGHT
    from unigym.scenes.articulation import ArticulationSpec

    profile = build_robot_profile({"name": "g1"})

    cfg = realize_g1(profile, spec=ArticulationSpec(name="g1"), params={})

    assert cfg.init_state.pos == (0.0, 0.0, G1_SPAWN_HEIGHT)
    assert cfg.init_state.rot == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_g1_realizer_adds_the_hands_group_only_for_the_hand_face():
    """The hand actuator group follows the profile's declared action face."""
    from unigym.backends.isaaclab2.robots.g1 import realize_g1
    from unigym.robots import build_robot_profile
    from unigym.scenes.articulation import ArticulationSpec

    params = {"actuate_hands": True}
    profile = build_robot_profile({"name": "g1", "params": params})

    cfg = realize_g1(profile, spec=ArticulationSpec(name="g1"), params=params)

    assert "hands" in cfg.actuators


def test_g1_realizer_uses_the_profile_asset_override():
    """A declared USD asset override reaches the spawn config."""
    from unigym.backends.isaaclab2.robots.g1 import realize_g1
    from unigym.robots import build_robot_profile
    from unigym.scenes.articulation import ArticulationSpec

    params = {"usd_path": "/tmp/g1_override.usd"}
    profile = build_robot_profile({"name": "g1", "params": params})

    cfg = realize_g1(profile, spec=ArticulationSpec(name="g1"), params=params)

    assert cfg.spawn.usd_path == "/tmp/g1_override.usd"


def test_isaaclab_realizer_rejects_a_profile_without_a_realization():
    """GR1 has no isaaclab realization and the seam fails fast."""
    from unigym.backends.isaaclab2.robots import build_robot_realization
    from unigym.robots import build_robot_profile
    from unigym.scenes.articulation import ArticulationSpec

    profile = build_robot_profile({"name": "gr1"})

    with pytest.raises(NotImplementedError, match="no robot realizer.*gr1"):
        build_robot_realization(profile, spec=ArticulationSpec(name="gr1"), params={})
