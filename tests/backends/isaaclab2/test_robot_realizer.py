"""The isaaclab robot realizer builds the expected G1 ArticulationCfg."""

from __future__ import annotations

import math

import pytest


def test_g1_realizer_builds_the_expected_articulation_cfg():
    """Spawn, contact sensors, joints and actuator groups come from the profile."""
    from isaaclab.assets import ArticulationCfg

    from unigym.backends.isaaclab2.robots.g1 import realize_g1
    from unigym.robots import build_robot_profile
    from unigym.robots.g1_constants import G1_EFFORT_LEGS

    params = {
        "spawn_height": 0.91,
        "spawn_x": 1.0,
        "spawn_y": 2.0,
        "spawn_yaw_deg": 90.0,
    }
    profile = build_robot_profile({"name": "g1", "params": params})

    cfg = realize_g1(profile, params=params)

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


def test_g1_realizer_adds_the_hands_group_only_for_the_hand_face():
    """The hand actuator group follows the profile's declared action face."""
    from unigym.backends.isaaclab2.robots.g1 import realize_g1
    from unigym.robots import build_robot_profile

    params = {"actuate_hands": True}
    profile = build_robot_profile({"name": "g1", "params": params})

    cfg = realize_g1(profile, params=params)

    assert "hands" in cfg.actuators


def test_g1_realizer_uses_the_profile_asset_override():
    """A declared USD asset override reaches the spawn config."""
    from unigym.backends.isaaclab2.robots.g1 import realize_g1
    from unigym.robots import build_robot_profile

    params = {"usd_path": "/tmp/g1_override.usd"}
    profile = build_robot_profile({"name": "g1", "params": params})

    cfg = realize_g1(profile, params=params)

    assert cfg.spawn.usd_path == "/tmp/g1_override.usd"


def test_isaaclab_realizer_rejects_a_profile_without_a_realization():
    """GR1 has no isaaclab realization and the seam fails fast."""
    from unigym.backends.isaaclab2.robots import build_robot_realization
    from unigym.robots import build_robot_profile

    profile = build_robot_profile({"name": "gr1"})

    with pytest.raises(NotImplementedError, match="no robot realizer.*gr1"):
        build_robot_realization(profile, params={})
