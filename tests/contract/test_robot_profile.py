"""Neutral robot profiles are backend-free data with the expected facts."""

from __future__ import annotations

import dataclasses
import inspect
import sys

import pytest

from unigym.backends.registry import BackendInfo, backend_info
from unigym.robots import build_robot_profile
from unigym.robots import g1_constants as g1c
from unigym.robots import gr1_constants as gr1c
from unigym.robots.base import RobotAsset, RobotProfile
from unigym.robots.g1 import make_g1_profile
from unigym.robots.gr1 import make_gr1_profile

_ENGINE_ROOTS = ("isaaclab", "isaaclab_assets", "omni", "isaacsim", "mujoco", "mujoco_warp")


def test_building_neutral_profiles_imports_no_engine():
    """Robot import/profile construction stays independent of every backend."""
    before = set(sys.modules)
    profile = build_robot_profile({"name": "g1"})
    build_robot_profile({"name": "gr1"})

    assert isinstance(profile, RobotProfile)
    added = {
        name.split(".")[0]
        for name in set(sys.modules) - before
        if name.split(".")[0] in _ENGINE_ROOTS
    }
    assert added == set()


def test_g1_neutral_profile_matches_the_deployment_facts():
    """The neutral G1 profile preserves joint sets, q0, feet and body parts."""
    profile = build_robot_profile({"name": "g1"})

    assert profile.name == "g1"
    assert profile.joint_sets["body"] == tuple(g1c.G1_JOINT_ORDER_IL)
    assert profile.joint_sets["hand"] == tuple(g1c.G1_HAND_JOINTS)
    assert tuple(profile.foot_bodies) == tuple(g1c.G1_FOOT_BODIES)
    assert dict(profile.body_parts) == g1c.G1_BODY_PARTS
    expected_q0 = dict(g1c.G1_DEFAULT_Q)
    expected_q0.update({joint: 0.0 for joint in g1c.G1_HAND_JOINTS})
    assert dict(profile.default_joint_pos) == expected_q0
    assert profile.joint_order == list(g1c.G1_JOINT_ORDER_IL)
    assert profile.full_joint_names == list(g1c.G1_JOINT_ORDER_FULL)
    assert dict(profile.action_faces) == {}
    assert profile.assets == ()


def test_g1_hand_face_and_usd_asset_are_declared_as_profile_data():
    """``actuate_hands`` and ``usd_path`` change neutral data, not a backend flag."""
    profile = build_robot_profile(
        {
            "name": "g1",
            "params": {"actuate_hands": True, "usd_path": "/tmp/g1.usd"},
        }
    )

    assert dict(profile.action_faces) == {"hand": tuple(g1c.G1_HAND_JOINTS)}
    assert profile.assets == (RobotAsset(format="usd", ref="/tmp/g1.usd"),)


def test_g1_default_q_patterns_expand_on_the_neutral_profile():
    """Regex q0 overrides become literal joint names before the realizer."""
    profile = build_robot_profile(
        {
            "name": "g1",
            "params": {"default_q": {".*_hip_pitch_joint": -0.125}},
        }
    )

    assert profile.default_joint_pos["left_hip_pitch_joint"] == pytest.approx(-0.125)
    assert profile.default_joint_pos["right_hip_pitch_joint"] == pytest.approx(-0.125)


def test_gr1_neutral_profile_matches_the_tabletop_facts():
    """The neutral GR1 profile preserves joint sets, q0, body parts and faces."""
    profile = build_robot_profile({"name": "gr1"})

    assert profile.name == "gr1"
    assert profile.joint_sets == {
        "body": tuple(gr1c.GR1_JOINT_ORDER),
        "right_hand": tuple(gr1c.GR1_RIGHT_HAND_DRIVE_ORDER),
        "left_hand": tuple(gr1c.GR1_LEFT_HAND_DRIVE_ORDER),
    }
    assert tuple(profile.foot_bodies) == ()
    assert dict(profile.body_parts) == gr1c.GR1_BODY_PARTS
    assert profile.full_joint_names == list(gr1c.GR1_JOINT_ORDER_MJ)
    assert dict(profile.action_faces) == {
        "right_hand": tuple(gr1c.GR1_RIGHT_HAND_DRIVE_ORDER),
        "left_hand": tuple(gr1c.GR1_LEFT_HAND_DRIVE_ORDER),
    }
    assert profile.assets == ()
    hand_joints = [name for name in gr1c.GR1_JOINT_ORDER_MJ if name not in gr1c.GR1_JOINT_ORDER]
    expected_q0 = dict(gr1c.GR1_INIT_QPOS)
    expected_q0.update({joint: 0.0 for joint in hand_joints})
    assert dict(profile.default_joint_pos) == expected_q0


def test_robot_factories_have_no_backend_flag():
    """Neither robot factory accepts ``with_isaaclab`` any more."""
    assert "with_isaaclab" not in inspect.signature(make_g1_profile).parameters
    assert "with_isaaclab" not in inspect.signature(make_gr1_profile).parameters


@pytest.mark.parametrize("robot", ["g1", "gr1"])
def test_profile_carries_no_backend_object_or_switch(robot):
    """A neutral profile has no ArticulationCfg field and no backend flag."""
    fields = {item.name for item in dataclasses.fields(RobotProfile)}
    assert "articulation_cfg" not in fields
    assert "with_isaaclab" not in fields

    profile = build_robot_profile({"name": robot})
    assert not hasattr(profile, "articulation_cfg")
    assert not hasattr(profile, "with_isaaclab")

    info_fields = {item.name for item in dataclasses.fields(BackendInfo)}
    assert "recipe_with_isaaclab" not in info_fields


def test_backend_robot_realizer_is_the_only_backend_seam():
    """Only backends with an engine robot object declare a realizer."""
    assert backend_info("mujoco").robot_realizer is None
    assert backend_info("mjwarp").robot_realizer is None
    assert backend_info("isaaclab2").robot_realizer is not None
    assert backend_info("isaaclab3").robot_realizer is not None


def test_legacy_recipe_names_are_removed():
    """The cutover leaves no compatibility shim for the old recipe API."""
    import unigym.robots as robots
    import unigym.robots.base as base

    assert not hasattr(robots, "build_robot_recipe")
    assert not hasattr(robots, "RobotRecipe")
    assert not hasattr(base, "RobotRecipe")
