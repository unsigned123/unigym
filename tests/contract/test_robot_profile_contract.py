"""Generic neutral-profile and backend-seam contracts independent of robot packs."""

from __future__ import annotations

import dataclasses
import sys

import pytest

from unigym.backends.registry import BackendInfo, backend_info
from unigym.robots import build_robot_profile, register_robot, restore_robots, snapshot_robots
from unigym.robots.base import RobotProfile
from tests._support.profiles import fixture_profile

_ENGINE_ROOTS = ("isaaclab", "isaaclab_assets", "omni", "isaacsim", "mujoco", "mujoco_warp")


@pytest.fixture(autouse=True)
def _fixture_robot():
    """Register one neutral factory for the resolution path."""
    snapshot = snapshot_robots()
    register_robot("fixture_arm", fixture_profile)
    yield
    restore_robots(snapshot)


def test_building_a_neutral_profile_imports_no_engine():
    """Neutral profile construction stays independent of every backend."""
    before = set(sys.modules)
    profile = build_robot_profile({"name": "fixture_arm"})

    assert isinstance(profile, RobotProfile)
    added = {
        name.split(".")[0]
        for name in set(sys.modules) - before
        if name.split(".")[0] in _ENGINE_ROOTS
    }
    assert added == set()


def test_neutral_profile_carries_no_backend_object_or_switch():
    """A neutral profile has no backend object, body_parts or RobotAsset field."""
    fields = {item.name for item in dataclasses.fields(RobotProfile)}
    assert "articulation_cfg" not in fields
    assert "with_isaaclab" not in fields
    assert "body_parts" not in fields
    assert "assets" in fields

    profile = fixture_profile()
    assert not hasattr(profile, "articulation_cfg")
    assert not hasattr(profile, "with_isaaclab")
    assert not hasattr(profile, "body_parts")

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
    assert not hasattr(robots, "RobotAsset")
    assert not hasattr(base, "RobotAsset")


@pytest.mark.parametrize(
    ("backend", "asset_format"),
    [("mujoco", "mjcf"), ("mjwarp", "mjcf"), ("isaaclab2", "usd"), ("isaaclab3", "usd")],
)
def test_every_backend_declares_its_asset_format(backend: str, asset_format: str):
    """The selected asset format is backend-owned and complete."""
    assert backend_info(backend).asset_format == asset_format
