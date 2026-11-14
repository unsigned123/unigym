"""Passive-joint authoring sugar over a real robot profile."""

from __future__ import annotations

import pytest

from unigym.robots import register_robot, restore_robot_registry, snapshot_robot_registry
from unigym.run.passive import declare_passive_joints
from tests._support.profiles import fixture_profile


@pytest.fixture(autouse=True)
def _fixture_robot():
    """Expose the fixture profile to passive-joint resolution."""
    snapshot = snapshot_robot_registry()
    register_robot("fixture_arm", fixture_profile)
    yield
    restore_robot_registry(snapshot)


def test_declare_passive_joints_covers_undriven_profile_joints():
    """Authoring sugar marks every undriven profile joint as passive."""
    profile = fixture_profile()
    actuated: list[str] = []
    for joints in profile.joint_sets.values():
        for joint in joints:
            if joint not in actuated:
                actuated.append(joint)
    driven = profile.joint_sets["body"][0]
    entry = {"name": "fixture_arm"}

    declare_passive_joints(
        entry,
        [{"name": "fixture_arm", "articulation": "fixture_arm", "joints": [driven]}],
    )

    assert set(entry["passive_joints"]) == set(actuated) - {driven}
    assert len(entry["passive_joints"]) == len(actuated) - 1


def test_declare_passive_joints_keeps_explicit_declaration():
    """An explicit caller passive_joints declaration always wins."""
    entry = {"name": "fixture_arm", "passive_joints": ["custom"]}

    declare_passive_joints(
        entry, [{"name": "fixture_arm", "articulation": "fixture_arm", "joints": []}]
    )

    assert entry["passive_joints"] == ["custom"]
