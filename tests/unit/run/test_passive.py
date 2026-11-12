"""Passive-joint authoring sugar over a real robot profile."""

from __future__ import annotations

from unigym.robots import build_robot_profile
from unigym.run.passive import declare_passive_joints


def test_declare_passive_joints_covers_undriven_profile_joints():
    """Authoring sugar marks every undriven profile joint as passive."""
    profile = build_robot_profile({"name": "g1"})
    actuated: list[str] = []
    for joints in profile.joint_sets.values():
        for joint in joints:
            if joint not in actuated:
                actuated.append(joint)
    driven = profile.joint_sets["body"][0]
    entry = {"name": "g1"}

    declare_passive_joints(
        entry,
        [{"name": "g1", "articulation": "g1", "joints": [driven]}],
    )

    assert set(entry["passive_joints"]) == set(actuated) - {driven}
    assert len(entry["passive_joints"]) == len(actuated) - 1


def test_declare_passive_joints_keeps_explicit_declaration():
    """An explicit caller passive_joints declaration always wins."""
    entry = {"name": "g1", "passive_joints": ["custom"]}

    declare_passive_joints(entry, [{"name": "g1", "articulation": "g1", "joints": []}])

    assert entry["passive_joints"] == ["custom"]
