"""Pure-data control declaration contract (no engine involved)."""
from __future__ import annotations

import pytest

from unigym.runtime.controls import (
    ControlError,
    ControlGroup,
    target_for_drive,
    validate_controls,
)
from unigym.scenes.articulation import ArticulationSpec, JointSpec


def _group(**overrides):
    entry = {
        "name": "body",
        "adapter": "manual",
        "articulation": "arm",
        "joints": ["shoulder", "elbow"],
        "drive": "position_pd",
        "target": "position",
    }
    entry.update(overrides)
    return entry


def test_group_accepts_explicit_joints_and_joint_set():
    """Explicit joints and a named joint set are the two legal selectors."""
    by_joints = ControlGroup.from_mapping(_group())
    assert by_joints.joints == ("shoulder", "elbow")
    assert by_joints.joint_set is None
    by_set = ControlGroup.from_mapping(_group(joints=None, joint_set="body"))
    assert by_set.joint_set == "body"
    assert by_set.joints is None


@pytest.mark.parametrize(
    "drive,target",
    [("position_pd", "position"), ("position_servo", "position"),
     ("torque", "torque"), ("velocity", "velocity"),
     ("effort", "effort"), ("current", "current")],
)
def test_drive_target_pairs_map_explicitly(drive, target):
    """Every legal drive maps to exactly one target kind."""
    assert target_for_drive(drive) == target
    group = ControlGroup.from_mapping(_group(drive=drive, target=target))
    assert group.target == target


def test_drive_target_mismatch_fails_fast():
    """A drive may not be paired with a different target kind."""
    with pytest.raises(ControlError, match="requires target 'torque'"):
        ControlGroup.from_mapping(_group(drive="torque", target="position"))


def test_exactly_one_joint_selector_required():
    """Both selectors and neither selector are rejected."""
    with pytest.raises(ControlError, match="exactly one"):
        ControlGroup.from_mapping(_group(joint_set="body"))
    with pytest.raises(ControlError, match="exactly one"):
        ControlGroup.from_mapping(_group(joints=[]))
    with pytest.raises(ControlError, match="exactly one"):
        ControlGroup.from_mapping(_group(joints=None))


def test_duplicate_joints_and_unknown_keys_rejected():
    """Joint lists are duplicate-free and the entry key set is closed."""
    with pytest.raises(ControlError, match="duplicate"):
        ControlGroup.from_mapping(
            _group(joints=["shoulder", "shoulder"]))
    with pytest.raises(ControlError, match="unknown keys"):
        ControlGroup.from_mapping(_group(mode="policy"))


def test_validate_controls_coverage_and_passive_joints():
    """Coverage is exact: driven + passive must equal the declared joints."""
    arm = ArticulationSpec(
        name="arm",
        joints=(JointSpec("shoulder"), JointSpec("elbow")),
    )
    partial = ControlGroup.from_mapping(_group(joints=["shoulder"]))
    with pytest.raises(ControlError, match="neither covered"):
        validate_controls((arm,), (partial,))

    passive = ArticulationSpec(
        name="arm",
        joints=(JointSpec("shoulder"), JointSpec("elbow")),
        passive_joints=("elbow",),
    )
    validate_controls((passive,), (partial,))


def test_validate_controls_unknown_and_overlapping_joints():
    """Unknown joints and cross-group overlap are rejected."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("shoulder"),))
    unknown = ControlGroup.from_mapping(_group(joints=["wrist"]))
    with pytest.raises(ControlError, match="not declared"):
        validate_controls((arm,), (unknown,))

    first = ControlGroup.from_mapping(_group(name="a", joints=["shoulder"]))
    second = ControlGroup.from_mapping(_group(name="b", joints=["shoulder"]))
    with pytest.raises(ControlError, match="overlap"):
        validate_controls((arm,), (first, second))
