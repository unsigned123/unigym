"""Pure-data action/control contract (no engine involved)."""

from __future__ import annotations

import pytest

from unigym.runtime.actions import OrderedActionSpec, OrderedActionGroup, PolicyBinding
from unigym.runtime.controllers import validate_action_coverage, validate_policy_bindings
from unigym.runtime.controls import ControlError, target_for_drive
from unigym.scenes.articulation import ArticulationSpec, JointSpec


def _group(**overrides):
    entry = {
        "name": "body",
        "articulation": "arm",
        "joints": ["shoulder", "elbow"],
        "target": "position",
    }
    entry.update(overrides)
    return entry


def test_group_requires_explicit_ordered_joints():
    """The action surface has no joint_set selector; joints are explicit."""
    group = OrderedActionGroup.from_mapping(_group())
    assert group.joints == ("shoulder", "elbow")
    assert group.role == "aux"

    with pytest.raises(ValueError, match="joint_set"):
        OrderedActionGroup.from_mapping({**_group(), "joint_set": "body"})


def test_group_role_is_explicit_body_or_aux():
    """role is optional (default aux) and validated when present."""
    assert OrderedActionGroup.from_mapping(_group(role="body")).role == "body"
    with pytest.raises(ValueError, match="role must be one of"):
        OrderedActionGroup.from_mapping(_group(role="prop"))


@pytest.mark.parametrize(
    "drive,target",
    [
        ("position_pd", "position"),
        ("position_servo", "position"),
        ("torque", "torque"),
        ("velocity", "velocity"),
        ("effort", "effort"),
        ("current", "current"),
    ],
)
def test_drive_target_pairs_map_explicitly(drive, target):
    """Every legal drive maps to exactly one target kind."""
    assert target_for_drive(drive) == target


def test_action_spec_rejects_overlap_and_duplicate_identity():
    """Groups are disjoint inside one articulation and unique by identity."""
    shoulder = OrderedActionGroup("body", "arm", ("shoulder",), "position")
    with pytest.raises(ValueError, match="overlapping joints"):
        OrderedActionSpec((shoulder, OrderedActionGroup("aux", "arm", ("shoulder",), "position")))
    with pytest.raises(ValueError, match="duplicate group identity"):
        OrderedActionSpec((shoulder, OrderedActionGroup("body", "arm", ("elbow",), "position")))


def test_policy_binding_references_declared_groups_only():
    """A binding may not name an absent group or share one with another policy."""
    body = OrderedActionGroup("body", "arm", ("shoulder",), "position")

    with pytest.raises(ControlError, match="undeclared action group"):
        validate_policy_bindings((body,), (PolicyBinding("manual", (("arm", "hand"),)),))
    with pytest.raises(ControlError, match="owned by two policies"):
        validate_policy_bindings(
            (body,),
            (
                PolicyBinding("a", (("arm", "body"),)),
                PolicyBinding("b", (("arm", "body"),)),
            ),
        )
    validate_policy_bindings((body,), (PolicyBinding("manual", (("arm", "body"),)),))


def test_action_coverage_requires_driven_plus_passive_to_equal_actuated():
    """Coverage is exact: driven + passive must equal the declared joints."""
    arm = ArticulationSpec(
        name="arm",
        joints=(JointSpec("shoulder"), JointSpec("elbow")),
    )
    partial = OrderedActionSpec((OrderedActionGroup("body", "arm", ("shoulder",), "position"),))
    with pytest.raises(ControlError, match="neither driven by an action group"):
        validate_action_coverage((arm,), {}, partial)

    passive = ArticulationSpec(
        name="arm",
        joints=(JointSpec("shoulder"), JointSpec("elbow")),
        passive_joints=("elbow",),
    )
    validate_action_coverage((passive,), {}, partial)


def test_action_coverage_rejects_unknown_joints():
    """Joints outside the articulation's actuated vocabulary are rejected."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("shoulder"),))
    unknown = OrderedActionSpec(
        (OrderedActionGroup("body", "arm", ("shoulder", "wrist"), "position"),)
    )
    with pytest.raises(ControlError, match="not in the articulation"):
        validate_action_coverage((arm,), {}, unknown)
