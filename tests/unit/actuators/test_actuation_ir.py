"""Actuation group construction and exactly-once coverage invariants."""

from __future__ import annotations

import pytest

from unigym.actuators.ir import (
    ActuationError,
    ActuationGroup,
    validate_actuation_groups,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"joints": ("j1",), "joint_set": "arm"},
        {},
        {"joints": ()},
    ],
)
def test_actuation_group_requires_exactly_one_addressing_mode(kwargs):
    """A group declares non-empty joints or joint_set, never both and never neither."""
    with pytest.raises(ActuationError, match="exactly one"):
        ActuationGroup(name="arm", law="standard_pd", **kwargs)


def test_actuation_group_rejects_unknown_law_and_lists_registered_laws():
    """law must name a registered actuator law, with the legal set in the error."""
    with pytest.raises(ActuationError, match=r"not registered.*standard_pd"):
        ActuationGroup(name="arm", law="bogus_law", joints=("j1",))


@pytest.mark.parametrize(
    ("joints", "match"),
    [
        (("j1", "j1"), "duplicate names"),
        (("",), "empty names"),
        ((" ",), "empty names"),
        ((" j1",), "surrounding whitespace"),
        (("j1", 1), "must contain strings"),
    ],
)
def test_actuation_group_rejects_duplicate_and_empty_joint_names(joints, match):
    """Joint addressing accepts only unique non-empty names."""
    with pytest.raises(ActuationError, match=match):
        ActuationGroup(name="arm", law="standard_pd", joints=joints)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"gains": {"kp": float("nan")}}, "must be finite"),
        ({"gains": {"kp": True}}, "finite number"),
        ({"effort_limit": -1.0}, "non-negative"),
        ({"velocity_limit": {"j1": -0.1}}, "non-negative"),
        ({"command": []}, "command must be a mapping"),
        ({"parts": ("part", "part")}, "duplicate names"),
        ({"parts": ("",)}, "empty names"),
    ],
)
def test_actuation_group_rejects_malformed_generic_payload(kwargs, match):
    """Gains, limits, command and parts fail fast on a malformed neutral shape."""
    with pytest.raises(ActuationError, match=match):
        ActuationGroup(name="arm", law="standard_pd", joints=("j1",), **kwargs)


def test_actuation_group_accepts_scalar_and_mapping_limits_and_copies_command():
    """The generic payload keeps scalar or joint-keyed limits and copies command."""
    command = {"mode": "position"}

    group = ActuationGroup(
        name="arm",
        law="standard_pd",
        joints=("j1",),
        gains={"kp": 1.0},
        effort_limit=10.0,
        velocity_limit={"j1": 2.0},
        command=command,
        parts=("body",),
    )

    assert group.gains == {"kp": 1.0}
    assert group.effort_limit == 10.0
    assert group.velocity_limit == {"j1": 2.0}
    assert group.command == {"mode": "position"}
    assert group.command is not command
    assert group.parts == ("body",)


def test_actuation_groups_reject_overlapping_joint_coverage():
    """One joint may be covered by only one group."""
    groups = (
        ActuationGroup(name="a", law="standard_pd", joints=("j1", "j2")),
        ActuationGroup(name="b", law="standard_pd", joints=("j2", "j3")),
    )

    with pytest.raises(ActuationError, match=r"overlaps.*'j2'"):
        validate_actuation_groups(groups, joints=("j1", "j2", "j3"))


def test_actuation_groups_reject_missing_joint_coverage():
    """Every non-passive joint must be covered exactly once."""
    groups = (ActuationGroup(name="a", law="standard_pd", joints=("j1",)),)

    with pytest.raises(ActuationError, match=r"do not cover joints \['j2'\]"):
        validate_actuation_groups(groups, joints=("j1", "j2"))


def test_actuation_groups_reject_unknown_explicit_joint():
    """An explicit joint outside the articulation universe fails."""
    groups = (ActuationGroup(name="a", law="standard_pd", joints=("ghost",)),)

    with pytest.raises(ActuationError, match=r"names unknown joints \['ghost'\]"):
        validate_actuation_groups(groups, joints=("j1",))


def test_actuation_groups_reject_missing_and_unknown_joint_sets():
    """A joint_set must resolve; its name and members are explicit."""
    unknown_set = (ActuationGroup(name="a", law="standard_pd", joint_set="ghost"),)
    with pytest.raises(
        ActuationError,
        match=r"joint_set 'ghost' is unknown.*\['arm'\]",
    ):
        validate_actuation_groups(
            unknown_set,
            joints=("j1",),
            joint_sets={"arm": ("j1",)},
        )

    unknown_member = (ActuationGroup(name="a", law="standard_pd", joint_set="arm"),)
    with pytest.raises(
        ActuationError,
        match=r"joint_set 'arm' names unknown joints \['ghost'\]",
    ):
        validate_actuation_groups(
            unknown_member,
            joints=("j1",),
            joint_sets={"arm": ("ghost",)},
        )


def test_actuation_groups_reject_passive_joint_coverage():
    """A passive joint may not be covered by an actuation group."""
    groups = (ActuationGroup(name="a", law="standard_pd", joints=("j1",)),)

    with pytest.raises(ActuationError, match=r"cover passive joints \['j1'\]"):
        validate_actuation_groups(
            groups,
            joints=("j1", "j2"),
            passive_joints=("j1",),
        )


def test_actuation_joint_set_resolution_and_passive_omission_accept():
    """A resolved joint set plus a passive joint is complete coverage."""
    groups = (ActuationGroup(name="arm", law="standard_pd", joint_set="arm"),)

    validate_actuation_groups(
        groups,
        joints=("shoulder", "elbow", "finger"),
        joint_sets={"arm": ("shoulder", "elbow")},
        passive_joints=("finger",),
    )


def test_actuation_group_names_must_be_unique():
    """Group names are the coverage identity and must be unique."""
    groups = (
        ActuationGroup(name="a", law="standard_pd", joints=("j1",)),
        ActuationGroup(name="a", law="standard_pd", joints=("j2",)),
    )

    with pytest.raises(ActuationError, match="names must be unique"):
        validate_actuation_groups(groups, joints=("j1", "j2"))


def test_empty_actuation_passes_without_driven_joints():
    """An empty group list is valid when every joint is passive (or absent)."""
    validate_actuation_groups((), joints=())
    validate_actuation_groups((), joints=("j1",), passive_joints=("j1",))


def test_empty_actuation_fails_with_driven_joints():
    """An empty group list fails when a non-passive joint exists."""
    with pytest.raises(ActuationError, match="do not cover joints"):
        validate_actuation_groups((), joints=("j1",))
