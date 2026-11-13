"""Actuation group construction and exactly-once coverage invariants."""

from __future__ import annotations

import pytest

from unigym.actuators.ir import (
    ActuationError,
    ActuationGroup,
    ResolvedActuationGroup,
    normalize_actuation_group,
    normalize_actuation_groups,
    resolve_actuation_groups,
    resolve_group_joints,
    validate_actuation_group,
    validate_actuation_groups,
    validate_drive_law_matrix,
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


def test_actuation_group_accepts_scalar_and_joint_keyed_gain_values():
    """A gain value is a finite scalar or a non-empty joint-keyed finite map."""
    group = ActuationGroup(
        name="arm",
        law="standard_pd",
        joints=("j1", "j2"),
        gains={"kp": {"j1": 10.0, "j2": 20.0}, "kd": 1.0},
    )

    assert group.gains == {"kp": {"j1": 10.0, "j2": 20.0}, "kd": 1.0}


@pytest.mark.parametrize(
    ("gains", "match"),
    [
        ({"kp": {}}, "non-empty joint-keyed map"),
        ({"kp": {"j1": float("nan")}}, "must be finite"),
        ({"kp": {"": 1.0}}, "keys must be non-empty"),
    ],
)
def test_actuation_group_rejects_malformed_gain_values(gains, match):
    """Nested gain maps must be non-empty, non-blank-keyed and finite."""
    with pytest.raises(ActuationError, match=match):
        ActuationGroup(name="arm", law="standard_pd", joints=("j1",), gains=gains)


def test_normalize_actuation_group_accepts_mapping_and_unknown_keys_fail():
    """Mapping declarations normalize; the group key set is closed."""
    group = normalize_actuation_group(
        {"name": "arm", "law": "standard_pd", "joint_set": "body", "gains": {"kp": 1.0}}
    )

    assert isinstance(group, ActuationGroup)
    assert group.joint_set == "body"
    with pytest.raises(ActuationError, match=r"unknown keys \['bogus'\]"):
        normalize_actuation_group(
            {"name": "arm", "law": "standard_pd", "joints": ["j1"], "bogus": 1}
        )


def test_normalize_actuation_groups_rejects_duplicate_names_and_non_lists():
    """The list helper rejects duplicate group names and non-list bundles."""
    with pytest.raises(ActuationError, match="names must be unique"):
        normalize_actuation_groups(
            [
                {"name": "a", "law": "standard_pd", "joints": ["j1"]},
                {"name": "a", "law": "standard_pd", "joints": ["j2"]},
            ]
        )
    with pytest.raises(ActuationError, match="must be a list"):
        normalize_actuation_groups({"name": "a"})


def test_resolve_actuation_groups_returns_declaration_order_and_joint_set():
    """Resolution expands joint_set and keeps the declaration order."""
    groups = (
        ActuationGroup(name="hand", law="standard_pd", joints=("finger",)),
        ActuationGroup(name="body", law="computed_torque_pd", joint_set="body"),
    )

    resolved = resolve_actuation_groups(
        groups,
        articulation="arm",
        joints=("shoulder", "elbow", "finger"),
        joint_sets={"body": ("shoulder", "elbow")},
    )

    assert [row.group.name for row in resolved] == ["hand", "body"]
    assert [row.joint_names for row in resolved] == [("finger",), ("shoulder", "elbow")]
    assert all(row.articulation == "arm" for row in resolved)


def test_resolve_group_joints_expands_joint_sets_and_passes_explicit_joints():
    """The realizer helper resolves sets and keeps explicit joints in order."""
    groups = (
        ActuationGroup(name="legs", law="standard_pd", joint_set="legs"),
        ActuationGroup(name="arm", law="standard_pd", joints=("shoulder",)),
    )

    resolved = resolve_group_joints(
        groups,
        articulation="g1",
        joint_sets={"legs": ("hip", "knee"), "arm": ("shoulder", "elbow")},
    )

    assert [row.group.name for row in resolved] == ["legs", "arm"]
    assert [row.joint_names for row in resolved] == [("hip", "knee"), ("shoulder",)]
    assert all(row.articulation == "g1" for row in resolved)


def test_resolve_group_joints_unknown_joint_set_names_the_declared_sets():
    """An unknown joint_set fails with the profile vocabulary in the error."""
    groups = (ActuationGroup(name="legs", law="standard_pd", joint_set="feet"),)

    with pytest.raises(
        ActuationError,
        match=r"joint_set 'feet' is unknown.*\['legs'\]",
    ):
        resolve_group_joints(groups, articulation="g1", joint_sets={"legs": ("hip",)})


def test_resolve_group_joints_does_not_validate_joint_coverage():
    """Partial coverage resolves: driven-joint coverage belongs to run/build."""
    groups = (ActuationGroup(name="one", law="standard_pd", joints=("hip",)),)

    resolved = resolve_group_joints(
        groups,
        articulation="g1",
        joint_sets={"body": ("hip", "knee", "ankle")},
    )

    assert [row.joint_names for row in resolved] == [("hip",)]


def test_validate_actuation_group_law_matrix():
    """Law payloads validate per backend and law."""
    standard = ActuationGroup(
        name="body", law="standard_pd", joints=("j1",), gains={"kp": 1.0, "kd": 2.0}
    )
    validate_actuation_group(standard, backend="mujoco")
    validate_actuation_group(standard, backend="isaaclab2")
    with pytest.raises(ActuationError, match="not supported by backend 'isaaclab2'"):
        validate_actuation_group(
            ActuationGroup(
                name="body",
                law="computed_torque_pd",
                joints=("j1",),
                gains={"kp": 1.0, "kd": 2.0},
            ),
            backend="isaaclab2",
        )
    with pytest.raises(ActuationError, match="missing gains"):
        validate_actuation_group(
            ActuationGroup(name="body", law="standard_pd", joints=("j1",), gains={"kp": 1.0}),
            backend="mujoco",
        )
    with pytest.raises(ActuationError, match="unknown gains"):
        validate_actuation_group(
            ActuationGroup(
                name="body",
                law="standard_pd",
                joints=("j1",),
                gains={"kp": 1.0, "kd": 1.0, "bogus": 1.0},
            ),
            backend="mujoco",
        )
    with pytest.raises(ActuationError, match="parts are only valid"):
        validate_actuation_group(
            ActuationGroup(
                name="body",
                law="standard_pd",
                joints=("j1",),
                gains={"kp": 1.0, "kd": 1.0},
                parts=("j1",),
            ),
            backend="mujoco",
        )
    with pytest.raises(ActuationError, match="backend-owned limits"):
        validate_actuation_group(
            ActuationGroup(
                name="body",
                law="standard_pd",
                joints=("j1",),
                gains={"kp": 1.0, "kd": 1.0},
                effort_limit=1.0,
            ),
            backend="mujoco",
        )


def _resolved(law: str, joints: tuple[str, ...]) -> tuple[ResolvedActuationGroup, ...]:
    return (
        ResolvedActuationGroup(
            articulation="arm",
            group=ActuationGroup(name="body", law=law, joints=joints),
            joint_names=joints,
        ),
    )


def _drive(drive: str, joints: tuple[str, ...] = ("j1",)):
    from types import SimpleNamespace

    return SimpleNamespace(
        name="body",
        articulation="arm",
        drive=drive,
        joint_names=joints,
        prop=False,
    )


@pytest.mark.parametrize("law", ["standard_pd", "computed_torque_pd"])
def test_drive_law_matrix_position_pd_accepts_position_laws(law):
    """position_pd is served by both position-PD laws."""
    validate_drive_law_matrix(_resolved(law, ("j1",)), (_drive("position_pd"),))


def test_drive_law_matrix_position_servo_requires_native_servo():
    """position_servo needs native_servo, and native_servo needs position_servo."""
    validate_drive_law_matrix(_resolved("native_servo", ("j1",)), (_drive("position_servo"),))
    with pytest.raises(ActuationError, match="requires native_servo"):
        validate_drive_law_matrix(_resolved("standard_pd", ("j1",)), (_drive("position_servo"),))
    with pytest.raises(ActuationError, match="position_pd.*standard_pd or computed_torque_pd"):
        validate_drive_law_matrix(_resolved("native_servo", ("j1",)), (_drive("position_pd"),))


@pytest.mark.parametrize("drive", ["torque", "velocity", "effort", "current"])
def test_drive_law_matrix_rejects_direct_drives(drive):
    """D4 has no direct-drive law; the teaching error names the drive."""
    with pytest.raises(ActuationError, match="no D4 actuation law"):
        validate_drive_law_matrix(_resolved("standard_pd", ("j1",)), (_drive(drive),))
