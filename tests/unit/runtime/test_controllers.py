"""Control runtime construction: validation, profile resolution and coverage."""

from __future__ import annotations

import pytest
import torch

from unigym.robots.base import RobotProfile
from unigym.runtime.controllers import (
    ArticulationRuntime,
    ControlRuntime,
    build_control_runtimes,
    validate_resolved_coverage,
)
from unigym.runtime.controls import ControlError, ControlGroup
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
    return ControlGroup.from_mapping(entry)


def _profile(joint_sets, defaults):
    return RobotProfile(
        name="arm",
        joint_sets=joint_sets,
        default_joint_pos=defaults,
        assets=None,
        action_faces={},
        base_body=None,
        foot_bodies=(),
        body_sets={},
        body_bindings={},
    )


def _runtime(name: str, articulation: str, joints: tuple[str, ...]) -> ControlRuntime:
    return ControlRuntime(
        name=name,
        action_face=name,
        adapter="manual",
        articulation=articulation,
        joint_names=joints,
        q0=(0.0,) * len(joints),
    )


@pytest.mark.parametrize(
    ("drive", "target", "match"),
    [
        ("bogus", "position", "drive must be one of"),
        ("torque", "position", "requires target 'torque'"),
        ("position_pd", "velocity", "requires target 'position'"),
    ],
)
def test_control_runtime_rejects_invalid_drive_target_pair(drive, target, match):
    """A runtime accepts only a legal drive and its required target kind."""
    with pytest.raises(ControlError, match=match):
        ControlRuntime(
            name="body",
            action_face="body",
            adapter="manual",
            articulation="arm",
            joint_names=("shoulder",),
            q0=(0.0,),
            drive=drive,
            target=target,
        )


@pytest.mark.parametrize(
    ("joint_names", "match"),
    [((), "owns no joints"), (("shoulder", "shoulder"), "contains duplicates")],
    ids=["empty", "duplicate"],
)
def test_control_runtime_rejects_empty_or_duplicate_joint_names(joint_names, match):
    """A control runtime owns a non-empty, duplicate-free joint list."""
    with pytest.raises(ControlError, match=match):
        ControlRuntime(
            name="body",
            action_face="body",
            adapter="manual",
            articulation="arm",
            joint_names=joint_names,
            q0=(0.0,) * len(joint_names),
        )


@pytest.mark.parametrize("q0", [None, (0.0,)], ids=["missing", "short"])
def test_control_runtime_requires_q0_aligned_with_joint_names(q0):
    """The q0 vector must have exactly one entry per owned joint."""
    with pytest.raises(ControlError, match=r"q0 has \("):
        ControlRuntime(
            name="body",
            action_face="body",
            adapter="manual",
            articulation="arm",
            joint_names=("shoulder", "elbow"),
            q0=q0,
        )


def test_build_control_runtimes_uses_explicit_joints_and_q0_precedence():
    """Explicit joints project in declaration order with group q0 winning."""
    arm = ArticulationSpec(
        name="arm",
        joints=(JointSpec("a"), JointSpec("b"), JointSpec("c")),
    )
    profile = _profile({"body": ("a", "b", "c")}, {"a": 0.1, "b": 0.2})
    group = _group(joints=["a", "b", "c"], q0={"b": 0.9})

    runtime = build_control_runtimes((group,), (arm,), {"arm": profile})[0]

    assert runtime.joint_names == ("a", "b", "c")
    assert runtime.q0.tolist() == pytest.approx([0.1, 0.9, 0.0])
    assert runtime.source == "config.joints"
    assert runtime.strategy_name == "position_pd"


def test_build_control_runtimes_rejects_missing_q0_for_non_position_drive():
    """Only position drives may default a missing q0 to zero."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("shoulder"),))
    group = _group(joints=["shoulder"], drive="torque", target="torque")

    with pytest.raises(ControlError, match="only position drives may default to 0"):
        build_control_runtimes((group,), (arm,), {"arm": _profile({"body": ("shoulder",)}, {})})


def test_build_control_runtimes_expands_profile_joint_set():
    """A named joint set is expanded once against the real profile."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("finger"),))
    profile = _profile({"hand": ("finger",)}, {"finger": 0.25})
    group = _group(joint_set="hand", joints=None)

    runtime = build_control_runtimes((group,), (arm,), {"arm": profile})[0]

    assert runtime.joint_names == ("finger",)
    assert runtime.q0.tolist() == pytest.approx([0.25])
    assert runtime.source == "profile:arm.hand"
    assert runtime.joint_set == "hand"


@pytest.mark.parametrize(
    ("profiles", "match"),
    [
        ({}, "has no RobotProfile"),
        ({"arm": _profile({"body": ("shoulder",)}, {})}, "unknown joint_set 'hand'"),
    ],
    ids=["no-profile", "unknown-set"],
)
def test_build_control_runtimes_rejects_unresolvable_joint_set(profiles, match):
    """A joint_set selector fails fast without a matching profile set."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("shoulder"),))
    group = _group(joint_set="hand", joints=None)

    with pytest.raises(ControlError, match=match):
        build_control_runtimes((group,), (arm,), profiles)


def _coverage_case(kind: str):
    if kind == "missing":
        arm = ArticulationSpec(name="arm", joints=(JointSpec("j"), JointSpec("k")))
        return (
            (arm,),
            (_group(joints=["j"]),),
            (_runtime("body", "arm", ("j",)),),
            "neither driven by a control group",
        )
    if kind == "overlap":
        arm = ArticulationSpec(name="arm", joints=(JointSpec("j"),))
        return (
            (arm,),
            (_group(joints=["j"]), _group(name="aux", joints=["j"])),
            (_runtime("body", "arm", ("j",)), _runtime("aux", "arm", ("j",))),
            "control groups overlap on joints",
        )
    if kind == "passive-conflict":
        arm = ArticulationSpec(
            name="arm",
            joints=(JointSpec("j"),),
            passive_joints=("j",),
        )
        return (
            (arm,),
            (_group(joints=["j"]),),
            (_runtime("body", "arm", ("j",)),),
            "both driven by a control group",
        )
    if kind == "unknown":
        arm = ArticulationSpec(name="arm", joints=(JointSpec("j"),))
        return (
            (arm,),
            (_group(joints=["j", "z"]),),
            (_runtime("body", "arm", ("j", "z")),),
            "not in the articulation's actuated joint vocabulary",
        )
    raise AssertionError(f"unknown coverage case {kind!r}")


@pytest.mark.parametrize(
    "kind",
    ["missing", "overlap", "passive-conflict", "unknown"],
)
def test_validate_resolved_coverage_rejects_incomplete_or_conflicting(kind):
    """Resolved coverage is exact and disjoint against the actuated vocabulary."""
    arts, controls, runtimes, match = _coverage_case(kind)

    with pytest.raises(ControlError, match=match):
        validate_resolved_coverage(arts, {}, controls, runtimes)


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "inf", "-inf"],
)
def test_control_runtime_rejects_non_finite_q0(bad):
    """A control runtime rejects NaN/inf q0 naming the runtime and q0 field."""
    with pytest.raises(ControlError, match=r"control 'body' q0 must be finite"):
        ControlRuntime(
            name="body",
            action_face="body",
            adapter="manual",
            articulation="arm",
            joint_names=("shoulder", "elbow"),
            q0=(0.0, bad),
        )


def test_articulation_runtime_rejects_non_finite_q0():
    """An articulation runtime uses the same non-finite q0 rejection."""
    with pytest.raises(ControlError, match=r"articulation 'arm' q0 must be finite"):
        ArticulationRuntime(
            name="arm",
            joint_names=("shoulder",),
            q0=(float("nan"),),
        )


def test_control_runtime_rejects_non_numeric_q0():
    """A non-numeric q0 entry fails as a ControlError, not a raw TypeError."""
    with pytest.raises(ControlError, match=r"control 'body' q0 must be numeric"):
        ControlRuntime(
            name="body",
            action_face="body",
            adapter="manual",
            articulation="arm",
            joint_names=("shoulder",),
            q0=("not-a-number",),
        )


def test_control_runtime_rejects_non_finite_tensor_q0():
    """The tensor q0 path applies the same finiteness check."""
    with pytest.raises(ControlError, match=r"control 'body' q0 must be finite"):
        ControlRuntime(
            name="body",
            action_face="body",
            adapter="manual",
            articulation="arm",
            joint_names=("shoulder", "elbow"),
            q0=torch.tensor([0.0, float("nan")]),
        )


def test_control_runtime_checks_q0_shape_before_finiteness():
    """A wrong q0 shape reports shape, not a downstream finite error."""
    with pytest.raises(ControlError, match=r"q0 has \(1,\) entries"):
        ControlRuntime(
            name="body",
            action_face="body",
            adapter="manual",
            articulation="arm",
            joint_names=("shoulder", "elbow"),
            q0=(float("nan"),),
        )


def test_articulation_runtime_keeps_native_and_canonical_base_body():
    """base_body is the backend-native name; base_body_id is the canonical id."""
    from unigym.scenes.articulation import ContactSensorInventory

    runtime = ArticulationRuntime(
        name="g1",
        joint_names=("j",),
        q0=(0.0,),
        foot_bodies=("left_ankle_roll",),
        base_body="pelvis",
        base_body_id="base",
        body_sets={"grasp": ("finger",)},
        contacts=ContactSensorInventory(mode="none"),
    )

    assert runtime.base_body == "pelvis"
    assert runtime.base_body_id == "base"
    assert runtime.foot_bodies == ("left_ankle_roll",)
    assert runtime.body_sets == {"grasp": ("finger",)}
    assert runtime.contacts is not None and runtime.contacts.mode == "none"
    assert not hasattr(runtime, "feet")
    assert not hasattr(runtime, "body_parts")
