"""Control runtime construction: validation, recipe resolution and coverage."""

from __future__ import annotations

import pytest

from unigym.robots.base import RobotRecipe
from unigym.runtime.controllers import (
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


def _recipe(joint_sets, defaults):
    return RobotRecipe(
        articulation_cfg=None,
        joint_sets=joint_sets,
        default_joint_pos=defaults,
        foot_bodies=[],
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
    recipe = _recipe({"body": ("a", "b", "c")}, {"a": 0.1, "b": 0.2})
    group = _group(joints=["a", "b", "c"], q0={"b": 0.9})

    runtime = build_control_runtimes((group,), (arm,), {"arm": recipe})[0]

    assert runtime.joint_names == ("a", "b", "c")
    assert runtime.q0.tolist() == pytest.approx([0.1, 0.9, 0.0])
    assert runtime.source == "config.joints"
    assert runtime.strategy_name == "position_pd"


def test_build_control_runtimes_rejects_missing_q0_for_non_position_drive():
    """Only position drives may default a missing q0 to zero."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("shoulder"),))
    group = _group(joints=["shoulder"], drive="torque", target="torque")

    with pytest.raises(ControlError, match="only position drives may default to 0"):
        build_control_runtimes((group,), (arm,), {"arm": _recipe({"body": ("shoulder",)}, {})})


def test_build_control_runtimes_expands_recipe_joint_set():
    """A named joint set is expanded once against the real recipe."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("finger"),))
    recipe = _recipe({"hand": ("finger",)}, {"finger": 0.25})
    group = _group(joint_set="hand", joints=None)

    runtime = build_control_runtimes((group,), (arm,), {"arm": recipe})[0]

    assert runtime.joint_names == ("finger",)
    assert runtime.q0.tolist() == pytest.approx([0.25])
    assert runtime.source == "recipe:arm.hand"
    assert runtime.joint_set == "hand"


@pytest.mark.parametrize(
    ("recipes", "match"),
    [
        ({}, "has no RobotRecipe"),
        ({"arm": _recipe({"body": ("shoulder",)}, {})}, "unknown joint_set 'hand'"),
    ],
    ids=["no-recipe", "unknown-set"],
)
def test_build_control_runtimes_rejects_unresolvable_joint_set(recipes, match):
    """A joint_set selector fails fast without a matching recipe set."""
    arm = ArticulationSpec(name="arm", joints=(JointSpec("shoulder"),))
    group = _group(joint_set="hand", joints=None)

    with pytest.raises(ControlError, match=match):
        build_control_runtimes((group,), (arm,), recipes)


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
