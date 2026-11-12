"""Explicit control declarations: numeric boundaries and cross-group validation."""

from __future__ import annotations

import pytest

from unigym.runtime.controls import (
    ControlError,
    ControlGroup,
    normalize_controls,
    validate_controls,
)
from unigym.scenes.articulation import ArticulationSpec, JointSpec


def _entry(**overrides):
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


def _group(**overrides):
    return ControlGroup.from_mapping(_entry(**overrides))


def _arm(*joints: str, passive: tuple[str, ...] = ()) -> ArticulationSpec:
    return ArticulationSpec(
        name="arm",
        joints=tuple(JointSpec(name) for name in joints),
        passive_joints=passive,
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("q0", {"shoulder": float("nan")}, "must be finite"),
        ("q0", {"shoulder": "not-a-number"}, "must be a number"),
    ],
)
def test_control_group_rejects_non_finite_numeric_maps(field, value, match):
    """Per-joint q0 rejects non-numeric and non-finite values."""
    with pytest.raises(ControlError, match=match):
        ControlGroup.from_mapping(_entry(**{field: value}))


def test_control_group_rejects_removed_gains_key():
    """Physical gains belong to ArticulationSpec.actuation, not controls[]."""
    with pytest.raises(ControlError, match="unknown keys.*gains"):
        ControlGroup.from_mapping(_entry(gains={"kp": 1.0}))


@pytest.mark.parametrize("control_dt", [0.0, -0.02, float("nan")])
def test_control_group_rejects_non_positive_control_dt(control_dt):
    """A control period must be finite and strictly greater than zero."""
    with pytest.raises(ControlError, match="control_dt must be (finite|> 0)"):
        ControlGroup.from_mapping(_entry(control_dt=control_dt))


@pytest.mark.parametrize(
    "entries",
    [{"name": "body"}, "body", 3],
    ids=["mapping", "string", "integer"],
)
def test_normalize_controls_rejects_non_list_bundle(entries):
    """The run-level controls block is a list, never a mapping or scalar."""
    with pytest.raises(ControlError, match="controls must be a list"):
        normalize_controls(entries)


def test_normalize_controls_rejects_duplicate_group_names():
    """Two controls groups may not share one action-face name."""
    with pytest.raises(ControlError, match="control group names must be unique"):
        normalize_controls((_group(), _group()))


def test_validate_controls_rejects_undeclared_articulation():
    """A group naming an articulation absent from the run fails fast."""
    with pytest.raises(ControlError, match="which is not declared"):
        validate_controls((_arm("shoulder", "elbow"),), (_group(articulation="hand"),))


def test_validate_controls_rejects_inconsistent_adapter_control_dt():
    """All declared groups of one adapter must agree on control_dt."""
    arm = _arm("shoulder")
    other = ArticulationSpec(name="other", joints=(JointSpec("elbow"),))
    first = _group(joints=["shoulder"], control_dt=0.02)
    second = _group(name="aux", articulation="other", joints=["elbow"], control_dt=0.04)

    with pytest.raises(ControlError, match="inconsistent control_dt"):
        validate_controls((arm, other), (first, second))


def test_validate_controls_rejects_joint_both_driven_and_passive():
    """A joint cannot be covered by a group and listed as passive."""
    arm = _arm("shoulder", "elbow", passive=("shoulder",))

    with pytest.raises(ControlError, match="both driven and passive"):
        validate_controls((arm,), (_group(),))


def test_validate_controls_defers_coverage_for_joint_set_group():
    """Joint-level coverage is deferred when a recipe joint_set is selected."""
    arm = _arm("shoulder", "elbow")
    groups = (
        _group(joint_set="body", joints=None),
        _group(name="aux", joints=["shoulder"]),
    )

    validate_controls((arm,), groups)
