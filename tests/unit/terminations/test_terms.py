"""Fall, joint-limit and base-contact thresholds and missing-state behavior."""

from __future__ import annotations

import numpy as np
import pytest

from unigym.terminations import (
    BaseContactTermination,
    FallTermination,
    JointLimitTermination,
    TerminationState,
)


def _state(inner: dict, **extra) -> TerminationState:
    return TerminationState(articulations={"g1": inner}, **extra)


@pytest.mark.parametrize(
    ("base_z", "gravity", "expected"),
    [
        (0.15, None, False),
        (0.149, None, True),
        (0.5, [0.0, 0.0, -1.0], False),
        (0.5, [0.0, 0.0, -0.1], True),
    ],
)
def test_fall_fires_below_height_or_past_tilt_threshold(base_z, gravity, expected):
    """The fall rule is strict at the height boundary and skips tilt without gravity."""
    inner = {"base_pos_w": np.array([[0.0, 0.0, base_z]])}
    if gravity is not None:
        inner["gravity_b"] = np.array([gravity], dtype=float)
    term = FallTermination({"articulation": "g1"})

    assert term.check(_state(inner)).tolist() == [expected]


def test_fall_missing_base_position_fails_fast():
    """A fall check without base position names the articulation and field."""
    term = FallTermination({"articulation": "g1"})
    state = _state({"gravity_b": np.array([[0.0, 0.0, -1.0]])})

    with pytest.raises(ValueError, match="requires 'g1' base_pos_w"):
        term.check(state)


@pytest.mark.parametrize(
    ("joint_pos", "limits", "expected"),
    [
        ([[0.05]], (0.0, 1.0), [False]),
        ([[0.049]], (0.0, 1.0), [True]),
        ([[0.95]], (0.0, 1.0), [False]),
        ([[0.951]], (0.0, 1.0), [True]),
        ([[10.0]], None, [False]),
        ([[10.0]], (None, 1.0), [False]),
        ([[-10.0]], (0.0, None), [False]),
    ],
)
def test_joint_limit_margin_boundaries_and_missing_limits(joint_pos, limits, expected):
    """The margin band is strict at its inner edge and no limits means no fire."""
    term = JointLimitTermination({"articulation": "g1"})
    state = _state(
        {"joint_pos": np.array(joint_pos)},
        joint_pos_limits={"g1": limits} if limits is not None else None,
    )

    assert term.check(state).tolist() == expected


@pytest.mark.parametrize(
    ("base_z", "gravity_z", "expected"),
    [
        (0.05, -0.1, True),
        (0.05, -1.0, False),
        (0.5, -0.1, False),
    ],
)
def test_base_contact_approximation_requires_low_and_tilted(base_z, gravity_z, expected):
    """The geometry fallback fires only for the low-base AND tilted conjunction."""
    term = BaseContactTermination({"articulation": "g1"})
    state = _state(
        {
            "base_pos_w": np.array([[0.0, 0.0, base_z]]),
            "gravity_b": np.array([[0.0, 0.0, gravity_z]]),
        }
    )

    assert term.check(state).tolist() == [expected]


@pytest.mark.parametrize(
    ("force", "expected"),
    [
        (4.9, False),
        (5.0, False),
        (5.1, True),
    ],
)
def test_base_contact_force_threshold_is_strict(force, expected):
    """The force path fires only for a net magnitude strictly above the threshold."""
    term = BaseContactTermination({"articulation": "g1", "body_names": ["base"]})
    state = _state(
        {
            "base_pos_w": np.array([[0.0, 0.0, 0.5]]),
            "gravity_b": np.array([[0.0, 0.0, -1.0]]),
        },
        body_contact_forces=np.array([[[force, 0.0, 0.0]]]),
        body_names=("base",),
    )

    assert term.check(state).tolist() == [expected]


def test_base_contact_absent_requested_body_falls_back_to_geometry():
    """When no monitored body reaches the force sensor, the geometry fallback decides."""
    term = BaseContactTermination({"articulation": "g1", "body_names": ["base"]})
    state = _state(
        {
            "base_pos_w": np.array([[0.0, 0.0, 0.05]]),
            "gravity_b": np.array([[0.0, 0.0, -0.1]]),
        },
        body_contact_forces=np.zeros((1, 1, 3)),
        body_names=("left_foot",),
    )

    assert term.check(state).tolist() == [True]


def test_joint_limit_missing_joint_position_fails_fast():
    """A joint-limit check without joint positions names the articulation and field."""
    term = JointLimitTermination({"articulation": "g1"})

    with pytest.raises(ValueError, match="requires 'g1' joint_pos"):
        term.check(_state({}))


def test_base_contact_without_any_state_fails_fast():
    """A base-contact check with neither signal names the articulation and fields."""
    term = BaseContactTermination({"articulation": "g1"})

    with pytest.raises(ValueError, match="requires 'g1' base_pos_w or gravity_b"):
        term.check(_state({}))
