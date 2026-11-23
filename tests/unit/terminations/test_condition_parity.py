"""Condition-backed terminations preserve thresholds, OR/invert and fail-fast."""

from __future__ import annotations

import pytest
import torch

from unigym.evaluation import EnvView, EvalContext
from unigym.runtime.controllers import ArticulationRuntime
from unigym.terminations import (
    FallTermination,
    InvertedTermination,
    JointLimitTermination,
    TerminationGroup,
)


def _runtime(*, limits: object = None) -> ArticulationRuntime:
    art = ArticulationRuntime(
        name="robot",
        root="fixed",
        joint_names=("j0", "j1"),
        q0=(0.0, 0.0),
    )
    if limits is not None:
        setattr(art, "joint_pos_limits", limits)
    return art


def _ctx(inner: dict, *, limits: object = None) -> EvalContext:
    record = {f"articulation/robot/{key}": value for key, value in inner.items()}
    return EvalContext(
        env=EnvView(
            record=record,
            obs=None,
            env_id=None,
            num_envs=1,
            device="cpu",
            articulations={"robot": _runtime(limits=limits)},
        )
    )


@pytest.mark.parametrize(
    ("base_z", "gravity", "tilt_threshold", "expected"),
    [
        (0.15, None, None, False),
        (0.149, None, None, True),
        (0.5, [0.0, 0.0, -0.1], 0.8, True),
        (0.5, [0.0, 0.0, -1.0], 0.8, False),
        (0.5, [0.0, 0.0, -0.1], None, False),
        (0.1, [0.0, 0.0, -1.0], 0.8, True),
    ],
)
def test_fall_thresholds_and_or_semantics(base_z, gravity, tilt_threshold, expected):
    """The delegated height OR tilt rule keeps strict thresholds and disablement."""
    inner = {"base_pos_w": torch.tensor([[0.0, 0.0, base_z]])}
    if gravity is not None:
        inner["gravity_b"] = torch.tensor([gravity])
    term = FallTermination(
        {
            "articulation": "robot",
            "height_threshold": 0.15,
            "tilt_threshold": tilt_threshold,
        }
    )

    assert term.check(_ctx(inner)).tolist() == [expected]


def test_fall_missing_base_position_still_fails_fast_with_its_message():
    """A missing base position keeps the termination's own teaching error."""
    term = FallTermination({"articulation": "robot"})
    ctx = _ctx({"gravity_b": torch.tensor([[0.0, 0.0, -1.0]])})

    with pytest.raises(ValueError, match="requires 'robot' base_pos_w"):
        term.check(ctx)


def test_fall_group_or_and_invert_parity():
    """Group OR and inversion still operate on the delegated condition masks."""
    narrow = FallTermination(
        {"articulation": "robot", "height_threshold": 0.15, "tilt_threshold": None},
        name="narrow",
    )
    wide = FallTermination(
        {"articulation": "robot", "height_threshold": 0.6, "tilt_threshold": None},
        name="wide",
    )
    ctx = _ctx({"base_pos_w": torch.tensor([[0.0, 0.0, 0.5]])})

    assert TerminationGroup([narrow, wide], mode="or").check(ctx).tolist() == [True]
    assert TerminationGroup([narrow, wide], mode="and").check(ctx).tolist() == [False]
    assert InvertedTermination(wide).check(ctx).tolist() == [False]


@pytest.mark.parametrize(
    ("joint_pos", "limits", "expected"),
    [
        ([[0.05]], (0.0, 1.0), False),
        ([[0.049]], (0.0, 1.0), True),
        ([[0.95]], (0.0, 1.0), False),
        ([[0.951]], (0.0, 1.0), True),
        ([[10.0]], None, False),
        ([[10.0]], (None, 1.0), False),
    ],
)
def test_joint_limit_margin_boundaries_and_missing_limits(joint_pos, limits, expected):
    """The delegated band keeps its strict edges and all-false missing limits."""
    runtime_limits = None
    if limits is not None:
        lower, upper = limits
        runtime_limits = (
            None if lower is None else torch.tensor([[lower]]),
            None if upper is None else torch.tensor([[upper]]),
        )
    term = JointLimitTermination({"articulation": "robot"})

    mask = term.check(_ctx({"joint_pos": torch.tensor(joint_pos)}, limits=runtime_limits))

    assert mask.tolist() == [expected]


def test_joint_limit_missing_joint_position_still_fails_fast_with_its_message():
    """A missing joint-position row keeps the termination's own teaching error."""
    term = JointLimitTermination({"articulation": "robot"})

    with pytest.raises(ValueError, match="requires 'robot' joint_pos"):
        term.check(_ctx({"gravity_b": torch.zeros(1, 3)}))
