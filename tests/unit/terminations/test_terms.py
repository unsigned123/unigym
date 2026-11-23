"""Fall, joint-limit and base-contact thresholds and missing-state behavior."""

from __future__ import annotations

import torch
import pytest

from unigym.evaluation import EnvView, EvalContext
from unigym.runtime.controllers import ArticulationRuntime
from unigym.terminations import (
    BaseContactTermination,
    FallTermination,
    JointLimitTermination,
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


def _ctx(
    inner: dict,
    *,
    num_envs: int = 1,
    limits: object = None,
    body_contact_forces: torch.Tensor | None = None,
    body_names: tuple[str, ...] | None = None,
    articulations: dict[str, ArticulationRuntime] | None = None,
) -> EvalContext:
    record = {f"articulation/robot/{key}": value for key, value in inner.items()}
    if body_contact_forces is not None:
        record["body_contact_forces"] = body_contact_forces
    if body_names is not None:
        record["body_names"] = body_names
    arts = {"robot": _runtime(limits=limits)} if articulations is None else articulations
    return EvalContext(
        env=EnvView(
            record=record,
            obs=None,
            env_id=None,
            num_envs=num_envs,
            device="cpu",
            articulations=arts,
        )
    )


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
    inner = {"base_pos_w": torch.tensor([[0.0, 0.0, base_z]])}
    if gravity is not None:
        inner["gravity_b"] = torch.tensor([gravity])
    term = FallTermination({"articulation": "robot"})

    assert term.check(_ctx(inner)).tolist() == [expected]


def test_fall_missing_base_position_fails_fast():
    """A fall check without base position names the articulation and field."""
    term = FallTermination({"articulation": "robot"})
    ctx = _ctx({"gravity_b": torch.tensor([[0.0, 0.0, -1.0]])})

    with pytest.raises(ValueError, match="requires 'robot' base_pos_w"):
        term.check(ctx)


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
    term = JointLimitTermination({"articulation": "robot"})
    runtime_limits = None
    if limits is not None:
        lower, upper = limits
        runtime_limits = (
            None if lower is None else torch.tensor([[lower]]),
            None if upper is None else torch.tensor([[upper]]),
        )
    ctx = _ctx(
        {"joint_pos": torch.tensor(joint_pos)},
        limits=runtime_limits,
    )

    assert term.check(ctx).tolist() == expected


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
    term = BaseContactTermination({"articulation": "robot"})
    ctx = _ctx(
        {
            "base_pos_w": torch.tensor([[0.0, 0.0, base_z]]),
            "gravity_b": torch.tensor([[0.0, 0.0, gravity_z]]),
        }
    )

    assert term.check(ctx).tolist() == [expected]


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
    term = BaseContactTermination({"articulation": "robot", "body_names": ["base"]})
    ctx = _ctx(
        {
            "base_pos_w": torch.tensor([[0.0, 0.0, 0.5]]),
            "gravity_b": torch.tensor([[0.0, 0.0, -1.0]]),
        },
        body_contact_forces=torch.tensor([[[force, 0.0, 0.0]]]),
        body_names=("base",),
    )

    assert term.check(ctx).tolist() == [expected]


def test_base_contact_absent_requested_body_falls_back_to_geometry():
    """When no monitored body reaches the force sensor, the geometry fallback decides."""
    term = BaseContactTermination({"articulation": "robot", "body_names": ["base"]})
    ctx = _ctx(
        {
            "base_pos_w": torch.tensor([[0.0, 0.0, 0.05]]),
            "gravity_b": torch.tensor([[0.0, 0.0, -0.1]]),
        },
        body_contact_forces=torch.zeros(1, 1, 3),
        body_names=("left_foot",),
    )

    assert term.check(ctx).tolist() == [True]


def test_joint_limit_missing_joint_position_fails_fast():
    """A joint-limit check without joint positions names the articulation and field."""
    term = JointLimitTermination({"articulation": "robot"})

    with pytest.raises(ValueError, match="requires 'robot' joint_pos"):
        term.check(_ctx({"gravity_b": torch.zeros(1, 3)}))


def test_base_contact_without_any_state_fails_fast():
    """A base-contact check with neither signal names the articulation and fields."""
    term = BaseContactTermination({"articulation": "robot"})

    with pytest.raises(ValueError, match="requires 'robot' base_pos_w or gravity_b"):
        term.check(_ctx({"joint_pos": torch.zeros(1, 1)}))
