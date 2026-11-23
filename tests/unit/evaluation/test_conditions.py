"""Canonical condition masks: boundaries, modes, missing sources and order."""

from __future__ import annotations

import math

import pytest
import torch

from unigym.conditions.base import ConditionError
from unigym.conditions.height import HeightCondition
from unigym.conditions.joint_limit import JointLimitCondition
from unigym.conditions.tilt import TiltCondition
from unigym.evaluation import EnvView, EvalContext
from unigym.runtime.controllers import ArticulationRuntime


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
    record: dict,
    *,
    num_envs: int = 1,
    limits: object = None,
) -> EvalContext:
    return EvalContext(
        env=EnvView(
            record=record,
            obs=None,
            env_id=None,
            num_envs=num_envs,
            device="cpu",
            articulations={"robot": _runtime(limits=limits)},
        )
    )


@pytest.mark.parametrize(
    ("base_z", "expected"),
    [
        (0.15, False),
        (0.149, True),
    ],
)
def test_height_termination_threshold_is_strict(base_z, expected):
    """The threshold form fires only strictly below the height threshold."""
    record = {"articulation/robot/base_pos_w": torch.tensor([[0.0, 0.0, base_z]])}

    mask = HeightCondition(articulation="robot", threshold=0.15).evaluate(_ctx(record))

    assert mask.dtype == torch.bool
    assert tuple(mask.shape) == (1,)
    assert mask.tolist() == [expected]


@pytest.mark.parametrize(
    ("op", "threshold", "base_z", "expected"),
    [
        ("gt", 0.5, 1.0, True),
        ("lt", 0.5, 0.5, False),
        ("gte", 0.5, 0.5, True),
        ("abs_gt", 0.5, -1.0, True),
        ("abs_lte", 0.5, -0.5, True),
    ],
)
def test_height_watcher_ops_match_the_comparison_contract(op, threshold, base_z, expected):
    """Every watcher comparison operator is applied to the height component."""
    record = {"base_pos_w": torch.tensor([[0.0, 0.0, base_z]])}
    condition = HeightCondition(record_key="base_pos_w", op=op, threshold=threshold)

    assert condition.evaluate(_ctx(record)).tolist() == [expected]


@pytest.mark.parametrize(
    ("base_z", "expected"),
    [
        (0.05, False),
        (0.051, True),
        (0.0, False),
    ],
)
def test_height_band_is_inclusive(base_z, expected):
    """The watcher band fires only outside the inclusive [low, high] range."""
    record = {"base_pos_w": torch.tensor([[0.0, 0.0, base_z]])}
    condition = HeightCondition(record_key="base_pos_w", band=(0.0, 0.05))

    assert condition.evaluate(_ctx(record)).tolist() == [expected]


def test_height_disabled_threshold_returns_all_false():
    """A disabled threshold yields an all-false mask of the source shape."""
    record = {"base_pos_w": torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])}

    mask = HeightCondition(record_key="base_pos_w", threshold=None).evaluate(
        _ctx(record, num_envs=2)
    )

    assert mask.tolist() == [False, False]


def test_height_missing_record_key_names_the_source():
    """A missing record source is a condition error naming the leaf."""
    condition = HeightCondition(record_key="base_pos_w", threshold=0.15)

    with pytest.raises(ConditionError, match="base_pos_w"):
        condition.evaluate(_ctx({}))


def test_tilt_gravity_threshold_is_strict_at_the_boundary():
    """Projected-gravity tilt fires only past -cos(threshold)."""
    boundary = -math.cos(0.8)
    record = {
        "articulation/robot/gravity_b": torch.tensor(
            [[0.0, 0.0, boundary], [0.0, 0.0, boundary + 1e-9], [0.0, 0.0, -1.0]],
            dtype=torch.float64,
        )
    }
    condition = TiltCondition(source="gravity_b", articulation="robot", threshold=0.8)

    assert condition.evaluate(_ctx(record, num_envs=3)).tolist() == [False, True, False]


def test_tilt_disabled_returns_all_false_without_a_source():
    """A disabled tilt condition does not read a missing source."""
    condition = TiltCondition(
        source="gravity_b",
        articulation="robot",
        threshold=None,
        missing="false",
    )

    mask = condition.evaluate(_ctx({}, num_envs=2))

    assert mask.tolist() == [False, False]


def test_tilt_missing_source_policy_is_explicit():
    """``missing='false'`` suppresses an absent source; ``raise`` does not."""
    missing_false = TiltCondition(
        source="gravity_b",
        articulation="robot",
        threshold=0.8,
        missing="false",
    )
    missing_raise = TiltCondition(
        source="gravity_b",
        articulation="robot",
        threshold=0.8,
        missing="raise",
    )

    partial = {"articulation/robot/base_pos_w": torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])}
    assert missing_false.evaluate(_ctx(partial, num_envs=2)).tolist() == [False, False]
    with pytest.raises(ConditionError, match="gravity_b"):
        missing_raise.evaluate(_ctx(partial))


def test_tilt_unit_conversion_matches_the_native_unit():
    """Gravity degrees and quaternion radians resolve to the same thresholds."""
    boundary = -math.cos(0.8)
    gravity = {
        "articulation/robot/gravity_b": torch.tensor(
            [[0.0, 0.0, boundary + 1e-9]], dtype=torch.float64
        )
    }
    gravity_deg = TiltCondition(
        source="gravity_b",
        articulation="robot",
        threshold=math.degrees(0.8),
        unit="deg",
    )
    quaternion = {
        "base_quat_w": torch.tensor([[0.7071068, 0.0, 0.7071068, 0.0]], dtype=torch.float64)
    }
    quaternion_rad = TiltCondition(
        source="base_quat_w",
        record_key="base_quat_w",
        threshold=math.radians(60.0),
        unit="rad",
    )

    assert gravity_deg.evaluate(_ctx(gravity)).tolist() == [True]
    assert quaternion_rad.evaluate(_ctx(quaternion)).tolist() == [True]


def test_joint_limit_default_margins_match_each_mode():
    """Band mode defaults to the termination margin; magnitude to the watcher."""
    assert JointLimitCondition(articulation="robot").margin == 0.05
    assert JointLimitCondition(articulation="robot", mode="magnitude").margin == 0.1


def test_tilt_quaternion_uses_degrees():
    """The quaternion source compares the converted tilt in degrees."""
    record = {
        "base_quat_w": torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.7071068, 0.0, 0.7071068, 0.0],
            ],
            dtype=torch.float64,
        )
    }
    condition = TiltCondition(
        source="base_quat_w",
        record_key="base_quat_w",
        threshold=60.0,
        unit="deg",
    )

    assert condition.evaluate(_ctx(record, num_envs=2)).tolist() == [False, True]


@pytest.mark.parametrize(
    ("joint_pos", "limits", "expected"),
    [
        ([[0.05]], (0.0, 1.0), False),
        ([[0.049]], (0.0, 1.0), True),
        ([[0.95]], (0.0, 1.0), False),
        ([[0.951]], (0.0, 1.0), True),
        ([[10.0]], None, False),
        ([[10.0]], (None, 1.0), False),
        ([[-10.0]], (0.0, None), False),
    ],
)
def test_joint_limit_band_boundaries_and_missing_limits(joint_pos, limits, expected):
    """The termination band is strict inside its edges; no limits means no fire."""
    runtime_limits = None
    if limits is not None:
        lower, upper = limits
        runtime_limits = (
            None if lower is None else torch.tensor([[lower]]),
            None if upper is None else torch.tensor([[upper]]),
        )
    record = {"articulation/robot/joint_pos": torch.tensor(joint_pos)}
    condition = JointLimitCondition(articulation="robot", mode="band")

    assert condition.evaluate(_ctx(record, limits=runtime_limits)).tolist() == [expected]


def test_joint_limit_band_and_magnitude_modes_differ_for_asymmetric_limits():
    """The watcher magnitude mode is not the termination band on asymmetric limits."""
    limits = (torch.tensor([[-2.0]]), torch.tensor([[1.0]]))
    record = {"articulation/robot/joint_pos": torch.tensor([[-1.9], [0.92]])}
    band = JointLimitCondition(articulation="robot", mode="band", margin=0.1)
    magnitude = JointLimitCondition(articulation="robot", mode="magnitude", margin=0.1)

    assert band.evaluate(_ctx(record, num_envs=2, limits=limits)).tolist() == [False, True]
    assert magnitude.evaluate(_ctx(record, num_envs=2, limits=limits)).tolist() == [True, False]


def test_joint_limit_record_key_source_matches_the_articulation_leaf():
    """An explicit record key reads the same policy-order joint positions."""
    limits = (torch.tensor([[-0.5, -1.57]]), torch.tensor([[1.0, 1.57]]))
    record = {"articulation/robot/joint_pos": torch.tensor([[0.0, 1.6]])}
    leaf = JointLimitCondition(articulation="robot", mode="band")
    keyed = JointLimitCondition(
        articulation="robot",
        record_key="articulation/robot/joint_pos",
        mode="band",
    )

    assert leaf.evaluate(_ctx(record, limits=limits)).tolist() == [True]
    assert keyed.evaluate(_ctx(record, limits=limits)).tolist() == [True]


def test_joint_limit_missing_joint_position_names_the_source():
    """A missing joint-position source is a condition error naming the leaf."""
    condition = JointLimitCondition(articulation="robot", mode="band")
    record = {"articulation/robot/gravity_b": torch.zeros(1, 3)}

    with pytest.raises(ConditionError, match="joint_pos"):
        condition.evaluate(_ctx(record))
