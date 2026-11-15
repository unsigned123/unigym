"""Composite termination groups: OR/AND reduction, empty groups and reference state."""

from __future__ import annotations

import numpy as np
import pytest

from unigym.terminations import FallTermination, TerminationGroup, TerminationState


def _state() -> TerminationState:
    return TerminationState(
        articulations={
            "robot": {
                "base_pos_w": np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.5]]),
                "gravity_b": np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
            }
        }
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("or", [True, True]),
        ("and", [True, False]),
    ],
)
def test_composite_group_combines_masks_with_its_declared_mode(mode, expected):
    """OR fires when any term fires; AND fires only when every term fires."""
    low_only = FallTermination(
        {"articulation": "robot", "height_threshold": 0.15, "tilt_threshold": None},
        name="low_only",
    )
    wide_height = FallTermination(
        {"articulation": "robot", "height_threshold": 0.6, "tilt_threshold": None},
        name="wide_height",
    )
    group = TerminationGroup([low_only, wide_height], mode=mode)

    assert group.check(_state()).tolist() == expected


def test_composite_empty_group_never_fires():
    """An empty group is inactive: it reports an all-False mask."""
    group = TerminationGroup([])

    assert group.check(_state()).tolist() == [False, False]


@pytest.mark.parametrize(
    ("ref", "expected_shape"),
    [
        (np.array([0.0, 1.0, 2.0]), (3,)),
        (np.zeros((3, 4)), (3,)),
    ],
    ids=["1d", "2d"],
)
def test_composite_empty_group_mask_keeps_batch_shape(ref, expected_shape):
    """Empty group mask keeps the reference's leading batch shape, not 0-D."""
    mask = TerminationGroup([]).check(TerminationState(batch=ref))

    assert mask.shape == expected_shape
    assert mask.dtype == np.bool_
    assert not mask.any()


def test_composite_empty_group_mask_keeps_torch_type():
    """The empty group mask stays a torch tensor for a torch reference."""
    torch = pytest.importorskip("torch")
    mask = TerminationGroup([]).check(TerminationState(batch=torch.zeros(4)))

    assert isinstance(mask, torch.Tensor)
    assert tuple(mask.shape) == (4,)
    assert not bool(mask.any())


def test_composite_group_rejects_unknown_mode():
    """Only OR and AND are legal; any other mode fails naming the value."""
    with pytest.raises(ValueError, match="mode must be 'or' or 'and', got 'xor'"):
        TerminationGroup([], mode="xor")
