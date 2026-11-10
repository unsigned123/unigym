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


def test_composite_group_rejects_unknown_mode():
    """Only OR and AND are legal; any other mode fails naming the value."""
    with pytest.raises(ValueError, match="mode must be 'or' or 'and', got 'xor'"):
        TerminationGroup([], mode="xor")
