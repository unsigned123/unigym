"""Generic shard-series operations retained for the analysis CLI."""

from __future__ import annotations

import numpy as np
import pytest

from unigym.analysis.series import SERIES_OPS, SeriesError, apply_op


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("max", 4.0),
        ("min", 1.0),
        ("mean", 2.5),
        ("std", float(np.std([1.0, 2.0, 3.0, 4.0]))),
    ],
)
def test_reduction_ops_return_one_scalar_and_no_aux(op: str, expected: float) -> None:
    """Value reductions collapse the series and report no auxiliary index."""
    value, aux = apply_op(np.array([1.0, 2.0, 3.0, 4.0]), op)

    assert value == pytest.approx(expected)
    assert aux is None


def test_relative_and_norm_ops_follow_the_retained_semantics() -> None:
    """rel_* subtracts frame 0; norm_* is the per-frame L2 reduction."""
    assert apply_op(np.array([1.0, 2.0, 4.0]), "rel_max") == (3.0, None)
    assert apply_op(np.array([1.0, 2.0, 4.0]), "rel_min") == (0.0, None)
    assert apply_op(np.array([[-3.0, 1.0]]), "abs_max") == (3.0, None)
    assert apply_op(np.array([[-3.0, 1.0]]), "abs_min") == (1.0, None)
    assert apply_op(np.array([[3.0, 4.0], [6.0, 8.0]]), "norm_max") == (10.0, None)
    assert apply_op(np.array([[3.0, 4.0], [6.0, 8.0]]), "norm_mean") == (7.5, None)


def test_arg_ops_return_an_index_and_no_value() -> None:
    """arg* ops report the plateau tick, not a scalar reduction."""
    series = np.array([1.0, 3.0, 2.0, 4.0])

    assert apply_op(series, "argmax") == (None, 3)
    assert apply_op(series, "argmin") == (None, 0)


def test_first_ops_return_the_first_crossing_tick() -> None:
    """first_* reports the first tick crossing the requested threshold."""
    series = np.array([1.0, 2.0, 3.0, 4.0])

    assert apply_op(series, "first_gt", 2.5) == (None, 2)
    assert apply_op(series, "first_lt", 2.5) == (None, 0)
    assert apply_op(series, "first_ge", 4.0) == (None, 3)
    assert apply_op(series, "first_le", 0.5) == (None, None)


def test_unknown_empty_and_thresholdless_series_fail_fast() -> None:
    """Unsupported ops, empty input and a missing first_* threshold raise."""
    assert SERIES_OPS

    with pytest.raises(SeriesError, match="unknown --op"):
        apply_op(np.array([1.0]), "median")
    with pytest.raises(SeriesError, match="empty series"):
        apply_op(np.array([]), "max")
    with pytest.raises(SeriesError, match="requires --value"):
        apply_op(np.array([1.0]), "first_gt")
