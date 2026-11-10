"""Explicit articulation ownership: required params and fail-fast state reads."""

from __future__ import annotations

import pytest

from unigym.terminations import (
    BaseContactTermination,
    FallTermination,
    JointLimitTermination,
    TerminationState,
)


@pytest.mark.parametrize(
    "term_type", [FallTermination, BaseContactTermination, JointLimitTermination]
)
def test_articulation_termination_requires_explicit_articulation(term_type):
    """Every per-articulation term rejects construction without an articulation."""
    with pytest.raises(ValueError, match=rf"{term_type.__name__} requires explicit 'articulation'"):
        term_type({})
    with pytest.raises(ValueError, match=rf"{term_type.__name__} requires explicit 'articulation'"):
        term_type({"articulation": ""})


@pytest.mark.parametrize(
    "term_type", [FallTermination, BaseContactTermination, JointLimitTermination]
)
def test_articulation_termination_missing_state_fails_fast(term_type):
    """A term fails with its own name when the requested articulation is absent."""
    term = term_type({"articulation": "robot"})

    with pytest.raises(
        ValueError,
        match=rf"{term_type.__name__}: state has no articulation 'robot' keys; available: \[\]",
    ):
        term.check(TerminationState())
