"""Explicit articulation ownership: required params and fail-fast live reads."""

from __future__ import annotations

import pytest

from unigym.evaluation import EnvView, EvalContext, EvalViewError
from unigym.runtime.controllers import ArticulationRuntime
from unigym.terminations import (
    BaseContactTermination,
    FallTermination,
    JointLimitTermination,
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


def _ctx(*, articulations: dict[str, ArticulationRuntime], record: dict) -> EvalContext:
    return EvalContext(
        env=EnvView(
            record=record,
            obs=None,
            env_id=None,
            num_envs=1,
            device="cpu",
            articulations=articulations,
        )
    )


@pytest.mark.parametrize(
    "term_type", [FallTermination, BaseContactTermination, JointLimitTermination]
)
def test_articulation_termination_missing_articulation_fails_fast(term_type):
    """A term fails when the requested articulation is absent from the record."""
    term = term_type({"articulation": "robot"})
    ctx = _ctx(articulations={}, record={})

    with pytest.raises(
        EvalViewError,
        match="articulation 'robot' is not in the evaluation record; available: <none>",
    ):
        term.check(ctx)
