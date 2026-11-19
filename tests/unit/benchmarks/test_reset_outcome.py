"""ResetOutcome: structured legality plus the legacy placements wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from unigym.initializers.base import (
    InitializerBase,
    ResetOutcome,
    wrap_initializer_result,
)


class _Legacy(InitializerBase):
    name = "legacy_test"

    def apply(self):
        return {"ball": {"pos": [0.0, 0.0, 1.0]}}


class _Structured(_Legacy):
    name = "structured_test"

    def apply(self):
        return ResetOutcome(
            valid=False,
            reason="no_free_space",
            metrics={"clearance": 0.0},
            sample={"articulation/g1/joint_pos": np.zeros(3)},
            placements={"ball": {"pos": [0.0, 0.0, 0.2]}},
        )


def test_legacy_placements_are_wrapped():
    outcome = wrap_initializer_result({"ball": {"pos": [0, 0, 1]}})
    assert outcome.valid is None and outcome.reason is None and outcome.sample is None
    assert outcome.placements == {"ball": {"pos": [0, 0, 1]}}


def test_structured_outcome_passes_through():
    outcome = ResetOutcome(valid=True, metrics={"clearance": 0.4})
    assert wrap_initializer_result(outcome) is outcome


def test_apply_outcome_wraps_legacy_initializer():
    init = _Legacy({}, None, None)
    outcome = init.apply_outcome()
    assert isinstance(outcome, ResetOutcome)
    assert outcome.valid is None
    assert "ball" in outcome.placements


def test_apply_outcome_keeps_structured_initializer_legality():
    init = _Structured({}, None, None)
    outcome = init.apply_outcome()
    assert outcome.valid is False
    assert outcome.reason == "no_free_space"
    assert outcome.metrics == {"clearance": 0.0}
    assert outcome.sample["articulation/g1/joint_pos"].shape == (3,)


def test_reset_outcome_validation():
    with pytest.raises(ValueError):
        ResetOutcome(valid=False, metrics={"clearance": float("nan")})
    with pytest.raises(TypeError):
        ResetOutcome(valid="yes")
