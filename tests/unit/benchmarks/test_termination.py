"""Core-neutral termination facts: vocabulary, validation, precedence."""

from __future__ import annotations

import numpy as np
import pytest

from unigym.benchmarks.termination import (
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationExtension,
    TerminationSource,
    code_counts,
    ensure_json_value,
    outcome_counts,
    source_counts,
    validate_code,
)


def _term(
    outcome=EpisodeOutcome.FAILURE,
    code="env.fall",
    source=TerminationSource.ENVIRONMENT,
    **kw,
):
    return Termination(outcome=outcome, code=code, source=source, **kw)


def test_code_must_be_namespaced_and_stable():
    assert validate_code("env.force_limit") == "env.force_limit"
    assert validate_code("runtime.episode_horizon")
    for bad in ("force_limit", "Env.Force", "unknown.foo", "env.", ""):
        with pytest.raises(ValueError):
            validate_code(bad)


def test_extensions_are_json_safe_and_namespaced():
    ext = TerminationExtension(
        namespace="pack", schema_version=1, data={"reason": "x", "values": [1, 2.5]}
    )
    term = _term(extensions={"pack": ext})
    assert term.to_json()["extensions"]["pack"]["data"]["reason"] == "x"

    with pytest.raises(TypeError):
        TerminationExtension(namespace="pack", schema_version=1, data={"x": np.array([1])})
    with pytest.raises(ValueError):
        TerminationExtension(namespace="pack", schema_version=1, data={"x": float("nan")})
    with pytest.raises(ValueError):
        _term(extensions={"other": ext})


def test_metrics_must_be_finite_floats():
    assert _term(metrics={"force": 12.5}).metrics == {"force": 12.5}
    with pytest.raises(ValueError):
        _term(metrics={"force": float("inf")})
    with pytest.raises(TypeError):
        _term(metrics={"force": object()})


def test_ensure_json_value_rejects_live_objects():
    assert ensure_json_value({"a": [1, True, None, "x"]}) == {"a": [1, True, None, "x"]}
    with pytest.raises(TypeError):
        ensure_json_value({"tensor": object()})


def test_core_precedence_and_priority_tiebreak():
    invalid = _term(
        outcome=EpisodeOutcome.INVALID_RESET,
        code="initializer.invalid_reset",
        source=TerminationSource.INITIALIZER,
        step_index=0,
        time_s=0.0,
    )
    success = _term(
        outcome=EpisodeOutcome.SUCCESS, code="task.success", source=TerminationSource.CUSTOM
    )
    deadline = _term(
        outcome=EpisodeOutcome.DEADLINE,
        code="runtime.task_deadline",
        source=TerminationSource.RUNTIME,
        step_index=10,
        time_s=0.5,
    )
    bundle = EpisodeTermination.from_terminations([deadline, success, invalid])
    assert bundle.primary is invalid
    assert bundle.all == (deadline, success, invalid)

    low = _term(metrics={"v": 0.0}, priority=0)
    high = _term(code="env.other", metrics={"v": 1.0}, priority=5)
    assert EpisodeTermination.from_terminations([low, high]).primary is high


def test_all_must_contain_primary():
    a = _term()
    b = _term(code="env.other")
    with pytest.raises(ValueError):
        EpisodeTermination(primary=a, all=(b,))


def test_counts_ignore_none():
    a = _term(outcome=EpisodeOutcome.FAILURE)
    b = _term(
        outcome=EpisodeOutcome.DEADLINE,
        code="runtime.task_deadline",
        source=TerminationSource.RUNTIME,
    )
    assert outcome_counts([a, b, None]) == {"failure": 1, "deadline": 1}
    assert code_counts([a, b]) == {"env.fall": 1, "runtime.task_deadline": 1}
    assert source_counts([a, b]) == {"environment": 1, "runtime": 1}


def test_outcome_vocabulary_has_no_task_or_gym_flags():
    values = {o.value for o in EpisodeOutcome}
    assert "terminated" not in values and "truncated" not in values
