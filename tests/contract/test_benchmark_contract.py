"""Benchmark episode contract: schema, neutrality and isolation checks."""

from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path

import pytest

from unigym.benchmarks.base import (
    TASK_REGISTRY,
    EpisodeData,
    EpisodeResult,
    Task,
    build_tasks,
)
from unigym.benchmarks import reporting as benchmark_reporting
from unigym.benchmarks.episode import EpisodeLimits
from unigym.benchmarks.termination import EpisodeOutcome, Termination
from unigym.contacts.base import ContactGroupSpec
from unigym.initializers.base import ResetOutcome, wrap_initializer_result
from unigym.packs.runtime import discover


def test_report_and_results_have_stable_schemas():
    assert benchmark_reporting.REPORT_SCHEMA == "benchmark_report"
    assert benchmark_reporting.RESULTS_SCHEMA == "results"


def test_episode_data_carries_contract_fields():
    fields = EpisodeData.__dataclass_fields__
    for name in (
        "policy_dt",
        "sim_dt",
        "step_index",
        "time_s",
        "reset_index",
        "reset_sample",
        "reset_valid",
        "reset_reason",
        "terminal_index",
        "termination",
        "terminations",
        "ticks",
    ):
        assert name in fields
    assert fields["ticks"].default is None
    assert "termination" in EpisodeResult.__dataclass_fields__


def test_task_trigger_and_on_tick_are_the_opt_in_cadence_surface():
    """``trigger`` defaults to episode and ``on_tick`` is a real no-op hook."""
    from unigym.benchmarks import TickResult, TickTrace

    assert Task.trigger == "episode"
    on_tick = inspect.signature(Task.on_tick)
    assert list(on_tick.parameters) == ["self", "ctx"]

    result = TickResult()
    assert result.metrics == {}
    assert result.fired is None
    trace = TickTrace()
    assert trace.metrics == {}
    assert trace.fired is None


def test_public_export_contract_is_buffer_only_and_exposes_tick_types():
    """Ticks are public benchmark types; evaluation re-exports only the buffer."""
    import importlib.util

    import unigym.benchmarks as benchmarks
    import unigym.runner.evaluation as evaluation

    assert {"TickResult", "TickTrace"} <= set(benchmarks.__all__)
    assert set(evaluation.__all__) == {"EpisodeBuffer", "TakenEpisode", "TickSample"}
    assert "EvaluationSession" not in evaluation.__all__
    assert importlib.util.find_spec("unigym.runner.evaluation.session") is not None
    assert "BenchmarkSink" not in benchmarks.__all__


def test_termination_module_is_core_neutral():
    source = inspect.getsource(__import__("unigym.benchmarks.termination", fromlist=["x"]))
    assert "import gymnasium" not in source
    assert "from gymnasium" not in source
    values = {outcome.value for outcome in EpisodeOutcome}
    assert values == {"success", "failure", "deadline", "invalid_reset", "engine", "manual"}


def test_episode_limits_and_reset_outcome_are_exported():
    assert EpisodeLimits(timeout_steps=10).validate_against(0.02) == 10
    wrapped = wrap_initializer_result({"ball": {"pos": [0, 0, 1]}})
    assert isinstance(wrapped, ResetOutcome)
    assert wrapped.valid is None


def test_contact_spec_exposes_owner_and_aggregation():
    spec = ContactGroupSpec(
        name="support",
        sensor_refs=[{"owner": "environment", "body": "world"}],
        frame="world",
        aggregation="sum_magnitudes",
    )
    assert spec.metadata()["owner"] == "environment"
    assert spec.metadata()["sensor_refs"] == [{"owner": "environment", "body": "world"}]
    assert spec.metadata()["mode"] == "general"
    assert spec.force_key() == "contact/support/force"
    assert spec.active_bodies_key() == "contact/support/active_bodies"


def test_isaaclab3_common_contacts_use_the_typed_ref_schema():
    """The shared IsaacLab 3 contact face no longer consumes legacy body lists."""
    from unigym.backends.shared.isaaclab3_common import contacts

    names = {field.name for field in fields(ContactGroupSpec)}
    assert {"sensor_refs", "filter_refs"} <= names
    assert "sensor_bodies" not in names
    assert "filter_bodies" not in names
    source = Path(contacts.__file__).read_text(encoding="utf-8")
    assert "spec.sensor_bodies" not in source
    assert "spec.filter_bodies" not in source
    assert "ContactInventory" in source


def test_termination_extension_payload_is_json_safe():
    term = Termination(
        outcome=EpisodeOutcome.FAILURE,
        code="env.force_limit",
        source="environment",
        step_index=1,
        time_s=0.02,
    )
    assert term.to_json()["step_index"] == 1
    assert term.to_json()["metrics"] == {}


class _LegacyScoreTask(Task):
    """A pre-live task: one positional ``ep`` argument."""

    name = "contract_legacy_score"

    def score(self, ep):
        del ep
        return EpisodeResult(metrics={}, success=None)


class _LegacyDetailTask(Task):
    """A pre-live detail face: ``retain_frames`` as a second positional arg."""

    name = "contract_legacy_detail"

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={}, success=None)

    def score_detail(self, ep, retain_frames=False):
        del ep, retain_frames
        return EpisodeResult(metrics={}, success=None), {}, {}


class _MissingRetainTask(Task):
    """An override that forgot the keyword-only ``retain_frames`` parameter."""

    name = "contract_missing_retain"

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={}, success=None)

    def score_detail(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={}, success=None), {}, {}


def test_task_public_scoring_signatures_are_live_first():
    """The base protocol is ``score(ep, ctx)`` with a keyword-only detail flag."""
    score = inspect.signature(Task.score)
    assert list(score.parameters) == ["self", "ep", "ctx"]

    detail = inspect.signature(Task.score_detail)
    assert list(detail.parameters) == ["self", "ep", "ctx", "retain_frames"]
    retained = detail.parameters["retain_frames"]
    assert retained.kind is inspect.Parameter.KEYWORD_ONLY
    assert retained.default is False


def _build_registered(cls: type[Task]) -> list[Task]:
    TASK_REGISTRY[cls.name] = cls
    try:
        return build_tasks([{"name": cls.name}], {"articulations": {}}, discover([]))
    finally:
        TASK_REGISTRY.pop(cls.name, None)


def test_build_tasks_rejects_the_legacy_score_signature():
    """An old ``score(self, ep)`` task fails at build, not silently in the sink."""
    with pytest.raises(ValueError, match=r"score\(self, ep, ctx\)"):
        _build_registered(_LegacyScoreTask)


def test_build_tasks_rejects_the_legacy_score_detail_signature():
    """An old positional ``retain_frames`` detail face fails at build."""
    with pytest.raises(ValueError, match=r"score_detail\(self, ep, ctx"):
        _build_registered(_LegacyDetailTask)


def test_build_tasks_rejects_an_override_without_retain_frames():
    """An override that drops ``retain_frames`` would fail at sink time."""
    with pytest.raises(ValueError, match=r"must declare keyword-only `retain_frames`"):
        _build_registered(_MissingRetainTask)
