"""EvaluationSession per-tick dispatch, run-level disable and rewind truth."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unigym.benchmarks.base import EpisodeResult, Task, TickResult
from unigym.benchmarks.reporting import BenchmarkReporter
from unigym.evaluation import EnvView, EvalViewError
from unigym.runner.evaluation.buffer import EpisodeBuffer
from unigym.runner.evaluation.scoring import EpisodeEvaluator
from unigym.runner.evaluation.session import EvaluationSession
from unigym.runtime.rewind_providers import EvaluationSessionProvider


def _meta(num_envs: int = 1) -> dict:
    return {"run_name": "r", "num_envs": num_envs, "policy_dt": 0.02, "seed": 7}


class _VectorTickTask(Task):
    """Both-cadence task recording the vector context and its trace."""

    name = "acme/vector"
    trigger = "both"

    def __init__(self, *, num_envs: int = 1, log: list | None = None) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.log = [] if log is None else log

    def on_tick(self, ctx):
        self.log.append(("tick", ctx.env_id, ctx.t, ctx.env.scope_size))
        return TickResult(
            metrics={"det": torch.full((self.num_envs,), 2.0)},
            fired=torch.ones(self.num_envs, dtype=torch.bool),
        )

    def score(self, ep, ctx):
        del ctx
        self.log.append(("score", ep.episode_id, sorted(ep.ticks)))
        trace = ep.ticks[self.name]
        return EpisodeResult(metrics={"ticks": float(trace.metrics["det"].shape[0])}, success=True)


class _OtherTickTask(Task):
    """A second both task proving dispatch ordering and isolation."""

    name = "acme/other"
    trigger = "both"

    def __init__(self, *, num_envs: int = 1) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.calls = 0

    def on_tick(self, ctx):
        del ctx
        self.calls += 1
        return TickResult(
            metrics={"det": torch.zeros(self.num_envs)},
            fired=torch.zeros(self.num_envs, dtype=torch.bool),
        )

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={"other": 1.0}, success=True)


class _RaisingTickTask(Task):
    """Both task whose per-tick hook raises on every call."""

    name = "acme/raising"
    trigger = "both"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def on_tick(self, ctx):
        del ctx
        self.calls += 1
        raise RuntimeError("tick boom")

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={"raising": 1.0}, success=True)


class _ScalarTickTask(Task):
    """Both task returning a scalar (no env axis) metric."""

    name = "acme/scalar"
    trigger = "both"

    def on_tick(self, ctx):
        del ctx
        return TickResult(metrics={"bad": torch.tensor(1.0)})

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={"scalar": 1.0}, success=True)


class _WrongTypeTickTask(Task):
    """Both task returning a non-TickResult object."""

    name = "acme/wrong_type"
    trigger = "both"

    def on_tick(self, ctx):
        del ctx
        return "not a tick result"

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={"wrong": 1.0}, success=True)


class _BadFiredTask(Task):
    """Both task returning a non-bool ``fired`` mask."""

    name = "acme/bad_fired"
    trigger = "both"

    def on_tick(self, ctx):
        del ctx
        return TickResult(fired=torch.zeros(1, dtype=torch.int64))

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={"bad_fired": 1.0}, success=True)


class _ScalarFiredTask(Task):
    """Both task returning a scalar (no env axis) ``fired`` mask."""

    name = "acme/scalar_fired"
    trigger = "both"

    def on_tick(self, ctx):
        del ctx
        return TickResult(fired=torch.tensor(True))

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={"scalar_fired": 1.0}, success=True)


class _NonTensorMetricTask(Task):
    """Both task returning a non-tensor metric value."""

    name = "acme/non_tensor"
    trigger = "both"

    def on_tick(self, ctx):
        del ctx
        return TickResult(metrics={"bad": 1.0})

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={"non_tensor": 1.0}, success=True)


class _EpisodeTask(Task):
    """Episode-only task for the report-schema comparison."""

    name = "acme/episode"

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={"episode": 1.0}, success=True)


class _RecordTickTask(Task):
    """Both-cadence task recording the record keys it observes."""

    name = "acme/record"
    trigger = "both"

    def __init__(self, *, num_envs: int = 1) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.keys: list[tuple[str, ...]] = []

    def on_tick(self, ctx):
        self.keys.append(tuple(sorted(ctx.env.record())))
        return TickResult(metrics={"det": torch.zeros(self.num_envs)})

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={"record": 1.0}, success=True)


def _step_bundle(episode_id: int = 0, num_envs: int = 1) -> dict:
    record = {
        "episode_id": torch.full((num_envs,), episode_id, dtype=torch.int64),
        "time_in_episode": torch.full((num_envs,), 0.02),
    }
    return {
        "record": record,
        "raw": None,
        "cmd": None,
        "view": EnvView(
            record=record,
            obs=None,
            env_id=None,
            num_envs=num_envs,
            backend="record-only",
        ),
    }


def _end_bundle(episode_id: int = 0, num_envs: int = 1) -> dict:
    bundle = _step_bundle(episode_id, num_envs)
    bundle["terminated"] = torch.zeros(num_envs, dtype=torch.bool)
    bundle["reset_facts"] = {}
    bundle["terminations"] = {}
    return bundle


def _session(tmp_path, tasks, *, num_envs: int = 1, min_episodes: int = 1):
    meta = _meta(num_envs)
    reporter = BenchmarkReporter(
        tasks=list(tasks),
        meta=meta,
        report_path=str(tmp_path / "benchmark_report.json"),
    )
    return EvaluationSession(
        tasks=list(tasks),
        buffer=EpisodeBuffer(num_envs),
        evaluator=EpisodeEvaluator(tasks=list(tasks), meta=meta),
        reporter=reporter,
        meta=meta,
        num_envs=num_envs,
        min_episodes=min_episodes,
        max_steps=10,
    )


def test_on_tick_receives_one_vector_context_at_the_policy_tick(tmp_path):
    """The per-tick context is vector-scoped and carries the policy tick."""
    task = _VectorTickTask(num_envs=2)
    session = _session(tmp_path, [task], num_envs=2)

    session.on_step(4, _step_bundle(num_envs=2))

    assert task.log == [("tick", None, 4, 2)]


def test_tick_context_receives_the_normalized_record(tmp_path):
    """The ``with_record`` child carries CollectSink's normalized record keys."""
    task = _RecordTickTask(num_envs=1)
    session = _session(tmp_path, [task])
    record = {
        "episode_id": torch.tensor([0], dtype=torch.int64),
        "time_in_episode": torch.tensor([0.02]),
        "articulation/g1/base_pos_w": torch.zeros(1, 3),
    }
    bundle = {
        "record": record,
        "raw": torch.zeros(1, 2),
        "cmd": torch.zeros(1, 1),
        "view": EnvView(
            record=record,
            obs=None,
            env_id=None,
            num_envs=1,
            backend="record-only",
        ),
    }

    session.on_step(0, bundle)

    keys = task.keys[0]
    assert "articulation/g1/base_pos" in keys
    assert "articulation/g1/base_pos_w" not in keys
    assert "action_raw" in keys
    assert "command" in keys


def test_tick_dispatch_does_not_close_the_shared_boundary_view(tmp_path):
    """Tick dispatch closes only its derived child, never ``bundle["view"]``."""
    session = _session(tmp_path, [_VectorTickTask(num_envs=1)])
    bundle = _step_bundle()
    parent = bundle["view"]

    session.on_step(0, bundle)

    assert parent.num_envs == 1
    assert "episode_id" in parent.record()


def test_tick_dispatch_requires_the_runner_boundary_view(tmp_path):
    """A bundle without ``view`` fails fast before any task ticks."""
    session = _session(tmp_path, [_VectorTickTask(num_envs=1)])
    bundle = _step_bundle()
    del bundle["view"]

    with pytest.raises(EvalViewError, match=r"bundle\['view'\]"):
        session.on_step(0, bundle)


def test_tick_metrics_are_sliced_per_env_to_cpu_numpy(tmp_path):
    """A ``(N,)`` metric and bool fired reach each env's trace as numpy."""
    task = _VectorTickTask(num_envs=2)
    session = _session(tmp_path, [task], num_envs=2)

    session.on_step(0, _step_bundle(num_envs=2))
    session.on_episode_end(torch.tensor([0, 1]), _end_bundle(num_envs=2))

    episode = session.reporter.episodes[0]
    trace = episode.ticks["acme/vector"]
    assert trace.metrics["det"].shape == (1,)
    assert trace.metrics["det"].dtype == np.float32
    assert trace.fired.shape == (1,)
    assert trace.fired.dtype == np.bool_


def test_multiple_both_tasks_share_one_tick_dispatch(tmp_path):
    """Every active both task ticks on the same step and buffers its trace."""
    first = _VectorTickTask(num_envs=1)
    second = _OtherTickTask(num_envs=1)
    session = _session(tmp_path, [first, second])

    session.on_step(0, _step_bundle())
    session.on_episode_end(torch.tensor([0]), _end_bundle())

    episode = session.reporter.episodes[0]
    assert set(episode.ticks) == {"acme/vector", "acme/other"}
    assert session.metrics_progress()["by_task"] == {"acme/vector": 1, "acme/other": 1}
    assert second.calls == 1


def test_ticks_dispatch_before_the_episode_end_score(tmp_path):
    """The final tick is buffered before ``score`` observes the trace."""
    task = _VectorTickTask(num_envs=1)
    session = _session(tmp_path, [task])

    session.on_step(3, _step_bundle())
    assert task.log == [("tick", None, 3, 1)]

    session.on_episode_end(torch.tensor([0]), _end_bundle())

    assert task.log == [("tick", None, 3, 1), ("score", 0, ["acme/vector"])]


@pytest.mark.parametrize(
    "bad_task",
    [
        _RaisingTickTask(),
        _ScalarTickTask(),
        _WrongTypeTickTask(),
        _BadFiredTask(),
        _ScalarFiredTask(),
        _NonTensorMetricTask(),
    ],
)
def test_tick_contract_failure_disables_only_that_task_for_the_run(tmp_path, bad_task):
    """A bad ``on_tick`` disables the task for later episodes; others score."""
    good = _OtherTickTask(num_envs=1)
    session = _session(tmp_path, [bad_task, good], min_episodes=10)

    session.on_step(0, _step_bundle())
    session.on_episode_end(torch.tensor([0]), _end_bundle())
    session.on_step(1, _step_bundle())
    session.on_episode_end(torch.tensor([0]), _end_bundle(episode_id=1))

    report = session.on_run_end()

    assert report["tasks"][bad_task.name]["n_episodes"] == 0
    assert report["tasks"]["acme/other"]["n_episodes"] == 2
    assert good.calls == 2
    if isinstance(bad_task, _RaisingTickTask):
        assert bad_task.calls == 1  # disabled: not called on the second episode


def test_report_schema_is_unchanged_for_a_both_run(tmp_path):
    """A both run keeps the episode-only report key set (no tick fields)."""
    episode_session = _session(tmp_path / "episode", [_EpisodeTask()])
    episode_session.on_step(0, _step_bundle())
    episode_session.on_episode_end(torch.tensor([0]), _end_bundle())
    episode_report = episode_session.on_run_end()

    tick_session = _session(tmp_path / "both", [_VectorTickTask(num_envs=1)])
    tick_session.on_step(0, _step_bundle())
    tick_session.on_episode_end(torch.tensor([0]), _end_bundle())
    tick_report = tick_session.on_run_end()

    assert set(tick_report) == set(episode_report)
    assert "ticks" not in tick_report
    assert set(tick_report["tasks"]) == {"acme/vector"}


def test_rewind_restores_open_buffer_tick_traces(tmp_path):
    """A buffer snapshot/restore round-trips the per-step tick samples."""
    task = _VectorTickTask(num_envs=1)
    session = _session(tmp_path, [task])
    provider = EvaluationSessionProvider(session)

    session.on_step(0, _step_bundle())
    state = provider.rewind_state()
    session.on_step(1, _step_bundle())
    assert len(session.episode_buffer.ticks[0]) == 2

    provider.apply_rewind_state(state)

    assert len(session.episode_buffer.ticks[0]) == 1
    sample = session.episode_buffer.ticks[0][0]["acme/vector"]
    assert float(sample.metrics["det"]) == pytest.approx(2.0)
    assert bool(sample.fired)


def test_rewind_does_not_re_enable_a_disabled_task(tmp_path):
    """The run-level disable set is process truth and outlives a rewind."""
    bad = _RaisingTickTask()
    good = _OtherTickTask(num_envs=1)
    session = _session(tmp_path, [bad, good], min_episodes=10)
    provider = EvaluationSessionProvider(session)
    empty = provider.rewind_state()

    session.on_step(0, _step_bundle())
    assert bad.calls == 1

    provider.apply_rewind_state(empty)
    session.on_step(1, _step_bundle())
    assert bad.calls == 1
    session.on_episode_end(torch.tensor([0]), _end_bundle())

    report = session.on_run_end()
    assert report["tasks"][bad.name]["n_episodes"] == 0
