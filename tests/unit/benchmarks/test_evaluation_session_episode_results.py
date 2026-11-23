"""EvaluationSession publishes per-env EpisodeResultView tuples at episode end."""

from __future__ import annotations

import pytest
import torch

from tests._support.fakes.benchmark_env import FakeBenchmarkEnv
from unigym.benchmarks.base import EpisodeResult, Task
from unigym.benchmarks.reporting import BenchmarkReporter
from unigym.core.episode_result import EpisodeResultView
from unigym.env.obs import ObsSnapshot, ObservationSchema, ObservationSpec
from unigym.evaluation import (
    EnvView,
    EvalQueryUnavailableError,
    EvalViewError,
    EvalViewStaleError,
)
from unigym.runner.evaluation.buffer import EpisodeBuffer
from unigym.runner.evaluation.scoring import EpisodeEvaluator
from unigym.runner.evaluation.session import EvaluationSession


class _ScoredTask(Task):
    """Successful task with a fixed metric value."""

    def __init__(self, name: str, value: float, *, success: bool | None = True) -> None:
        super().__init__()
        self.name = name
        self.value = value
        self.success = success

    def score(self, ep, ctx) -> EpisodeResult:
        del ep, ctx
        return EpisodeResult(metrics={"score": self.value}, success=self.success)


class _EnvTask(Task):
    """Successful task whose metric is the scored env id."""

    name = "acme/env"

    def score(self, ep, ctx) -> EpisodeResult:
        del ctx
        return EpisodeResult(metrics={"env_id": float(ep.env_id)}, success=True)


class _FailingTask(Task):
    """Task that fails scoring and must be omitted from the views."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def score(self, ep, ctx) -> EpisodeResult:
        del ep, ctx
        raise RuntimeError("task scoring failed")


def _session(tmp_path, tasks, *, num_envs: int = 1) -> EvaluationSession:
    tasks = list(tasks)
    meta = {"num_envs": num_envs, "policy_dt": 0.02}
    return EvaluationSession(
        tasks=tasks,
        buffer=EpisodeBuffer(num_envs),
        evaluator=EpisodeEvaluator(tasks=tasks, meta=meta),
        reporter=BenchmarkReporter(
            tasks=tasks,
            meta=meta,
            report_path=str(tmp_path / "benchmark_report.json"),
        ),
        meta=meta,
        num_envs=num_envs,
        min_episodes=1,
        max_steps=10,
    )


def _view(
    record: dict,
    *,
    num_envs: int,
    obs: ObsSnapshot | None = None,
    env: FakeBenchmarkEnv | None = None,
) -> EnvView:
    if env is not None:
        return env.view(record, obs=obs)
    return EnvView(
        record=record,
        obs=obs,
        env_id=None,
        num_envs=num_envs,
        backend="record-only",
    )


def _step_bundle(
    *episode_ids: int,
    env: FakeBenchmarkEnv | None = None,
    obs: ObsSnapshot | None = None,
) -> dict:
    count = len(episode_ids)
    record = {
        "episode_id": torch.tensor(episode_ids, dtype=torch.int64),
        "time_in_episode": torch.full((count,), 0.02),
    }
    bundle = {
        "record": record,
        "cmd": None,
        "terminated": torch.zeros(count, dtype=torch.bool),
        "view": _view(record, num_envs=count, obs=obs, env=env),
    }
    if obs is not None:
        bundle["obs"] = obs
    return bundle


def _episode_bundle(
    *episode_ids: int,
    env: FakeBenchmarkEnv | None = None,
    obs: ObsSnapshot | None = None,
) -> dict:
    bundle = _step_bundle(*episode_ids, env=env, obs=obs)
    bundle["reset_facts"] = {
        ei: {
            "reset_index": 0,
            "reset_sample": None,
            "reset_valid": None,
            "reset_reason": None,
        }
        for ei in range(len(episode_ids))
    }
    return bundle


def test_session_publishes_successful_views_in_declaration_order(tmp_path):
    """Views follow declaration order, not name order, and skip failing tasks."""
    tasks = [
        _ScoredTask("acme/zeta", 1.0),
        _FailingTask("acme/broken_first"),
        _ScoredTask("acme/alpha", 2.0, success=False),
        _FailingTask("acme/broken_middle"),
        _ScoredTask("acme/mike", 3.0, success=None),
        _FailingTask("acme/broken_last"),
    ]
    sink = _session(tmp_path, tasks)
    bundle = _step_bundle(7)
    sink.on_step(0, bundle)

    sink.on_episode_end(torch.tensor([0]), bundle)

    views = bundle["episode_results"][0]
    assert [view.task for view in views] == ["acme/zeta", "acme/alpha", "acme/mike"]
    assert [view.success for view in views] == [True, False, None]
    assert [dict(view.metrics) for view in views] == [
        {"score": 1.0},
        {"score": 2.0},
        {"score": 3.0},
    ]
    assert all(isinstance(view, EpisodeResultView) for view in views)


def test_session_keys_views_by_env_id_not_position(tmp_path):
    """Two due envs are keyed by their real ids in arrival order."""
    sink = _session(tmp_path, [_EnvTask()], num_envs=2)
    bundle = _step_bundle(7, 8)
    sink.on_step(0, bundle)

    sink.on_episode_end(torch.tensor([1, 0]), bundle)

    results = bundle["episode_results"]
    assert set(results) == {0, 1}
    assert results[1][0].metrics["env_id"] == 1.0
    assert results[0][0].metrics["env_id"] == 0.0


def test_session_publishes_an_empty_tuple_for_a_due_env_without_scored_steps(tmp_path):
    """A boundary with no buffered frames still names the env with no views."""
    sink = _session(tmp_path, [_ScoredTask("acme/good", 1.0)])
    bundle = _episode_bundle(0)

    sink.on_episode_end(torch.tensor([0]), bundle)

    assert bundle["episode_results"] == {0: ()}


def test_session_episode_results_are_isolated_from_scored_metrics(tmp_path):
    """The published view owns a copy; later metric mutation cannot leak in."""
    sink = _session(tmp_path, [_ScoredTask("acme/good", 1.5)])
    bundle = _step_bundle(7)
    sink.on_step(0, bundle)
    sink.on_episode_end(torch.tensor([0]), bundle)

    result = sink.reporter.results["acme/good"][0]
    result.metrics["score"] = 99.0

    assert dict(bundle["episode_results"][0][0].metrics) == {"score": 1.5}


def test_session_episode_results_keys_are_actual_env_ids(tmp_path):
    """The mapping is keyed by env id even when due ids are not the full set."""
    sink = _session(tmp_path, [_EnvTask()], num_envs=3)
    bundle = _step_bundle(7, 8, 9)
    sink.on_step(0, bundle)

    sink.on_episode_end(torch.tensor([2]), bundle)

    results = bundle["episode_results"]
    assert list(results) == [2]
    assert results[2][0].metrics["env_id"] == 2.0
    assert 0 not in results


class _ContextTask(Task):
    """Scores from the scoped record and the read-only query."""

    name = "acme/context"

    def __init__(self, seen: list[object]) -> None:
        super().__init__()
        self._seen = seen

    def score(self, ep, ctx) -> EpisodeResult:
        self._seen.append(ctx)
        record = ctx.env.record()
        return EpisodeResult(
            metrics={
                "env_id": float(ctx.env_id),
                "rows": float(record["articulation/g1/joint_pos"].shape[0]),
                "query_backend": float(ctx.env.query.backend == "fake"),
            },
            success=True,
        )


class _CrossEnvTask(Task):
    """A per-env task that tries to read another env through the query."""

    name = "acme/cross_env"

    def __init__(self, errors: list[Exception]) -> None:
        super().__init__()
        self._errors = errors

    def score(self, ep, ctx) -> EpisodeResult:
        del ep
        try:
            ctx.env.query.state.root_pose("g1", env=0)
        except Exception as exc:
            self._errors.append(exc)
            raise
        return EpisodeResult(metrics={"ok": 1.0}, success=True)


class _HasQueryTask(Task):
    """Record-only context reports query unavailability explicitly."""

    name = "acme/has_query"

    def score(self, ep, ctx) -> EpisodeResult:
        del ep
        return EpisodeResult(metrics={"has_query": float(ctx.env.has_query)}, success=True)


class _QueryCallTask(Task):
    """Touching the unavailable query fails and is isolated."""

    name = "acme/query_call"

    def __init__(self, errors: list[Exception]) -> None:
        super().__init__()
        self._errors = errors

    def score(self, ep, ctx) -> EpisodeResult:
        del ep
        try:
            ctx.env.query
        except Exception as exc:
            self._errors.append(exc)
            raise
        return EpisodeResult(metrics={"unexpected": 1.0}, success=True)


class _ObsTask(Task):
    """Reads the per-env scoped observation snapshot."""

    name = "acme/obs"

    def score(self, ep, ctx) -> EpisodeResult:
        del ep
        obs = ctx.env.obs()
        assert obs is not None
        value = obs.value("articulation/g1/joint_pos")
        return EpisodeResult(
            metrics={"obs_rows": float(value.shape[0]), "obs_value": float(value[0, 0])},
            success=True,
        )


def _live_step_bundle(
    count: int,
    *,
    env: FakeBenchmarkEnv | None = None,
    obs: ObsSnapshot | None = None,
) -> dict:
    record = {
        "episode_id": torch.arange(count, dtype=torch.int64),
        "time_in_episode": torch.full((count,), 0.02),
        "articulation/g1/joint_pos": torch.zeros(count, 2),
    }
    bundle = {
        "record": record,
        "cmd": None,
        "terminated": torch.zeros(count, dtype=torch.bool),
        "view": _view(record, num_envs=count, obs=obs, env=env),
    }
    if obs is not None:
        bundle["obs"] = obs
    return bundle


def test_session_builds_a_per_env_live_context_and_closes_it(tmp_path):
    """Tasks see the due env id, a ``(1, ...)`` record and a stale context."""
    seen: list[object] = []
    env = FakeBenchmarkEnv(num_envs=2)
    sink = _session(tmp_path, [_ContextTask(seen)], num_envs=2)
    bundle = _live_step_bundle(2, env=env)
    sink.on_step(0, bundle)

    sink.on_episode_end(torch.tensor([1]), bundle)

    result = sink.reporter.results["acme/context"][0]
    assert result.metrics == {"env_id": 1.0, "rows": 1.0, "query_backend": 1.0}
    assert env.query_reads == 1
    with pytest.raises(EvalViewStaleError):
        seen[0].env.num_envs
    # episode scoring closes only its derived per-env child
    assert bundle["view"].num_envs == 2


def test_session_scopes_the_observation_snapshot_to_the_due_env(tmp_path):
    """``ctx.env.obs()`` carries only the due env's row, not the vector."""
    env = FakeBenchmarkEnv(num_envs=2)
    sink = _session(tmp_path, [_ObsTask()], num_envs=2)
    schema = ObservationSchema(
        {"articulation/g1/joint_pos": ObservationSpec("articulation/g1/joint_pos", (2,), "f32")}
    )
    obs = ObsSnapshot(
        {"articulation/g1/joint_pos": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        schema=schema,
        num_envs=2,
        device="cpu",
    )
    bundle = _live_step_bundle(2, env=env, obs=obs)
    sink.on_step(0, bundle)

    sink.on_episode_end(torch.tensor([1]), bundle)

    result = sink.reporter.results["acme/obs"][0]
    assert result.metrics == {"obs_rows": 1.0, "obs_value": 3.0}


def test_session_does_not_touch_the_query_for_a_context_ignoring_task(tmp_path):
    """The query factory is lazy; a pure task never opens the query handle."""
    env = FakeBenchmarkEnv(num_envs=1)
    sink = _session(tmp_path, [_ScoredTask("acme/pure", 2.0)], num_envs=1)
    bundle = _live_step_bundle(1, env=env)
    sink.on_step(0, bundle)

    sink.on_episode_end(torch.tensor([0]), bundle)

    assert sink.reporter.results["acme/pure"][0].metrics == {"score": 2.0}
    assert env.query_reads == 0


def test_record_only_context_has_no_query_and_isolates_query_callers(tmp_path):
    """Without a live env the context is explicitly query-unavailable."""
    errors: list[Exception] = []
    sink = _session(tmp_path, [_HasQueryTask(), _QueryCallTask(errors)], num_envs=1)
    bundle = _live_step_bundle(1)
    sink.on_step(0, bundle)

    sink.on_episode_end(torch.tensor([0]), bundle)

    assert sink.reporter.results["acme/has_query"][0].metrics == {"has_query": 0.0}
    assert sink.reporter.results["acme/query_call"] == []
    assert len(errors) == 1
    assert isinstance(errors[0], EvalQueryUnavailableError)


def test_per_env_query_rejects_another_env_and_isolates_the_task(tmp_path):
    """A scoped query read for another env is an error, not a cross-env read."""
    errors: list[Exception] = []
    env = FakeBenchmarkEnv(num_envs=2)
    sink = _session(tmp_path, [_CrossEnvTask(errors)], num_envs=2)
    bundle = _live_step_bundle(2, env=env)
    sink.on_step(0, bundle)

    sink.on_episode_end(torch.tensor([1]), bundle)

    assert sink.reporter.results["acme/cross_env"] == []
    assert env.query_reads == 1
    assert len(errors) == 1
    assert isinstance(errors[0], EvalViewError)
