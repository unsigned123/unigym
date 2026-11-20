"""BenchmarkSink publishes per-env EpisodeResultView tuples at episode end."""

from __future__ import annotations

import torch

from unigym.benchmarks.base import EpisodeResult, Task
from unigym.core.episode_result import EpisodeResultView
from unigym.runner.sinks.benchmark import BenchmarkSink


class _ScoredTask(Task):
    """Successful task with a fixed metric value."""

    def __init__(self, name: str, value: float, *, success: bool | None = True) -> None:
        super().__init__()
        self.name = name
        self.value = value
        self.success = success

    def score(self, ep) -> EpisodeResult:
        del ep
        return EpisodeResult(metrics={"score": self.value}, success=self.success)


class _EnvTask(Task):
    """Successful task whose metric is the scored env id."""

    name = "acme/env"

    def score(self, ep) -> EpisodeResult:
        return EpisodeResult(metrics={"env_id": float(ep.env_id)}, success=True)


class _FailingTask(Task):
    """Task that fails scoring and must be omitted from the views."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def score(self, ep) -> EpisodeResult:
        del ep
        raise RuntimeError("task scoring failed")


def _sink(tmp_path, tasks, *, num_envs: int = 1) -> BenchmarkSink:
    return BenchmarkSink(
        tasks=list(tasks),
        meta={"num_envs": num_envs, "policy_dt": 0.02},
        min_episodes=1,
        max_steps=10,
        report_path=str(tmp_path / "benchmark_report.json"),
    )


def _step_bundle(*episode_ids: int) -> dict:
    count = len(episode_ids)
    return {
        "record": {
            "episode_id": torch.tensor(episode_ids, dtype=torch.int64),
            "time_in_episode": torch.full((count,), 0.02),
        },
        "cmd": None,
        "terminated": torch.zeros(count, dtype=torch.bool),
    }


def _episode_bundle(*episode_ids: int) -> dict:
    bundle = _step_bundle(*episode_ids)
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


def test_sink_publishes_successful_views_in_declaration_order(tmp_path):
    """Views follow declaration order, not name order, and skip failing tasks."""
    tasks = [
        _ScoredTask("acme/zeta", 1.0),
        _FailingTask("acme/broken_first"),
        _ScoredTask("acme/alpha", 2.0, success=False),
        _FailingTask("acme/broken_middle"),
        _ScoredTask("acme/mike", 3.0, success=None),
        _FailingTask("acme/broken_last"),
    ]
    sink = _sink(tmp_path, tasks)
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


def test_sink_keys_views_by_env_id_not_position(tmp_path):
    """Two due envs are keyed by their real ids in arrival order."""
    sink = _sink(tmp_path, [_EnvTask()], num_envs=2)
    bundle = _step_bundle(7, 8)
    sink.on_step(0, bundle)

    sink.on_episode_end(torch.tensor([1, 0]), bundle)

    results = bundle["episode_results"]
    assert set(results) == {0, 1}
    assert results[1][0].metrics["env_id"] == 1.0
    assert results[0][0].metrics["env_id"] == 0.0


def test_sink_publishes_an_empty_tuple_for_a_due_env_without_scored_steps(tmp_path):
    """A boundary with no buffered frames still names the env with no views."""
    sink = _sink(tmp_path, [_ScoredTask("acme/good", 1.0)])
    bundle = _episode_bundle(0)

    sink.on_episode_end(torch.tensor([0]), bundle)

    assert bundle["episode_results"] == {0: ()}


def test_sink_episode_results_are_isolated_from_scored_metrics(tmp_path):
    """The published view owns a copy; later metric mutation cannot leak in."""
    sink = _sink(tmp_path, [_ScoredTask("acme/good", 1.5)])
    bundle = _step_bundle(7)
    sink.on_step(0, bundle)
    sink.on_episode_end(torch.tensor([0]), bundle)

    result = sink.results["acme/good"][0]
    result.metrics["score"] = 99.0

    assert dict(bundle["episode_results"][0][0].metrics) == {"score": 1.5}


def test_sink_episode_results_keys_are_actual_env_ids(tmp_path):
    """The mapping is keyed by env id even when due ids are not the full set."""
    sink = _sink(tmp_path, [_EnvTask()], num_envs=3)
    bundle = _step_bundle(7, 8, 9)
    sink.on_step(0, bundle)

    sink.on_episode_end(torch.tensor([2]), bundle)

    results = bundle["episode_results"]
    assert list(results) == [2]
    assert results[2][0].metrics["env_id"] == 2.0
    assert 0 not in results
