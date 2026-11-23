"""EvaluationSession lifecycle, injected collaborators and metrics feed."""

from __future__ import annotations

import importlib

import pytest
import torch

from unigym.benchmarks.base import EpisodeResult, Task
from unigym.benchmarks.reporting import BenchmarkReporter, EpisodeReporter
from unigym.runner.evaluation.buffer import EpisodeBuffer
from unigym.runner.evaluation.scoring import EpisodeEvaluator
from unigym.runner.evaluation.session import EvaluationSession
from unigym.runner.sinks.fanout import FanOutSink
from unigym.runtime.bridge import MetricsSource


class _RewardTask(Task):
    """Successful task carrying the bridge's reward slot."""

    name = "acme/reward"

    def score(self, ep, ctx) -> EpisodeResult:
        del ep, ctx
        return EpisodeResult(metrics={"reward": 0.5}, success=True)


class _FailingTask(Task):
    """Task whose scoring failure must not advance the session counter."""

    name = "acme/failing"

    def score(self, ep, ctx) -> EpisodeResult:
        del ep, ctx
        raise RuntimeError("boom")


class _MarkedDirector:
    """Director double that marks every episode directed."""

    def __init__(self) -> None:
        self.seen: list[int] = []

    def is_episode_directed(self, env_id: int) -> bool:
        self.seen.append(int(env_id))
        return True


class _RecordingReporter:
    """Reporter double with no benchmark reporter internals."""

    def __init__(self, *, draft: dict | None = None) -> None:
        self.draft = {"schema": "fake_report"} if draft is None else draft
        self.episodes: list[tuple[object, bool]] = []
        self.progress: list[tuple[int, int]] = []
        self.finished: tuple[int, int, float] | None = None

    def on_episode(self, evaluation, *, directed: bool) -> None:
        self.episodes.append((evaluation, directed))

    def note_progress(self, *, episodes_total: int, steps_total: int) -> None:
        self.progress.append((episodes_total, steps_total))

    def draft_report(self) -> dict:
        return self.draft

    def finish(self, *, episodes_total: int, steps_total: int, wall_time_s: float) -> dict:
        self.finished = (episodes_total, steps_total, wall_time_s)
        return {"schema": "fake_report", "episodes_total": episodes_total}


def _meta(num_envs: int = 1) -> dict:
    return {"run_name": "r", "num_envs": num_envs, "policy_dt": 0.02, "seed": 7}


def _collaborators(tasks, *, num_envs=1, reporter=None, env=None, retain_frames=False):
    meta = _meta(num_envs)
    return (
        EpisodeBuffer(num_envs),
        EpisodeEvaluator(tasks=tasks, meta=meta, env=env, retain_frames=retain_frames),
        reporter,
        meta,
    )


def _session(
    tmp_path,
    tasks,
    *,
    director=None,
    min_episodes: int = 1,
    num_envs: int = 1,
    reporter=None,
    env=None,
) -> EvaluationSession:
    if reporter is None:
        reporter = BenchmarkReporter(
            tasks=list(tasks),
            meta=_meta(num_envs),
            report_path=str(tmp_path / "benchmark_report.json"),
        )
    buffer, evaluator, reporter, meta = _collaborators(
        list(tasks), num_envs=num_envs, reporter=reporter, env=env
    )
    return EvaluationSession(
        tasks=list(tasks),
        buffer=buffer,
        evaluator=evaluator,
        reporter=reporter,
        meta=meta,
        num_envs=num_envs,
        min_episodes=min_episodes,
        max_steps=10,
        director=director,
    )


def _step_bundle(episode_id: int = 0) -> dict:
    return {
        "record": {
            "episode_id": torch.tensor([episode_id], dtype=torch.int64),
            "time_in_episode": torch.tensor([0.02]),
        },
        "raw": None,
        "cmd": None,
    }


def _end_bundle(episode_id: int = 0) -> dict:
    bundle = _step_bundle(episode_id)
    bundle["terminated"] = torch.tensor([False])
    bundle["reset_facts"] = {
        0: {
            "reset_index": 0,
            "reset_sample": None,
            "reset_valid": True,
            "reset_reason": None,
        }
    }
    bundle["terminations"] = {}
    return bundle


def test_session_uses_injected_collaborators_and_reporter_protocol(tmp_path):
    """The session drives the exact injected objects, not concrete classes."""
    tasks = [_RewardTask()]
    reporter = _RecordingReporter()
    buffer, evaluator, reporter, meta = _collaborators(tasks, reporter=reporter)

    session = EvaluationSession(
        tasks=tasks,
        buffer=buffer,
        evaluator=evaluator,
        reporter=reporter,
        meta=meta,
        num_envs=1,
        min_episodes=2,
        max_steps=10,
    )

    assert isinstance(reporter, EpisodeReporter)
    assert session.episode_buffer is buffer
    assert session.evaluator is evaluator
    assert session.reporter is reporter
    assert not isinstance(session.reporter, BenchmarkReporter)

    session.on_step(0, _step_bundle())
    session.on_episode_end(torch.tensor([0]), _end_bundle())

    assert reporter.progress == [(0, 1), (1, 1)]
    assert len(reporter.episodes) == 1
    assert reporter.episodes[0][0].scores[0].task == "acme/reward"
    assert session.metrics_report() is reporter.draft


def test_session_metrics_source_uses_the_documented_feed(tmp_path):
    """The sink adapter satisfies MetricsSource without reporter internals."""
    director = _MarkedDirector()
    session = _session(tmp_path, [_RewardTask()], director=director)

    assert isinstance(session, MetricsSource)
    assert session.metrics_task() == {"pack": "acme", "name": "reward"}
    assert session.metrics_progress() == {
        "episodes": 0,
        "min_episodes": 1,
        "by_task": {"acme/reward": 0},
    }
    assert session.metrics_drain() == []

    session.on_step(0, _step_bundle())
    session.on_episode_end(torch.tensor([0]), _end_bundle())

    assert director.seen == [0]
    assert session.metrics_drain() == [
        {"reward": 0.5, "success": True, "directed": True, "rewound": False}
    ]
    assert session.metrics_drain() == []
    assert session.metrics_progress()["by_task"] == {"acme/reward": 1}


def test_task_counters_follow_successful_scores_only(tmp_path):
    """The session's by_task counters advance only for successful scores."""
    reporter = _RecordingReporter()
    tasks = [_RewardTask(), _FailingTask()]
    session = _session(tmp_path, tasks, reporter=reporter)
    session.on_step(0, _step_bundle())
    session.on_episode_end(torch.tensor([0]), _end_bundle())

    assert session.metrics_progress()["by_task"] == {"acme/reward": 1, "acme/failing": 0}


def test_metrics_report_and_task_are_absent_without_tasks(tmp_path):
    """A taskless session reports no task identity and no draft report."""
    reporter = _RecordingReporter()
    session = _session(tmp_path, [], reporter=reporter)

    assert session.metrics_task() is None
    assert session.metrics_report() is None


def test_episodes_done_fires_exactly_at_the_min_episodes_threshold(tmp_path):
    """Early stop is the inclusive n_episodes >= min_episodes predicate."""
    session = _session(tmp_path, [_RewardTask()], min_episodes=3)

    session.n_episodes = 2
    assert session.episodes_done() is False
    session.n_episodes = 3
    assert session.episodes_done() is True


def test_on_run_end_returns_the_reporter_finish_document(tmp_path):
    """The session's on_run_end is the injected reporter's finish result."""
    reporter = _RecordingReporter()
    session = _session(tmp_path, [_RewardTask()], reporter=reporter)
    session.on_step(0, _step_bundle())
    session.on_episode_end(torch.tensor([0]), _end_bundle())

    report = session.on_run_end()

    assert report == {"schema": "fake_report", "episodes_total": 1}
    assert reporter.finished is not None
    assert reporter.finished[:2] == (1, 1)
    assert reporter.finished[2] >= 0.0


def test_fanout_summary_key_is_the_session_class(tmp_path):
    """FanOutSink names the benchmark summary by its sink class."""
    session = _session(tmp_path, [_RewardTask()], reporter=_RecordingReporter())

    summary = FanOutSink(session).on_run_end()

    assert set(summary) == {"EvaluationSession"}
    assert summary["EvaluationSession"]["episodes_total"] == 0


def test_benchmark_sink_module_is_gone():
    """The breaking change removed the module and the exported sink name."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("unigym.runner.sinks.benchmark")
    sinks = importlib.import_module("unigym.runner.sinks")
    assert sinks.__all__ == ["CollectSink", "FanOutSink"]
