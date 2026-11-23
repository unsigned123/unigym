"""EpisodeEvaluator assembles stable per-task tick traces into EpisodeData."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unigym.benchmarks.base import EpisodeData, EpisodeResult, Task, TickResult, TickTrace
from unigym.evaluation import EnvView
from unigym.runner.evaluation.buffer import TakenEpisode, TickSample
from unigym.runner.evaluation.scoring import EpisodeEvaluator


class _EpisodeTask(Task):
    """Episode-only task: never contributes a tick trace."""

    name = "episode_only"

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={"episode": 1.0}, success=True)


class _TickTask(Task):
    """Both-cadence task that reads its own trace at score time."""

    name = "tick_task"
    trigger = "both"

    def on_tick(self, ctx):
        del ctx
        return TickResult()

    def score(self, ep, ctx):
        del ctx
        trace = ep.ticks[self.name]
        return EpisodeResult(
            metrics={
                "tick_count": float(trace.metrics["det"].shape[0]),
                "fired_sum": float(np.asarray(trace.fired).sum()),
            },
            success=True,
        )


class _OtherTickTask(Task):
    """A second both task used to prove failure isolation."""

    name = "other_tick"
    trigger = "both"

    def on_tick(self, ctx):
        del ctx
        return TickResult()

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={"other": 1.0}, success=True)


def _steps(length: int) -> list[dict]:
    return [
        {
            "episode_id": torch.tensor(5),
            "time_in_episode": torch.tensor(0.02 * (index + 1)),
        }
        for index in range(length)
    ]


def _ticks(
    length: int,
    samples: dict[str, list[dict]],
) -> list[dict[str, TickSample]]:
    """Build one per-step task sample dict from per-step metric values."""
    out: list[dict[str, TickSample]] = []
    for index in range(length):
        step: dict[str, TickSample] = {}
        for task_name, per_step in samples.items():
            value = per_step[index]
            step[task_name] = TickSample(
                metrics={key: np.asarray(item) for key, item in value["metrics"].items()},
                fired=(None if value.get("fired") is None else np.asarray(value["fired"])),
            )
        out.append(step)
    return out


def _taken(
    *,
    length: int = 3,
    samples: dict[str, list[dict]] | None = None,
) -> TakenEpisode:
    return TakenEpisode(
        steps=_steps(length),
        rewinds=[],
        ticks=_ticks(length, samples or {}),
    )


def _bundle(*, episode_id: int = 5) -> dict:
    record = {
        "episode_id": torch.tensor([episode_id]),
        "time_in_episode": torch.tensor([0.02]),
    }
    return {
        "record": record,
        "terminated": torch.tensor([False]),
        "reset_facts": {},
        "terminations": {},
        "view": EnvView(
            record=record,
            obs=None,
            env_id=None,
            num_envs=1,
            backend="record-only",
        ),
    }


def _evaluator(tasks: list[Task]) -> EpisodeEvaluator:
    return EpisodeEvaluator(tasks=tasks, meta={"num_envs": 1, "policy_dt": 0.02, "sim_dt": 0.005})


def test_episode_only_task_has_no_tick_trace():
    """A run without a both task carries ``EpisodeData.ticks is None``."""
    evaluation = _evaluator([_EpisodeTask()]).evaluate(_taken(), _bundle(), 0)

    assert evaluation.episode.ticks is None
    assert [score.task for score in evaluation.scores] == ["episode_only"]


def test_both_task_trace_has_a_time_leading_axis():
    """Sample values stack to ``(T,)``/``(T, ...)`` with a ``(T,)`` bool fired."""
    samples = {
        "tick_task": [
            {"metrics": {"scalar": 1.0, "row": [0.0, 1.0]}, "fired": True},
            {"metrics": {"scalar": 2.0, "row": [2.0, 3.0]}, "fired": False},
            {"metrics": {"scalar": 3.0, "row": [4.0, 5.0]}, "fired": True},
        ]
    }

    evaluation = _evaluator([_TickTask()]).evaluate(_taken(samples=samples), _bundle(), 0)

    trace = evaluation.episode.ticks["tick_task"]
    assert trace.metrics["scalar"].shape == (3,)
    assert trace.metrics["row"].shape == (3, 2)
    assert trace.fired.shape == (3,)
    assert trace.fired.dtype == np.bool_
    assert trace.fired.tolist() == [True, False, True]


def test_score_runs_for_a_both_task_and_sees_its_trace():
    """``score`` still runs and reads the assembled tick history."""
    samples = {
        "tick_task": [
            {"metrics": {"det": 0.0}, "fired": True},
            {"metrics": {"det": 0.0}, "fired": True},
            {"metrics": {"det": 0.0}, "fired": False},
        ]
    }

    evaluation = _evaluator([_TickTask()]).evaluate(_taken(samples=samples), _bundle(), 0)

    assert evaluation.scores[0].task == "tick_task"
    assert evaluation.scores[0].result.metrics == {"tick_count": 3.0, "fired_sum": 2.0}


def test_metric_key_drift_fails_only_the_offending_task():
    """An unstable key set omits that task from scores, views and ticks."""
    samples = {
        "tick_task": [
            {"metrics": {"a": 1.0}, "fired": True},
            {"metrics": {"b": 2.0}, "fired": True},
            {"metrics": {"a": 3.0}, "fired": True},
        ],
        "other_tick": [
            {"metrics": {"det": 1.0}, "fired": True},
            {"metrics": {"det": 1.0}, "fired": True},
            {"metrics": {"det": 1.0}, "fired": True},
        ],
    }

    evaluation = _evaluator([_TickTask(), _OtherTickTask()]).evaluate(
        _taken(samples=samples), _bundle(), 0
    )

    assert [score.task for score in evaluation.scores] == ["other_tick"]
    assert [view.task for view in evaluation.views] == ["other_tick"]
    assert set(evaluation.episode.ticks) == {"other_tick"}


def test_metric_shape_drift_fails_the_task():
    """A metric whose trailing shape changes cannot be stacked into a trace."""
    samples = {
        "tick_task": [
            {"metrics": {"det": [0.0, 1.0]}, "fired": True},
            {"metrics": {"det": [0.0, 1.0, 2.0]}, "fired": True},
            {"metrics": {"det": [0.0, 1.0]}, "fired": True},
        ]
    }

    evaluation = _evaluator([_TickTask(), _EpisodeTask()]).evaluate(
        _taken(samples=samples), _bundle(), 0
    )

    assert [score.task for score in evaluation.scores] == ["episode_only"]
    assert evaluation.episode.ticks == {}


def test_fired_presence_drift_fails_the_task():
    """``fired`` must be present or absent for the whole trace."""
    samples = {
        "tick_task": [
            {"metrics": {"det": 1.0}, "fired": True},
            {"metrics": {"det": 1.0}, "fired": None},
            {"metrics": {"det": 1.0}, "fired": True},
        ]
    }

    evaluation = _evaluator([_TickTask(), _EpisodeTask()]).evaluate(
        _taken(samples=samples), _bundle(), 0
    )

    assert [score.task for score in evaluation.scores] == ["episode_only"]
    assert evaluation.episode.ticks == {}


def test_disabled_both_task_contributes_neither_score_nor_trace():
    """A task in the session's run-level disable set is skipped everywhere."""
    samples = {
        "tick_task": [
            {"metrics": {"det": 1.0}, "fired": True},
            {"metrics": {"det": 1.0}, "fired": True},
            {"metrics": {"det": 1.0}, "fired": True},
        ]
    }

    evaluation = _evaluator([_EpisodeTask(), _TickTask()]).evaluate(
        _taken(samples=samples), _bundle(), 0, skip_tasks=frozenset({"tick_task"})
    )

    assert [score.task for score in evaluation.scores] == ["episode_only"]
    assert evaluation.episode.ticks == {}


def test_trace_step_count_mismatch_fails_the_task():
    """A trace with fewer tick steps than episode frames fails the task."""
    taken = TakenEpisode(
        steps=_steps(3),
        rewinds=[],
        ticks=_ticks(2, {"tick_task": [{"metrics": {"det": 1.0}}, {"metrics": {"det": 1.0}}]}),
    )

    evaluation = _evaluator([_TickTask(), _EpisodeTask()]).evaluate(taken, _bundle(), 0)

    assert [score.task for score in evaluation.scores] == ["episode_only"]
    assert evaluation.episode.ticks == {}


def test_episode_data_rejects_a_trace_with_the_wrong_time_axis():
    """The EpisodeData contract validates trace leading axes against length."""
    with pytest.raises(ValueError, match=r"does not start with the episode length"):
        EpisodeData(
            data={},
            env_id=0,
            episode_id=1,
            length=2,
            duration_s=0.04,
            ended_by_termination=False,
            ticks={"t": TickTrace(metrics={"m": np.zeros(3)}, fired=None)},
        )
