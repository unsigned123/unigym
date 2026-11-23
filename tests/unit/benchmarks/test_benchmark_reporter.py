"""BenchmarkReporter owns the benchmark run report, results and metrics.

The fixed scenarios below prove the report/results/metrics behavior is
unchanged by the reporter extraction: exact report key set, terminal counts,
task aggregation, per-episode terminations, directed/rewound documents,
results episode order and ``metrics.npz`` keys.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from unigym.benchmarks.base import EpisodeData, EpisodeResult, Task
from unigym.benchmarks.episode import RunIdentity
from unigym.benchmarks.reporting import (
    METRICS_NPZ,
    REPORT_SCHEMA,
    RESULTS_JSON,
    RESULTS_SCHEMA,
    BenchmarkReporter,
    EpisodeReporter,
)
from unigym.benchmarks.termination import (
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationSource,
)
from unigym.runner.sinks.benchmark import BenchmarkSink

_REPORT_KEYS = [
    "schema",
    "run_name",
    "meta",
    "tasks",
    "episodes_total",
    "steps_total",
    "terminal_outcome_counts",
    "terminal_code_counts",
    "terminal_source_counts",
    "episode_terminations",
    "directed",
    "episodes_directed",
    "rewound",
    "episodes_rewound",
    "wall_time_s",
]

_RESULTS_KEYS = [
    "schema",
    "run",
    "run_dir",
    "backend",
    "episodes",
    "source",
    "terminal_outcome_counts",
    "terminal_code_counts",
    "terminal_source_counts",
]


class _GoodTask(Task):
    """Scores both episodes; episode 0 declares a task success termination."""

    name = "acme/good"

    def __init__(self) -> None:
        super().__init__({}, {})

    def score(self, ep, ctx) -> EpisodeResult:
        del ctx
        return self._episode_result(ep)

    def score_detail(self, ep, ctx, *, retain_frames: bool = False):
        del ctx
        result = self._episode_result(ep)
        frames = {"quality_trace": np.array([float(ep.episode_id), 2.0])} if retain_frames else {}
        return result, {"reached": True}, frames

    @staticmethod
    def _episode_result(ep) -> EpisodeResult:
        if ep.episode_id == 0:
            return EpisodeResult(
                metrics={"quality": 1.0, "reward": 0.5},
                success=True,
                termination=Termination(
                    outcome=EpisodeOutcome.SUCCESS,
                    code="task.success",
                    source=TerminationSource.CUSTOM,
                    step_index=ep.length,
                    time_s=0.02,
                ),
            )
        return EpisodeResult(metrics={"quality": 3.0, "reward": 0.1}, success=False)


class _BrokenTask(Task):
    """Always fails scoring; the evaluator omits it from every episode doc."""

    name = "acme/broken"

    def score(self, ep, ctx) -> EpisodeResult:
        del ep, ctx
        raise RuntimeError("task scoring failed")


class _FirstEpisodeDirector:
    """Marks the first episode as directed, then releases the label."""

    def __init__(self) -> None:
        self.directed = True
        self.calls: list[int] = []

    def is_episode_directed(self, env_id: int) -> bool:
        self.calls.append(env_id)
        return self.directed


def _identity() -> RunIdentity:
    return RunIdentity(
        backend="mujoco",
        device="cpu",
        num_envs=1,
        seed=7,
        policy_dt=0.02,
        sim_dt=0.005,
    )


def _sink(
    tmp_path,
    tasks,
    *,
    director=None,
    retain_frames: bool = False,
) -> BenchmarkSink:
    return BenchmarkSink(
        tasks=list(tasks),
        meta={
            "run_name": "r",
            "num_envs": 1,
            "policy_dt": 0.02,
            "seed": 7,
            "backend": "mujoco",
        },
        min_episodes=2,
        max_steps=10,
        report_path=str(tmp_path / "benchmark_report.json"),
        director=director,
        retain_frames=retain_frames,
        run_identity=_identity(),
    )


def _step_bundle(episode_id: int) -> dict:
    return {
        "record": {
            "episode_id": torch.tensor([episode_id], dtype=torch.int64),
            "time_in_episode": torch.tensor([0.02]),
        },
        "raw": None,
        "cmd": None,
        "targets": [],
    }


def _end_bundle(
    episode_id: int,
    *,
    reset_sample: dict | None = None,
    terminations: dict | None = None,
) -> dict:
    bundle = _step_bundle(episode_id)
    bundle["terminated"] = torch.tensor([False])
    bundle["reset_facts"] = {
        0: {
            "reset_index": 0,
            "reset_sample": reset_sample,
            "reset_valid": True,
            "reset_reason": None,
        }
    }
    bundle["terminations"] = terminations or {}
    return bundle


def _env_fall() -> EpisodeTermination:
    return EpisodeTermination(
        primary=Termination(
            outcome=EpisodeOutcome.FAILURE,
            code="env.fall",
            source=TerminationSource.ENVIRONMENT,
            step_index=1,
            time_s=0.02,
        )
    )


def _two_episode_sink(tmp_path) -> tuple[BenchmarkSink, _FirstEpisodeDirector]:
    """Two episodes, two tasks (one failing), one directed, one rewound."""
    director = _FirstEpisodeDirector()
    sink = _sink(
        tmp_path,
        [_GoodTask(), _BrokenTask()],
        director=director,
        retain_frames=True,
    )
    # Episode 0: directed, reset sample recorded, environment failure fact.
    sink.on_step(0, _step_bundle(0))
    sink.on_episode_end(
        torch.tensor([0]),
        _end_bundle(
            0,
            reset_sample={"articulation/g1/joint_pos": torch.zeros(2)},
            terminations={0: _env_fall()},
        ),
    )
    director.directed = False
    # Episode 1: straddles a timeline fork (rewound), no reset sample.
    sink.on_step(1, _step_bundle(1))
    sink.on_rewind({"tick": 5, "restored_tick": 4, "snap_id": "snap"})
    sink.on_episode_end(torch.tensor([0]), _end_bundle(1))
    return sink, director


def test_fixed_scenario_writes_the_exact_report_and_results(tmp_path):
    """Report keys/counts/aggregation and results order are behavior-stable."""
    sink, _ = _two_episode_sink(tmp_path)

    report = sink.on_run_end()

    assert list(report) == _REPORT_KEYS
    assert report["schema"] == REPORT_SCHEMA
    assert report["episodes_total"] == 2
    assert report["steps_total"] == 2

    good = report["tasks"]["acme/good"]
    assert good["params"] == {}
    assert good["n_episodes"] == 2
    assert good["success_rate"] == pytest.approx(0.5)
    assert good["metrics"]["quality"] == {
        "mean": pytest.approx(2.0),
        "std": pytest.approx(1.0),
        "min": pytest.approx(1.0),
        "max": pytest.approx(3.0),
        "p10": pytest.approx(1.2),
        "p90": pytest.approx(2.8),
    }
    assert good["metrics"]["reward"]["mean"] == pytest.approx(0.3)
    # the failing task is a zero-episode report entry, never silently removed
    assert report["tasks"]["acme/broken"] == {
        "params": {},
        "n_episodes": 0,
        "metrics": {},
        "success_rate": None,
    }

    assert report["terminal_outcome_counts"] == {"success": 1}
    assert report["terminal_code_counts"] == {"task.success": 1}
    assert report["terminal_source_counts"] == {"custom": 1}
    assert [doc["episode_id"] for doc in report["episode_terminations"]] == [0, 1]
    assert [doc["env"] for doc in report["episode_terminations"]] == [0, 0]
    assert [(doc["primary"] or {}).get("code") for doc in report["episode_terminations"]] == [
        "task.success",
        None,
    ]
    assert [term["code"] for term in report["episode_terminations"][0]["all"]] == [
        "env.fall",
        "task.success",
    ]
    assert report["episode_terminations"][1]["all"] == []

    assert report["directed"] is True
    assert report["episodes_directed"] == [
        {"episode_id": 0, "env_id": 0, "directed": True},
        {"episode_id": 1, "env_id": 0, "directed": False},
    ]
    assert report["rewound"] is True
    assert report["episodes_rewound"] == [
        {"episode_id": 0, "env_id": 0, "rewound": False},
        {"episode_id": 1, "env_id": 0, "rewound": True},
    ]

    on_disk = json.loads((tmp_path / "benchmark_report.json").read_text())
    assert list(on_disk) == _REPORT_KEYS
    assert on_disk["tasks"] == report["tasks"]

    results = json.loads((tmp_path / RESULTS_JSON).read_text())
    assert list(results) == _RESULTS_KEYS
    assert results["schema"] == RESULTS_SCHEMA
    assert [doc["episode_id"] for doc in results["episodes"]] == [0, 1]
    assert [doc["gate"] for doc in results["episodes"]] == ["acme/good", "acme/good"]
    assert [doc["verdict"] for doc in results["episodes"]] == ["success", "fail"]
    assert [doc["rewound"] for doc in results["episodes"]] == [False, True]
    assert results["episodes"][0]["episode_id"] == 0
    assert results["episodes"][0]["predicates"] == {"reached": True}
    assert results["episodes"][0]["metrics"] == {"quality": 1.0, "reward": 0.5}
    assert results["episodes"][0]["termination"]["primary"]["code"] == "task.success"
    assert results["episodes"][0]["reset"]["sample_keys"] == ["articulation/g1/joint_pos"]
    assert results["episodes"][1]["termination"] is None
    assert results["episodes"][1]["reset"]["sample_keys"] == []
    # retained frames are in metrics.npz, not results.json
    assert "frames" not in results["episodes"][0]

    frames = np.load(tmp_path / METRICS_NPZ)
    assert sorted(frames.files) == ["metric/e0/quality_trace", "metric/e1/quality_trace"]
    np.testing.assert_array_equal(frames["metric/e0/quality_trace"], np.array([0.0, 2.0]))
    np.testing.assert_array_equal(frames["metric/e1/quality_trace"], np.array([1.0, 2.0]))

    # reporter accumulation (internal, behavior-identical to the old sink lists)
    assert [ep.episode_id for ep in sink.reporter.episodes] == [0, 1]
    assert sink.reporter.episode_directed == [True, False]
    assert sink.reporter.results["acme/broken"] == []
    assert len(sink.reporter.results["acme/good"]) == 2


def test_empty_run_keeps_the_report_shape_without_results_files(tmp_path):
    """A run with no completed episode still writes the report but no results."""
    sink = _sink(tmp_path, [_GoodTask()])

    report = sink.on_run_end()

    assert list(report) == _REPORT_KEYS
    assert report["episodes_total"] == 0
    assert report["steps_total"] == 0
    assert report["tasks"]["acme/good"] == {
        "params": {},
        "n_episodes": 0,
        "metrics": {},
        "success_rate": None,
    }
    assert report["terminal_outcome_counts"] == {}
    assert report["terminal_code_counts"] == {}
    assert report["terminal_source_counts"] == {}
    assert report["episode_terminations"] == []
    assert report["directed"] is False
    assert report["episodes_directed"] == []
    assert report["rewound"] is False
    assert report["episodes_rewound"] == []

    on_disk = json.loads((tmp_path / "benchmark_report.json").read_text())
    assert on_disk["schema"] == REPORT_SCHEMA
    assert not (tmp_path / RESULTS_JSON).exists()
    assert not (tmp_path / METRICS_NPZ).exists()


def test_draft_report_drops_growing_lists_but_keeps_the_aggregate(tmp_path):
    """The bounded live draft has the final aggregates and no per-episode rows."""
    sink, _ = _two_episode_sink(tmp_path)

    draft = sink.metrics_report()
    report = sink.on_run_end()

    assert draft is not None
    assert "episodes_directed" not in draft
    assert "episodes_rewound" not in draft
    assert "episode_terminations" not in draft
    assert draft["tasks"] == report["tasks"]
    assert draft["episodes_total"] == report["episodes_total"] == 2
    assert draft["steps_total"] == report["steps_total"] == 2
    assert draft["terminal_outcome_counts"] == report["terminal_outcome_counts"]
    assert draft["wall_time_s"] is None
    assert report["wall_time_s"] is not None
    # the final artifact keeps the growing detail the draft omitted
    assert report["episodes_directed"]
    assert report["episodes_rewound"]
    assert report["episode_terminations"]


class _AggTask(_GoodTask):
    """A task-local reducer replaces the default run aggregation."""

    name = "acme/agg"

    def aggregate(self, results: list[EpisodeResult]) -> dict:
        return {
            "n_episodes": len(results),
            "success_rate": 0.25,
            "custom": {"flag": 1.0},
            "note": "not a metric block",
        }


def test_task_aggregate_override_is_honored(tmp_path):
    """The reporter delegates run-level aggregation to ``Task.aggregate``."""
    sink = _sink(tmp_path, [_AggTask()])
    sink.on_step(0, _step_bundle(0))
    sink.on_episode_end(torch.tensor([0]), _end_bundle(0))

    report = sink.on_run_end()

    assert report["tasks"]["acme/agg"] == {
        "params": {},
        "n_episodes": 1,
        "metrics": {"custom": {"flag": 1.0}},
        "success_rate": 0.25,
    }


def test_reporter_schema_constants_and_episode_fields():
    """The reporting vocabulary is stable and the reporter is an EpisodeReporter."""
    assert (REPORT_SCHEMA, RESULTS_SCHEMA) == ("benchmark_report", "results")
    assert (METRICS_NPZ, RESULTS_JSON) == ("metrics.npz", "results.json")
    data_fields = set(EpisodeData.__dataclass_fields__)
    assert {
        "data",
        "env_id",
        "episode_id",
        "length",
        "ended_by_termination",
        "rewound",
        "reset_valid",
        "terminal_index",
        "termination",
        "terminations",
    } <= data_fields
    assert {"metrics", "success", "termination", "debug"} <= set(EpisodeResult.__dataclass_fields__)
    reporter = BenchmarkReporter(tasks=[], meta={}, report_path="report.json")
    assert isinstance(reporter, EpisodeReporter)
