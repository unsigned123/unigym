"""RunIdentity + benchmark report (identity, counts, extensions)."""

from __future__ import annotations

import json

import pytest
import torch

from unigym.benchmarks.base import EpisodeResult, Task
from unigym.benchmarks.episode import EpisodeLimits, RobotIdentity, RunIdentity
from unigym.benchmarks.termination import (
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationExtension,
    TerminationSource,
)
from unigym.benchmarks.reporting import BenchmarkReporter
from unigym.runner.evaluation.buffer import EpisodeBuffer
from unigym.runner.evaluation.scoring import EpisodeEvaluator
from unigym.runner.evaluation.session import EvaluationSession


def _identity() -> RunIdentity:
    return RunIdentity(
        backend="mujoco",
        device="cpu",
        num_envs=1,
        seed=7,
        policy_dt=0.02,
        sim_dt=0.005,
        reset_seed=2,
        world_id="flat",
        articulations=(RobotIdentity(name="g1", profile="g1"),),
        observation_schema_version=3,
        model_revision=4,
    )


class _DummyTask(Task):
    name = "dummy"

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(
            metrics={"quality": 1.0},
            success=True,
            termination=Termination(
                outcome=EpisodeOutcome.SUCCESS,
                code="task.success",
                source=TerminationSource.CUSTOM,
                step_index=1,
                time_s=0.02,
                extensions={
                    "dummy": TerminationExtension(
                        namespace="dummy", schema_version=1, data={"reason": "all_good"}
                    )
                },
            ),
        )


def test_run_identity_json_roundtrip():
    ident = _identity()
    doc = ident.to_json()
    assert doc["backend"] == "mujoco"
    assert doc["articulations"] == [{"name": "g1", "profile": "g1"}]
    assert RunIdentity.from_mapping(doc).to_json() == doc


def test_run_identity_rejects_bad_periods():
    with pytest.raises(ValueError):
        RunIdentity(
            backend="mujoco",
            device="cpu",
            num_envs=1,
            seed=0,
            policy_dt=0.0,
            sim_dt=0.005,
        )


class _TermTask(Task):
    name = "term_task"

    def __init__(self, step_index):
        super().__init__({}, {})
        self._step_index = step_index

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(
            metrics={"quality": 1.0},
            success=True,
            termination=Termination(
                outcome=EpisodeOutcome.SUCCESS,
                code="task.success",
                source=TerminationSource.CUSTOM,
                step_index=self._step_index,
                time_s=None if self._step_index is None else 0.02,
            ),
        )


def _session(
    tmp_path,
    tasks,
    *,
    meta=None,
    limits=None,
    run_identity=None,
    report_path=None,
):
    tasks = list(tasks)
    if meta is None:
        meta = {"run_name": "r", "num_envs": 1, "policy_dt": 0.02, "seed": 7}
    if report_path is None:
        report_path = str(tmp_path / "benchmark_report.json")
    return EvaluationSession(
        tasks=tasks,
        buffer=EpisodeBuffer(int(meta.get("num_envs", 1))),
        evaluator=EpisodeEvaluator(tasks=tasks, meta=meta, limits=limits),
        reporter=BenchmarkReporter(
            tasks=tasks,
            meta=meta,
            report_path=report_path,
            run_identity=run_identity,
        ),
        meta=meta,
        num_envs=int(meta.get("num_envs", 1)),
        min_episodes=1,
        max_steps=10,
    )


def _run_single_step_episode(tmp_path, task):
    sink = _session(tmp_path, [task], run_identity=_identity())
    step = _step_bundle()
    sink.on_step(0, step)
    end = dict(step)
    end["terminated"] = torch.tensor([True])
    end["reset_facts"] = {
        0: {
            "reset_index": 0,
            "reset_sample": None,
            "reset_valid": True,
            "reset_reason": None,
        }
    }
    end["terminations"] = {}
    sink.on_episode_end(torch.tensor([0]), end)
    return sink


def test_task_termination_outside_episode_axis_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"episode-local axis 0\.\.1"):
        _run_single_step_episode(tmp_path, _TermTask(99))


@pytest.mark.parametrize("step_index", [None, 1])
def test_task_termination_within_episode_axis_is_accepted(tmp_path, step_index):
    sink = _run_single_step_episode(tmp_path, _TermTask(step_index))
    assert sink.reporter.episodes[0].terminal_index == step_index
    assert sink.reporter.episode_docs[0]["termination"]["primary"]["step_index"] == step_index


def _step_bundle():
    return {
        "record": {
            "episode_id": torch.tensor([0]),
            "time_in_episode": torch.tensor([0.02]),
        },
        "raw": None,
        "cmd": None,
        "targets": [],
    }


def test_report_has_run_identity_counts_and_extensions(tmp_path):
    sink = _session(
        tmp_path,
        [_DummyTask(params={}, meta={})],
        meta={"run_name": "r", "num_envs": 1, "policy_dt": 0.02, "seed": 7, "backend": "mujoco"},
        limits=EpisodeLimits(timeout_steps=1),
        run_identity=_identity(),
    )
    step = _step_bundle()
    sink.on_step(0, step)

    end = dict(step)
    end["terminated"] = torch.tensor([True])
    end["reset_facts"] = {
        0: {
            "reset_index": 0,
            "reset_sample": {"articulation/g1/joint_pos": torch.zeros(2)},
            "reset_valid": True,
            "reset_reason": None,
        }
    }
    end["terminations"] = {
        0: EpisodeTermination(
            primary=Termination(
                outcome=EpisodeOutcome.FAILURE,
                code="env.fall",
                source=TerminationSource.ENVIRONMENT,
                step_index=1,
                time_s=0.02,
            )
        )
    }
    sink.on_episode_end(torch.tensor([0]), end)
    report = sink.on_run_end()

    assert report["schema"] == "benchmark_report"
    run = report["meta"]["run"]
    assert run["backend"] == "mujoco"
    assert run["reset_seed"] == 2
    assert run["observation_schema_version"] == 3
    assert run["model_revision"] == 4
    # SUCCESS (task) outranks FAILURE and DEADLINE under core precedence
    assert report["terminal_outcome_counts"] == {"success": 1}
    assert report["terminal_code_counts"] == {"task.success": 1}
    assert report["terminal_source_counts"] == {"custom": 1}
    assert report["episode_terminations"][0]["primary"]["extensions"]["dummy"]["data"] == {
        "reason": "all_good"
    }
    codes = {term["code"] for term in report["episode_terminations"][0]["all"]}
    assert {"task.success", "env.fall", "runtime.task_deadline"} <= codes

    on_disk = json.loads((tmp_path / "benchmark_report.json").read_text())
    assert on_disk["schema"] == "benchmark_report"
    assert on_disk["meta"]["run"]["world_id"] == "flat"

    results = json.loads((tmp_path / "results.json").read_text())
    assert results["schema"] == "results"
    assert results["run"]["backend"] == "mujoco"
    assert results["episodes"][0]["termination"]["primary"]["outcome"] == "success"
    assert {"env", "verdict", "gate"} <= set(results["episodes"][0])


def test_deadline_is_not_stamped_after_overrun(tmp_path):
    sink = _session(
        tmp_path,
        [_DummyTask(params={}, meta={})],
        limits=EpisodeLimits(timeout_steps=1),
        run_identity=_identity(),
    )
    step = _step_bundle()
    sink.on_step(0, step)
    sink.on_step(1, step)  # episode overran the 1-step deadline
    end = dict(step)
    end["terminated"] = torch.tensor([False])
    end["reset_facts"] = {
        0: {
            "reset_index": 0,
            "reset_sample": None,
            "reset_valid": True,
            "reset_reason": None,
        }
    }
    end["terminations"] = {}
    sink.on_episode_end(torch.tensor([0]), end)
    report = sink.on_run_end()
    codes = {term["code"] for episode in report["episode_terminations"] for term in episode["all"]}
    assert "runtime.task_deadline" not in codes


_EXPECTED_RUN = {
    "schema_version": 2,
    "backend": "mujoco",
    "device": "cpu",
    "num_envs": 1,
    "seed": 7,
    "policy_dt": 0.02,
    "sim_dt": 0.005,
    "reset_seed": 2,
    "world_id": "flat",
    "articulations": [{"name": "g1", "profile": "g1"}],
    "observation_schema_version": 3,
    "model_revision": 4,
}

_ENV_FALL = {
    "outcome": "failure",
    "code": "env.fall",
    "source": "environment",
    "step_index": 1,
    "time_s": 0.02,
    "metrics": {},
    "extensions": {},
    "priority": 0,
}

_DEADLINE = {
    "outcome": "deadline",
    "code": "runtime.task_deadline",
    "source": "runtime",
    "step_index": 1,
    "time_s": 0.02,
    "metrics": {"timeout_steps": 1.0},
    "extensions": {},
    "priority": 0,
}

_TASK_SUCCESS = {
    "outcome": "success",
    "code": "task.success",
    "source": "custom",
    "step_index": 1,
    "time_s": 0.02,
    "metrics": {},
    "extensions": {
        "dummy": {
            "namespace": "dummy",
            "schema_version": 1,
            "data": {"reason": "all_good"},
        }
    },
    "priority": 0,
}

_EXPECTED_REPORT = {
    "schema": "benchmark_report",
    "run_name": "r",
    "meta": {
        "run": _EXPECTED_RUN,
        "joint_names_count": 0,
        "policy_dt": 0.02,
        "sim_dt": None,
        "seed": 7,
    },
    "tasks": {
        "dummy": {
            "params": {},
            "n_episodes": 1,
            "metrics": {
                "quality": {
                    "mean": 1.0,
                    "std": 0.0,
                    "min": 1.0,
                    "max": 1.0,
                    "p10": 1.0,
                    "p90": 1.0,
                }
            },
            "success_rate": 1.0,
        }
    },
    "episodes_total": 1,
    "steps_total": 1,
    "terminal_outcome_counts": {"success": 1},
    "terminal_code_counts": {"task.success": 1},
    "terminal_source_counts": {"custom": 1},
    "episode_terminations": [
        {
            "env": 0,
            "episode_id": 0,
            "primary": _TASK_SUCCESS,
            "all": [_ENV_FALL, _DEADLINE, _TASK_SUCCESS],
        }
    ],
    "directed": False,
    "episodes_directed": [{"episode_id": 0, "env_id": 0, "directed": False}],
    "rewound": False,
    "episodes_rewound": [{"episode_id": 0, "env_id": 0, "rewound": False}],
    "wall_time_s": None,
}

_EXPECTED_RESULTS = {
    "schema": "results",
    "run": _EXPECTED_RUN,
    "run_dir": "sub",
    "backend": "mujoco",
    "episodes": [
        {
            "env": 0,
            "episode_id": 0,
            "verdict": "success",
            "gate": "dummy",
            "predicates": {},
            "metrics": {"quality": 1.0},
            "rewound": False,
            "termination": {
                "primary": _TASK_SUCCESS,
                "all": [_ENV_FALL, _DEADLINE, _TASK_SUCCESS],
            },
            "reset": {
                "index": 0,
                "valid": True,
                "reason": None,
                "sample_keys": ["articulation/g1/joint_pos"],
            },
        }
    ],
    "source": "engine",
    "terminal_outcome_counts": {"success": 1},
    "terminal_code_counts": {"task.success": 1},
    "terminal_source_counts": {"custom": 1},
}


def test_report_and_results_serialization_is_unchanged(tmp_path, monkeypatch):
    """The fixed-scenario report/results bytes are stable across the refactor."""
    monkeypatch.chdir(tmp_path)
    sink = _session(
        tmp_path,
        [_DummyTask(params={}, meta={})],
        meta={"run_name": "r", "num_envs": 1, "policy_dt": 0.02, "seed": 7, "backend": "mujoco"},
        limits=EpisodeLimits(timeout_steps=1),
        run_identity=_identity(),
        report_path="sub/benchmark_report.json",
    )
    step = _step_bundle()
    sink.on_step(0, step)
    end = dict(step)
    end["terminated"] = torch.tensor([True])
    end["reset_facts"] = {
        0: {
            "reset_index": 0,
            "reset_sample": {"articulation/g1/joint_pos": torch.zeros(2)},
            "reset_valid": True,
            "reset_reason": None,
        }
    }
    end["terminations"] = {
        0: EpisodeTermination(
            primary=Termination(
                outcome=EpisodeOutcome.FAILURE,
                code="env.fall",
                source=TerminationSource.ENVIRONMENT,
                step_index=1,
                time_s=0.02,
            )
        )
    }
    sink.on_episode_end(torch.tensor([0]), end)
    sink.on_run_end()

    results_text = (tmp_path / "sub" / "results.json").read_text()
    assert results_text == json.dumps(_EXPECTED_RESULTS, indent=1, ensure_ascii=False)

    on_disk = json.loads((tmp_path / "sub" / "benchmark_report.json").read_text())
    on_disk["wall_time_s"] = None
    assert json.dumps(on_disk, indent=2, ensure_ascii=False) == json.dumps(
        _EXPECTED_REPORT, indent=2, ensure_ascii=False
    )
