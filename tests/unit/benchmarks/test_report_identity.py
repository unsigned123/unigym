"""RunIdentity + benchmark report v2 (identity, counts, extensions)."""

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
from unigym.runner.sinks.benchmark import BenchmarkSink


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

    def score(self, ep):
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

    def score(self, ep):
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


def _run_single_step_episode(tmp_path, task):
    sink = BenchmarkSink(
        tasks=[task],
        meta={"run_name": "r", "num_envs": 1, "policy_dt": 0.02, "seed": 7},
        min_episodes=1,
        max_steps=10,
        report_path=str(tmp_path / "benchmark_report.json"),
        run_identity=_identity(),
    )
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
    assert sink.episodes[0].terminal_index == step_index
    assert sink.episode_docs[0]["termination"]["primary"]["step_index"] == step_index


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


def test_report_v2_has_run_identity_counts_and_extensions(tmp_path):
    sink = BenchmarkSink(
        tasks=[_DummyTask(params={}, meta={})],
        meta={"run_name": "r", "num_envs": 1, "policy_dt": 0.02, "seed": 7, "backend": "mujoco"},
        min_episodes=1,
        max_steps=10,
        report_path=str(tmp_path / "benchmark_report.json"),
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

    assert report["schema"] == "benchmark_report.v2"
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
    assert on_disk["schema"] == "benchmark_report.v2"
    assert on_disk["meta"]["run"]["world_id"] == "flat"

    results = json.loads((tmp_path / "results.json").read_text())
    assert results["schema"] == "results.v2"
    assert results["run"]["backend"] == "mujoco"
    assert results["episodes"][0]["termination"]["primary"]["outcome"] == "success"
    assert {"env", "verdict", "gate"} <= set(results["episodes"][0])


def test_deadline_is_not_stamped_after_overrun(tmp_path):
    sink = BenchmarkSink(
        tasks=[_DummyTask(params={}, meta={})],
        meta={"run_name": "r", "num_envs": 1, "policy_dt": 0.02, "seed": 7},
        min_episodes=1,
        max_steps=10,
        report_path=str(tmp_path / "benchmark_report.json"),
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
