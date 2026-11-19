"""Benchmark episode contract: schema, neutrality and isolation checks."""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import pytest

from unigym.benchmarks.base import EpisodeData, EpisodeResult
from unigym.benchmarks.episode import EpisodeLimits
from unigym.benchmarks.termination import EpisodeOutcome, Termination
from unigym.contacts.base import ContactGroupSpec
from unigym.initializers.base import ResetOutcome, wrap_initializer_result
from unigym.runner.sinks import benchmark as benchmark_sink

REPO = Path(__file__).resolve().parents[2]


def test_report_and_results_have_stable_schemas():
    assert benchmark_sink.REPORT_SCHEMA == "benchmark_report"
    assert benchmark_sink.RESULTS_SCHEMA == "results"


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
    ):
        assert name in fields
    assert "termination" in EpisodeResult.__dataclass_fields__


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
        sensor_bodies=["floor"],
        frame="world",
        owner="environment",
        aggregation="sum_magnitudes",
    )
    assert spec.metadata()["owner"] == "environment"
    assert spec.force_key() == "contact/support/force"
    assert spec.active_bodies_key() == "contact/support/active_bodies"


def test_no_isaaclab3_files_touched():
    if not (REPO / ".git").exists():
        pytest.skip("not a git worktree")
    out = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "status",
            "--porcelain",
            "--",
            "unigym/backends/isaaclab3",
            "unigym/backends/shared/isaaclab3_common",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == ""


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
