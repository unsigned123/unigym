"""Rewind provider wiring and initializer state capture/restore boundaries."""

from __future__ import annotations

import pytest
import torch

from unigym.benchmarks.reporting import BenchmarkReporter
from unigym.initializers.base import InitializerBase
from unigym.runner.evaluation.buffer import EpisodeBuffer
from unigym.runner.evaluation.scoring import EpisodeEvaluator
from unigym.runner.evaluation.session import EvaluationSession
from unigym.runner.sinks.fanout import FanOutSink
from unigym.runtime.rewind import RewindGateError
from unigym.runtime.rewind_providers import (
    EvaluationSessionProvider,
    InitializerStateProvider,
    build_rewind_providers,
)


def _initializer(seed: int = 0) -> InitializerBase:
    return InitializerBase({"seed": seed}, None, None)


def _session(tmp_path, *, num_envs: int = 1) -> EvaluationSession:
    tasks: list = []
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
        min_episodes=5,
        max_steps=10,
    )


def _step_bundle(*, episode_id: int = 0, num_envs: int = 1) -> dict:
    return {
        "record": {
            "episode_id": torch.full((num_envs,), episode_id, dtype=torch.int64),
            "time_in_episode": torch.zeros(num_envs),
        },
        "cmd": None,
    }


def test_build_rewind_providers_mounts_only_capable_initializer():
    """An initializer joins the rewind package only when it can restore."""
    capable = _initializer()
    providers = build_rewind_providers(None, initializer=capable)

    assert [provider.name for provider in providers] == ["initializer"]
    assert build_rewind_providers(None, initializer=None) == []
    assert build_rewind_providers(None, initializer=object()) == []


def test_initializer_provider_snapshot_is_isolated_from_live_state():
    """A captured snapshot is a deep copy; mutating it leaves the RNG untouched."""
    initializer = _initializer(seed=11)
    provider = InitializerStateProvider(initializer)
    snapshot = provider.rewind_state()

    snapshot["rng"]["state"]["state"] += 1

    reference = _initializer(seed=11)
    assert initializer.rng.integers(0, 1_000_000) == reference.rng.integers(0, 1_000_000)


def test_initializer_provider_restore_replays_the_same_stream():
    """Applying a captured state replays the initializer's RNG stream exactly."""
    initializer = _initializer(seed=5)
    provider = InitializerStateProvider(initializer)
    state = provider.rewind_state()

    expected = [int(initializer.rng.integers(0, 1000)) for _ in range(3)]
    provider.apply_rewind_state(state)

    assert [int(initializer.rng.integers(0, 1000)) for _ in range(3)] == expected


@pytest.mark.parametrize(
    ("initializer", "state", "match"),
    [
        (_initializer(), None, "rewind state is missing"),
        (object(), {}, "cannot restore rewind state"),
    ],
    ids=["missing-state", "unrestorable-initializer"],
)
def test_initializer_provider_validate_rejects_unrestorable_state(initializer, state, match):
    """Pre-flight refuses state an initializer could not restore."""
    provider = InitializerStateProvider(initializer)

    with pytest.raises(ValueError, match=match):
        provider.validate_rewind_state(state)


def test_evaluation_session_provider_mounts_on_episode_buffer(tmp_path):
    """A session exposing an episode_buffer mounts the benchmark provider."""
    sink = _session(tmp_path)

    providers = build_rewind_providers(sink)

    assert [provider.name for provider in providers] == ["benchmark"]
    assert isinstance(providers[0], EvaluationSessionProvider)


def test_evaluation_session_provider_restore_returns_open_steps_and_clears_marks(tmp_path):
    """Restoring the snapshot drops the abandoned branch's step and rewind mark."""
    sink = _session(tmp_path)
    sink.on_step(0, _step_bundle(episode_id=0))
    provider = build_rewind_providers(sink)[0]
    state = provider.rewind_state()

    sink.on_step(1, _step_bundle(episode_id=0))
    sink.on_rewind({"tick": 5, "restored_tick": 2, "snap_id": 1})
    assert len(sink.episode_buffer.bufs[0]) == 2
    assert len(sink.episode_buffer.rewinds[0]) == 1

    provider.apply_rewind_state(state)

    assert len(sink.episode_buffer.bufs[0]) == 1
    assert sink.episode_buffer.rewinds[0] == []
    assert provider.rewind_state()["episode_stamp"] == 0


def test_evaluation_session_provider_restores_episode_fuse(tmp_path):
    """n_episodes and the derived _stop latch ride the fork back."""
    sink = _session(tmp_path)
    sink.n_episodes = 2
    sink._stop = True
    provider = build_rewind_providers(sink)[0]
    state = provider.rewind_state()

    sink.n_episodes = 9
    sink._stop = False
    provider.apply_rewind_state(state)

    assert sink.n_episodes == 2
    assert sink._stop is True


def test_evaluation_session_provider_refuses_cross_episode_rewind(tmp_path):
    """A checkpoint from another episode is refused by both rewind phases."""
    sink = _session(tmp_path)
    sink.on_step(0, _step_bundle(episode_id=0))
    provider = build_rewind_providers(sink)[0]
    state = provider.rewind_state()
    sink.on_step(1, _step_bundle(episode_id=1))

    with pytest.raises(RewindGateError, match="cross-episode"):
        provider.validate_rewind_state(state)
    with pytest.raises(RewindGateError, match="cross-episode"):
        provider.apply_rewind_state(state)


class _RewindRecorder:
    def __init__(self):
        self.marks: list[dict] = []

    def on_rewind(self, info: dict) -> None:
        self.marks.append(dict(info))


def test_nested_fanout_forwards_on_rewind_once_per_child():
    """Every nested child sees exactly one on_rewind call."""
    first, second, third = _RewindRecorder(), _RewindRecorder(), _RewindRecorder()
    fan = FanOutSink(FanOutSink(first), FanOutSink(second, FanOutSink(third)))
    info = {"tick": 3, "restored_tick": 1, "snap_id": 7}

    fan.on_rewind(info)

    assert first.marks == [info]
    assert second.marks == [info]
    assert third.marks == [info]


def test_nested_fanout_marks_a_real_benchmark_buffer_once(tmp_path):
    """An evaluation session below a nested fan-out receives the rewind mark."""
    sink = _session(tmp_path)
    sink.on_step(0, _step_bundle(episode_id=0))

    FanOutSink(FanOutSink(sink)).on_rewind({"tick": 1, "restored_tick": 0, "snap_id": 2})

    assert len(sink.episode_buffer.rewinds[0]) == 1


def test_nested_fanout_skips_children_without_on_rewind():
    """A sink without on_rewind does not break the fan-out notification."""
    recorder = _RewindRecorder()
    fan = FanOutSink(object(), FanOutSink(recorder))

    fan.on_rewind({"tick": 1})

    assert recorder.marks == [{"tick": 1}]
