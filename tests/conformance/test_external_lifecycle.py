"""Terminate/reset lifecycle agreement over the fake backend boundary."""

from __future__ import annotations

import pytest
import torch

from tests._support.fakes.backend import ARTICULATION_NAME, FakeBackend
from unigym.adapters.manual import ManualAdapter
from unigym.core.episode_termination import EpisodeOutcome, Termination, TerminationSource
from unigym.runner.sync import SyncRunner
from unigym.runtime.actions import NamedAction
from unigym.runtime.rewind_providers import RunnerLifecycleProvider
from unigym.terminations import TerminationGroup


def _ids(env_ids) -> list[int]:
    tolist = getattr(env_ids, "tolist", None)
    values = tolist() if callable(tolist) else list(env_ids)
    return [int(value) for value in values]


class _LifecycleSink:
    """Minimal reset/terminal-aware sink for the runner lifecycle boundary."""

    def __init__(self) -> None:
        self.pending: dict[int, Termination] = {}
        self.episode_ends: list[list[int]] = []
        self.reset_calls: list[list[int]] = []
        self.drain_calls = 0

    def on_step(self, t: int, bundle: dict) -> None:
        del t, bundle

    def on_episode_end(self, env_ids, bundle: dict) -> None:
        del bundle
        self.episode_ends.append(_ids(env_ids))

    def on_step_end(self, t: int, bundle: dict) -> None:
        del t, bundle

    def on_run_end(self) -> dict:
        return {}

    def on_reset(self, env_ids) -> None:
        self.reset_calls.append(_ids(env_ids))

    def drain_terminations(self) -> dict[int, Termination]:
        self.drain_calls += 1
        pending = self.pending
        self.pending = {}
        return pending


def _actions() -> tuple[NamedAction, ...]:
    return (
        NamedAction(
            ARTICULATION_NAME,
            torch.zeros(1, 1),
            ("shoulder",),
            "position",
        ),
    )


def _runner(env: FakeBackend, sink, *, auto_reset: bool = True) -> SyncRunner:
    return SyncRunner(
        cfg={"collect": {"steps": 4}},
        profile=None,
        adapter=ManualAdapter(),
        sampler=None,
        term_group=TerminationGroup(),
        env=env,
        sink=sink,
        auto_reset=auto_reset,
        control_dt=0.02,
    )


def _termination() -> Termination:
    return Termination(EpisodeOutcome.MANUAL, "custom.stop", TerminationSource.CUSTOM)


def test_terminate_reset_false_keeps_backend_terminal_until_explicit_reset():
    """A closed env is not stepped or reset until the driver resets it."""
    env = FakeBackend()
    sink = _LifecycleSink()
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions())
        runner.terminate([0], {0: _termination()}, reset=False)

        terminal = env.record()

        assert terminal["episode_id"].tolist() == [0]
        assert terminal["time_in_episode"].tolist() == [1.0]
        with pytest.raises(RuntimeError, match="closed"):
            runner.tick(actions_override=_actions())

        runner.reset([0])
        fresh = env.record()
        assert fresh["episode_id"].tolist() == [1]
        assert fresh["time_in_episode"].tolist() == [0.0]

        runner.tick(actions_override=_actions())
        assert env.record()["time_in_episode"].tolist() == [1.0]
    finally:
        env.close()


def test_terminate_reset_true_applies_the_backend_reset_before_the_next_step():
    """reset=True runs the backend reset chain in the same boundary."""
    env = FakeBackend()
    sink = _LifecycleSink()
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions())
        runner.terminate([0], {0: _termination()}, reset=True)

        assert env.record()["episode_id"].tolist() == [1]
        assert env.record()["time_in_episode"].tolist() == [0.0]

        runner.tick(actions_override=_actions())
        assert env.record()["episode_id"].tolist() == [1]
        assert env.record()["time_in_episode"].tolist() == [1.0]
    finally:
        env.close()


def test_sink_terminal_and_explicit_terminate_compose():
    """A sink terminal ends/resets once; a later explicit close does not reset."""
    env = FakeBackend()
    sink = _LifecycleSink()
    sink.pending = {0: _termination()}
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions())
        assert sink.drain_calls == 1
        assert sink.episode_ends == [[0]]
        assert sink.reset_calls == [[0]]
        assert env.record()["episode_id"].tolist() == [1]

        runner.tick(actions_override=_actions())
        runner.terminate([0], {0: _termination()}, reset=False)
        assert sink.episode_ends == [[0], [0]]
        assert sink.reset_calls == [[0]]
        assert env.record()["episode_id"].tolist() == [1]
        with pytest.raises(RuntimeError, match="closed"):
            runner.tick(actions_override=_actions())

        runner.reset([0])
        assert sink.reset_calls == [[0], [0]]
        assert env.record()["episode_id"].tolist() == [2]
    finally:
        env.close()


@pytest.mark.parametrize("auto_reset", [True, False])
def test_tick_due_notification_follows_the_auto_reset_policy(auto_reset):
    """auto_reset controls whether the due tick resets and notifies on_reset."""
    env = FakeBackend(due_rows=(0,))
    sink = _LifecycleSink()
    runner = _runner(env, sink, auto_reset=auto_reset)
    try:
        runner.tick(actions_override=_actions())
        if auto_reset:
            assert sink.reset_calls == [[0]]
            assert env.record()["episode_id"].tolist() == [1]
            assert env.record()["time_in_episode"].tolist() == [0.0]
        else:
            assert sink.reset_calls == []
            assert env.record()["episode_id"].tolist() == [0]
            assert env.record()["time_in_episode"].tolist() == [1.0]
    finally:
        env.close()


def test_rewind_restore_does_not_replay_the_reset_notification():
    """Restoring runner lifecycle state never re-runs a sink on_reset."""
    env = FakeBackend(due_rows=(0,))
    sink = _LifecycleSink()
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions())
        provider = RunnerLifecycleProvider(runner)
        snapshot = provider.rewind_state()
        assert sink.reset_calls == [[0]]

        runner.tick(actions_override=_actions())
        assert sink.reset_calls == [[0], [0]]

        provider.apply_rewind_state(snapshot)
    finally:
        env.close()

    assert sink.reset_calls == [[0], [0]]
