"""Explicit terminate/reset lifecycle over the fake backend.

The state machine: open -> terminate(reset=True) -> open, open ->
terminate(reset=False) -> closed -> explicit reset -> open.  Every invalid id,
fact or capability is refused before any mutation.
"""

from __future__ import annotations

import pytest
import torch

from tests._support.fakes.backend import ARTICULATION_NAME, FakeBackend
from tests._support.fakes.external_sink import (
    LifecycleProbeSink,
    ResetAwareLifecycleSink,
)
from unigym.adapters.manual import ManualAdapter
from unigym.core.episode_termination import EpisodeOutcome, Termination, TerminationSource
from unigym.evaluation import EvalViewStaleError
from unigym.runner.sync import SyncRunner
from unigym.runtime.actions import NamedAction
from unigym.terminations import TerminationGroup


class _CommandAdapter(ManualAdapter):
    """A self-commanding adapter with one fixed generated command."""

    generates_command = True

    def __init__(self) -> None:
        super().__init__()
        self.last_command = torch.tensor([[0.25]])


class _ResetViewProbeAdapter(ManualAdapter):
    """Records whether the boundary view is already stale at env reset time."""

    def __init__(self, sink: ResetAwareLifecycleSink) -> None:
        super().__init__()
        self.sink = sink
        self.stale_at_reset: list[bool] = []

    def on_reset(self, env_ids) -> None:
        view = self.sink.episode_end_views[-1]
        try:
            view.record()
        except EvalViewStaleError:
            self.stale_at_reset.append(True)
        else:
            self.stale_at_reset.append(False)


def _actions(num_envs: int = 1) -> tuple[NamedAction, ...]:
    return (
        NamedAction(
            ARTICULATION_NAME,
            torch.zeros(num_envs, 1),
            ("shoulder",),
            "position",
        ),
    )


def _termination(code: str = "custom.manual_stop", **kwargs) -> Termination:
    return Termination(EpisodeOutcome.MANUAL, code, TerminationSource.CUSTOM, **kwargs)


def _runner(
    env: FakeBackend,
    sink,
    *,
    adapter=None,
    auto_reset: bool = True,
    reset_mask_fn=None,
) -> SyncRunner:
    return SyncRunner(
        cfg={"collect": {"steps": 4}},
        profile=None,
        adapter=ManualAdapter() if adapter is None else adapter,
        sampler=None,
        term_group=TerminationGroup(),
        env=env,
        sink=sink,
        auto_reset=auto_reset,
        reset_mask_fn=reset_mask_fn,
        control_dt=0.02,
    )


def _force_due_zero(_bundle) -> torch.Tensor:
    return torch.tensor([0], dtype=torch.int64)


@pytest.mark.parametrize(
    ("env_ids", "error", "match"),
    [
        ([], ValueError, "at least one env"),
        ([1], ValueError, r"outside \[0, 1\)"),
        ([-1], ValueError, r"outside \[0, 1\)"),
        ([0, 0], ValueError, "duplicate env ids"),
        ([True], TypeError, "bool"),
        (["0"], TypeError, "not an integer"),
    ],
)
def test_terminate_rejects_invalid_env_ids(env_ids, error, match):
    """An invalid id set fails before the episode ends."""
    env = FakeBackend()
    sink = ResetAwareLifecycleSink()
    runner = _runner(env, sink)
    try:
        with pytest.raises(error, match=match):
            runner.terminate(env_ids, {}, reset=True)
        assert env.record()["episode_id"].tolist() == [0]
        assert sink.episode_end_calls == []
    finally:
        env.close()


@pytest.mark.parametrize(
    ("terminations", "error", "match"),
    [
        ({}, ValueError, "missing \\[0\\]"),
        ({0: "custom.x"}, TypeError, "expected a Termination"),
        (
            {0: Termination(EpisodeOutcome.FAILURE, "custom.x", TerminationSource.ENVIRONMENT)},
            ValueError,
            "TerminationSource.CUSTOM",
        ),
        (
            {0: Termination(EpisodeOutcome.FAILURE, "env.x", TerminationSource.CUSTOM)},
            ValueError,
            "custom.<name>",
        ),
        ({0: _termination(step_index=4)}, ValueError, "episode-local step"),
        ({0: _termination(time_s=1.0)}, ValueError, "episode-local time"),
    ],
)
def test_terminate_rejects_invalid_or_mismatched_terminations(terminations, error, match):
    """The termination mapping must exactly match the ids and carry custom facts."""
    env = FakeBackend()
    sink = ResetAwareLifecycleSink()
    runner = _runner(env, sink)
    try:
        with pytest.raises(error, match=match):
            runner.terminate([0], terminations, reset=True)
        assert env.record()["episode_id"].tolist() == [0]
        assert sink.episode_end_calls == []
    finally:
        env.close()


def test_terminate_rejects_a_fact_for_an_unrequested_env():
    """A termination key outside ``env_ids`` is an exact-mapping error."""
    env = FakeBackend(num_envs=2)
    sink = ResetAwareLifecycleSink()
    runner = _runner(env, sink)
    try:
        with pytest.raises(ValueError, match="unexpected \\[1\\]"):
            runner.terminate([0], {0: _termination(), 1: _termination()}, reset=True)
        assert env.record()["episode_id"].tolist() == [0, 0]
    finally:
        env.close()


def test_terminate_reset_false_requires_reset_aware_sink_before_mutation():
    """A non-reset-aware sink is refused without ending or closing the episode."""
    env = FakeBackend()
    sink = LifecycleProbeSink()
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions())
        before = env.record()["episode_id"].clone()

        with pytest.raises(RuntimeError, match="ResetAwareSink"):
            runner.terminate([0], {0: _termination()}, reset=False)

        assert torch.equal(env.record()["episode_id"], before)
        assert sink.episode_end_calls == []
        assert [call[0] for call in sink.order] == ["on_step", "on_step_end"]
        runner.tick(actions_override=_actions())
    finally:
        env.close()


def test_terminate_reset_false_closes_until_explicit_reset_reopens():
    """reset=False ends the episode without resetting; step refuses until reset."""
    env = FakeBackend()
    sink = ResetAwareLifecycleSink()
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions())
        result = runner.terminate([0], {0: _termination()}, reset=False)

        assert result.reset_env_ids.tolist() == [0]
        assert result.terminated.tolist() == [True]
        assert sink.episode_end_calls[0][0] == (0,)
        assert sink.reset_calls == []
        assert env.record()["episode_id"].tolist() == [0]
        assert runner.ticks_done == 1

        with pytest.raises(RuntimeError, match="closed"):
            runner.tick(actions_override=_actions())
        with pytest.raises(ValueError, match="already closed"):
            runner.terminate([0], {0: _termination()}, reset=False)

        reopened = runner.reset([0])
        assert reopened.reset_env_ids.tolist() == [0]
        assert sink.reset_calls == [[0]]
        assert env.record()["episode_id"].tolist() == [1]
        runner.tick(actions_override=_actions())
        assert runner.ticks_done == 2
    finally:
        env.close()


def test_terminate_reset_true_runs_the_reset_chain_and_reopens():
    """reset=True ends and immediately starts the next episode in one call."""
    env = FakeBackend()
    sink = ResetAwareLifecycleSink()
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions())
        result = runner.terminate([0], {0: _termination()}, reset=True)

        assert result.reset_env_ids.tolist() == [0]
        assert result.terminated.tolist() == [True]
        assert sink.episode_end_calls[0][0] == (0,)
        assert sink.reset_calls == [[0]]
        assert env.record()["episode_id"].tolist() == [1]
        assert runner.ticks_done == 1
        runner.tick(actions_override=_actions())
        assert runner.ticks_done == 2
    finally:
        env.close()


def test_reset_of_an_open_env_restarts_the_episode():
    """Reset keeps its documented mid-run behavior: reset and reopen, no end."""
    env = FakeBackend()
    sink = ResetAwareLifecycleSink()
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions())
        assert env.record()["time_in_episode"].tolist() == [1.0]

        result = runner.reset([0])

        assert result.reset_env_ids.tolist() == [0]
        assert result.terminated is None
        assert sink.episode_end_calls == []
        assert sink.reset_calls == [[0]]
        assert env.record()["episode_id"].tolist() == [1]
        assert env.record()["time_in_episode"].tolist() == [0.0]
        assert runner.ticks_done == 1
        runner.tick(actions_override=_actions())
        assert runner.ticks_done == 2
    finally:
        env.close()


def test_tick_view_is_closed_before_the_reset_chain_starts():
    """on_episode_end reads the live view; env reset only sees a closed view."""
    env = FakeBackend(due_rows=(0,))
    sink = ResetAwareLifecycleSink()
    adapter = _ResetViewProbeAdapter(sink)
    runner = _runner(env, sink, adapter=adapter)
    try:
        runner.tick(actions_override=_actions())
    finally:
        env.close()

    assert sink.episode_end_views[0] is not None
    assert adapter.stale_at_reset == [True]
    assert sink.step_end_bundles[0]["view"] is None


def test_terminate_view_is_closed_before_the_reset_chain_starts():
    """terminate(reset=True) also resets only after closing the boundary view."""
    env = FakeBackend()
    sink = ResetAwareLifecycleSink()
    adapter = _ResetViewProbeAdapter(sink)
    runner = _runner(env, sink, adapter=adapter)
    try:
        runner.tick(actions_override=_actions())
        runner.terminate([0], {0: _termination()}, reset=True)
    finally:
        env.close()

    assert sink.episode_end_views[-1] is not None
    assert adapter.stale_at_reset == [True]


def test_terminate_uses_the_adapter_generated_command_like_tick():
    """Terminate archives the same effective command as the last tick."""
    env = FakeBackend()
    sink = ResetAwareLifecycleSink()
    adapter = _CommandAdapter()
    runner = _runner(env, sink, adapter=adapter)
    try:
        runner.tick(actions_override=_actions())
        tick_cmd = sink.step_bundles[0]["cmd"]
        runner.terminate([0], {0: _termination()}, reset=False)
    finally:
        env.close()

    end_bundle = sink.episode_end_calls[0][1]
    final_step = sink.final_step_contexts[0]
    assert tick_cmd is adapter.last_command
    assert end_bundle["cmd"] is tick_cmd
    assert final_step.cmd is tick_cmd


def test_pre_start_reset_all_remains_legal():
    """The initial start may still reset every env before the first tick."""
    env = FakeBackend()
    sink = ResetAwareLifecycleSink()
    runner = _runner(env, sink)
    try:
        result = runner.reset()
        assert result.reset_env_ids.tolist() == [0]
        assert env.record()["episode_id"].tolist() == [1]
        assert sink.reset_calls == [[0]]
    finally:
        env.close()


def test_tick_auto_reset_notifies_the_sink_after_the_reset_chain():
    """on_episode_end -> reset chain -> on_reset -> on_step_end on a due tick."""
    env = FakeBackend(due_rows=(0,))
    sink = ResetAwareLifecycleSink()
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions())
    finally:
        env.close()

    names = [call[0] for call in sink.order]
    assert names.index("on_episode_end") < names.index("on_reset")
    assert names.index("on_reset") < names.index("on_step_end")
    assert sink.reset_calls == [[0]]
    assert env.record()["time_in_episode"].tolist() == [0.0]


def test_terminate_episode_end_facts_match_a_tick_end_at_the_same_step():
    """A standalone terminate builds the same end facts as the tick boundary."""
    tick_env = FakeBackend()
    tick_sink = ResetAwareLifecycleSink()
    tick_runner = _runner(tick_env, tick_sink, reset_mask_fn=_force_due_zero)
    terminate_env = FakeBackend()
    terminate_sink = ResetAwareLifecycleSink()
    terminate_runner = _runner(terminate_env, terminate_sink)
    try:
        tick_runner.tick(actions_override=_actions())
        terminate_runner.tick(actions_override=_actions())
        terminate_runner.terminate([0], {0: _termination()}, reset=False)
    finally:
        tick_env.close()
        terminate_env.close()

    tick_bundle = tick_sink.episode_end_calls[0][1]
    terminate_bundle = terminate_sink.episode_end_calls[0][1]
    assert tick_bundle["t"] == terminate_bundle["t"] == 0
    assert torch.equal(tick_bundle["step_index"], terminate_bundle["step_index"])
    assert torch.equal(tick_bundle["time_s"], terminate_bundle["time_s"])
    assert torch.equal(
        tick_bundle["record"]["time_in_episode"],
        terminate_bundle["record"]["time_in_episode"],
    )
    for bundle in (tick_bundle, terminate_bundle):
        assert "final_step" in bundle
        assert set(bundle["reset_facts"]) == {0}
