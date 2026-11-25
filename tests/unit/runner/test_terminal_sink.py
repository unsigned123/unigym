"""Runner sink-terminal drain: due union, validation, stamping and rewind."""

from __future__ import annotations

import pytest
import torch

from tests._support.fakes.backend import ARTICULATION_NAME, FakeBackend
from tests._support.fakes.command_sampler import RecordingSampler
from tests._support.fakes.external_sink import ScriptedTerminalSink
from unigym.adapters.manual import ManualAdapter
from unigym.core.episode_termination import EpisodeOutcome, Termination, TerminationSource
from unigym.runner.sync import SyncRunner
from unigym.runtime.actions import NamedAction
from unigym.terminations import JointLimitTermination, TerminationGroup


def _actions(num_envs: int, target: float = 0.0) -> tuple[NamedAction, ...]:
    """One joint-addressed action batch; env 0 gets *target*."""
    values = torch.zeros(num_envs, 1)
    values[0] = target
    return (NamedAction(ARTICULATION_NAME, values, ("shoulder",), "position"),)


def _termination(code: str = "custom.reach", **kwargs) -> Termination:
    """A sink-owned terminal fact (custom provenance and namespace)."""
    return Termination(EpisodeOutcome.FAILURE, code, TerminationSource.CUSTOM, **kwargs)


def _runner(
    env,
    sink,
    *,
    term_group=None,
    auto_reset: bool = True,
    sampler=None,
    reset_mask_fn=None,
) -> SyncRunner:
    """Build a SyncRunner over a manual driver and the fake backend."""
    return SyncRunner(
        cfg={"collect": {"steps": 4}},
        profile=None,
        adapter=ManualAdapter(),
        sampler=sampler,
        term_group=TerminationGroup() if term_group is None else term_group,
        env=env,
        sink=sink,
        auto_reset=auto_reset,
        reset_mask_fn=reset_mask_fn,
        control_dt=0.02,
    )


def test_sink_terminal_env_joins_due_and_triggers_episode_end():
    """A sink terminal env is due, reset and reported like a clock episode end."""
    env = FakeBackend(num_envs=2)
    sink = ScriptedTerminalSink()
    sink.pending = {1: _termination()}
    runner = _runner(env, sink)
    try:
        result = runner.tick(actions_override=_actions(2))
    finally:
        env.close()

    assert sink.drain_calls == 1
    assert sink.pending == {}
    assert len(sink.episode_end_calls) == 1
    env_ids, _bundle = sink.episode_end_calls[0]
    assert env_ids.tolist() == [1]
    assert result.reset_env_ids.tolist() == [1]
    # terminated stays the core term-group mask; the sink did not terminate.
    assert result.terminated.tolist() == [False, False]


def test_sink_terminal_is_stamped_and_merged_into_terminations():
    """The broadcast bundle carries the sinking fact with the episode-local stamp."""
    env = FakeBackend()
    sink = ScriptedTerminalSink()
    sink.pending = {0: _termination()}
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions(1))
    finally:
        env.close()

    bundle = sink.step_bundles[0]
    assert set(bundle["terminations"]) == {0}
    primary = bundle["terminations"][0].primary
    assert primary.code == "custom.reach"
    assert primary.source is TerminationSource.CUSTOM
    assert primary.step_index == int(bundle["step_index"][0])
    assert primary.time_s == pytest.approx(float(bundle["time_s"][0]))


def test_provided_sink_time_accepts_the_runner_float32_value():
    """A sink echoing the tick's canonical float32 time is accepted, not rejected."""
    env = FakeBackend()
    sink = ScriptedTerminalSink()
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions(1))
        runner.tick(actions_override=_actions(1))
        # The runner's bundle time_s is float32; at episode step 3 it is not
        # exactly the float64 literal 0.06.
        canonical = float(torch.tensor([3], dtype=torch.int64).to(torch.float32) * 0.02)
        assert canonical != 0.06
        sink.pending = {0: _termination(step_index=3, time_s=canonical)}
        runner.tick(actions_override=_actions(1))
    finally:
        env.close()

    primary = sink.episode_end_calls[0][1]["terminations"][0].primary
    assert primary.step_index == 3
    assert primary.time_s == canonical


def test_core_and_sink_terminations_compose_with_core_precedence():
    """Core and sink facts for one env merge; core precedence picks the primary."""
    env = FakeBackend(num_envs=2)
    env.articulations[ARTICULATION_NAME].joint_pos_limits = env.policy_joint_pos_limits
    core = JointLimitTermination({"articulation": ARTICULATION_NAME}, name="out_of_limits")
    sink = ScriptedTerminalSink()
    sink.pending = {0: _termination("custom.grasp_lost")}
    runner = _runner(env, sink, term_group=TerminationGroup([core]))
    try:
        result = runner.tick(actions_override=_actions(2, target=5.0))
    finally:
        env.close()

    bundle = sink.episode_end_calls[0][1]
    composed = bundle["terminations"][0]
    assert [term.code for term in composed.all] == [
        "env.out_of_limits",
        "custom.grasp_lost",
    ]
    assert composed.primary.code == "env.out_of_limits"
    assert result.reset_env_ids.tolist() == [0]
    assert result.terminated.tolist() == [True, False]


def test_sampler_on_episode_end_receives_sink_terminal_due_env():
    """Sampler feedback sees the sink-terminal env on the same boundary."""
    env = FakeBackend(num_envs=2)
    sampler = RecordingSampler(dim=1, device="cpu", num_envs=2)
    sink = ScriptedTerminalSink()
    sink.pending = {1: _termination()}
    runner = _runner(env, sink, sampler=sampler)
    try:
        runner.tick(actions_override=_actions(2))
    finally:
        env.close()

    assert len(sampler.episode_calls) == 1
    env_ids, results = sampler.episode_calls[0]
    assert env_ids == [1]
    assert dict(results) == {}


def test_reset_chain_ran_for_the_sink_terminal_env():
    """The sink-terminal env goes through env.reset_envs, not only bookkeeping."""
    env = FakeBackend(num_envs=2)
    sink = ScriptedTerminalSink()
    sink.pending = {1: _termination()}
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions(2))
        assert env.record()["episode_id"].tolist() == [0, 1]
    finally:
        env.close()


def test_sink_stamp_is_episode_local_after_a_prior_reset():
    """After a reset the stamp is the new episode step, never the global tick."""
    env = FakeBackend()
    sink = ScriptedTerminalSink()
    runner = _runner(env, sink)
    try:
        sink.pending = {0: _termination()}
        runner.tick(actions_override=_actions(1))
        sink.pending = {0: _termination()}
        runner.tick(actions_override=_actions(1))
    finally:
        env.close()

    primary = sink.episode_end_calls[1][1]["terminations"][0].primary
    assert primary.step_index == 1
    assert runner.ticks_done == 2
    assert primary.step_index != runner.ticks_done
    assert primary.time_s == pytest.approx(0.02)


def test_backend_due_and_sink_terminal_merge_on_one_env():
    """A backend-owned due env and a sink terminal compose one fact set."""
    env = FakeBackend(num_envs=2, due_rows=(0,))
    sink = ScriptedTerminalSink()
    sink.pending = {0: _termination("custom.grasp_lost")}
    runner = _runner(env, sink)
    try:
        result = runner.tick(actions_override=_actions(2))
    finally:
        env.close()

    assert result.reset_env_ids.tolist() == [0]
    composed = sink.episode_end_calls[0][1]["terminations"][0]
    codes = [term.code for term in composed.all]
    assert "runtime.episode_horizon" in codes
    assert "custom.grasp_lost" in codes
    # Core precedence still chooses: FAILURE outranks DEADLINE.
    assert composed.primary.code == "custom.grasp_lost"


def test_reset_mask_fn_observes_merged_sink_terminations():
    """The reset policy sees the sink facts on the bundle before due is final."""
    env = FakeBackend(num_envs=2)
    sink = ScriptedTerminalSink()
    sink.pending = {0: _termination()}
    observed: dict[str, object] = {}

    def _mask(bundle):
        observed["terminations"] = bundle["terminations"]
        return torch.tensor([1], dtype=torch.int64)

    runner = _runner(env, sink, reset_mask_fn=_mask)
    try:
        result = runner.tick(actions_override=_actions(2))
    finally:
        env.close()

    merged = observed["terminations"]
    assert set(merged) == {0}
    assert merged[0].primary.code == "custom.reach"
    assert result.reset_env_ids.tolist() == [0, 1]


@pytest.mark.parametrize(
    ("pending", "error", "match"),
    [
        ({1: _termination()}, ValueError, r"env id 1 is outside \[0, 1\)"),
        ({-1: _termination()}, ValueError, r"env id -1 is outside \[0, 1\)"),
        ({True: _termination()}, TypeError, "is a bool"),
        ({0: "custom.reach"}, TypeError, "expected a Termination"),
        ({0: _termination(step_index=99)}, ValueError, "episode-local step"),
        ({0: _termination(time_s=99.0)}, ValueError, "episode-local time"),
        (
            {0: Termination(EpisodeOutcome.FAILURE, "custom.reach", TerminationSource.ENVIRONMENT)},
            ValueError,
            "must use TerminationSource.CUSTOM",
        ),
        (
            {0: Termination(EpisodeOutcome.FAILURE, "env.force_limit", TerminationSource.CUSTOM)},
            ValueError,
            "must be namespaced 'custom.<name>'",
        ),
        (None, TypeError, "expected a Mapping"),
        ([0], TypeError, "expected a Mapping"),
    ],
    ids=[
        "high-env-id",
        "negative-env-id",
        "bool-env-id",
        "non-termination",
        "global-step",
        "global-time",
        "non-custom-source",
        "non-custom-code",
        "none-drain",
        "list-drain",
    ],
)
def test_invalid_sink_drain_fails_before_episode_end(pending, error, match):
    """A malformed drain fails with a teaching error and no episode end."""
    env = FakeBackend()
    sink = ScriptedTerminalSink()
    sink.pending = pending
    runner = _runner(env, sink)
    try:
        with pytest.raises(error, match=match):
            runner.tick(actions_override=_actions(1))
    finally:
        env.close()

    assert sink.drain_calls == 1
    assert sink.episode_end_calls == []


def test_sink_terminal_without_auto_reset_raises_before_due_mutation():
    """auto_reset=False plus a non-empty drain is an inconsistent terminal end."""
    env = FakeBackend()
    sink = ScriptedTerminalSink()
    sink.pending = {0: _termination()}
    runner = _runner(env, sink, auto_reset=False)
    try:
        with pytest.raises(RuntimeError, match="auto_reset=False") as info:
            runner.tick(actions_override=_actions(1))
        assert env.record()["episode_id"].tolist() == [0]
    finally:
        env.close()

    assert "terminal env ids [0]" in str(info.value)
    assert sink.drain_calls == 1
    assert sink.episode_end_calls == []


def test_rewind_boundary_skips_drain_and_episode_end():
    """An abandoned branch never drains, never ends the episode and never resets."""
    env = FakeBackend()
    sink = ScriptedTerminalSink(rewind_next=5)
    sink.pending = {0: _termination()}
    runner = _runner(env, sink)
    try:
        result = runner.tick(actions_override=_actions(1))
        assert env.record()["episode_id"].tolist() == [0]
    finally:
        env.close()

    assert sink.rewinds == [{}]
    assert sink.drain_calls == 0
    assert set(sink.pending) == {0}
    assert sink.episode_end_calls == []
    assert result.reset_env_ids.numel() == 0
    assert runner.ticks_done == 5
