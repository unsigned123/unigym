"""SyncRunner builds step/final value contexts at the episode boundary."""

from __future__ import annotations

import pytest
import torch

from tests._support.fakes.backend import ARTICULATION_NAME, JOINT_NAME, FakeBackend
from tests._support.fakes.external_sink import ValuePublishingSink, ViewProbeSink
from unigym.adapters.manual import ManualAdapter
from unigym.evaluation import EnvStepContext, EvalViewStaleError
from unigym.runner.sinks import FanOutSink
from unigym.runner.sync import SyncRunner
from unigym.runtime.actions import NamedAction
from unigym.terminations import TerminationGroup
from unigym.values import StepValues, ValueStore

ACTIONS = (NamedAction.zeros(1, ARTICULATION_NAME, (JOINT_NAME,), "position"),)


def _runner(env: FakeBackend, sink, *, values: ValueStore | None) -> SyncRunner:
    return SyncRunner(
        cfg={"collect": {"steps": 4}},
        profile=None,
        adapter=ManualAdapter(num_envs=1),
        sampler=None,
        term_group=TerminationGroup(),
        env=env,
        sink=sink,
        control_dt=0.02,
        values=values,
    )


def test_step_and_final_contexts_share_the_tick_view_and_publications():
    """Step values are pre-on_step; final values include the sink publication."""
    store = ValueStore(num_envs=1, device="cpu")
    publishing = ValuePublishingSink(store.publisher("sink"), num_envs=1)
    probe = ViewProbeSink()
    env = FakeBackend(due_rows=(0,))
    runner = _runner(env, FanOutSink(publishing, probe), values=store)
    try:
        runner.tick(actions_override=ACTIONS)
    finally:
        env.close()

    step_ctx = probe.step_contexts[0]
    final_ctx = probe.final_step_contexts[0]
    assert isinstance(step_ctx, EnvStepContext)
    assert isinstance(final_ctx, EnvStepContext)
    assert step_ctx.view is probe.step_views[0]
    assert final_ctx.view is probe.episode_end_views[0]
    assert publishing.value not in step_ctx.values
    assert publishing.value in final_ctx.values
    assert float(final_ctx.values[publishing.value][0, 0]) == 1.0

    stored = store.get(publishing.value)
    assert stored is not None
    assert torch.equal(stored, torch.tensor([[1.0]]))


def test_context_keys_are_none_after_the_tick_and_at_on_step_end():
    """The closed boundary drops both context keys before on_step_end."""
    store = ValueStore(num_envs=1, device="cpu")
    publishing = ValuePublishingSink(store.publisher("sink"), num_envs=1)
    probe = ViewProbeSink()
    env = FakeBackend(due_rows=(0,))
    runner = _runner(env, FanOutSink(publishing, probe), values=store)
    try:
        runner.tick(actions_override=ACTIONS)
    finally:
        env.close()

    assert probe.step_end_contexts == [None]
    end_bundle = probe.step_end_bundles[0]
    assert end_bundle["step"] is None
    assert end_bundle["final_step"] is None
    step_view = probe.step_views[0]
    assert step_view is not None
    with pytest.raises(EvalViewStaleError):
        step_view.record()


def test_no_due_tick_has_no_final_step_and_closes_cleanly():
    """A tick without episode ends never builds a final context."""
    store = ValueStore(num_envs=1, device="cpu")
    publishing = ValuePublishingSink(store.publisher("sink"), num_envs=1)
    probe = ViewProbeSink()
    env = FakeBackend()
    runner = _runner(env, FanOutSink(publishing, probe), values=store)
    try:
        runner.tick(actions_override=ACTIONS)
    finally:
        env.close()

    assert probe.episode_end_bundles == []
    assert probe.final_step_contexts == []
    end_bundle = probe.step_end_bundles[0]
    assert "final_step" not in end_bundle
    assert end_bundle["step"] is None
    step_view = probe.step_views[0]
    assert step_view is not None
    with pytest.raises(EvalViewStaleError):
        step_view.record()


def test_values_none_yields_an_empty_snapshot_and_creates_no_store():
    """Direct construction without a store stays empty and instance-local."""
    probe = ViewProbeSink()
    env = FakeBackend()
    runner = _runner(env, probe, values=None)
    try:
        runner.tick(actions_override=ACTIONS)
    finally:
        env.close()

    assert runner.values is None
    step_ctx = probe.step_contexts[0]
    assert isinstance(step_ctx, EnvStepContext)
    assert isinstance(step_ctx.values, StepValues)
    assert len(step_ctx.values) == 0


def test_rewind_boundary_has_no_final_context():
    """A rewind abandons the branch before any final context exists."""
    store = ValueStore(num_envs=1, device="cpu")
    probe = ViewProbeSink(rewind_next=2)
    env = FakeBackend(due_rows=(0,))
    runner = _runner(env, probe, values=store)
    try:
        result = runner.tick(actions_override=ACTIONS)
    finally:
        env.close()

    assert result.reset_env_ids.numel() == 0
    assert len(probe.step_contexts) == 1
    assert probe.final_step_contexts == []
    assert probe.step_end_bundles == []
    assert "final_step" not in probe.step_bundles[0]
    assert probe.step_bundles[0]["step"] is None


def test_restored_store_state_is_visible_in_the_next_step_context():
    """A rewind restore changes the next step context's pre-on_step values."""
    store = ValueStore(num_envs=1, device="cpu")
    publishing = ValuePublishingSink(store.publisher("sink"), num_envs=1)
    probe = ViewProbeSink()
    env = FakeBackend()
    runner = _runner(env, FanOutSink(publishing, probe), values=store)
    try:
        runner.tick(actions_override=ACTIONS)
        checkpoint = store.rewind_state()
        runner.tick(actions_override=ACTIONS)

        overwritten = store.get(publishing.value)
        assert overwritten is not None
        assert float(overwritten[0, 0]) == 2.0

        store.apply_rewind_state(checkpoint)
        runner.tick(actions_override=ACTIONS)
    finally:
        env.close()

    step_ctx = probe.step_contexts[-1]
    assert isinstance(step_ctx, EnvStepContext)
    restored = step_ctx.values[publishing.value]
    assert float(restored[0, 0]) == 1.0
    current = store.get(publishing.value)
    assert current is not None
    assert float(current[0, 0]) == 3.0
