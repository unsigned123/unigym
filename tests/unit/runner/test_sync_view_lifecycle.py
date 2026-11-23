"""SyncRunner owns one boundary view per tick, valid until the reset chain."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from tests._support.fakes.backend import ARTICULATION_NAME, FakeBackend
from tests._support.fakes.external_sink import ViewProbeSink
from unigym.evaluation import EvalContext, EvalViewStaleError
from unigym.runtime.actions import NamedAction
from unigym.runner.sync import SyncRunner
from unigym.terminations import Termination, TerminationGroup


class _OneJointAdapter:
    """Minimal adapter: one joint, one action column, no history."""

    control_dt = 0.02
    generates_command = False

    def compute_obs(self, obs, env) -> torch.Tensor:
        return obs.robot(ARTICULATION_NAME).joint_pos

    def compute_obs_collect(self, frame, snapshot, env) -> torch.Tensor:
        del snapshot, env
        return frame

    def act(self, obs) -> torch.Tensor:
        del obs
        return torch.zeros(1, 1)

    def postprocess_action(self, raw: torch.Tensor) -> Sequence[NamedAction]:
        return (
            NamedAction(
                ARTICULATION_NAME,
                raw.reshape(1, 1),
                ("shoulder",),
                "position",
            ),
        )

    def on_reset(self, env_ids) -> None:
        del env_ids


class _RaisingResetAdapter(_OneJointAdapter):
    """Adapter whose reset-chain callback fails after env.reset_envs ran."""

    def on_reset(self, env_ids) -> None:
        del env_ids
        raise RuntimeError("adapter reset failure")


class _ProbeTermination(Termination):
    """All-false live term that records the boundary context it received."""

    def __init__(self) -> None:
        super().__init__(None, "probe")
        self.seen: EvalContext | None = None
        self.control_keys: tuple[tuple[str, str], ...] = ()

    def check(self, ctx: EvalContext) -> torch.Tensor:
        self.seen = ctx
        self.control_keys = tuple(ctx.env.controls)
        return torch.zeros(ctx.num_envs, dtype=torch.bool, device=ctx.env.device)


class _RaisingTermination(Termination):
    """Live term that fails before any sink callback can observe the bundle."""

    def __init__(self) -> None:
        super().__init__(None, "boom")
        self.seen: EvalContext | None = None
        self.calls = 0

    def check(self, ctx: EvalContext) -> torch.Tensor:
        self.calls += 1
        self.seen = ctx
        raise RuntimeError("termination failure")


def _runner(
    env: FakeBackend,
    sink,
    *,
    adapter=None,
    term_group: TerminationGroup | None = None,
    reset_mask_fn=None,
    reset_on_termination: bool = False,
) -> SyncRunner:
    return SyncRunner(
        cfg={"collect": {"steps": 4}},
        profile=None,
        adapter=_OneJointAdapter() if adapter is None else adapter,
        sampler=None,
        term_group=TerminationGroup() if term_group is None else term_group,
        env=env,
        sink=sink,
        reset_mask_fn=reset_mask_fn,
        reset_on_termination=reset_on_termination,
        control_dt=0.02,
    )


def _force_due_to_env_zero(_bundle) -> torch.Tensor:
    return torch.tensor([0], dtype=torch.int64)


def test_one_view_reaches_on_step_and_on_episode_end():
    """The termination view is the same object handed to both sink callbacks."""
    env = FakeBackend()
    term = _ProbeTermination()
    sink = ViewProbeSink()
    runner = _runner(
        env,
        sink,
        term_group=TerminationGroup([term]),
        reset_mask_fn=_force_due_to_env_zero,
    )
    try:
        runner.tick()
    finally:
        env.close()

    assert term.seen is not None
    assert term.control_keys == tuple(env.controls)
    assert len(sink.step_bundles) == 1
    assert len(sink.episode_end_bundles) == 1
    step_view = sink.step_views[0]
    assert step_view is term.seen.env
    assert sink.episode_end_views[0] is step_view
    assert step_view is not None
    assert sink.step_view_scopes == [(None, env.num_envs)]
    assert sink.episode_end_view_scopes == [(None, env.num_envs)]


def test_view_is_stale_after_the_tick_and_none_at_on_step_end():
    """The closed view is stale while on_step_end keeps the pre-reset facts."""
    env = FakeBackend()
    sink = ViewProbeSink()
    runner = _runner(env, sink, reset_mask_fn=_force_due_to_env_zero)
    try:
        runner.tick()
    finally:
        env.close()

    step_bundle = sink.step_bundles[0]
    end_bundle = sink.step_end_bundles[0]
    step_view = sink.step_views[0]
    assert step_view is not None
    with pytest.raises(EvalViewStaleError):
        step_view.record()
    assert "view" in end_bundle
    assert end_bundle["view"] is None
    assert sink.step_end_views == [None]
    # on_step_end still carries the pre-reset record/obs facts.
    assert end_bundle["record"] is step_bundle["record"]
    assert end_bundle["obs"] is step_bundle["obs"]
    assert step_bundle["record"]["time_in_episode"].tolist() == [1.0]


def test_no_due_tick_still_closes_the_view_before_on_step_end():
    """A tick that resets nothing closes the view before on_step_end."""
    env = FakeBackend()
    sink = ViewProbeSink()
    runner = _runner(env, sink)
    try:
        runner.tick()
    finally:
        env.close()

    assert sink.episode_end_bundles == []
    assert len(sink.step_end_bundles) == 1
    assert sink.step_end_bundles[0]["view"] is None
    step_view = sink.step_views[0]
    assert step_view is not None
    with pytest.raises(EvalViewStaleError):
        step_view.record()


def test_termination_exception_closes_the_view_and_skips_sink_callbacks():
    """A failing termination check closes the view and calls no sink callback."""
    env = FakeBackend()
    term = _RaisingTermination()
    sink = ViewProbeSink()
    runner = _runner(env, sink, term_group=TerminationGroup([term]))
    try:
        with pytest.raises(RuntimeError, match="termination failure"):
            runner.tick()
    finally:
        env.close()

    assert term.calls == 1
    assert sink.step_bundles == []
    assert sink.episode_end_bundles == []
    assert sink.step_end_bundles == []
    assert term.seen is not None
    with pytest.raises(EvalViewStaleError):
        term.seen.env.record()


def test_on_step_exception_closes_the_view():
    """A sink on_step failure still closes the boundary view."""
    env = FakeBackend()
    sink = ViewProbeSink(raise_at="on_step")
    runner = _runner(env, sink, reset_mask_fn=_force_due_to_env_zero)
    try:
        with pytest.raises(RuntimeError, match="view probe on_step failure"):
            runner.tick()
    finally:
        env.close()

    assert len(sink.step_bundles) == 1
    assert sink.episode_end_bundles == []
    assert sink.step_end_bundles == []
    step_view = sink.step_views[0]
    assert step_view is not None
    with pytest.raises(EvalViewStaleError):
        step_view.record()


def test_on_episode_end_exception_closes_the_view_before_the_reset_chain():
    """A failing on_episode_end closes the view and skips reset and on_step_end."""
    env = FakeBackend()
    sink = ViewProbeSink(raise_at="on_episode_end")
    runner = _runner(env, sink, reset_mask_fn=_force_due_to_env_zero)
    try:
        with pytest.raises(RuntimeError, match="view probe on_episode_end failure"):
            runner.tick()
        assert env.record()["episode_id"].tolist() == [0]
    finally:
        env.close()

    assert len(sink.step_bundles) == 1
    assert len(sink.episode_end_bundles) == 1
    assert sink.step_end_bundles == []
    episode_view = sink.episode_end_views[0]
    assert episode_view is not None
    with pytest.raises(EvalViewStaleError):
        episode_view.record()


def test_reset_mask_exception_closes_the_view():
    """A failing reset-mask calculation closes the view before anything else."""

    def _raising_mask(_bundle):
        raise RuntimeError("reset mask failure")

    env = FakeBackend()
    sink = ViewProbeSink()
    runner = _runner(env, sink, reset_mask_fn=_raising_mask)
    try:
        with pytest.raises(RuntimeError, match="reset mask failure"):
            runner.tick()
    finally:
        env.close()

    assert len(sink.step_bundles) == 1
    assert sink.episode_end_bundles == []
    assert sink.step_end_bundles == []
    step_view = sink.step_views[0]
    assert step_view is not None
    with pytest.raises(EvalViewStaleError):
        step_view.record()


def test_reset_chain_exception_closes_the_view():
    """A reset-chain failure happens after the view is already closed."""
    env = FakeBackend()
    sink = ViewProbeSink()
    runner = _runner(
        env,
        sink,
        adapter=_RaisingResetAdapter(),
        reset_mask_fn=_force_due_to_env_zero,
    )
    try:
        with pytest.raises(RuntimeError, match="adapter reset failure"):
            runner.tick()
        assert env.record()["episode_id"].tolist() == [1]
    finally:
        env.close()

    assert len(sink.step_bundles) == 1
    assert len(sink.episode_end_bundles) == 1
    assert sink.step_end_bundles == []
    episode_view = sink.episode_end_views[0]
    assert episode_view is not None
    with pytest.raises(EvalViewStaleError):
        episode_view.record()


def test_rewind_boundary_closes_the_view_and_skips_end_callbacks():
    """A rewind closes first, then adopts the restored tick and ends the tick."""
    env = FakeBackend()
    sink = ViewProbeSink(rewind_next=7)
    runner = _runner(
        env,
        sink,
        term_group=TerminationGroup([_ProbeTermination()]),
        reset_on_termination=True,
    )
    try:
        result = runner.tick()
        assert env.record()["episode_id"].tolist() == [0]
    finally:
        env.close()

    assert runner.ticks_done == 7
    assert sink.rewinds == [{}]
    assert len(sink.step_bundles) == 1
    assert sink.episode_end_bundles == []
    assert sink.step_end_bundles == []
    assert result.reset_env_ids.numel() == 0
    step_view = sink.step_views[0]
    assert step_view is not None
    with pytest.raises(EvalViewStaleError):
        step_view.record()
