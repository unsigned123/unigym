"""Runner termination boundary: one live context, one detail pass, no term_state."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tests._support.fakes.backend import ARTICULATION_NAME, FakeBackend
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


class _RecordTermination(Termination):
    """Custom live term that reads the boundary record."""

    def __init__(self) -> None:
        super().__init__(None, "record_probe")
        self.calls = 0
        self.seen: EvalContext | None = None

    def check(self, ctx: EvalContext) -> torch.Tensor:
        self.calls += 1
        self.seen = ctx
        joint_pos = ctx.env.record()[f"articulation/{ARTICULATION_NAME}/joint_pos"]
        return joint_pos[:, 0] > 0.5


class _AlwaysTermination(Termination):
    """Live term that fires for every row in the vector context."""

    def __init__(self) -> None:
        super().__init__(None, "always")
        self.calls = 0

    def check(self, ctx: EvalContext) -> torch.Tensor:
        self.calls += 1
        return torch.ones(ctx.num_envs, dtype=torch.bool, device=ctx.env.device)


class _CaptureSink:
    """Capture the bundle and prove the context is already closed."""

    def __init__(self, term: _RecordTermination) -> None:
        self.term = term
        self.bundle: dict | None = None
        self.context_closed = False

    def on_step(self, t: int, bundle: dict) -> None:
        del t
        self.bundle = bundle
        ctx = self.term.seen
        assert ctx is not None
        try:
            ctx.env.record()
        except EvalViewStaleError:
            self.context_closed = True

    def on_episode_end(self, env_ids, bundle: dict) -> None:
        del env_ids, bundle

    def on_step_end(self, t: int, bundle: dict) -> None:
        del t, bundle

    def on_run_end(self) -> dict:
        return {}


def test_runner_evaluates_one_live_detail_pass_and_closes_before_sink():
    """A non-empty group derives all facts from one detail pass and no state object."""
    env = FakeBackend()
    record_term = _RecordTermination()
    always_term = _AlwaysTermination()
    sink = _CaptureSink(record_term)
    runner = SyncRunner(
        cfg={"collect": {"steps": 1}},
        profile=None,
        adapter=_OneJointAdapter(),
        sampler=None,
        term_group=TerminationGroup([record_term, always_term]),
        env=env,
        sink=sink,
        control_dt=0.02,
    )
    try:
        result = runner.tick()
    finally:
        env.close()

    assert record_term.calls == 1
    assert always_term.calls == 1
    assert sink.bundle is not None
    assert "term_state" not in sink.bundle
    assert sink.context_closed is True
    assert isinstance(result.terminated, torch.Tensor)
    assert tuple(result.terminated.shape) == (env.num_envs,)
    assert result.terminated.dtype == torch.bool
    assert result.terminated.tolist() == [True]
    assert sink.bundle["alive"].tolist() == [False]
    assert sink.bundle["term_metrics"] == {
        "terminated": 1.0,
        "fallen": 1.0,
        "term_record_probe": 0.0,
        "term_always": 1.0,
    }
    terminations = sink.bundle["terminations"]
    assert set(terminations) == {0}
    assert [term.code for term in terminations[0].all] == ["env.always"]
