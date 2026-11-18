"""Command/prev_action observation timing is post-sampler, not pre-step."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tests._support.fakes.backend import ARTICULATION_NAME, FakeBackend
from unigym.adapters.base import Adapter
from unigym.runner.sync import SyncRunner
from unigym.runtime.actions import NamedAction
from unigym.terminations import TerminationGroup


class _OneJointAdapter(Adapter):
    """Minimal adapter: one joint, one action column, no history."""

    obs_dim = 1

    @property
    def control_dt(self) -> float:
        """50 Hz policy period."""
        return 0.02

    def compute_obs(self, obs, env) -> torch.Tensor:
        """Read the typed robot view."""
        return obs.robot(ARTICULATION_NAME).joint_pos

    def act(self, obs) -> torch.Tensor:
        """Emit one zero action column."""
        return torch.zeros(1, 1)

    def postprocess_action(self, raw: torch.Tensor) -> Sequence[NamedAction]:
        """Wrap the raw action as the owned joint target."""
        return (
            NamedAction(
                ARTICULATION_NAME,
                raw.reshape(1, 1),
                ("shoulder",),
                "position",
            ),
        )


class _AdvancingSampler:
    """A command clock whose update is visible in the tick's observation."""

    def __init__(self) -> None:
        self.value = torch.zeros(1, 1)

    def reset(self, env_ids) -> torch.Tensor:
        """Reset the command to zero for the selected rows."""
        del env_ids
        self.value = torch.zeros(1, 1)
        return self.value

    def update(self, dt: float, cmd: torch.Tensor) -> torch.Tensor:
        """Advance the command (the post-update value is the observable one)."""
        del dt
        self.value = cmd + 1.0
        return self.value


class _CaptureSink:
    """Capture the tick bundle and do no recording."""

    def __init__(self) -> None:
        self.bundle: dict | None = None

    def on_step(self, t: int, bundle: dict) -> None:
        self.bundle = bundle

    def on_episode_end(self, env_ids, bundle: dict) -> None:
        """No episode bookkeeping."""

    def on_step_end(self, t: int, bundle: dict) -> None:
        """No metrics bookkeeping."""

    def on_run_end(self) -> dict:
        return {}


def test_command_observation_carries_the_post_sampler_value() -> None:
    env = FakeBackend()
    sampler = _AdvancingSampler()
    sink = _CaptureSink()
    runner = SyncRunner(
        cfg={"collect": {"steps": 1}},
        profile=None,
        adapter=_OneJointAdapter(),
        sampler=sampler,
        term_group=TerminationGroup(),
        env=env,
        sink=sink,
        control_dt=0.02,
    )
    try:
        result = runner.tick()
        assert sink.bundle is not None
        snapshot = sink.bundle["obs"]
        assert torch.equal(snapshot.command.value, torch.ones(1, 1))
        assert torch.equal(env.obs.command.value, torch.ones(1, 1))
        assert not torch.equal(result.obs.command.value, torch.zeros(1, 1))
    finally:
        env.close()


def test_prev_action_observation_carries_the_adapter_raw_action() -> None:
    env = FakeBackend()
    sink = _CaptureSink()
    runner = SyncRunner(
        cfg={"collect": {"steps": 1}},
        profile=None,
        adapter=_OneJointAdapter(),
        sampler=None,
        term_group=TerminationGroup(),
        env=env,
        sink=sink,
    )
    try:
        runner.tick()
        assert sink.bundle is not None
        assert torch.equal(sink.bundle["obs"].prev_action.value, torch.zeros(1, 1))
        assert "prev_action" not in sink.bundle["record"]
        assert "command" not in sink.bundle["record"]
    finally:
        env.close()
