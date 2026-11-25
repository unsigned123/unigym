"""Sink-published values: polling is two-tick, boundary subscription is one-tick."""

from __future__ import annotations

import torch

from tests._support.fakes.backend import ARTICULATION_NAME, JOINT_NAME, FakeBackend
from tests._support.fakes.command_sampler import ClosedLoopValueSampler
from tests._support.fakes.external_sink import ValuePublishingSink
from unigym.adapters.manual import ManualAdapter
from unigym.commands.base import CommandSampler
from unigym.runner.sync import SyncRunner
from unigym.runtime.actions import NamedAction
from unigym.terminations import TerminationGroup
from unigym.values import ValueStore, ValueView


def _drive(ticks: int) -> tuple[ValueStore, ClosedLoopValueSampler, ValuePublishingSink]:
    """Run *ticks* real SyncRunner ticks over the fake backend and one store."""
    store = ValueStore(num_envs=1, device="cpu")
    sampler = ClosedLoopValueSampler(dim=1, device="cpu", num_envs=1)
    sampler.bind_values(store)
    sink = ValuePublishingSink(store.publisher("sink"), num_envs=1, value=sampler.value)
    env = FakeBackend()
    runner = SyncRunner(
        cfg={"collect": {"steps": ticks}},
        profile=None,
        adapter=ManualAdapter(num_envs=1),
        sampler=sampler,
        term_group=TerminationGroup(),
        env=env,
        sink=sink,
        control_dt=0.02,
        values=store,
    )
    actions = (NamedAction.zeros(1, ARTICULATION_NAME, (JOINT_NAME,), "position"),)
    try:
        for _ in range(ticks):
            runner.tick(actions_override=actions)
    finally:
        env.close()
    return store, sampler, sink


def _values(values: list[torch.Tensor | None]) -> list[float | None]:
    return [None if value is None else float(value[0, 0]) for value in values]


def test_sink_publishes_once_per_tick_and_sampler_reads_the_next_update() -> None:
    """A value published at ``on_step(t)`` is visible to ``update(t+1)`` only."""
    store, sampler, sink = _drive(3)

    assert sink.published == [(0, 1.0), (1, 2.0), (2, 3.0)]
    assert _values(sampler.observations) == [None, 1.0, 2.0]
    snapshot = store.snapshot()
    assert list(snapshot) == [sink.value]
    assert float(snapshot[sink.value][0, 0]) == 3.0


def test_value_published_at_tick_t_acts_at_tick_t_plus_two() -> None:
    """The two-tick latency is real: publish t -> update t+1 -> env.step t+2."""
    _, sampler, sink = _drive(3)

    # update(t) receives the command that tick's env.step used.
    assert _values(sampler.used_commands) == [0.0, 0.0, 1.0]
    assert sink.published[0] == (0, 1.0)
    assert sampler.observations[1] is not None
    assert float(sampler.observations[1][0, 0]) == 1.0
    assert _values(sampler.used_commands)[2] == 1.0


class _BoundarySubscribingSampler(CommandSampler):
    """Sampler double that consumes a published value in its own callback.

    It returns the owned ``self.cmd`` buffer from ``update`` (the runner holds
    that alias for the next ``env.step``) and, unless ``rebind`` is set,
    mutates it in place from the callback.  ``rebind`` replaces the buffer
    instead, proving the alias requirement.  No rewind capability.
    """

    def __init__(
        self,
        dim: int,
        device: str,
        num_envs: int,
        *,
        value: str = "acme/probe",
        rebind: bool = False,
    ) -> None:
        super().__init__(dim, device, num_envs)
        self.value = value
        self.rebind = rebind
        self.used_commands: list[torch.Tensor | None] = []
        self._unsubscribe = None

    def bind_values(self, values: ValueView) -> None:
        assert isinstance(values, ValueStore)
        self._unsubscribe = values.subscribe(self._on_value, names=frozenset({self.value}))

    def reset(self, env_ids=None) -> torch.Tensor:
        self.cmd[self._resolve_ids(env_ids)] = 0.0
        return self.cmd

    def update(self, step_dt: float, cmd: torch.Tensor) -> torch.Tensor:
        del step_dt
        self.used_commands.append(None if cmd is None else cmd.detach().clone())
        return self.cmd

    def _on_value(self, name: str, value) -> None:
        if not isinstance(value, torch.Tensor):
            return
        target = value.to(device=self.device, dtype=torch.float32).reshape(self.num_envs, self.dim)
        if self.rebind:
            self.cmd = target.clone()
        else:
            self.cmd.copy_(target)


def _drive_subscribing(
    ticks: int, *, rebind: bool = False
) -> tuple[ValueStore, _BoundarySubscribingSampler, ValuePublishingSink]:
    """Run *ticks* real SyncRunner ticks with a subscribing sampler."""
    store = ValueStore(num_envs=1, device="cpu")
    sampler = _BoundarySubscribingSampler(1, "cpu", 1, rebind=rebind)
    sampler.bind_values(store)
    sink = ValuePublishingSink(store.publisher("sink"), num_envs=1, value=sampler.value)
    env = FakeBackend()
    runner = SyncRunner(
        cfg={"collect": {"steps": ticks}},
        profile=None,
        adapter=ManualAdapter(num_envs=1),
        sampler=sampler,
        term_group=TerminationGroup(),
        env=env,
        sink=sink,
        control_dt=0.02,
        values=store,
    )
    actions = (NamedAction.zeros(1, ARTICULATION_NAME, (JOINT_NAME,), "position"),)
    try:
        for _ in range(ticks):
            runner.tick(actions_override=actions)
    finally:
        env.close()
    return store, sampler, sink


def test_subscribing_sampler_acts_on_a_publication_one_tick_later() -> None:
    """A publish at on_step(t) mutates the live buffer and acts at env.step(t+1)."""
    _, sampler, sink = _drive_subscribing(3)

    assert sink.published == [(0, 1.0), (1, 2.0), (2, 3.0)]
    # update(t) receives the command that tick's env.step used.  Tick 0 still
    # sees the reset command; tick 1 already sees on_step(0)'s published value.
    assert _values(sampler.used_commands) == [0.0, 1.0, 2.0]


def test_rebinding_the_command_tensor_in_the_callback_does_not_propagate() -> None:
    """Reassigning self.cmd breaks the runner alias; the change waits for polling."""
    _, sampler, _ = _drive_subscribing(3, rebind=True)

    assert _values(sampler.used_commands) == [0.0, 0.0, 1.0]
