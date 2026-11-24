"""Sink-published signals reach the sampler one update later (two-tick command latency)."""

from __future__ import annotations

import torch

from tests._support.fakes.backend import ARTICULATION_NAME, JOINT_NAME, FakeBackend
from tests._support.fakes.command_sampler import ClosedLoopSampler
from tests._support.fakes.external_sink import SignalPublishingSink
from unigym.adapters.manual import ManualAdapter
from unigym.runner.sync import SyncRunner
from unigym.runtime.actions import NamedAction
from unigym.signals import SignalBus
from unigym.terminations import TerminationGroup


def _drive(ticks: int) -> tuple[SignalBus, ClosedLoopSampler, SignalPublishingSink]:
    """Run *ticks* real SyncRunner ticks over the fake backend and one bus."""
    bus = SignalBus(num_envs=1, device="cpu")
    sampler = ClosedLoopSampler(dim=1, device="cpu", num_envs=1)
    sampler.bind_signals(bus)
    sink = SignalPublishingSink(bus, num_envs=1, signal=sampler.signal)
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
    )
    actions = (NamedAction.zeros(1, ARTICULATION_NAME, (JOINT_NAME,), "position"),)
    try:
        for _ in range(ticks):
            runner.tick(actions_override=actions)
    finally:
        env.close()
    return bus, sampler, sink


def _values(values: list[torch.Tensor | None]) -> list[float | None]:
    return [None if value is None else float(value[0, 0]) for value in values]


def test_sink_publishes_once_per_tick_and_sampler_reads_the_next_update() -> None:
    """A value published at ``on_step(t)`` is visible to ``update(t+1)`` only."""
    bus, sampler, sink = _drive(3)

    assert sink.published == [(0, 1.0), (1, 2.0), (2, 3.0)]
    assert _values(sampler.observations) == [None, 1.0, 2.0]
    snapshot = bus.snapshot()
    assert list(snapshot) == [sink.signal]
    assert float(snapshot[sink.signal][0, 0]) == 3.0


def test_signal_published_at_tick_t_acts_at_tick_t_plus_two() -> None:
    """The two-tick latency is real: publish t -> update t+1 -> env.step t+2."""
    _, sampler, sink = _drive(3)

    # update(t) receives the command that tick's env.step used.
    assert _values(sampler.used_commands) == [0.0, 0.0, 1.0]
    assert sink.published[0] == (0, 1.0)
    assert sampler.observations[1] is not None
    assert float(sampler.observations[1][0, 0]) == 1.0
    assert _values(sampler.used_commands)[2] == 1.0
