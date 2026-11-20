"""Rewind gate and payload wiring for sampler state."""

from __future__ import annotations

import pytest
import torch

from unigym.commands.base import CommandSampler
from unigym.commands.fixed import FixedSampler
from unigym.commands.lhs import LHSSampler
from unigym.commands.uniform import UniformSampler
from unigym.runtime.rewind import RewindController, RewindGateError


class _Obs:
    """Observation-registry double with passthrough rewind state."""

    def __init__(self) -> None:
        self.state: dict = {}

    def rewind_state(self) -> dict:
        return dict(self.state)

    def restore_rewind_state(self, state: dict) -> None:
        self.state = dict(state)


class _Env:
    """Engine snapshot/restore double."""

    def __init__(self) -> None:
        self.obs = _Obs()
        self.value = 0

    def snapshot(self) -> dict:
        return {"value": self.value}

    def restore(self, state: dict) -> None:
        self.value = state["value"]


class _Adapter:
    """Rewindable adapter double."""

    rewindable = True

    def __init__(self) -> None:
        self.value = 0
        self.rewound_to: int | None = None

    def on_rewind(self, restored_tick: int) -> None:
        self.rewound_to = restored_tick

    def snapshot_internal(self) -> dict:
        return {"value": self.value}

    def restore_internal(self, state: dict) -> None:
        self.value = state["value"]


class _Bridge:
    """Bridge double: command table + loop bookkeeping slots."""

    def __init__(self) -> None:
        self.commands: dict = {}
        self._bundle: dict = {"obs_frame": None, "cmd": None, "record": {}}
        self._last_step = 5
        self.rewind_log: list = []
        self.out_dir = None
        self.paused = False

    def register_command(self, name: str, handler) -> None:
        self.commands[name] = handler

    def pause_for_rewind(self) -> None:
        self.paused = True


class _Engine:
    """Runner-loop double: sampler seam + iteration state."""

    def __init__(self, sampler) -> None:
        self.sampler = sampler
        self.replay = None
        self._tick = 0
        self._rewind_state = None


class _CapableSampler(CommandSampler):
    """Real sampler with explicit, restorable command state."""

    rewind_capable = True

    def __init__(self) -> None:
        super().__init__(dim=2, device="cpu", num_envs=1)
        self.value = 0
        self.fail_on: int | None = None

    def reset(self, env_ids=None) -> torch.Tensor:
        self.cmd[self._resolve_ids(env_ids)] = float(self.value)
        return self.cmd

    def update(self, step_dt: float, cmd: torch.Tensor) -> torch.Tensor:
        del step_dt
        self.cmd = cmd
        return self.cmd

    def rewind_state(self) -> dict:
        return {"value": self.value, "cmd": self.cmd.detach().cpu().tolist()}

    def apply_rewind_state(self, state: dict) -> None:
        if self.fail_on is not None and state["value"] == self.fail_on:
            raise ValueError("sampler cannot restore this state")
        self.value = int(state["value"])
        self.cmd = torch.tensor(state["cmd"], dtype=torch.float32)


class _OrderProbe:
    """Provider double asserting the sampler is restored before providers."""

    name = "probe"

    def __init__(self, sampler: _CapableSampler) -> None:
        self.sampler = sampler
        self.seen: list[int] = []

    def rewind_state(self) -> dict:
        return {"ok": True}

    def apply_rewind_state(self, state: dict) -> None:
        del state
        self.seen.append(self.sampler.value)


def _controller(sampler, providers=None):
    env = _Env()
    adapter = _Adapter()
    engine = _Engine(sampler)
    bridge = _Bridge()
    return (
        RewindController(
            bridge,
            env=env,
            adapter=adapter,
            engine=engine,
            backend_label="mujoco",
            providers=providers,
        ),
        env,
        adapter,
        engine,
        bridge,
    )


def test_rewind_gate_refuses_a_sampler_without_rewind_capability():
    """A random resampler carries outside state: the gate refuses it."""
    controller, *_ = _controller(
        UniformSampler(dim=1, device="cpu", num_envs=1, ranges=[[0.0, 1.0]])
    )

    with pytest.raises(RewindGateError, match="not rewindable"):
        controller.cmd_snapshot(0, {})


def test_rewind_gate_refuses_an_lhs_sampler():
    """LHS keeps decks/timers outside the snapshot: still non-rewindable."""
    controller, *_ = _controller(LHSSampler(dim=1, device="cpu", num_envs=1, ranges=[[0.0, 1.0]]))

    with pytest.raises(RewindGateError, match="not rewindable"):
        controller.cmd_snapshot(0, {})


def test_rewind_snapshot_accepts_an_inert_sampler_without_packaging_state():
    """FixedSampler is inert: rewind works and packages no sampler state."""
    controller, *_ = _controller(FixedSampler(dim=1, device="cpu", num_envs=1))

    ack = controller.cmd_snapshot(0, {})
    record = controller._snaps[ack["snap_id"]]

    assert record["payload"]["sampler"] is None


def test_rewind_restores_capable_sampler_command_and_value():
    """A capable sampler's cmd buffer and own state round-trip together."""
    sampler = _CapableSampler()
    sampler.value = 2
    sampler.reset()
    controller, env, adapter, engine, bridge = _controller(sampler)
    ack = controller.cmd_snapshot(0, {})

    sampler.value = 5
    sampler.reset()
    env.value = 1
    adapter.value = 1
    bridge._last_step = 9
    controller.cmd_rewind(0, {"snap_id": ack["snap_id"]})

    assert sampler.value == 2
    assert sampler.cmd.tolist() == [[2.0, 2.0]]
    assert env.value == 0
    assert adapter.value == 0
    assert engine._tick == ack["tick"] + 1


def test_rewind_restores_the_sampler_before_providers():
    """Provider state application observes the already-restored sampler."""
    sampler = _CapableSampler()
    probe = _OrderProbe(sampler)
    controller, *_ = _controller(sampler, providers=[probe])
    ack = controller.cmd_snapshot(0, {})
    sampler.value = 4
    sampler.reset()

    controller.cmd_rewind(0, {"snap_id": ack["snap_id"]})

    assert probe.seen == [0]


def test_rewind_preflight_refuses_a_capable_sampler_without_snapshot_state():
    """A capable sampler with a state-less snapshot is refused before restore."""
    sampler = _CapableSampler()
    controller, env, adapter, engine, _ = _controller(sampler)
    ack = controller.cmd_snapshot(0, {})
    controller._snaps[ack["snap_id"]]["payload"]["sampler"] = None
    sampler.value = 1
    sampler.reset()
    env.value = 1
    adapter.value = 1

    with pytest.raises(RewindGateError, match="has no sampler state"):
        controller.cmd_rewind(0, {"snap_id": ack["snap_id"]})

    assert (sampler.value, env.value, adapter.value) == (1, 1, 1)
    assert engine._tick == 0


def test_rewind_sampler_restore_failure_rolls_back_the_safety_snapshot():
    """A failing sampler apply rolls env/adapter/sampler back and re-raises."""
    sampler = _CapableSampler()
    controller, env, adapter, engine, bridge = _controller(sampler)
    ack = controller.cmd_snapshot(0, {})
    sampler.value = 1
    sampler.reset()
    env.value = 1
    adapter.value = 1
    sampler.fail_on = 0

    with pytest.raises(ValueError, match="cannot restore this state"):
        controller.cmd_rewind(0, {"snap_id": ack["snap_id"]})

    assert sampler.value == 1
    assert sampler.cmd.tolist() == [[1.0, 1.0]]
    assert env.value == 1
    assert adapter.value == 1
    assert engine._tick == 0
