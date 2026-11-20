"""SignalBus rewind provider: round-trip, preflight and zero-change refusal."""

from __future__ import annotations

import pytest
import torch

from unigym.runtime.rewind import RewindController, RewindGateError, RewindStateProvider
from unigym.signals import SignalBus


class _Obs:
    """Observation-registry double with passthrough rewind state."""

    def rewind_state(self) -> dict:
        return {}

    def restore_rewind_state(self, state: dict) -> None:
        del state


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

    def on_rewind(self, restored_tick: int) -> None:
        del restored_tick

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

    def register_command(self, name: str, handler) -> None:
        self.commands[name] = handler

    def pause_for_rewind(self) -> None:
        pass


class _Engine:
    """Runner-loop double: no sampler, no replay."""

    sampler = None
    replay = None

    def __init__(self) -> None:
        self._tick = 0
        self._rewind_state = None


def _controller(providers):
    env = _Env()
    adapter = _Adapter()
    engine = _Engine()
    bridge = _Bridge()
    controller = RewindController(
        bridge,
        env=env,
        adapter=adapter,
        engine=engine,
        backend_label="mujoco",
        providers=providers,
    )
    return controller, env, adapter, engine


def test_signal_bus_is_a_named_rewind_provider():
    """The bus satisfies the provider protocol under the ``signals`` key."""
    bus = SignalBus(num_envs=2, device="cpu")

    assert bus.name == "signals"
    assert isinstance(bus, RewindStateProvider)


def test_signal_bus_rewind_state_round_trips_with_private_copies():
    """State capture/restore replaces the store and cannot alias it."""
    bus = SignalBus(num_envs=2, device="cpu")
    bus.publish("a", torch.tensor([1.0, 2.0]))

    state = bus.rewind_state()
    bus.publish("a", torch.tensor([9.0, 9.0]))
    bus.publish("b", torch.tensor([3.0, 3.0]))

    bus.apply_rewind_state(state)

    assert torch.equal(bus.get("a"), torch.tensor([1.0, 2.0]))
    assert bus.get("b") is None
    captured = state["a"]
    assert isinstance(captured, torch.Tensor)
    captured[0] = 42.0
    assert torch.equal(bus.get("a"), torch.tensor([1.0, 2.0]))


def test_controller_rewind_restores_signal_values():
    """The bus joins the checkpoint payload and is restored by the controller."""
    bus = SignalBus(num_envs=2, device="cpu")
    bus.publish("a", torch.tensor([1.0, 2.0]))
    controller, *_ = _controller([bus])
    ack = controller.cmd_snapshot(0, {})
    record = controller._snaps[ack["snap_id"]]
    assert torch.equal(record["payload"]["providers"]["signals"]["a"], bus.get("a"))

    bus.publish("a", torch.tensor([5.0, 5.0]))
    bus.publish("b", torch.tensor([3.0, 3.0]))
    controller.cmd_rewind(0, {"snap_id": ack["snap_id"]})

    assert torch.equal(bus.get("a"), torch.tensor([1.0, 2.0]))
    assert bus.get("b") is None


def test_invalid_signal_state_is_refused_before_any_restore():
    """A malformed provider payload is a zero-change rejection."""
    bus = SignalBus(num_envs=2, device="cpu")
    bus.publish("a", torch.tensor([1.0, 2.0]))
    controller, env, adapter, engine = _controller([bus])
    ack = controller.cmd_snapshot(0, {})
    record = controller._snaps[ack["snap_id"]]
    record["payload"]["providers"]["signals"] = {"a": torch.zeros(3)}

    bus.publish("a", torch.tensor([5.0, 5.0]))
    env.value = 1
    adapter.value = 1

    with pytest.raises(RewindGateError, match="signals rewind state"):
        controller.cmd_rewind(0, {"snap_id": ack["snap_id"]})

    assert env.value == 1
    assert adapter.value == 1
    assert engine._tick == 0
    assert torch.equal(bus.get("a"), torch.tensor([5.0, 5.0]))
