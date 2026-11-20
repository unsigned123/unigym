"""WatcherEngine rewind provider: specs, trigger state and windows."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from unigym.runtime.rewind import RewindGateError, RewindStateProvider
from unigym.signals import SignalBus
from unigym.watchers import WatchResult, WatcherEngine


class _ObsSnapshot:
    """Minimal observation snapshot double (``values`` face)."""

    def __init__(self, values: dict) -> None:
        self.values = values


class SetupWatcher:
    """Callable double that only checks after its framework setup ran."""

    def setup(self, ctx) -> None:
        del ctx
        self.ready = True

    def check(self, ctx):
        del ctx
        if not getattr(self, "ready", False):
            raise RuntimeError("setup did not run")
        return WatchResult(fired=True, message="ready")


def _bundle(x: float) -> dict:
    return {"obs": _ObsSnapshot({"x": torch.tensor([x, x], dtype=torch.float32)})}


def _engine() -> tuple[WatcherEngine, SignalBus]:
    bus = SignalBus(num_envs=2, device="cpu")
    engine = WatcherEngine(run_id="r", out_dir=None, signals=bus)
    engine.add(
        {
            "name": "base",
            "expr": ("WatchResult(fired=True, signals={'user/n': " "torch.tensor([1.0, 1.0])})"),
        }
    )
    engine.add(
        {
            "name": "win",
            "template": "key_window",
            "params": {
                "key": "obs/x",
                "op": "gt",
                "value": 0.0,
                "window": 2,
                "reduce": "mean",
            },
        }
    )
    return engine, bus


def test_engine_is_a_named_rewind_provider():
    """The engine satisfies the provider protocol under ``watchers``."""
    engine = WatcherEngine(run_id="r", out_dir=None)

    assert engine.name == "watchers"
    assert isinstance(engine, RewindStateProvider)


def test_rewind_round_trip_restores_specs_trigger_state_and_windows():
    """A checkpoint reverts hot-add/clear and restores the producer state."""
    engine, bus = _engine()
    engine.on_tick(0, _bundle(2.0))
    engine.on_tick(1, _bundle(4.0))

    state = engine.rewind_state()
    bus_state = bus.rewind_state()
    assert [row["name"] for row in engine.list()] == ["base", "win"]

    # diverge: hot-add, mutate the window, clear a spec, disable the base
    engine.on_tick(2, _bundle(8.0))
    engine.add({"name": "hot", "expr": "True"})
    engine.clear("win")
    engine.update("base", {"enabled": False})
    bus.publish("user/n", torch.tensor([9.0, 9.0]))
    assert sorted(row["name"] for row in engine.list()) == ["base", "hot"]
    assert next(row for row in engine.list() if row["name"] == "base")["state"] == "disabled"

    engine.validate_rewind_state(state)
    engine.apply_rewind_state(state)
    bus.apply_rewind_state(bus_state)

    rows = {row["name"]: row for row in engine.list()}
    assert sorted(rows) == ["base", "win"]
    assert rows["base"]["state"] == "armed"
    assert rows["base"]["fired"] == 1
    assert rows["base"]["last_tick"] == 0
    window = engine.window_cache[("obs/x", 2)]
    assert list(window["buf"]) == [2.0, 4.0]
    assert torch.equal(bus.get("user/n"), torch.tensor([1.0, 1.0]))

    # the restored edge state still suppresses the same firing edge
    engine.on_tick(60, _bundle(8.0))
    assert next(row for row in engine.list() if row["name"] == "base")["fired"] == 1


def test_malformed_state_is_refused_before_any_change():
    """A bad trigger-state field is a zero-change refusal."""
    engine, _ = _engine()
    engine.on_tick(0, _bundle(2.0))
    state = engine.rewind_state()
    before = engine.list()
    raw_states = state["states"]
    assert isinstance(raw_states, dict)
    raw_state = raw_states["base"]
    assert isinstance(raw_state, dict)
    raw_state["fired"] = "one"

    with pytest.raises(RewindGateError, match="fired must be int"):
        engine.validate_rewind_state(state)
    with pytest.raises(RewindGateError, match="fired must be int"):
        engine.apply_rewind_state(state)

    assert engine.list() == before


def test_malformed_spec_payload_is_refused_before_any_change():
    """A spec that no longer parses is refused read-only."""
    engine, _ = _engine()
    engine.on_tick(0, _bundle(2.0))
    state = engine.rewind_state()
    before = engine.list()
    state["specs"] = [{"name": "broken"}]

    with pytest.raises(RewindGateError, match="invalid spec"):
        engine.validate_rewind_state(state)
    with pytest.raises(RewindGateError, match="invalid spec"):
        engine.apply_rewind_state(state)

    assert engine.list() == before


def test_rewind_rearms_callable_setup_instead_of_trusting_setup_done():
    """A re-created callable must run setup again before its next check."""
    engine = WatcherEngine(run_id="r", out_dir=None)
    engine.add(
        {
            "name": "setup",
            "callable": f"{Path(__file__).resolve()}:SetupWatcher",
        }
    )
    engine.on_tick(0, {})
    assert engine.list()[0]["fired"] == 1

    state = engine.rewind_state()
    raw_states = state["states"]
    assert isinstance(raw_states, dict)
    assert raw_states["setup"]["setup_done"] is True

    engine.clear()
    assert engine.list() == []

    engine.validate_rewind_state(state)
    engine.apply_rewind_state(state)
    restored = engine.rewind_state()["states"]
    assert isinstance(restored, dict)
    assert restored["setup"]["setup_done"] is False

    engine.on_tick(5, {})
    row = engine.list()[0]
    assert row["name"] == "setup"
    assert row["state"] == "armed"
    assert row["strikes"] == 0
    assert row["last_message"] == "ready"
    assert row["fired"] == 1
