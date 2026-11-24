"""Watchers are debug-only: results carry and publish no values/signals."""

from __future__ import annotations

import inspect

import pytest
import torch

from tests.unit.watchers._step_fixtures import tick_bundle
from unigym.values import ValueStore
from unigym.watchers import WatchResult, WatcherEngine, attach


def test_watch_result_has_no_signal_field_or_log_key():
    """The result dataclass and its log shape expose no publication face."""
    assert "signals" not in WatchResult.__dataclass_fields__
    result = WatchResult(fired=True, message="observed")

    assert not hasattr(result, "signals")
    assert result.as_dict() == {
        "fired": True,
        "level": "info",
        "message": "observed",
        "data": {},
        "actions": [],
    }

    with pytest.raises(TypeError):
        WatchResult(fired=True, signals={"user/x": object()})  # type: ignore[call-arg]


def test_engine_and_attach_have_no_signal_transport_parameters():
    """Neither the engine nor the wiring takes a bus-only transport face."""
    engine_params = inspect.signature(WatcherEngine.__init__).parameters
    attach_params = inspect.signature(attach).parameters

    assert "signals" not in engine_params
    assert "signals" not in attach_params
    assert "env" not in attach_params

    engine = WatcherEngine(run_id="r", out_dir=None)
    assert not hasattr(engine, "signals")
    assert not hasattr(engine, "_publish_signals")
    assert not hasattr(engine, "env")


def test_step_values_are_readable_but_never_published_back():
    """A watcher observes the step values; the store gains nothing from it."""
    store = ValueStore(num_envs=2, device="cpu")
    store.publish("sink", "user/value", torch.tensor([1.0, 2.0]))
    engine = WatcherEngine(run_id="r", out_dir=None)
    engine.add(
        {
            "name": "read-values",
            "expr": "WatchResult(fired=(values.get('user/value') is not None))",
        }
    )
    bundle, _step = tick_bundle(values=store)

    engine.on_tick(0, bundle)

    assert engine.list()[0]["fired"] == 1
    assert sorted(store.snapshot()) == ["user/value"]
    assert store.get("user/value").tolist() == [1.0, 2.0]
