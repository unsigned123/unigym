"""Watcher-produced signals: level publication and fail-safe behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from unigym.signals import SignalBus
from unigym.watchers import WatchResult, WatcherEngine


class _Events:
    """Event-log double recording the fail-safe event shape."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, dict]] = []

    def append(self, kind: str, name: str, **kwargs) -> None:
        self.rows.append((kind, name, kwargs))


class SignalWatcher:
    """Callable watcher double returning a user-defined per-env signal."""

    def check(self, ctx):
        del ctx
        return WatchResult(
            fired=False,
            signals={"user/pos": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        )


def _engine(bus: SignalBus | None = None, event_log=None) -> WatcherEngine:
    engine = WatcherEngine(run_id="r", out_dir=None, event_log=event_log, signals=bus)
    return engine


def test_watch_result_defaults_to_no_signals_and_logs_names_shapes_only():
    """The default is empty; event data never carries raw tensors."""
    plain = WatchResult(fired=False)
    assert dict(plain.signals) == {}

    result = WatchResult(fired=True, signals={"user/ready": torch.zeros(2, 3)})
    doc = result.as_dict()

    assert doc["signals"] == {"user/ready": [2, 3]}
    assert isinstance(doc["signals"]["user/ready"], list)


def test_expr_signals_publish_on_a_non_firing_tick():
    """Signals are level outputs: fired=False still publishes."""
    bus = SignalBus(num_envs=2, device="cpu")
    engine = _engine(bus)
    engine.add(
        {
            "name": "producer",
            "expr": (
                "WatchResult(fired=False, signals={'user/ready': " "torch.tensor([True, False])})"
            ),
        }
    )

    engine.on_tick(0, {})

    ready = bus.get("user/ready")
    assert ready is not None
    assert ready.tolist() == [True, False]
    assert engine.list()[0]["fired"] == 0


@pytest.mark.parametrize(
    ("value", "dtype", "shape"),
    [
        ("torch.tensor([True, False, True])", torch.bool, (3,)),
        ("torch.tensor([[1, 2], [3, 4], [5, 6]])", torch.int64, (3, 2)),
    ],
)
def test_user_name_dtype_and_trailing_shape_pass_through(value, dtype, shape):
    """Core transports arbitrary names, dtypes and trailing shapes."""
    bus = SignalBus(num_envs=3, device="cpu")
    engine = _engine(bus)
    engine.add(
        {
            "name": "producer",
            "expr": f"WatchResult(fired=False, signals={{'user/custom': {value}}})",
        }
    )

    engine.on_tick(0, {})

    stored = bus.get("user/custom")
    assert stored is not None
    assert stored.dtype == dtype
    assert tuple(stored.shape) == shape


def test_callable_watcher_can_return_signals_directly():
    """A callable predicate's WatchResult rides the same publication path."""
    bus = SignalBus(num_envs=2, device="cpu")
    engine = _engine(bus)
    engine.add(
        {
            "name": "callable",
            "callable": f"{Path(__file__).resolve()}:SignalWatcher",
        }
    )

    engine.on_tick(0, {})

    stored = bus.get("user/pos")
    assert stored is not None
    assert stored.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_without_a_bus_publication_is_a_noop():
    """Offline engines have no transport; signals are ignored."""
    engine = _engine(None)
    engine.add(
        {
            "name": "producer",
            "expr": "WatchResult(fired=False, signals={'user/ready': torch.zeros(2)})",
        }
    )

    engine.on_tick(0, {})

    assert engine.list()[0]["strikes"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        "{'user/bad': torch.zeros(3)}",
        "{'user/bad': [1.0, 2.0]}",
        "[]",
    ],
)
def test_invalid_signals_follow_the_fail_safe_path(payload):
    """A malformed signal mapping is a watcher error, never a crash."""
    bus = SignalBus(num_envs=2, device="cpu")
    events = _Events()
    engine = _engine(bus, events)
    engine.add(
        {
            "name": "producer",
            "expr": f"WatchResult(fired=False, signals={payload})",
        }
    )

    engine.on_tick(0, {})

    armed = engine.list()[0]
    assert armed["strikes"] == 1
    assert armed["fired"] == 0
    assert events.rows[-1][0] == "watcher_error"


def test_one_bad_signal_does_not_partially_publish_the_tick():
    """The mapping is preflighted before the first bus store."""
    bus = SignalBus(num_envs=2, device="cpu")
    engine = _engine(bus)
    engine.add(
        {
            "name": "producer",
            "expr": (
                "WatchResult(fired=False, signals={"
                "'user/good': torch.ones(2), 'user/bad': torch.zeros(3)})"
            ),
        }
    )

    engine.on_tick(0, {})

    assert bus.get("user/good") is None
    assert engine.list()[0]["strikes"] == 1
