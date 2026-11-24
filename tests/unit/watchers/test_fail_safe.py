"""Watcher fail-safe: runtime errors strike, data errors degrade without one."""

from __future__ import annotations

from pathlib import Path

from tests.unit.watchers._step_fixtures import tick_bundle
from unigym.watchers import WatchResult, WatcherEngine


class _Events:
    """Event-log double recording every appended watcher row."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, dict]] = []

    def append(self, kind: str, name: str, **kwargs) -> None:
        self.rows.append((kind, name, kwargs))


class RaisingWatcher:
    """Callable double whose check raises a non-data runtime error."""

    def check(self, ctx):
        del ctx
        raise RuntimeError("boom")


class MissingKeyWatcher:
    """Callable double whose check raises the data-shaped WatchError."""

    def check(self, ctx):
        return WatchResult(fired=ctx.key("record/nope") is not None)


def _callable(name: str) -> str:
    return f"{Path(__file__).resolve()}:{name}"


def test_runtime_error_emits_watcher_error_and_increments_strikes():
    """A raising predicate is a strike and never fires the watcher."""
    events = _Events()
    engine = WatcherEngine(run_id="r", out_dir=None, event_log=events)
    engine.add({"name": "boom", "callable": _callable("RaisingWatcher")})
    bundle, _step = tick_bundle()

    engine.on_tick(0, bundle)

    row = engine.list()[0]
    assert row["strikes"] == 1
    assert row["fired"] == 0
    assert row["state"] == "armed"
    assert events.rows[-1][0] == "watcher_error"
    assert events.rows[-1][1] == "boom"
    assert events.rows[-1][2]["level"] == "error"
    assert "RuntimeError: boom" in events.rows[-1][2]["data"]["error"]


def test_watch_error_degrades_without_a_strike_and_is_rate_limited():
    """A missing key degrades (no strike); repeated ticks do not spam events."""
    events = _Events()
    engine = WatcherEngine(run_id="r", out_dir=None, event_log=events)
    engine.add({"name": "missing", "callable": _callable("MissingKeyWatcher")})

    engine.on_tick(0, tick_bundle(t=0)[0])
    engine.on_tick(1, tick_bundle(t=1)[0])

    row = engine.list()[0]
    assert row["strikes"] == 0
    assert row["fired"] == 0
    assert row["state"] == "armed"
    degrade = [event for event in events.rows if event[0] == "watcher_degrade"]
    assert len(degrade) == 1
    assert degrade[0][1] == "missing"
    assert degrade[0][2]["level"] == "warn"
    assert "WatchError" in degrade[0][2]["data"]["error"]


def test_three_consecutive_runtime_errors_disable_the_watch():
    """The strike threshold auto-disables after three failing evaluations."""
    events = _Events()
    engine = WatcherEngine(run_id="r", out_dir=None, event_log=events)
    engine.add({"name": "boom", "callable": _callable("RaisingWatcher"), "every": 1})

    for t in range(3):
        engine.on_tick(t, tick_bundle(t=t)[0])

    row = engine.list()[0]
    assert row["strikes"] == 3
    assert row["fired"] == 0
    assert row["state"] == "disabled"
    assert row["disabled_reason"].startswith("fail-safe:")
    assert [event[0] for event in events.rows].count("watcher_error") == 3
