"""Rewind provider wiring and initializer state capture/restore boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from unigym.initializers.base import InitializerBase
from unigym.io.writer import ShardWriter
from unigym.runner.sinks.fanout import FanOutSink
from unigym.runtime.rewind import RewindGateError
from unigym.runtime.rewind_providers import (
    CollectSinkProvider,
    InitializerStateProvider,
    build_rewind_providers,
)


def _initializer(seed: int = 0) -> InitializerBase:
    return InitializerBase({"seed": seed}, None, None)


def test_build_rewind_providers_mounts_only_capable_initializer():
    """An initializer joins the rewind package only when it can restore."""
    capable = _initializer()
    providers = build_rewind_providers(None, initializer=capable)

    assert [provider.name for provider in providers] == ["initializer"]
    assert build_rewind_providers(None, initializer=None) == []
    assert build_rewind_providers(None, initializer=object()) == []


def test_initializer_provider_snapshot_is_isolated_from_live_state():
    """A captured snapshot is a deep copy; mutating it leaves the RNG untouched."""
    initializer = _initializer(seed=11)
    provider = InitializerStateProvider(initializer)
    snapshot = provider.rewind_state()

    snapshot["rng"]["state"]["state"] += 1

    reference = _initializer(seed=11)
    assert initializer.rng.integers(0, 1_000_000) == reference.rng.integers(0, 1_000_000)


def test_initializer_provider_restore_replays_the_same_stream():
    """Applying a captured state replays the initializer's RNG stream exactly."""
    initializer = _initializer(seed=5)
    provider = InitializerStateProvider(initializer)
    state = provider.rewind_state()

    expected = [int(initializer.rng.integers(0, 1000)) for _ in range(3)]
    provider.apply_rewind_state(state)

    assert [int(initializer.rng.integers(0, 1000)) for _ in range(3)] == expected


@pytest.mark.parametrize(
    ("initializer", "state", "match"),
    [
        (_initializer(), None, "rewind state is missing"),
        (object(), {}, "cannot restore rewind state"),
    ],
    ids=["missing-state", "unrestorable-initializer"],
)
def test_initializer_provider_validate_rejects_unrestorable_state(initializer, state, match):
    """Pre-flight refuses state an initializer could not restore."""
    provider = InitializerStateProvider(initializer)

    with pytest.raises(ValueError, match=match):
        provider.validate_rewind_state(state)


class _RewindRecorder:
    def __init__(self):
        self.marks: list[dict] = []

    def on_rewind(self, info: dict) -> None:
        self.marks.append(dict(info))


def test_nested_fanout_forwards_on_rewind_once_per_child():
    """Every nested child sees exactly one on_rewind call."""
    first, second, third = _RewindRecorder(), _RewindRecorder(), _RewindRecorder()
    fan = FanOutSink(FanOutSink(first), FanOutSink(second, FanOutSink(third)))
    info = {"tick": 3, "restored_tick": 1, "snap_id": 7}

    fan.on_rewind(info)

    assert first.marks == [info]
    assert second.marks == [info]
    assert third.marks == [info]


def test_nested_fanout_skips_children_without_on_rewind():
    """A sink without on_rewind does not break the fan-out notification."""
    recorder = _RewindRecorder()
    fan = FanOutSink(object(), FanOutSink(recorder))

    fan.on_rewind({"tick": 1})

    assert recorder.marks == [{"tick": 1}]


class _WriterSink:
    """The minimal collect-sink face the provider reads: a fuse plus a writer."""

    def __init__(self, writer: ShardWriter):
        self.writer = writer
        self.n_episodes = 0


def _append_shard(writer: ShardWriter) -> None:
    """Append exactly one full shard (two rows at ``shard_steps=2``)."""
    writer.append({"x": np.zeros((1, 1), dtype=np.float32)})
    writer.append({"x": np.zeros((1, 1), dtype=np.float32)})


def test_collect_sink_provider_round_trips_fuse_and_shard_cursor(tmp_path):
    """The collect provider snapshots and restores both fuse and shard cursor."""
    writer = ShardWriter(str(tmp_path), shard_steps=2)
    _append_shard(writer)
    sink = _WriterSink(writer)
    sink.n_episodes = 2
    provider = CollectSinkProvider(sink)

    state = provider.rewind_state()

    assert set(state) == {"fuse", "shard_commit"}
    assert state["fuse"] == {"n_episodes": 2}
    assert state["shard_commit"] == {"committed_shards": 1, "committed_rows": 2}

    _append_shard(writer)
    sink.n_episodes = 9
    provider.validate_rewind_state(state)
    provider.apply_rewind_state(state)

    assert sink.n_episodes == 2
    assert writer.commit_state() == {"committed_shards": 1, "committed_rows": 2}
    assert provider.dump_attempt(str(tmp_path), "attempt") == []


def test_collect_sink_provider_rejects_malformed_state(tmp_path):
    """A malformed fuse or shard commit state is refused before mutation."""
    writer = ShardWriter(str(tmp_path), shard_steps=2)
    sink = _WriterSink(writer)
    provider = CollectSinkProvider(sink)

    with pytest.raises(RewindGateError, match="fuse must be a mapping"):
        provider.validate_rewind_state({"fuse": []})

    bad_commit = {
        "fuse": {"n_episodes": 0},
        "shard_commit": {"committed_shards": 0, "committed_rows": 5},
    }
    with pytest.raises(RewindGateError, match="shard commit state corrupt"):
        provider.validate_rewind_state(bad_commit)
