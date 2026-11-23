"""The public Sink protocol and its capability protocols."""

from __future__ import annotations

import inspect

import pytest

import unigym
import unigym.runner
import unigym.sink
from unigym.run.instance import Sink as InstanceSink
from unigym.runner.sinks import FanOutSink
from unigym.sink import RewindAwareSink, Sink, StoppingSink


class _RecordingSink:
    """Sink double that records every callback it receives."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def on_step(self, t, bundle) -> None:
        self.calls.append(("on_step", t, bundle))

    def on_episode_end(self, env_ids, bundle) -> None:
        self.calls.append(("on_episode_end", env_ids, bundle))

    def on_step_end(self, t, bundle) -> None:
        self.calls.append(("on_step_end", t, bundle))

    def on_run_end(self) -> dict:
        self.calls.append(("on_run_end",))
        return {"sink": type(self).__name__}


class _OtherRecordingSink(_RecordingSink):
    """Second sink class so the fan-out summary keys stay distinct."""


def test_root_runner_and_instance_reexport_the_canonical_sink():
    """Every advertised import path resolves to the one canonical object."""
    assert "Sink" in unigym.__all__
    assert "Sink" in unigym.runner.__all__
    assert unigym.Sink is unigym.sink.Sink
    assert unigym.runner.Sink is unigym.sink.Sink
    assert InstanceSink is unigym.sink.Sink


def test_runner_reexports_the_capability_protocols():
    """Runner re-exports the optional capability protocols by identity."""
    assert unigym.runner.RewindAwareSink is RewindAwareSink
    assert unigym.runner.StoppingSink is StoppingSink


def test_sink_declares_exactly_the_four_required_callbacks():
    """The required contract is exactly four callbacks, one control-flow seam."""
    declared = {name for name in dir(Sink) if not name.startswith("_")}
    assert declared == {"on_step", "on_episode_end", "on_step_end", "on_run_end"}


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("on_step", ["self", "t", "bundle"]),
        ("on_episode_end", ["self", "env_ids", "bundle"]),
        ("on_step_end", ["self", "t", "bundle"]),
        ("on_run_end", ["self"]),
    ],
)
def test_sink_required_callback_shape(name, parameters):
    """Each required callback keeps its documented parameter shape."""
    signature = inspect.signature(getattr(Sink, name))
    assert list(signature.parameters) == parameters


@pytest.mark.parametrize(
    ("protocol", "name", "parameters"),
    [
        (RewindAwareSink, "on_rewind", ["self", "info"]),
        (StoppingSink, "stop_when", ["self"]),
    ],
)
def test_capability_protocol_shape(protocol, name, parameters):
    """Each optional capability declares exactly its one documented method."""
    declared = {member for member in dir(protocol) if not member.startswith("_")}
    assert declared == {name}
    signature = inspect.signature(getattr(protocol, name))
    assert list(signature.parameters) == parameters


def test_required_callbacks_make_an_object_a_sink():
    """Structural checks accept a full sink and reject an incomplete one."""

    class _Partial:
        def on_step(self, t, bundle) -> None:
            del t, bundle

    assert isinstance(_RecordingSink(), Sink)
    assert not isinstance(_Partial(), Sink)


def test_capability_protocols_detect_only_declared_methods():
    """Capability checks depend on the optional method being present."""

    class _Capable(_RecordingSink):
        def on_rewind(self, info) -> None:
            del info

        def stop_when(self) -> bool:
            return True

    capable = _Capable()
    plain = _RecordingSink()
    assert isinstance(capable, RewindAwareSink)
    assert isinstance(capable, StoppingSink)
    assert not isinstance(plain, RewindAwareSink)
    assert not isinstance(plain, StoppingSink)


def test_fanout_satisfies_sink_and_forwards_every_callback():
    """FanOutSink is a Sink that forwards all four callbacks to each child."""
    first = _RecordingSink()
    second = _OtherRecordingSink()
    fanout = FanOutSink(first, second)
    bundle = {"t": 3}
    env_ids = object()

    assert isinstance(fanout, Sink)
    fanout.on_step(4, bundle)
    fanout.on_episode_end(env_ids, bundle)
    fanout.on_step_end(5, bundle)
    summary = fanout.on_run_end()

    expected_calls = [
        ("on_step", 4, bundle),
        ("on_episode_end", env_ids, bundle),
        ("on_step_end", 5, bundle),
        ("on_run_end",),
    ]
    assert first.calls == expected_calls
    assert second.calls == expected_calls
    assert summary == {
        "_RecordingSink": {"sink": "_RecordingSink"},
        "_OtherRecordingSink": {"sink": "_OtherRecordingSink"},
    }
