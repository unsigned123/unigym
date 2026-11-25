"""Public contract for the explicit terminate/reset lifecycle."""

from __future__ import annotations

import inspect

import unigym.runner
from unigym.runner.sinks import CollectSink, FanOutSink
from unigym.runner.sync import SyncRunner
from unigym.runtime.rewind import RewindStateProvider
from unigym.runtime.rewind_providers import RunnerLifecycleProvider
from unigym.sink import ResetAwareSink


def test_terminate_signature_requires_keyword_only_reset():
    """reset is required, keyword-only and has no default."""
    signature = inspect.signature(SyncRunner.terminate)
    parameters = signature.parameters

    assert list(parameters) == ["self", "env_ids", "terminations", "reset"]
    assert parameters["reset"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["reset"].default is inspect.Parameter.empty


def test_reset_aware_sink_declares_the_single_reset_callback():
    """The optional capability is exactly on_reset(env_ids)."""
    declared = {name for name in dir(ResetAwareSink) if not name.startswith("_")}

    assert declared == {"on_reset"}
    signature = inspect.signature(ResetAwareSink.on_reset)
    assert list(signature.parameters) == ["self", "env_ids"]


def test_runner_reexports_the_reset_aware_capability():
    """The runner namespace publishes the one canonical protocol object."""
    assert unigym.runner.ResetAwareSink is ResetAwareSink
    assert "ResetAwareSink" in unigym.runner.__all__


class _ResetChild:
    """Minimal child exposing the reset notification capability."""

    def on_reset(self, env_ids) -> None:
        del env_ids


def test_built_in_collect_and_fanout_are_reset_aware():
    """Collect declares the no-op callback; the composite reports the truth."""
    assert isinstance(CollectSink, type)
    assert callable(getattr(CollectSink, "on_reset", None))
    plain = FanOutSink(object())
    aware = FanOutSink(_ResetChild())
    nested = FanOutSink(FanOutSink(FanOutSink(_ResetChild())))
    assert isinstance(plain, ResetAwareSink)
    assert plain.reset_aware is False
    assert aware.reset_aware is True
    assert nested.reset_aware is True


def test_runner_lifecycle_provider_shape_and_name():
    """The provider is the runner_lifecycle rewind provider."""
    provider = RunnerLifecycleProvider(object())

    assert provider.name == "runner_lifecycle"
    assert callable(provider.rewind_state)
    assert callable(provider.apply_rewind_state)
    assert callable(provider.validate_rewind_state)
    assert isinstance(provider, RewindStateProvider)
