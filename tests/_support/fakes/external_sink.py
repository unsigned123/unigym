"""External Sink / factory-boundary test doubles.

Not collected.  Every class here implements the public
:class:`unigym.sink.Sink` callback contract structurally; the capability and
provider shapes mirror the construction boundary in ``unigym/run/sink.py``.
Construct a fresh instance per test; the counters are explicit and there is no
cross-test state.
"""

from __future__ import annotations

from unigym._typing import JsonValue


class RecordingSink:
    """Minimal four-callback sink recording every callback it receives."""

    def __init__(self, label: str = "recording") -> None:
        self.label = label
        self.calls: list[tuple[object, ...]] = []

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        self.calls.append(("on_step", t, bundle))

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        self.calls.append(("on_episode_end", env_ids, bundle))

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        self.calls.append(("on_step_end", t, bundle))

    def on_run_end(self) -> dict[str, object]:
        self.calls.append(("on_run_end",))
        return {"sink": type(self).__name__}


class StopPollingSink:
    """Four-callback sink whose ``stop_when`` records how often it is polled."""

    def __init__(self, value: bool, *, label: str = "stopping") -> None:
        self.label = label
        self.value = value
        self.polls = 0

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        del env_ids, bundle

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_run_end(self) -> dict[str, object]:
        return {}

    def stop_when(self) -> bool:
        self.polls += 1
        return self.value


class MetricsFeedSink:
    """Four-callback sink declaring the full bridge telemetry feed."""

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        del env_ids, bundle

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_run_end(self) -> dict[str, object]:
        return {}

    def metrics_drain(self) -> list[dict[str, JsonValue]]:
        return []

    def metrics_task(self) -> dict[str, JsonValue] | None:
        return None

    def metrics_progress(self) -> dict[str, JsonValue] | None:
        return None

    def metrics_report(self) -> JsonValue:
        return None


class PartialSink:
    """Sink-shaped object missing three of the four required callbacks."""

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle


class NonCallableStopSink:
    """Four-callback sink whose declared ``stop_when`` is not callable."""

    stop_when: object = None

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        del env_ids, bundle

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_run_end(self) -> dict[str, object]:
        return {}


class NonCallableRewindSink:
    """Four-callback sink whose declared ``on_rewind`` is not callable."""

    on_rewind: object = None

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        del env_ids, bundle

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_run_end(self) -> dict[str, object]:
        return {}


class NonCallableMetricsSink:
    """Four-callback sink with one non-callable metrics-feed member."""

    metrics_drain: object = None

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        del env_ids, bundle

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_run_end(self) -> dict[str, object]:
        return {}

    def metrics_task(self) -> dict[str, JsonValue] | None:
        return None

    def metrics_progress(self) -> dict[str, JsonValue] | None:
        return None

    def metrics_report(self) -> JsonValue:
        return None


class RewindProvider:
    """Valid rewind-provider payload shape (unique name + state callbacks)."""

    name = "acme.pick"

    def rewind_state(self) -> dict[str, object]:
        return {}

    def apply_rewind_state(self, state: dict[str, object]) -> None:
        del state


class IncompleteRewindProvider:
    """Provider shape missing ``apply_rewind_state`` (invalid)."""

    name = "acme.pick"

    def rewind_state(self) -> dict[str, object]:
        return {}


class NamelessRewindProvider:
    """Provider shape missing the payload-key ``name`` (invalid)."""

    def rewind_state(self) -> dict[str, object]:
        return {}

    def apply_rewind_state(self, state: dict[str, object]) -> None:
        del state


class ProviderSink:
    """Sink that exposes ``rewind_provider`` as a callable supplier."""

    def __init__(self, provider: object) -> None:
        self.provider = provider
        self.supplier_calls = 0

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        del env_ids, bundle

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_run_end(self) -> dict[str, object]:
        return {}

    def rewind_provider(self) -> object:
        self.supplier_calls += 1
        return self.provider


class DirectProviderSink:
    """Sink exposing a non-callable ``rewind_provider`` provider object."""

    def __init__(self, provider: object) -> None:
        self.rewind_provider = provider

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        del env_ids, bundle

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_run_end(self) -> dict[str, object]:
        return {}
