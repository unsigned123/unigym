"""External Sink / factory-boundary test doubles.

Not collected.  Every class here implements the public
:class:`unigym.sink.Sink` callback contract structurally; the capability and
provider shapes mirror the construction boundary in ``unigym/run/sink.py``.
Construct a fresh instance per test; the counters are explicit and there is no
cross-test state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from unigym._typing import JsonValue

if TYPE_CHECKING:
    from unigym.run.sink import SinkContext
    from unigym.sink import Sink


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


class ExternalSink:
    """Stateful external sink for a real run, built by :func:`make_external_sink`.

    It records every required lifecycle callback, counts completed episodes and
    stops after ``target_episodes``.  The sink is also its own namespaced rewind
    provider (``name`` + ``rewind_state`` / ``apply_rewind_state``), so a
    checkpoint captures its counters without a second wrapper object.
    """

    name = "acme.external"

    def __init__(self, *, target_episodes: int = 2, label: str = "external") -> None:
        if isinstance(target_episodes, bool) or target_episodes < 1:
            raise ValueError("target_episodes must be a positive int, got " f"{target_episodes!r}")
        self.target_episodes = int(target_episodes)
        self.label = label
        self.step_calls = 0
        self.episode_end_calls = 0
        self.step_end_calls = 0
        self.run_end_calls = 0
        self.episodes_completed = 0
        self.stop_polls = 0
        self.episode_env_ids: list[tuple[int, ...]] = []
        self.last_episode_end_tick: int | None = None

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle
        self.step_calls += 1

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        self.episode_end_calls += 1
        ids = _episode_env_ids(env_ids)
        self.episodes_completed += len(ids)
        self.episode_env_ids.append(ids)
        tick = bundle.get("t")
        if isinstance(tick, int) and not isinstance(tick, bool):
            self.last_episode_end_tick = tick

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle
        self.step_end_calls += 1

    def on_run_end(self) -> dict[str, object]:
        self.run_end_calls += 1
        return {
            "sink": type(self).__name__,
            "episodes_completed": self.episodes_completed,
            "steps": self.step_calls,
        }

    def stop_when(self) -> bool:
        self.stop_polls += 1
        return self.episodes_completed >= self.target_episodes

    def rewind_state(self) -> dict[str, object]:
        return {
            "episodes_completed": self.episodes_completed,
            "step_calls": self.step_calls,
            "episode_end_calls": self.episode_end_calls,
            "step_end_calls": self.step_end_calls,
            "run_end_calls": self.run_end_calls,
            "stop_polls": self.stop_polls,
            "episode_env_ids": [list(ids) for ids in self.episode_env_ids],
            "last_episode_end_tick": self.last_episode_end_tick,
        }

    def apply_rewind_state(self, state: Mapping[str, object]) -> None:
        for attribute in (
            "episodes_completed",
            "step_calls",
            "episode_end_calls",
            "step_end_calls",
            "run_end_calls",
            "stop_polls",
        ):
            value = state.get(attribute)
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(self, attribute, value)
        episode_ids = state.get("episode_env_ids")
        if isinstance(episode_ids, list):
            self.episode_env_ids = [
                tuple(int(raw) for raw in row)
                for row in episode_ids
                if isinstance(row, (list, tuple))
            ]
        tick = state.get("last_episode_end_tick")
        self.last_episode_end_tick = (
            tick if isinstance(tick, int) and not isinstance(tick, bool) else None
        )


def _episode_env_ids(env_ids: object) -> tuple[int, ...]:
    """Read the due-env ids from the runner's tensor without importing torch."""
    tolist = getattr(env_ids, "tolist", None)
    if not callable(tolist):
        return ()
    values: object = tolist()
    if not isinstance(values, list):
        return ()
    return tuple(int(raw) for raw in values if isinstance(raw, (int, float)))


def make_external_sink(ctx: SinkContext) -> Sink:
    """Build one :class:`ExternalSink` from the public construction context."""
    target = ctx.options.get("target_episodes", 2)
    if isinstance(target, bool) or not isinstance(target, int):
        raise TypeError("sink.options.target_episodes must be a positive int, " f"got {target!r}")
    label = ctx.options.get("label", "external")
    if not isinstance(label, str):
        raise TypeError(f"sink.options.label must be a string, got {label!r}")
    return ExternalSink(target_episodes=target, label=label)
