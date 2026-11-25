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

import torch

from unigym._typing import JsonValue
from unigym.core.episode_termination import EpisodeOutcome, Termination, TerminationSource
from unigym.evaluation import EnvView
from unigym.runtime.bridge import RewindBoundary

if TYPE_CHECKING:
    from unigym.evaluation import EnvStepContext
    from unigym.run.sink import SinkContext
    from unigym.sink import Sink
    from unigym.values import ValuePublisher


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


class ViewProbeSink:
    """Four-callback sink proving the runner's per-tick boundary-view lifetime.

    Records the bundle of every callback.  While a view is present the probe
    reads it, so a closed view fails the test at the callback that violated the
    contract.  ``raise_at`` injects a failure from one named callback;
    ``rewind_next`` raises ``RewindBoundary`` from ``on_step`` after recording.
    The stored bundles are test evidence only; a production sink must not
    retain the view past its callback.
    """

    _RAISE_POINTS = (None, "on_step", "on_episode_end")

    def __init__(self, *, raise_at: str | None = None, rewind_next: int | None = None) -> None:
        if raise_at not in self._RAISE_POINTS:
            raise ValueError(f"raise_at must be one of {self._RAISE_POINTS}, got {raise_at!r}")
        if rewind_next is not None and (
            isinstance(rewind_next, bool) or not isinstance(rewind_next, int) or rewind_next < 0
        ):
            raise ValueError(f"rewind_next must be None or a non-negative int, got {rewind_next!r}")
        self.raise_at = raise_at
        self.rewind_next = rewind_next
        self.step_bundles: list[dict[str, object]] = []
        self.episode_end_bundles: list[dict[str, object]] = []
        self.step_end_bundles: list[dict[str, object]] = []
        self.step_views: list[EnvView | None] = []
        self.episode_end_views: list[EnvView | None] = []
        self.step_end_views: list[EnvView | None] = []
        self.step_view_scopes: list[tuple[int | None, int]] = []
        self.episode_end_view_scopes: list[tuple[int | None, int]] = []
        self.step_contexts: list[EnvStepContext | None] = []
        self.final_step_contexts: list[EnvStepContext | None] = []
        self.step_end_contexts: list[EnvStepContext | None] = []
        self.rewinds: list[dict[str, object]] = []

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        del t
        self.step_bundles.append(bundle)
        context = bundle.get("step")
        self.step_contexts.append(context if context is not None else None)
        view = bundle.get("view")
        self.step_views.append(view if isinstance(view, EnvView) else None)
        if view is not None:
            assert isinstance(view, EnvView)
            view.record()  # only legal while the runner keeps the view open
            self.step_view_scopes.append((view.env_id, view.num_envs))
        if self.rewind_next is not None:
            raise RewindBoundary(self.rewind_next)
        if self.raise_at == "on_step":
            raise RuntimeError("view probe on_step failure")

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        del env_ids
        self.episode_end_bundles.append(bundle)
        context = bundle.get("final_step")
        self.final_step_contexts.append(context if context is not None else None)
        view = bundle.get("view")
        self.episode_end_views.append(view if isinstance(view, EnvView) else None)
        if view is not None:
            assert isinstance(view, EnvView)
            view.record()
            self.episode_end_view_scopes.append((view.env_id, view.num_envs))
        if self.raise_at == "on_episode_end":
            raise RuntimeError("view probe on_episode_end failure")

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        del t
        self.step_end_bundles.append(bundle)
        context = bundle.get("step")
        self.step_end_contexts.append(context if context is not None else None)
        view = bundle.get("view")
        self.step_end_views.append(view if isinstance(view, EnvView) else None)

    def on_run_end(self) -> dict[str, object]:
        return {}

    def on_rewind(self, info: dict[str, object]) -> None:
        self.rewinds.append(info)


class _LifecycleProbeBase:
    """Shared callback recording for the reset-lifecycle doubles."""

    def __init__(self, order: list[tuple[object, ...]] | None = None) -> None:
        self.order = [] if order is None else order
        self.step_calls = 0
        self.step_bundles: list[dict[str, object]] = []
        self.episode_end_calls: list[tuple[tuple[int, ...], dict[str, object]]] = []
        self.episode_end_views: list[object] = []
        self.final_step_contexts: list[object] = []
        self.step_end_bundles: list[dict[str, object]] = []
        self.step_end_calls = 0
        self.run_end_calls = 0

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        self.step_calls += 1
        self.step_bundles.append(bundle)
        self.order.append(("on_step", t))

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        ids = _episode_env_ids(env_ids)
        self.episode_end_calls.append((ids, bundle))
        self.episode_end_views.append(bundle.get("view"))
        self.final_step_contexts.append(bundle.get("final_step"))
        self.order.append(("on_episode_end", ids))

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        self.step_end_calls += 1
        self.step_end_bundles.append(bundle)
        self.order.append(("on_step_end", t))

    def on_run_end(self) -> dict[str, object]:
        self.run_end_calls += 1
        self.order.append(("on_run_end",))
        return {"sink": type(self).__name__}


class LifecycleProbeSink(_LifecycleProbeBase):
    """Four-callback sink without the reset notification capability."""


class ResetAwareLifecycleSink(_LifecycleProbeBase):
    """Four-callback sink that also records ``on_reset``."""

    def __init__(self, order: list[tuple[object, ...]] | None = None) -> None:
        super().__init__(order)
        self.reset_calls: list[list[int]] = []

    def on_reset(self, env_ids: object) -> None:
        if isinstance(env_ids, (list, tuple)):
            ids = [int(value) for value in env_ids]
        else:
            ids = [int(value) for value in _episode_env_ids(env_ids)]
        self.reset_calls.append(ids)
        self.order.append(("on_reset", tuple(ids)))


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


class TerminalDrainSink:
    """Four-callback sink exposing ``drain_terminations`` with a fixed payload.

    ``payload`` is returned verbatim on every drain, so a test can supply a
    mapping, ``None`` or a non-mapping and prove the fan-out boundary.  A real
    terminal sink is one-shot; this double only records how often it was
    drained.
    """

    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.drain_calls = 0

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        del env_ids, bundle

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_run_end(self) -> dict[str, object]:
        return {"sink": type(self).__name__}

    def drain_terminations(self) -> object:
        self.drain_calls += 1
        return self.payload


class ScriptedTerminalSink:
    """Four-callback sink that drains one queued terminal mapping per tick.

    ``pending`` is returned by the next ``drain_terminations`` call and then
    cleared, so the double keeps the one-shot drain contract.  A test may set
    it to ``None``, a non-mapping or a malformed ``env_id -> value`` payload to
    prove the runner's boundary validation.  ``rewind_next`` raises a
    ``RewindBoundary`` from ``on_step`` after recording, so the runner never
    reaches the drain.  The stored bundles are test evidence only.
    """

    def __init__(self, *, rewind_next: int | None = None) -> None:
        if rewind_next is not None and (
            isinstance(rewind_next, bool) or not isinstance(rewind_next, int) or rewind_next < 0
        ):
            raise ValueError(f"rewind_next must be None or a non-negative int, got {rewind_next!r}")
        self.rewind_next = rewind_next
        self.pending: object = {}
        self.drain_calls = 0
        self.step_calls = 0
        self.step_end_calls = 0
        self.step_bundles: list[dict[str, object]] = []
        self.episode_end_calls: list[tuple[object, dict[str, object]]] = []
        self.rewinds: list[dict[str, object]] = []

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        del t
        self.step_calls += 1
        self.step_bundles.append(bundle)
        if self.rewind_next is not None:
            raise RewindBoundary(self.rewind_next)

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        self.episode_end_calls.append((env_ids, bundle))

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle
        self.step_end_calls += 1

    def on_run_end(self) -> dict[str, object]:
        return {"sink": type(self).__name__}

    def drain_terminations(self) -> object:
        self.drain_calls += 1
        payload = self.pending
        self.pending = {}
        return payload

    def on_rewind(self, info: dict[str, object]) -> None:
        self.rewinds.append(info)


class NonCallableDrainSink:
    """Four-callback sink whose declared ``drain_terminations`` is not callable."""

    drain_terminations: object = None

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
        self.reset_calls: list[list[int]] = []
        self.last_episode_end_tick: int | None = None
        self.view_seen = False
        self.env_key_seen = False
        self.step_end_view_none = True

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        del t
        self._observe_boundary(bundle)
        self.step_calls += 1

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        self._observe_boundary(bundle)
        self.episode_end_calls += 1
        ids = _episode_env_ids(env_ids)
        self.episodes_completed += len(ids)
        self.episode_env_ids.append(ids)
        tick = bundle.get("t")
        if isinstance(tick, int) and not isinstance(tick, bool):
            self.last_episode_end_tick = tick

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        del t
        if bundle.get("view") is not None:
            self.step_end_view_none = False
        self.step_end_calls += 1

    def on_reset(self, env_ids: object) -> None:
        if isinstance(env_ids, (list, tuple)):
            self.reset_calls.append([int(value) for value in env_ids])
        else:
            self.reset_calls.append([int(value) for value in _episode_env_ids(env_ids)])

    def _observe_boundary(self, bundle: dict[str, object]) -> None:
        """Read the public boundary view and record the retired env key."""
        self.env_key_seen = self.env_key_seen or "env" in bundle
        view = bundle.get("view")
        if isinstance(view, EnvView):
            view.record()
            self.view_seen = True

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


class ValuePublishingSink:
    """Stateful custom sink that publishes one level value per tick.

    It stores the write-only publisher handed over by :class:`SinkContext`
    (never a read face) and publishes ``(num_envs, 1)`` CPU value ``t + 1``
    under ``value``.  The publication log is one row per ``on_step``.  The sink
    is also its own namespaced rewind provider so the build seam can discover
    the exact instance behind ``SinkContext.values``; its state restores the
    publication log, not the store (the store is a separate core provider).
    """

    name = "acme.values"

    def __init__(
        self,
        publisher: ValuePublisher,
        num_envs: int,
        *,
        value: str = "acme/probe",
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError(f"num_envs must be a positive int, got {num_envs!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"value must be a non-empty str, got {value!r}")
        self.publisher = publisher
        self.num_envs = num_envs
        self.value = value
        self.published: list[tuple[int, float]] = []
        self.step_calls = 0

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        del bundle
        self.step_calls += 1
        value = float(t + 1)
        self.publisher.publish(
            self.value,
            torch.full((self.num_envs, 1), value, dtype=torch.float32),
        )
        self.published.append((t, value))

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        del env_ids, bundle

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_run_end(self) -> dict[str, object]:
        return {"sink": type(self).__name__, "published": len(self.published)}

    def rewind_state(self) -> dict[str, object]:
        return {"published": [list(row) for row in self.published]}

    def apply_rewind_state(self, state: Mapping[str, object]) -> None:
        rows = state.get("published")
        self.published = (
            [
                (int(row[0]), float(row[1]))
                for row in rows
                if isinstance(row, (list, tuple)) and len(row) == 2
            ]
            if isinstance(rows, list)
            else []
        )


def make_value_publishing_sink(ctx: SinkContext) -> Sink:
    """Build one :class:`ValuePublishingSink` from the public context.

    A closed-loop sink cannot work without the publisher, so an absent
    ``ctx.values`` fails fast instead of silently degrading.
    """
    if ctx.values is None:
        raise RuntimeError("ValuePublishingSink requires SinkContext.values")
    return ValuePublishingSink(ctx.values, ctx.num_envs)


class TerminalRequestSink:
    """Stateful custom sink that requests one per-env terminal on a chosen tick.

    Records the per-tick ``time_in_episode`` / ``episode_id`` of ``env_id`` so an
    integration test can prove the real runner reset the row, plus the
    episode-end callback stream and drain count.  It is its own rewind provider
    (``name`` + state callbacks) so the composite sink chain can expose it; a
    pending drain is deliberately not restored -- this double proves integration
    wiring, not rewind state.
    """

    name = "acme.terminal_request"

    def __init__(
        self,
        *,
        terminal_after_steps: int = 1,
        env_id: int = 0,
        code: str = "custom.sink_terminal",
    ) -> None:
        if (
            isinstance(terminal_after_steps, bool)
            or not isinstance(terminal_after_steps, int)
            or terminal_after_steps < 1
        ):
            raise ValueError(
                f"terminal_after_steps must be a positive int, got {terminal_after_steps!r}"
            )
        if isinstance(env_id, bool) or not isinstance(env_id, int) or env_id < 0:
            raise ValueError(f"env_id must be a non-negative int, got {env_id!r}")
        if not isinstance(code, str) or not code.startswith("custom."):
            raise ValueError(f"code must be a 'custom.*' string, got {code!r}")
        self.terminal_after_steps = terminal_after_steps
        self.env_id = env_id
        self.code = code
        self.pending: dict[int, Termination] = {}
        self._requested = False
        self.step_calls = 0
        self.drain_calls = 0
        self.episode_end_calls = 0
        self.run_end_calls = 0
        self.episode_env_ids: list[tuple[int, ...]] = []
        self.step_times: list[float] = []
        self.step_episode_ids: list[int] = []

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        del t
        self.step_calls += 1
        record = bundle["record"]
        self.step_times.append(float(record["time_in_episode"][self.env_id]))
        self.step_episode_ids.append(int(record["episode_id"][self.env_id]))
        if not self._requested and self.step_calls >= self.terminal_after_steps:
            self.pending = {
                self.env_id: Termination(
                    EpisodeOutcome.FAILURE, self.code, TerminationSource.CUSTOM
                )
            }
            self._requested = True

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        del bundle
        self.episode_end_calls += 1
        self.episode_env_ids.append(tuple(int(raw) for raw in env_ids.tolist()))

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        del t, bundle

    def on_run_end(self) -> dict[str, object]:
        self.run_end_calls += 1
        return {"sink": type(self).__name__, "episodes": self.episode_end_calls}

    def drain_terminations(self) -> dict[int, Termination]:
        self.drain_calls += 1
        drained = self.pending
        self.pending = {}
        return drained

    def rewind_state(self) -> dict[str, object]:
        return {
            "requested": self._requested,
            "step_calls": self.step_calls,
            "drain_calls": self.drain_calls,
            "episode_end_calls": self.episode_end_calls,
            "run_end_calls": self.run_end_calls,
            "episode_env_ids": [list(ids) for ids in self.episode_env_ids],
            "step_times": list(self.step_times),
            "step_episode_ids": list(self.step_episode_ids),
        }

    def apply_rewind_state(self, state: Mapping[str, object]) -> None:
        for attribute in ("step_calls", "drain_calls", "episode_end_calls", "run_end_calls"):
            value = state.get(attribute)
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(self, attribute, value)
        requested = state.get("requested")
        if isinstance(requested, bool):
            self._requested = requested
        self.pending = {}


def make_terminal_request_sink(ctx: SinkContext) -> Sink:
    """Build one :class:`TerminalRequestSink` from the public context."""
    after = ctx.options.get("terminal_after_steps", 1)
    if isinstance(after, bool) or not isinstance(after, int):
        raise TypeError(f"sink.options.terminal_after_steps must be an int, got {after!r}")
    env_id = ctx.options.get("env_id", 0)
    if isinstance(env_id, bool) or not isinstance(env_id, int):
        raise TypeError(f"sink.options.env_id must be an int, got {env_id!r}")
    code = ctx.options.get("code", "custom.sink_terminal")
    if not isinstance(code, str):
        raise TypeError(f"sink.options.code must be a string, got {code!r}")
    return TerminalRequestSink(terminal_after_steps=after, env_id=env_id, code=code)
