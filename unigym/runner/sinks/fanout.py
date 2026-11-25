"""FanOutSink: dispatch the engine callback stream to multiple sinks.

The generic composition primitive for explicit ``sinks:`` selection.  Child
sinks are called synchronously in construction order and each sees the
identical bundle objects, so independent recording/scoring cannot diverge.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unigym.core.episode_termination import Termination


def _node_reset_aware(node: object) -> bool:
    """One child's reset-notification capability (composites answer recursively)."""
    declared = getattr(node, "reset_aware", None)
    if isinstance(declared, bool):
        return declared
    return callable(getattr(node, "on_reset", None))


class FanOutSink:
    """Compose sinks behind one object satisfying the sink contract.

    Two construction forms are supported:

    * ``FanOutSink(sink_a, sink_b, ...)`` keeps the historical unnamed form:
      ``on_run_end`` returns ``{sink_class_name: summary_dict}`` and two
      children of the same class collide rather than silently overwriting.
    * ``FanOutSink(named=(("collect", sink_a), ("bench", sink_b)))`` gives
      every child an explicit stable name; ``on_run_end`` returns
      ``{name: summary_dict}`` and duplicate names fail at construction.

    ``drain_terminations`` merges the children that expose the terminal
    capability.  The method is always present, so its presence alone does not
    mean the composite has a pending terminal.
    """

    def __init__(self, *sinks, named: Iterable[tuple[str, object]] | None = None):
        if named is not None:
            if sinks:
                raise TypeError(
                    "FanOutSink accepts either positional sinks or named=(...) "
                    "children, not both"
                )
            children: list = []
            names: list[str] = []
            for entry in named:
                try:
                    name, sink = entry
                except (TypeError, ValueError) as err:
                    raise TypeError(
                        "FanOutSink(named=...) entries must be (name, sink) pairs"
                    ) from err
                if not isinstance(name, str) or not name:
                    raise ValueError(
                        f"FanOutSink child names must be non-empty strings, got {name!r}"
                    )
                if name in names:
                    raise ValueError(f"FanOutSink: duplicate child name {name!r}")
                names.append(name)
                children.append(sink)
            self.sinks = children
            self._names: tuple[str, ...] | None = tuple(names)
        else:
            self.sinks = list(sinks)
            self._names = None

    def _source(self, index: int, sink: object) -> str:
        """A stable diagnostic label for one child (declared name or class)."""
        label = type(sink).__name__ if self._names is None else self._names[index]
        return f"child {index} ({label})"

    def on_step(self, t, bundle) -> None:
        for sink in self.sinks:
            sink.on_step(t, bundle)

    def on_rewind(self, info) -> None:
        for sink in self.sinks:
            fn = getattr(sink, "on_rewind", None)
            if callable(fn):
                fn(info)

    def on_reset(self, env_ids) -> None:
        for sink in self.sinks:
            fn = getattr(sink, "on_reset", None)
            if callable(fn):
                fn(env_ids)

    @property
    def reset_aware(self) -> bool:
        """True only when at least one child accepts the reset notification.

        ``on_reset`` is always callable (the composite forwards), so the
        runner uses this declared capability to refuse ``terminate(...,
        reset=False)`` when no child could observe the reopen.
        """
        return any(_node_reset_aware(sink) for sink in self.sinks)

    def on_episode_end(self, env_ids, bundle) -> None:
        for sink in self.sinks:
            sink.on_episode_end(env_ids, bundle)

    def on_step_end(self, t, bundle) -> None:
        for sink in self.sinks:
            sink.on_step_end(t, bundle)

    def on_run_end(self) -> dict:
        summaries: dict = {}
        sources: dict[str, str] = {}
        for index, sink in enumerate(self.sinks):
            name = type(sink).__name__ if self._names is None else self._names[index]
            source = self._source(index, sink)
            if name in summaries:
                raise ValueError(
                    f"FanOutSink.on_run_end: duplicate summary key {name!r} from "
                    f"{sources[name]} and {source}"
                )
            summaries[name] = sink.on_run_end()
            sources[name] = source
        return summaries

    def drain_terminations(self) -> dict[int, Termination]:
        """Merge the per-env terminal facts of every child that exposes them."""
        merged: dict[int, Termination] = {}
        sources: dict[int, str] = {}
        for index, sink in enumerate(self.sinks):
            fn = getattr(sink, "drain_terminations", None)
            if not callable(fn):
                continue
            drained = fn()
            source = self._source(index, sink)
            if not isinstance(drained, Mapping):
                raise TypeError(
                    f"FanOutSink.drain_terminations: {source} returned "
                    f"{type(drained).__name__}, expected a Mapping"
                )
            for env_id, termination in drained.items():
                if env_id in merged:
                    raise ValueError(
                        f"FanOutSink.drain_terminations: duplicate env id {env_id!r} "
                        f"from {sources[env_id]} and {source}"
                    )
                merged[env_id] = termination
                sources[env_id] = source
        return merged

    def stop_when(self) -> bool:
        """OR every child that declares ``stop_when``; read-only and cheap.

        A nested fan-out participates through its own ``stop_when`` method, so
        the composite never flattens or recurses here itself.
        """
        for sink in self.sinks:
            fn = getattr(sink, "stop_when", None)
            if callable(fn) and fn():
                return True
        return False
