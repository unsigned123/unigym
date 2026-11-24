"""FanOutSink: dispatch the engine callback stream to multiple sinks.

Used for ``mode: both`` (collect + benchmark in one rollout).  Sinks are
called synchronously in construction order -- CollectSink first, then
EvaluationSession -- and each sees the identical
bundle objects, so recording and scoring cannot diverge.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unigym.core.episode_termination import Termination


class FanOutSink:
    """Compose sinks behind one object satisfying the sink contract.

    ``on_run_end`` returns ``{sink_class_name: summary_dict}``; with the
    standard pair that is ``{"CollectSink": ..., "EvaluationSession": ...}``.
    Two children of the same class collide on that key and fail rather than
    silently overwriting.

    ``drain_terminations`` merges the children that expose the terminal
    capability.  The method is always present, so its presence alone does not
    mean the composite has a pending terminal.
    """

    def __init__(self, *sinks):
        self.sinks = list(sinks)

    def on_step(self, t, bundle) -> None:
        for sink in self.sinks:
            sink.on_step(t, bundle)

    def on_rewind(self, info) -> None:
        for sink in self.sinks:
            fn = getattr(sink, "on_rewind", None)
            if callable(fn):
                fn(info)

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
            name = type(sink).__name__
            source = f"child {index} ({name})"
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
            source = f"child {index} ({type(sink).__name__})"
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
