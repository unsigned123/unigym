"""FanOutSink: dispatch the engine callback stream to multiple sinks.

Used for ``mode: both`` (collect + benchmark in one rollout).  Sinks are
called synchronously in construction order -- CollectSink first, then
EvaluationSession -- and each sees the identical
bundle objects, so recording and scoring cannot diverge.
"""

from __future__ import annotations


class FanOutSink:
    """Compose sinks behind one object satisfying the sink contract.

    ``on_run_end`` returns ``{sink_class_name: summary_dict}``; with the
    standard pair that is ``{"CollectSink": ..., "EvaluationSession": ...}``.
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
        return {type(sink).__name__: sink.on_run_end() for sink in self.sinks}

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
