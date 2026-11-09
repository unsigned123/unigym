"""FanOutSink: dispatch the engine callback stream to multiple sinks.

Used for ``mode: both`` (collect + benchmark in one rollout).  Sinks are
called synchronously in construction order -- CollectSink first, then
BenchmarkSink -- and each sees the identical
bundle objects, so recording and scoring cannot diverge.
"""

from __future__ import annotations


class FanOutSink:
    """Compose sinks behind one object satisfying the sink contract.

    ``on_run_end`` returns ``{sink_class_name: summary_dict}``; with the
    standard pair that is ``{"CollectSink": ..., "BenchmarkSink": ...}``.
    """

    def __init__(self, *sinks):
        self.sinks = list(sinks)

    def on_step(self, t, bundle) -> None:
        for sink in self.sinks:
            sink.on_step(t, bundle)

    def on_episode_end(self, env_ids, bundle) -> None:
        for sink in self.sinks:
            sink.on_episode_end(env_ids, bundle)

    def on_step_end(self, t, bundle) -> None:
        for sink in self.sinks:
            sink.on_step_end(t, bundle)

    def on_run_end(self) -> dict:
        return {type(sink).__name__: sink.on_run_end() for sink in self.sinks}
