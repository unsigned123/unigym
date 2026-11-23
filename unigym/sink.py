"""The public Sink protocol: the runner's one consumer extension seam.

``SyncRunner`` drives the four required callbacks in a fixed order; a sink
consumes that callback stream and owns everything consumer-side (recording,
benchmark state, scoring, reporting).  A sink never owns a second physics loop
or mutates the backend.

The optional capability protocols (:class:`RewindAwareSink`,
:class:`StoppingSink`) name extra behavior a sink may implement; they add no
new control-flow concepts to the four-callback contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["RewindAwareSink", "Sink", "StoppingSink"]


@runtime_checkable
class Sink(Protocol):
    """The stable consumer contract; callbacks are driven by ``SyncRunner`` only."""

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        """Consume one pre-reset engine step."""
        ...

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        """Consume the episode-end data side before the reset chain."""
        ...

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        """Consume one post-reset engine step (metrics side)."""
        ...

    def on_run_end(self) -> dict[str, object]:
        """Flush the run and return the sink summary."""
        ...


@runtime_checkable
class RewindAwareSink(Protocol):
    """Optional capability: a sink that labels a rewind/fork mark.

    ``on_rewind`` only marks the abandoned branch.  A sink whose own state must
    survive a checkpoint exposes a rewind provider; ``on_rewind`` does not
    restore state.
    """

    def on_rewind(self, info: dict[str, object]) -> None:
        """Receive one rewind/fork mark."""
        ...


@runtime_checkable
class StoppingSink(Protocol):
    """Optional capability: a sink that can request an early run stop."""

    def stop_when(self) -> bool:
        """Return True when the run should stop; read-only and cheap."""
        ...
