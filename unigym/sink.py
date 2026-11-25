"""The public Sink protocol: the runner's one consumer extension seam.

``SyncRunner`` drives the four required callbacks in a fixed order; a sink
consumes that callback stream and owns everything consumer-side (recording,
benchmark state, scoring, reporting).  A sink never owns a second physics loop
or mutates the backend.

The optional capability protocols (:class:`RewindAwareSink`,
:class:`ResetAwareSink`, :class:`StoppingSink`, :class:`TerminalSink`) name
extra behavior a sink may implement; they add no new control-flow concepts to
the four-callback contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from unigym.core.episode_termination import Termination

__all__ = [
    "ResetAwareSink",
    "RewindAwareSink",
    "Sink",
    "StoppingSink",
    "TerminalSink",
]


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
class ResetAwareSink(Protocol):
    """Optional capability: a sink notified after every real reset chain.

    The runner calls ``on_reset`` after it resets the named envs (tick
    auto-reset, explicit ``reset`` and ``terminate(..., reset=True)``).  The
    callback is never a substitute for the reset chain and is not replayed on
    rewind; a sink whose state must survive a checkpoint exposes a rewind
    provider.  A sink without this capability still supports ``reset=True``;
    ``terminate(..., reset=False)`` requires it because the closed env is not
    reopened until an explicit reset.
    """

    def on_reset(self, env_ids: Sequence[int]) -> None:
        """Receive the ids that just completed the reset chain."""
        ...


@runtime_checkable
class StoppingSink(Protocol):
    """Optional capability: a sink that can request an early run stop."""

    def stop_when(self) -> bool:
        """Return True when the run should stop; read-only and cheap."""
        ...


@runtime_checkable
class TerminalSink(Protocol):
    """Optional capability: a sink that requests per-env episode ends.

    A terminal sink is still a four-callback :class:`Sink`; this adds one
    optional capability, not a fifth required callback.  ``drain_terminations``
    is a one-shot drain: after it returns, the sink holds no pending terminal
    for those envs.
    """

    def drain_terminations(self) -> Mapping[int, Termination]:
        """Return and clear the pending ``env_id -> Termination`` facts."""
        ...
