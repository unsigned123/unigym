"""Runner protocol data types (R0/R2: single-step results + the protocol).

``StepResult`` is what one policy tick produces.  ``SyncRunner.tick()``
returns it; ``SyncRunner.run()`` is the loop around it.  The fields are a
verbatim mapping of the values the engine already computes per tick -- no
new semantics, no extra RNG.

``Runner`` is the structural contract every driver implements:
four methods only, no ownership of the underlying ``RunInstance``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from unigym.runtime.actions import NamedAction


@dataclass
class StepResult:
    """One policy tick's product (see module docstring)."""

    t: int
    obs: dict
    extra: dict
    reset_env_ids: torch.Tensor
    terminated: torch.Tensor | None


@runtime_checkable
class Runner(Protocol):
    """Time/control-flow owner of one constructed run.

    A runner *borrows* a ``RunInstance``: it never closes it.  The four
    methods are the whole contract; new drivers compose them instead of
    extending this protocol.
    """

    def tick(
        self,
        actions_override: dict[str, NamedAction] | None = None,
    ) -> StepResult:
        """Advance exactly one policy tick.

        Args:
            actions_override: Optional external-driver action dict.  When
                supplied the adapter forward/postprocess is skipped and the
                named actions go straight to the env step; ``None`` runs the
                adapter-driven path.
        """
        ...

    def run(self, stop_when: Callable[[], bool] | None = None) -> dict:
        """Drive ticks until the run ends; return the sink summary."""
        ...

    def reset(self, env_ids: Sequence[int] | None = None) -> StepResult:
        """Reset the named envs (all when None)."""
        ...

    def close(self) -> None:
        """Stop the runner's loop/thread; does not close the instance."""
        ...

    @property
    def ticks_done(self) -> int:
        """Number of policy ticks completed so far (public clock read)."""
        ...
