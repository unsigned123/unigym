"""Staged, not-yet-lowered actions of the experimental simulation facade.

``act`` writes exactly one staged entry per control face; ``step`` lowers the
whole pending map to the existing ``tuple[NamedAction, ...]`` surface and
clears it only after the engine accepted the tick.  This module owns the
staging table and the lowering; the validation policy lives in
:class:`~unigym.sim.simulation.Simulation`.

``torch`` is only named under ``TYPE_CHECKING`` and imported inside
:meth:`PendingActions.lower`, so importing ``unigym.sim`` stays torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import PendingActionError

if TYPE_CHECKING:
    import torch

    from unigym.runtime.actions import NamedAction

    from .handles import ActionBinding

__all__ = ["PendingActions", "StagedAction"]


@dataclass(frozen=True)
class StagedAction:
    """One validated action face waiting for the next tick.

    Attributes:
        binding: the explicit face/kind/column-order binding the values obey.
        values: a ``(N, dim)`` floating tensor already normalized by ``act``.
    """

    binding: ActionBinding
    values: torch.Tensor


class PendingActions:
    """The per-face pending table of one simulation.

    Insertion order is the lowering order.  ``stage`` refuses a second entry
    for the same face instead of overwriting it; the caller decides when to
    clear (a successful tick, a reset, a fault or a close).
    """

    def __init__(self) -> None:
        self._entries: dict[str, StagedAction] = {}

    def stage(self, face: str, staged: StagedAction) -> None:
        """Stage one face; a duplicate face keeps the existing entry."""
        if face in self._entries:
            raise PendingActionError(
                f"action face {face!r} is already staged; exactly one pending entry per "
                f"face is allowed (staged faces: {sorted(self._entries)})"
            )
        self._entries[face] = staged

    def clear(self) -> None:
        """Drop every staged entry."""
        self._entries.clear()

    def faces(self) -> frozenset[str]:
        """The staged face names."""
        return frozenset(self._entries)

    def lower(self) -> tuple[NamedAction, ...]:
        """Lower the pending table to a self-describing action sequence.

        The result preserves staging order; an empty table lowers to ``()``.
        Each entry carries its owning articulation name and face (group)
        identity; the sequence order is not an action layout.
        """
        from unigym.runtime.actions import NamedAction

        return tuple(
            NamedAction(
                articulation=staged.binding.articulation.name,
                group=staged.binding.face,
                values=staged.values,
                joint_names=tuple(staged.binding.joint_names),
                kind=staged.binding.kind,
            )
            for staged in self._entries.values()
        )
