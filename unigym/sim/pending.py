"""Staged, not-yet-lowered actions of the experimental simulation facade.

``act`` writes exactly one staged entry per ``(articulation, group)`` identity;
``step`` lowers the whole pending map to the existing
``tuple[NamedAction, ...]`` surface and clears it only after the engine
accepted the tick.  This module owns the staging table and the lowering; the
validation policy lives in :class:`~unigym.sim.simulation.Simulation`.

``torch`` is only named under ``TYPE_CHECKING`` and imported inside
:meth:`PendingActions.lower`, so importing ``unigym.sim`` stays torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import PendingActionError

if TYPE_CHECKING:
    import torch

    from unigym.runtime.actions import NamedAction, OrderedActionGroup

__all__ = ["PendingActions", "StagedAction"]

ActionGroupIdentity = tuple[str, str]


@dataclass(frozen=True)
class StagedAction:
    """One validated action group waiting for the next tick.

    Attributes:
        group: the declared ordered action group the values obey.
        values: a ``(N, dim)`` floating tensor already normalized by ``act``.
    """

    group: OrderedActionGroup
    values: torch.Tensor


class PendingActions:
    """The per-action-group pending table of one simulation.

    Insertion order is the lowering order.  ``stage`` refuses a second entry
    for the same ``(articulation, group)`` identity instead of overwriting it;
    the caller decides when to clear (a successful tick, a reset, a fault or a
    close).
    """

    def __init__(self) -> None:
        self._entries: dict[ActionGroupIdentity, StagedAction] = {}

    def stage(self, staged: StagedAction) -> None:
        """Stage one action group; a duplicate identity keeps the existing entry."""
        identity = (staged.group.articulation, staged.group.name)
        if identity in self._entries:
            raise PendingActionError(
                f"action group {identity!r} is already staged; exactly one pending "
                f"entry per (articulation, group) is allowed (staged groups: "
                f"{sorted(self._entries)})"
            )
        self._entries[identity] = staged

    def clear(self) -> None:
        """Drop every staged entry."""
        self._entries.clear()

    def groups(self) -> frozenset[ActionGroupIdentity]:
        """The staged ``(articulation, group)`` identities."""
        return frozenset(self._entries)

    def lower(self) -> tuple[NamedAction, ...]:
        """Lower the pending table to a self-describing action sequence.

        The result preserves staging order; an empty table lowers to ``()``.
        Each entry carries its group's articulation, name, joint order and
        kind; the sequence order is not an action layout.
        """
        from unigym.runtime.actions import NamedAction

        return tuple(
            NamedAction(
                articulation=staged.group.articulation,
                group=staged.group.name,
                values=staged.values,
                joint_names=tuple(staged.group.joints),
                kind=staged.group.target,
            )
            for staged in self._entries.values()
        )
