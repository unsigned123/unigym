"""Composite termination group (OR / AND combination) + inversion wrapper."""
from __future__ import annotations

from typing import Any

from .base import Termination, TerminationState, all_false


class InvertedTermination(Termination):
    """Logical NOT of a wrapped termination.

    Turns a *fault* detector into a *health* detector, e.g. wrapping
    ``FallTermination`` yields "body upright and high enough" — the
    exit/health condition of recovery-style switching. Built from
    yaml via the per-entry ``invert: true`` flag (see
    :func:`terminations.build_terminations`).
    """

    def __init__(self, term: Termination, name: str | None = None):
        super().__init__(None, name or f"not_{term.name}")
        self.term = term
        # propagate so collect.py fetches joint limits when the wrapped
        # termination needs them (class attr shadowed by an instance attr)
        self.requires_joint_limits = bool(term.requires_joint_limits)

    def check(self, state: TerminationState):
        return ~self.term.check(state)


class TerminationGroup:
    """Combine several terminations into one mask.

    Args:
        terminations: list of :class:`Termination` instances.
        mode: ``"or"`` (default; any term fires -> fire) or ``"and"``
            (all terms must fire).

    An empty group never fires (all-False mask), matching the
    "no terminations configured -> nothing triggers" contract.
    """

    def __init__(self, terminations: list[Termination] | None = None,
                 mode: str = "or"):
        if mode not in ("or", "and"):
            raise ValueError(f"mode must be 'or' or 'and', got {mode!r}")
        self.terminations = list(terminations or [])
        self.mode = mode

    def __len__(self) -> int:
        return len(self.terminations)

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.terminations]

    def check_detail(self, state: TerminationState) -> dict:
        """Per-termination masks ``{name: (N,) bool}`` (diagnostics)."""
        return {t.name: t.check(state) for t in self.terminations}

    def combine(self, detail: dict) -> Any:
        """Reduce per-termination masks (from :meth:`check_detail`)."""
        if not detail:
            return None
        masks = list(detail.values())
        out = masks[0]
        for m in masks[1:]:
            out = (out | m) if self.mode == "or" else (out & m)
        return out

    def check(self, state: TerminationState) -> Any:
        """Combined (N,) bool mask (True = the group fires)."""
        ref = state.reference()
        if ref is None:
            raise ValueError("TerminationState has no arrays to judge")
        if not self.terminations:
            return all_false(ref)
        return self.combine(self.check_detail(state))
