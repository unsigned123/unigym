"""Joint-limit termination: joint positions beyond soft limits (+margin).

Fires when any joint sits outside ``[lower + margin, upper - margin]``.
Requires this articulation's ``joint_pos`` (``state.articulations[name]``)
and its ``(lower, upper)`` limits (``state.joint_pos_limits[name]``); without
limits the termination never fires.

v103.6: limits are keyed by articulation (no first binding); the run layer
supplies a ``{name: (lower, upper)}`` mapping.
"""
from __future__ import annotations

from .articulation import ArticulationTermination
from .base import TerminationState, all_false


class JointLimitTermination(ArticulationTermination):
    """Fire when any joint exceeds its soft limit by more than *margin*.

    params:
        articulation: REQUIRED articulation name (v103.6, no flat fallback).
        margin: radial margin inside the soft limits (default 0.05); the
            trigger band is ``joint_pos < lower + margin`` or
            ``joint_pos > upper - margin``.
    """

    requires_joint_limits = True

    def __init__(self, params: dict | None = None, name: str | None = None):
        super().__init__(params, name)
        self.margin = float(self.params.get("margin", 0.05))

    def check(self, state: TerminationState):
        source = self.source(state)
        joint_pos = source.get("joint_pos")
        if joint_pos is None:
            raise ValueError(
                f"JointLimitTermination requires {self.articulation!r} "
                f"joint_pos in state")
        limits = (state.joint_pos_limits or {}).get(self.articulation)
        if limits is None:
            return all_false(joint_pos)
        lower, upper = limits
        if lower is None or upper is None:
            return all_false(joint_pos)
        below = joint_pos < (lower + self.margin)
        above = joint_pos > (upper - self.margin)
        return (below | above).sum(-1) > 0
