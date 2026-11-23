"""Joint-limit termination: joint positions beyond soft limits (+margin).

Fires when any joint sits outside ``[lower + margin, upper - margin]``.
Requires this articulation's live ``joint_pos`` and its
``(lower, upper)`` limits from ``ctx.env.joint_pos_limits(name)``.  Limits and
positions share the record/body-control policy order; without limits the
termination never fires.
"""

from __future__ import annotations

import torch

from unigym.evaluation import EvalContext

from .articulation import ArticulationTermination


class JointLimitTermination(ArticulationTermination):
    """Fire when any joint exceeds its soft limit by more than *margin*.

    params:
        articulation: REQUIRED articulation name (no flat fallback).
        margin: radial margin inside the soft limits (default 0.05); the
            trigger band is ``joint_pos < lower + margin`` or
            ``joint_pos > upper - margin``.
    """

    def __init__(self, params: dict | None = None, name: str | None = None):
        super().__init__(params, name)
        self.margin = float(self.params.get("margin", 0.05))

    def check(self, ctx: EvalContext) -> torch.Tensor:
        source = self.source(ctx)
        joint_pos = source.get("joint_pos")
        if not isinstance(joint_pos, torch.Tensor):
            raise ValueError(
                f"JointLimitTermination requires {self.articulation!r} "
                f"joint_pos in the evaluation record"
            )
        limits = ctx.env.joint_pos_limits(self.articulation)
        if limits is None:
            return torch.zeros(joint_pos.shape[0], dtype=torch.bool, device=joint_pos.device)
        lower, upper = limits
        if lower is None or upper is None:
            return torch.zeros(joint_pos.shape[0], dtype=torch.bool, device=joint_pos.device)
        below = joint_pos < (lower + self.margin)
        above = joint_pos > (upper - self.margin)
        return (below | above).sum(-1) > 0
