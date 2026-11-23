"""Canonical joint-limit condition.

``JointLimitCondition`` reads a policy-order ``joint_pos`` row and the
articulation's ``(lower, upper)`` soft limits.  ``band`` mode matches the
termination soft-limit band; ``magnitude`` mode matches the watcher's absolute
limit approach.  Missing limits always produce an all-false mask.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from .base import _as_tensor

if TYPE_CHECKING:
    from unigym.evaluation import EvalContext

__all__ = ["JointLimitCondition"]

JointLimitMode = Literal["band", "magnitude"]


class JointLimitCondition:
    """A vector-scoped joint-limit predicate.

    Args:
        articulation: Articulation whose limits are read.
        record_key: Explicit record leaf key for ``joint_pos``; defaults to the
            articulation leaf.
        mode: ``band`` (termination) or ``magnitude`` (watcher).
        margin: Band margin; defaults to ``0.05`` for ``band`` and ``0.1`` for
            ``magnitude``.
    """

    def __init__(
        self,
        *,
        articulation: str,
        record_key: str | None = None,
        mode: JointLimitMode = "band",
        margin: float | None = None,
    ) -> None:
        if not isinstance(articulation, str) or not articulation:
            raise ValueError("JointLimitCondition articulation must be a non-empty string")
        if record_key is not None and (not isinstance(record_key, str) or not record_key):
            raise ValueError("JointLimitCondition record_key must be a non-empty string")
        if mode not in ("band", "magnitude"):
            raise ValueError(f"JointLimitCondition mode must be band or magnitude, got {mode!r}")
        self.articulation = articulation
        self.record_key = record_key
        self.mode = mode
        self.margin = float(margin if margin is not None else (0.05 if mode == "band" else 0.1))
        self._source_name = record_key or f"articulation/{articulation}/joint_pos"

    def evaluate(self, ctx: EvalContext) -> torch.Tensor:
        """Return the per-scope bool mask for *ctx*."""
        joint_pos = self._read(ctx)
        limits = ctx.env.joint_pos_limits(self.articulation)
        if limits is None:
            return self._empty(joint_pos)
        lower, upper = limits
        if lower is None or upper is None:
            return self._empty(joint_pos)
        if self.mode == "band":
            below = joint_pos < (lower + self.margin)
            above = joint_pos > (upper - self.margin)
            return (below | above).sum(-1) > 0
        limit = torch.maximum(lower.abs(), upper.abs())
        band = limit - self.margin
        return (joint_pos.abs() >= band).any(dim=-1)

    def _read(self, ctx: EvalContext) -> torch.Tensor:
        if self.record_key is not None:
            value = ctx.env.record().get(self.record_key)
        else:
            value = ctx.env.articulation(self.articulation).get("joint_pos")
        return _as_tensor(value, self._source_name)

    @staticmethod
    def _empty(joint_pos: torch.Tensor) -> torch.Tensor:
        return torch.zeros(joint_pos.shape[0], dtype=torch.bool, device=joint_pos.device)
