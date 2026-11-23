"""Canonical height condition.

``HeightCondition`` reads either an articulation's ``base_pos_w`` leaf or an
explicit record key and compares its height component (default index ``2``).
Termination uses the threshold form; watchers use the threshold/op or band form.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .base import ConditionError, _as_tensor

if TYPE_CHECKING:
    from unigym.evaluation import EvalContext

__all__ = ["HeightCondition"]

_OPS = ("gt", "lt", "gte", "lte", "eq", "abs_gt", "abs_lt", "abs_gte", "abs_lte")


def _compare(values: torch.Tensor, op: str, threshold: float) -> torch.Tensor:
    """Apply one watcher comparison op to *values*."""
    if op == "gt":
        return values > threshold
    if op == "lt":
        return values < threshold
    if op == "gte":
        return values >= threshold
    if op == "lte":
        return values <= threshold
    if op == "eq":
        return values == threshold
    compare = abs(values)
    if op == "abs_gt":
        return compare > threshold
    if op == "abs_lt":
        return compare < threshold
    if op == "abs_gte":
        return compare >= threshold
    return compare <= threshold


class HeightCondition:
    """A vector-scoped height predicate.

    Args:
        articulation: Articulation whose ``base_pos_w`` leaf is read.
        record_key: Explicit record leaf key (mutually exclusive with
            *articulation*).
        threshold: Threshold value for the comparison form.
        op: One of the watcher comparison operators (default ``lt``).
        band: Inclusive ``(low, high)`` band; fires outside the band.
        dim: Trailing-axis height component (default ``2``).
    """

    def __init__(
        self,
        *,
        articulation: str | None = None,
        record_key: str | None = None,
        threshold: float | None = None,
        op: str = "lt",
        band: Sequence[float] | None = None,
        dim: int = 2,
    ) -> None:
        if (articulation is None) == (record_key is None):
            raise ValueError("HeightCondition requires exactly one of articulation or record_key")
        if articulation is not None and (not isinstance(articulation, str) or not articulation):
            raise ValueError("HeightCondition articulation must be a non-empty string")
        if record_key is not None and (not isinstance(record_key, str) or not record_key):
            raise ValueError("HeightCondition record_key must be a non-empty string")
        if band is not None:
            if len(band) != 2:
                raise ValueError("HeightCondition band must be a [low, high] pair")
            low, high = float(band[0]), float(band[1])
            if low > high:
                raise ValueError(f"HeightCondition band must have low <= high, got {band!r}")
            self._band: tuple[float, float] | None = (low, high)
        else:
            if op not in _OPS:
                raise ValueError(f"HeightCondition op must be one of {_OPS}, got {op!r}")
            self._band = None
        self.articulation = articulation
        self.record_key = record_key
        self.threshold = None if threshold is None else float(threshold)
        self.op = op
        self.dim = int(dim)
        self._source_name = (
            f"articulation/{articulation}/base_pos_w"
            if articulation is not None
            else str(record_key)
        )

    def evaluate(self, ctx: EvalContext) -> torch.Tensor:
        """Return the per-scope bool mask for *ctx*."""
        values = self._read(ctx)
        z = self._height_component(values)
        if self._band is not None:
            low, high = self._band
            return ~((z >= low) & (z <= high))
        if self.threshold is None:
            return torch.zeros(z.shape, dtype=torch.bool, device=z.device)
        return _compare(z, self.op, self.threshold)

    def _read(self, ctx: EvalContext) -> torch.Tensor:
        if self.articulation is not None:
            source = ctx.env.articulation(self.articulation).get("base_pos_w")
        else:
            assert self.record_key is not None
            source = ctx.env.record().get(self.record_key)
        return _as_tensor(source, self._source_name)

    def _height_component(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 0:
            raise ConditionError(f"{self._source_name} must be a vector, got a scalar")
        size = int(values.shape[-1])
        if not (-size <= self.dim < size):
            raise ConditionError(
                f"{self._source_name} height index {self.dim} is outside the "
                f"trailing axis of size {size}"
            )
        return values[..., self.dim]
