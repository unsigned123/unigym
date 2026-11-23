"""Canonical tilt condition.

``TiltCondition`` reads projected gravity (``gravity_b``) or a quaternion
(``base_quat_w`` style record key) and fires when the body tilt exceeds a
threshold.  Gravity thresholds are radians by default; quaternion thresholds
are degrees by default, matching the termination and watcher callers.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch

from .base import ConditionError, _as_tensor

if TYPE_CHECKING:
    from unigym.evaluation import EvalContext

__all__ = ["TiltCondition"]

TiltSource = Literal["gravity_b", "base_quat_w"]
TiltUnit = Literal["rad", "deg"]
MissingPolicy = Literal["raise", "false"]


class TiltCondition:
    """A vector-scoped tilt predicate.

    Args:
        source: ``gravity_b`` (projected gravity) or ``base_quat_w``
            (quaternion).
        articulation: Articulation whose leaf is read.
        record_key: Explicit record leaf key (mutually exclusive with
            *articulation*).
        threshold: Tilt threshold in *unit*; ``None`` disables the condition.
        unit: ``rad`` or ``deg``; defaults to the natural unit for *source*.
        missing: ``raise`` or ``false`` when the source leaf is absent.
    """

    def __init__(
        self,
        *,
        source: TiltSource = "gravity_b",
        articulation: str | None = None,
        record_key: str | None = None,
        threshold: float | None = None,
        unit: TiltUnit | None = None,
        missing: MissingPolicy = "raise",
    ) -> None:
        if source not in ("gravity_b", "base_quat_w"):
            raise ValueError(
                f"TiltCondition source must be gravity_b or base_quat_w, got {source!r}"
            )
        if (articulation is None) == (record_key is None):
            raise ValueError("TiltCondition requires exactly one of articulation or record_key")
        if articulation is not None and (not isinstance(articulation, str) or not articulation):
            raise ValueError("TiltCondition articulation must be a non-empty string")
        if record_key is not None and (not isinstance(record_key, str) or not record_key):
            raise ValueError("TiltCondition record_key must be a non-empty string")
        if missing not in ("raise", "false"):
            raise ValueError(f"TiltCondition missing must be raise or false, got {missing!r}")
        natural_unit: TiltUnit = "rad" if source == "gravity_b" else "deg"
        resolved_unit = natural_unit if unit is None else unit
        if resolved_unit not in ("rad", "deg"):
            raise ValueError(f"TiltCondition unit must be rad or deg, got {resolved_unit!r}")
        self.source = source
        self.articulation = articulation
        self.record_key = record_key
        self.threshold = None if threshold is None else float(threshold)
        self.unit = resolved_unit
        self.missing = missing
        self._source_name = (
            f"articulation/{articulation}/{source}" if articulation is not None else str(record_key)
        )

    def evaluate(self, ctx: EvalContext) -> torch.Tensor:
        """Return the per-scope bool mask for *ctx*."""
        if self.threshold is None:
            return self._empty(ctx)
        values = self._read(ctx)
        if values is None:
            return self._empty(ctx)
        if self.source == "gravity_b":
            return self._gravity(values)
        return self._quaternion(values)

    def _read(self, ctx: EvalContext) -> torch.Tensor | None:
        if self.articulation is not None:
            source = ctx.env.articulation(self.articulation).get(self.source)
        else:
            assert self.record_key is not None
            source = ctx.env.record().get(self.record_key)
        if source is None and self.missing == "false":
            return None
        return _as_tensor(source, self._source_name)

    def _gravity(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 0 or int(values.shape[-1]) < 3:
            raise ConditionError(f"{self._source_name} must have a trailing size >= 3")
        assert self.threshold is not None
        threshold = self.threshold if self.unit == "rad" else math.radians(self.threshold)
        return values[..., 2] > -math.cos(threshold)

    def _quaternion(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 0 or int(values.shape[-1]) != 4:
            raise ConditionError(f"{self._source_name} must have a trailing quaternion size of 4")
        q = values.to(dtype=torch.float64)
        norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-9)
        q = q / norm
        y = q[..., 2]
        z = q[..., 3]
        up_z = 1.0 - 2.0 * (y * y + z * z)
        degrees = torch.rad2deg(torch.acos(torch.clamp(up_z, -1.0, 1.0)))
        assert self.threshold is not None
        threshold = self.threshold if self.unit == "deg" else math.degrees(self.threshold)
        return degrees > threshold

    def _empty(self, ctx: EvalContext) -> torch.Tensor:
        return torch.zeros(ctx.env.scope_size, dtype=torch.bool, device=ctx.env.device)
