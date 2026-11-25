"""Torch-free base protocol for canonical evaluation conditions.

A condition is a live, read-only, vector-scoped boolean predicate over an
:class:`~unigym.evaluation.EvalContext`.  It owns no result, effect, message or
per-env state; domain adapters own termination facts and watcher results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

    from unigym.evaluation import EvalContext

__all__ = ["Condition", "ConditionError"]


class ConditionError(ValueError):
    """A condition could not read its declared evaluation source."""


def _as_tensor(value: object, name: str) -> torch.Tensor:
    """Return *value* as a tensor, or raise :class:`ConditionError`."""
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if value is None:
        raise ConditionError(f"{name} is missing from the evaluation record")
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise ConditionError(f"{name} is not a tensor value: {type(value).__name__}") from exc


class Condition(Protocol):
    """A live, read-only, effect-free boolean condition.

    Implementations return a ``(ctx.env.scope_size,)`` bool mask on the context
    device.  They never mutate the environment, close the context or produce a
    domain result.
    """

    def evaluate(self, ctx: EvalContext) -> torch.Tensor:
        """Return the per-scope bool mask for *ctx*."""
        ...
