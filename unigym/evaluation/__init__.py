"""Read-only, boundary-scoped evaluation context for live runs.

The module is additive: it defines the context that termination, benchmark
scoring and (later) watchers consume, without switching any existing consumer.
See ``docs/contracts/evaluation_contract.md`` for lifetime, scope, query
subset and borrowed-value rules.
"""

from __future__ import annotations

from .context import (
    EnvView,
    EvalContext,
    EvalQueryUnavailableError,
    EvalViewError,
    EvalViewStaleError,
)
from .query import EvalQuery

__all__ = [
    "EnvView",
    "EvalContext",
    "EvalQuery",
    "EvalQueryUnavailableError",
    "EvalViewError",
    "EvalViewStaleError",
]
