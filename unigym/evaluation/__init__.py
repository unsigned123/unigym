"""Read-only, boundary-scoped evaluation context for live runs.

Termination and watchers consume this context.  Termination builds one view
per callback and closes it after the callback; watchers retain one
vector-scoped view per tick until the next tick replaces it.  The public
surface is minimal and typed: :class:`EvalContext`, :class:`EnvView`, the
read-only :class:`EvalQuery` adapter and the evaluation errors.  No accessor
returns a raw runtime ``Query``; ``EnvView.query`` returns the scoped
:class:`EvalQuery` subset.

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
from .step import EnvStepContext

__all__ = [
    "EnvStepContext",
    "EnvView",
    "EvalContext",
    "EvalQuery",
    "EvalQueryUnavailableError",
    "EvalViewError",
    "EvalViewStaleError",
]
