"""Read-only, boundary-scoped evaluation context for live runs.

Termination and benchmark scoring consume this context; watcher convergence is
deferred, so watchers keep their existing ``WatchContext`` and ``env``
expression surface.  The public surface is minimal and typed: :class:`EvalContext`,
:class:`EnvView`, the read-only :class:`EvalQuery` adapter and the evaluation
errors.  No accessor returns a raw runtime ``Query``; ``EnvView.query`` returns
the scoped :class:`EvalQuery` subset.

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
