"""Typed failures for the on-demand runtime query face.

Query is deliberately separate from ``env.obs``: an unsupported operation, an
unknown entity name, a malformed request or a read after close fails loudly
with one of these errors instead of degrading to a default value.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "QueryCapabilityError",
    "QueryError",
    "QueryReferenceError",
    "QueryShapeError",
    "QueryStaleError",
]


class QueryError(RuntimeError):
    """Base class for runtime-query failures."""


class QueryCapabilityError(QueryError):
    """A backend cannot answer one query operation at its declared support."""

    def __init__(self, backend: str, op: str, support: str, detail: str = "") -> None:
        """Record the backend, operation, support level and optional detail."""
        self.backend = backend
        self.op = op
        self.support = support
        self.detail = detail
        message = f"backend {backend!r} cannot query {op!r} (support={support})"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class QueryReferenceError(QueryError):
    """An entity name is unknown or ambiguous for the requested operation."""

    def __init__(
        self,
        op: str,
        reference: str | None,
        available: Iterable[str],
        detail: str = "",
    ) -> None:
        """Record the operation, missing reference and available names."""
        self.op = op
        self.reference = reference
        self.available = tuple(available)
        self.detail = detail
        message = (
            f"query {op!r} cannot resolve reference {reference!r}; "
            f"available: {list(self.available)}"
        )
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)


class QueryShapeError(QueryError, ValueError):
    """A request or response tensor has an invalid shape, dtype or value."""

    def __init__(self, field: str, expected: str, actual: str) -> None:
        """Record the field name, the expected shape and the actual shape."""
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(f"{field} must be {expected}, got {actual}")


class QueryStaleError(QueryError):
    """A query handle was used after the run, env or query was closed."""
