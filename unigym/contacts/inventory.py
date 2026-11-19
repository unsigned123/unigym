"""Backend-neutral contact body inventory and explicit reference resolution.

The inventory is a flat table of addressable, contact-capable body addresses:
robot links, declared prop bodies, scene supports and native environment
bodies.  A :class:`~unigym.contacts.refs.ContactBodyRef` resolves against this
table with NO global regex union: each ref is resolved independently and the
resolved lists concatenate in declaration order (backend-native order within a
ref), which is exactly the B/F axis order of a contact group.

Errors carry the candidate canonical keys of the relevant scope so a typo is
actionable without asset spelunking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .refs import ContactBodyRef, is_wildcard

__all__ = [
    "ContactBodyAddress",
    "ContactInventory",
    "resolve_refs",
]


@dataclass(frozen=True)
class ContactBodyAddress:
    """One backend-native addressable body.

    ``key`` is the canonical, owner-qualified address used in errors and
    metadata: ``robot/<articulation>/<body>``, ``object/<name>/<body>``,
    ``support/<name>/<body>``, ``fixture/<name>/<body>`` or
    ``environment/<native>``.
    """

    owner: str
    scope: Optional[str]
    body: str
    native: str
    prim_path: Optional[str]
    collision: bool
    contact_capable: bool
    key: str
    root: bool = False
    """True for the scope root body (default target of an omitted ``body``)."""

    def matches_body(self, value: str, match: str = "regex") -> bool:
        """True when this row's body matches *value* under *match*."""
        if match == "exact":
            return self.body == value
        try:
            return re.fullmatch(value, self.body) is not None
        except re.error as exc:  # pragma: no cover - surfaced as ValueError below
            raise ValueError(f"invalid contact body regex {value!r}: {exc}") from exc


@dataclass(frozen=True)
class ContactInventory:
    """One backend's resolved contact address table.

    ``body_sets`` carries declared named robot body sets as
    ``(articulation, set_name, native_bodies)`` triples so a robot
    ``body_set`` ref can expand without a global body-name search.
    """

    backend: str
    rows: tuple[ContactBodyAddress, ...]
    body_sets: tuple[tuple[str, str, tuple[str, ...]], ...] = ()

    # ---- helpers --------------------------------------------------------- #
    def body_set(self, articulation: Optional[str], name: str) -> Optional[tuple[str, ...]]:
        """Native bodies of one declared body set (``None`` when unknown)."""
        for scope, set_name, bodies in self.body_sets:
            if set_name != name:
                continue
            if articulation is None or scope == articulation:
                return bodies
        return None

    def add(self, rows: Iterable[ContactBodyAddress]) -> ContactInventory:
        """A new inventory with *rows* appended (keeps declaration order)."""
        return ContactInventory(self.backend, self.rows + tuple(rows), self.body_sets)

    def rows_for(self, owner: str) -> list[ContactBodyAddress]:
        """Rows of one owner (inventory order)."""
        return [row for row in self.rows if row.owner == owner]

    def robot_scopes(self) -> list[str]:
        """Distinct robot articulation scopes in inventory order."""
        scopes: list[str] = []
        for row in self.rows:
            if row.owner == "robot" and row.scope is not None and row.scope not in scopes:
                scopes.append(row.scope)
        return scopes

    def all_keys(self) -> list[str]:
        """Every canonical row key (inventory order)."""
        return [row.key for row in self.rows]

    # ---- resolution ------------------------------------------------------ #
    def resolve(self, ref) -> list[ContactBodyAddress]:
        """Resolve ONE explicit ref to addresses in backend-native order."""
        ref = ContactBodyRef.from_mapping(ref)
        if ref.owner == "robot":
            return self._resolve_robot(ref)
        if ref.owner == "environment":
            return self._resolve_environment(ref)
        return self._resolve_scoped(ref)

    def _fail(self, ref: ContactBodyRef, message: str, pool: Sequence[ContactBodyAddress]) -> None:
        candidates = [row.key for row in pool]
        raise ValueError(
            f"contact body ref {ref.label()!r}: {message}; candidate canonical "
            f"keys: {candidates}"
        )

    def _match(
        self,
        ref: ContactBodyRef,
        pool: Sequence[ContactBodyAddress],
        *,
        body: Optional[str],
        match: str,
    ) -> list[ContactBodyAddress]:
        if body is None:
            return [row for row in pool if row.root]
        if is_wildcard(body, match):
            return list(pool)
        selected = [row for row in pool if row.matches_body(body, match)]
        return selected

    def _resolve_robot(self, ref: ContactBodyRef) -> list[ContactBodyAddress]:
        scopes = self.robot_scopes()
        if ref.articulation is None:
            if len(scopes) != 1:
                pool = self.rows_for("robot")
                self._fail(
                    ref,
                    f"owner 'robot' must name articulation when the run declares " f"{scopes}",
                    pool,
                )
            articulation = scopes[0]
        else:
            articulation = ref.articulation
            if articulation not in scopes:
                self._fail(
                    ref,
                    f"unknown articulation {articulation!r} (declared: {scopes})",
                    self.rows_for("robot"),
                )
        pool = [row for row in self.rows if row.owner == "robot" and row.scope == articulation]
        if ref.body_set is not None:
            bodies = self.body_set(articulation, ref.body_set)
            if bodies is None:
                self._fail(
                    ref,
                    f"unknown body_set {ref.body_set!r} on articulation " f"{articulation!r}",
                    pool,
                )
            selected = [row for row in pool if row.body in set(bodies)]
        else:
            selected = self._match(ref, pool, body=ref.body, match=ref.match)
        if not selected:
            self._fail(ref, "matched no bodies in scope", pool)
        return selected

    def _resolve_scoped(self, ref: ContactBodyRef) -> list[ContactBodyAddress]:
        pool = [row for row in self.rows if row.owner == ref.owner and row.scope == ref.name]
        if not pool:
            # No row with this owner exists at all: show the full inventory so
            # the author can see the actual declared owners/scopes.
            self._fail(ref, f"unknown {ref.owner} scope {ref.name!r}", self.rows)
        selected = self._match(ref, pool, body=ref.body, match=ref.match)
        if not selected:
            self._fail(ref, "matched no bodies in scope", pool)
        return selected

    def _resolve_environment(self, ref: ContactBodyRef) -> list[ContactBodyAddress]:
        pool = self.rows_for("environment")
        if not pool:
            self._fail(ref, "no environment bodies are declared by this run", pool)
        if ref.path is not None:
            selected = [row for row in pool if row.prim_path == ref.path or row.native == ref.path]
        else:
            selected = self._match(ref, pool, body=ref.body, match=ref.match)
        if not selected:
            self._fail(ref, "matched no environment bodies", pool)
        return selected


def resolve_refs(
    inventory: ContactInventory,
    refs: Iterable,
    *,
    axis: str,
    group: str,
) -> list[ContactBodyAddress]:
    """Resolve refs in declaration order and reject duplicate addresses.

    ``axis`` is ``sensor_refs`` or ``filter_refs`` (teaching-error text).
    Within one ref the inventory order is backend-native; refs concatenate in
    declaration order and are never unioned or reordered.
    """
    out: list[ContactBodyAddress] = []
    for index, ref in enumerate(refs):
        resolved = inventory.resolve(ref)
        for row in resolved:
            if row in out:
                raise ValueError(
                    f"contact group {group!r}: {axis}[{index}] resolves duplicate "
                    f"body {row.key!r}; each resolved body must appear once"
                )
            out.append(row)
    return out
