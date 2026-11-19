"""Typed contact body references (breaking Scheme A).

A :class:`ContactBodyRef` is the ONLY way a contact group names the bodies it
observes.  There is no string-list legacy form, no ``auto`` owner, no
``{prop: ...}`` / ``{contact_group: ...}`` reference and no implicit regex
union across owners: every address is explicit and typed.

Owners:

* ``robot``       -- an articulation body / body set;
* ``object``      -- a declared free object prop;
* ``support``     -- a scene support body (table / counter / static scene);
* ``fixture``     -- a declared articulated fixture prop;
* ``environment`` -- a native environment body (world / ground / explicit path).

The ref is backend-neutral: resolution happens in
:class:`unigym.contacts.inventory.ContactInventory` over backend-native rows.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Optional

__all__ = [
    "MATCHES",
    "OWNERS",
    "ContactBodyRef",
    "is_wildcard",
]

OWNERS = ("robot", "object", "support", "fixture", "environment")
"""The closed owner vocabulary (declaration order is stable for metadata)."""

MATCHES = ("regex", "exact")
"""Body name matching modes; ``regex`` is the default (fullmatch)."""


def is_wildcard(value: Optional[str], match: str = "regex") -> bool:
    """True when *value* means "every body in this scope".

    ``body: "*"`` is the wildcard in regex mode; ``match: exact`` makes every
    string (including ``"*"``) literal.
    """
    return value == "*" and match == "regex"


def _check_string(value, what: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{what} must be a non-empty, trimmed string or None, got {value!r}")
    return value


@dataclass(frozen=True)
class ContactBodyRef:
    """One explicit body address inside a contact group.

    Exactly one address form is legal per owner:

    * ``robot``: exactly one of ``body`` / ``body_set``; ``articulation`` is
      required when the run declares more than one articulation; ``name`` and
      ``path`` are forbidden.
    * ``object`` / ``support`` / ``fixture``: ``name`` required, ``body``
      optional (the scope root body); ``articulation`` / ``body_set`` /
      ``path`` are forbidden.
    * ``environment``: exactly one of ``body`` / ``path``; ``name`` /
      ``body_set`` / ``articulation`` are forbidden.

    ``body: "*"`` selects every body in the scope.  ``match: exact`` makes the
    body literal.  ``path`` is the backend-native escape hatch.
    """

    owner: str
    articulation: Optional[str] = None
    name: Optional[str] = None
    body: Optional[str] = None
    body_set: Optional[str] = None
    path: Optional[str] = None
    match: str = "regex"

    def __post_init__(self) -> None:
        if self.owner not in OWNERS:
            raise ValueError(f"contact body ref owner must be one of {OWNERS}, got {self.owner!r}")
        if self.match not in MATCHES:
            raise ValueError(f"contact body ref match must be one of {MATCHES}, got {self.match!r}")
        for key in ("articulation", "name", "body", "body_set", "path"):
            object.__setattr__(self, key, _check_string(getattr(self, key), f"contact ref {key}"))
        if self.owner == "robot":
            self._validate_robot()
        elif self.owner == "environment":
            self._validate_environment()
        else:
            self._validate_scoped_owner()

    def _forbidden(self, *keys: str) -> None:
        present = [key for key in keys if getattr(self, key) is not None]
        if present:
            raise ValueError(
                f"contact body ref owner {self.owner!r} forbids "
                f"{present}; ref={self.to_dict()!r}"
            )

    def _validate_robot(self) -> None:
        self._forbidden("name", "path")
        if (self.body is None) == (self.body_set is None):
            raise ValueError(
                "contact body ref owner 'robot' requires exactly one of "
                f"body or body_set, got body={self.body!r}, body_set={self.body_set!r}"
            )

    def _validate_scoped_owner(self) -> None:
        if self.name is None:
            raise ValueError(
                f"contact body ref owner {self.owner!r} requires name "
                f"(the declared prop/support scope), got {self.to_dict()!r}"
            )
        self._forbidden("articulation", "body_set", "path")

    def _validate_environment(self) -> None:
        self._forbidden("articulation", "name", "body_set")
        if (self.body is None) == (self.path is None):
            raise ValueError(
                "contact body ref owner 'environment' requires exactly one of "
                f"body or path, got body={self.body!r}, path={self.path!r}"
            )

    # ---- parsing / serialization ---------------------------------------- #
    @classmethod
    def from_mapping(cls, value) -> ContactBodyRef:
        """Normalize a mapping (or an existing ref) into a ``ContactBodyRef``."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(
                "contact body ref must be a mapping "
                "{owner: robot|object|support|fixture|environment, ...}, got "
                f"{type(value).__name__}: {value!r}"
            )
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                f"contact body ref declares unknown keys {unknown}; allowed keys "
                f"are {sorted(allowed)}"
            )
        return cls(**dict(value))

    def to_dict(self) -> dict:
        """JSON-safe mapping (only explicitly supplied address fields)."""
        out = {"owner": self.owner}
        for key in ("articulation", "name", "body", "body_set", "path"):
            entry = getattr(self, key)
            if entry is not None:
                out[key] = entry
        if self.match != "regex":
            out["match"] = self.match
        return out

    def scope(self) -> Optional[str]:
        """The resolution scope (articulation name or declared prop/support name)."""
        if self.owner == "robot":
            return self.articulation
        if self.owner == "environment":
            return None
        return self.name

    def label(self) -> str:
        """Compact human label used in teaching errors."""
        if self.owner == "robot":
            address = self.body_set if self.body_set is not None else self.body
            scope = f"{self.articulation}." if self.articulation else ""
            return f"robot:{scope}{address}"
        if self.owner == "environment":
            return f"environment:{self.path or self.body}"
        return f"{self.owner}:{self.name}" + (f".{self.body}" if self.body else "")
