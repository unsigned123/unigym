"""Termination registry + builders (numpy/torch agnostic, unit-testable).

Yaml usage (top-level ``terminations`` section, optional; entries are
OR-combined — any term firing marks the env "terminated" for
logging/filtering; nothing actually resets in collection mode)::

    terminations:
      - name: fall
        articulation: g1            # REQUIRED for per-articulation terms
        params: {height_threshold: 0.15, tilt_threshold: 0.8}
      - name: base_contact
        articulation: g1
        params: {body_names: "base", force_threshold: 5.0}
      - name: joint_limits
        articulation: g1
        params: {margin: 0.05}

Explicit articulation ownership: every per-articulation term carries an
explicit ``articulation`` (top-level or under ``params``).  There is no
single-binding injection: a missing ``articulation`` fails fast at
construction, whether the run has one binding or several.
"""

from __future__ import annotations

from collections.abc import Callable

from unigym.provenance import (
    OriginTable,
    ProviderInfo,
    ProviderOrigin,
    RegistrySnapshot,
)

from .articulation import ArticulationTermination
from .base import Termination, TerminationState
from .composite import InvertedTermination, TerminationGroup
from .fall import (
    COS_DEFAULT_TILT,
    DEFAULT_HEIGHT_THRESHOLD,
    DEFAULT_TILT_THRESHOLD,
    FallTermination,
)
from .base_contact import BaseContactTermination
from .joint_limits import JointLimitTermination

TERMINATION_REGISTRY: dict[str, Callable[..., Termination]] = {
    "fall": FallTermination,
    "base_contact": BaseContactTermination,
    "joint_limits": JointLimitTermination,
}

_TERMINATION_ORIGINS = OriginTable(
    {
        "fall": ProviderOrigin(kind="core"),
        "base_contact": ProviderOrigin(kind="core"),
        "joint_limits": ProviderOrigin(kind="core"),
    },
)
"""Private provenance side table beside :data:`TERMINATION_REGISTRY`."""

TerminationRegistrySnapshot = RegistrySnapshot[Callable[..., Termination]]
"""The origin-aware termination registry snapshot type."""


def register_termination(
    name: str,
    cls: Callable[..., Termination],
    *,
    origin: ProviderOrigin | None = None,
) -> Callable[..., Termination]:
    """Register one termination class in the process-global registry.

    Args:
        name: Termination name the run config selects in ``terminations[].name``.
        cls: A callable returning a :class:`Termination`.
        origin: Internal/runtime-only provenance; pack authors never pass it.
            A registration without an origin reports
            :meth:`ProviderOrigin.unknown`.

    Returns:
        *cls*, so the helper can be used as a decorator.

    Raises:
        ValueError: The name is malformed or already registered, or *cls* is
            not callable.
        TypeError: *origin* is not a :class:`ProviderOrigin`.
    """
    if (
        not isinstance(name, str)
        or not name
        or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name)
    ):
        raise ValueError(f"termination name {name!r} must be lowercase snake_case")
    if name in TERMINATION_REGISTRY:
        raise ValueError(f"termination {name!r} registered twice")
    if not callable(cls):
        raise ValueError(f"termination {name!r} must be callable")
    if origin is not None and not isinstance(origin, ProviderOrigin):
        raise TypeError(
            f"termination {name!r}: origin must be a ProviderOrigin, "
            f"got {type(origin).__name__}"
        )
    TERMINATION_REGISTRY[name] = cls
    if origin is None:
        _TERMINATION_ORIGINS.remove(name)
    else:
        _TERMINATION_ORIGINS.set(name, origin)
    return cls


def attribute_termination(name: str, origin: ProviderOrigin) -> None:
    """Attach runtime-owned provenance to an already-registered termination.

    This is the pack runtime's activation-window attribution boundary, not
    pack-author API: pack code registers through :func:`register_termination`
    and never passes an origin.  It may only label an entry whose current
    origin is absent or ``unknown``.

    Raises:
        ValueError: The termination is unknown or already has an attributed
            origin.
        TypeError: *origin* is not a :class:`ProviderOrigin`.
    """
    if name not in TERMINATION_REGISTRY:
        raise ValueError(
            f"termination {name!r} is not registered "
            f"(registered terminations: {sorted(TERMINATION_REGISTRY)})"
        )
    if not isinstance(origin, ProviderOrigin):
        raise TypeError(
            f"termination {name!r}: origin must be a ProviderOrigin, "
            f"got {type(origin).__name__}"
        )
    existing = _TERMINATION_ORIGINS.get(name)
    if existing is not None and existing.kind != "unknown":
        raise ValueError(
            f"termination {name!r} already has a {existing.kind!r} origin; "
            "only an unattributed registration may be labeled"
        )
    _TERMINATION_ORIGINS.set(name, origin)


def termination_origin(name: str) -> ProviderOrigin:
    """Return the origin of one registered termination (never ``None``).

    The built-in ``fall``, ``base_contact`` and ``joint_limits`` terminations
    report ``kind="core"``.  A registered termination with no recorded origin
    returns :meth:`ProviderOrigin.unknown`; an unknown name raises a teaching
    ``ValueError``.
    """
    if name not in TERMINATION_REGISTRY:
        raise ValueError(
            f"termination {name!r} is not registered "
            f"(registered terminations: {sorted(TERMINATION_REGISTRY)})"
        )
    return _TERMINATION_ORIGINS.origin(name)


def termination_providers(kind: str | None = None) -> tuple[ProviderInfo, ...]:
    """Return the registered terminations as live :class:`ProviderInfo` records.

    ``kind`` selects the provider kind; the termination registry owns only
    ``"termination"``, so any other kind returns an empty tuple (``None``
    returns every record).  Records are ordered by name.
    """
    if kind is not None and kind != "termination":
        return ()
    return tuple(
        ProviderInfo(
            name=name,
            kind="termination",
            origin=_TERMINATION_ORIGINS.origin(name),
        )
        for name in sorted(TERMINATION_REGISTRY)
    )


def snapshot_termination_registry() -> TerminationRegistrySnapshot:
    """Return a total snapshot of the termination classes and their origins."""
    return RegistrySnapshot.capture(TERMINATION_REGISTRY, _TERMINATION_ORIGINS)


def restore_termination_registry(snapshot: TerminationRegistrySnapshot) -> None:
    """Restore the termination classes and origins to exactly *snapshot*."""
    if not isinstance(snapshot, RegistrySnapshot):
        raise TypeError(
            "restore_termination_registry expects a TerminationRegistrySnapshot, "
            f"got {type(snapshot).__name__}"
        )
    snapshot.restore(TERMINATION_REGISTRY, _TERMINATION_ORIGINS)


def build_terminations(cfg: list | None) -> list[Termination]:
    """Build a list of terminations from the yaml ``terminations`` section.

    Per-articulation terms must carry an explicit ``articulation`` (top-level
    or under ``params``); the term's own base class enforces it.  ``None``/
    empty cfg -> empty list.
    """
    terms: list[Termination] = []
    if not cfg:
        return terms
    for entry in cfg:
        name = entry.get("name")
        if name not in TERMINATION_REGISTRY:
            raise ValueError(
                f"unknown termination '{name}'; " f"available: {sorted(TERMINATION_REGISTRY)}"
            )
        params = dict(entry.get("params", {}))
        explicit = entry.get("articulation")
        if explicit is not None:
            params["articulation"] = explicit
        term = TERMINATION_REGISTRY[name](params, name=name)
        if entry.get("invert", False):  # NOT(term) health conditions
            term = InvertedTermination(term, name=name)
        terms.append(term)
    return terms


def build_termination_group(cfg: list | None, mode: str = "or") -> TerminationGroup:
    """Build an OR (default) TerminationGroup from the yaml section.

    ``None``/empty cfg -> empty group that never fires (no default fall
    group is synthesised).
    """
    return TerminationGroup(build_terminations(cfg), mode=mode)


__all__ = [
    "Termination",
    "ArticulationTermination",
    "TerminationState",
    "TerminationGroup",
    "InvertedTermination",
    "FallTermination",
    "BaseContactTermination",
    "JointLimitTermination",
    "TERMINATION_REGISTRY",
    "TerminationRegistrySnapshot",
    "DEFAULT_HEIGHT_THRESHOLD",
    "DEFAULT_TILT_THRESHOLD",
    "COS_DEFAULT_TILT",
    "attribute_termination",
    "build_terminations",
    "build_termination_group",
    "register_termination",
    "restore_termination_registry",
    "snapshot_termination_registry",
    "termination_origin",
    "termination_providers",
]
