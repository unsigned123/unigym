"""Shared provenance records and registry side-table helpers.

Every live registry entry may carry a :class:`ProviderOrigin` in a private side
table beside the registry's authoritative name -> value mapping.  This module
owns the record shape and the small helpers the registries share:

* :class:`OriginTable` -- the name -> origin side table;
* :class:`RegistrySnapshot` -- values plus origins, captured and restored
  together;
* :func:`attribution` -- the activation-window diff that labels names a pack
  registered while its entrypoint was running.

The module is standard-library only and imports no engine, backend or pack
module.  Provenance is process/diagnostic data: it is never serialized into a
run config, ledger, shard or ``to_config`` output.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, TypeVar

EntryT = TypeVar("EntryT")


@dataclass(frozen=True)
class ProviderOrigin:
    """Where one installed pack or one live registry entry came from.

    ``kind`` is one of:

    * ``"pack"`` -- a pack-level discovery origin; ``face`` and ``item_id`` are
      ``None``;
    * ``"manifest"`` -- a manifest-declared provider item (``face`` and
      ``item_id`` are set, and exactly one of ``resource``/``entrypoint``);
    * ``"pack-code"`` -- registered by pack Python during an activation window;
      ``item_id`` is ``None`` because the registration call carries no item id;
    * ``"core"`` -- a built-in registry entry (for example the ``manual``
      adapter);
    * ``"inline"`` -- a process-local inline provider (``World.inline``);
    * ``"unknown"`` -- a registration outside any attributable window.  This is
      a legal, explicit state, never a guess.

    ``root`` is the resolved pack root as a string.  ``discovered_by`` is the
    discovery source label.  ``location_imported`` records the entry-point
    location seam: an entry-point root may import its target module before any
    manifest is read, while explicit roots and configured content sources do
    not.
    """

    kind: str
    pack: str | None = None
    pack_version: str | None = None
    root: str | None = None
    discovered_by: str | None = None
    face: str | None = None
    item_id: str | None = None
    resource: str | None = None
    entrypoint: str | None = None
    activation_entrypoint: str | None = None
    location_imported: bool = False

    @classmethod
    def unknown(cls) -> ProviderOrigin:
        """Return the explicit unattributed-registration origin."""
        return cls(kind="unknown")

    @classmethod
    def inline(cls) -> ProviderOrigin:
        """Return the ``World.inline`` private-provider origin."""
        return cls(kind="inline")


@dataclass(frozen=True)
class ProviderInfo:
    """One provider entry: its kind, origin and (for installed items) catalog facts.

    ``active`` describes a live registry query: ``True`` for an entry returned
    by a registry query and ``None`` for an installed-catalog query.  A live
    entry leaves the installed-only fields (``pack``, ``pack_version``,
    ``face``, ``resource``, ``entrypoint``) unset; an installed entry fills
    them from the manifest and leaves ``active=None``.
    """

    name: str
    kind: str | None
    origin: ProviderOrigin
    active: bool | None = field(default=True, kw_only=True)
    pack: str | None = None
    pack_version: str | None = None
    face: str | None = None
    resource: str | None = None
    entrypoint: str | None = None


class OriginTable:
    """A private name -> :class:`ProviderOrigin` side table.

    A registry owns one table beside its authoritative value mapping.  The
    table owns no registry and never decides when an entry is registered.
    """

    def __init__(self, entries: Mapping[str, ProviderOrigin] | None = None) -> None:
        self._entries: dict[str, ProviderOrigin] = dict(entries or {})

    def set(self, name: str, origin: ProviderOrigin) -> None:
        """Record *origin* for *name*."""
        if not isinstance(origin, ProviderOrigin):
            raise TypeError(
                f"origin for {name!r} must be a ProviderOrigin, got {type(origin).__name__}"
            )
        self._entries[name] = origin

    def get(self, name: str) -> ProviderOrigin | None:
        """Return the recorded origin for *name*, or ``None`` when absent."""
        return self._entries.get(name)

    def origin(self, name: str) -> ProviderOrigin:
        """Return the recorded origin for *name*, or ``unknown()`` when absent."""
        return self._entries.get(name, ProviderOrigin.unknown())

    def remove(self, name: str) -> None:
        """Drop any recorded origin for *name* (absent names are a no-op)."""
        self._entries.pop(name, None)

    def snapshot(self) -> dict[str, ProviderOrigin]:
        """Return an independent copy of the current table."""
        return dict(self._entries)

    def restore(self, snapshot: Mapping[str, ProviderOrigin]) -> None:
        """Replace the table with exactly *snapshot*."""
        self._entries.clear()
        self._entries.update(snapshot)

    def names(self) -> tuple[str, ...]:
        """Return the recorded names in insertion order."""
        return tuple(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True)
class RegistrySnapshot(Generic[EntryT]):
    """A total snapshot of one registry's values and origins.

    ``restore`` is exact: it drops entries and origins registered after the
    snapshot and reinstates the captured ones.  The two mappings are read-only
    views so a snapshot cannot be mutated after capture.
    """

    entries: Mapping[str, EntryT]
    origins: Mapping[str, ProviderOrigin]

    @classmethod
    def capture(
        cls, entries: Mapping[str, EntryT], origins: OriginTable
    ) -> RegistrySnapshot[EntryT]:
        """Capture *entries* and *origins* as one immutable snapshot."""
        return cls(
            entries=MappingProxyType(dict(entries)),
            origins=MappingProxyType(origins.snapshot()),
        )

    def restore(self, entries: MutableMapping[str, EntryT], origins: OriginTable) -> None:
        """Restore *entries* and *origins* to exactly this snapshot."""
        entries.clear()
        entries.update(self.entries)
        origins.restore(self.origins)


def attribution(
    before: Mapping[str, object],
    after: Mapping[str, object],
    origin: ProviderOrigin,
) -> dict[str, ProviderOrigin]:
    """Label every name present in *after* but not in *before* with *origin*.

    This is the activation-window diff: it states factually that a name was
    registered while a pack entrypoint was running.  It is not per-call
    attribution and does not inspect a stack, context variable or import hook.
    """
    return {name: origin for name in after if name not in before}


__all__ = [
    "OriginTable",
    "ProviderInfo",
    "ProviderOrigin",
    "RegistrySnapshot",
    "attribution",
]
