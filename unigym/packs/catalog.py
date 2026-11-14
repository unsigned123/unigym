"""Public pack catalog over validated ``unigym.pack/1`` manifests.

The catalog is a pure, immutable view of what a set of manifests declares.
It preserves pack order and per-pack item order, rejects duplicate pack names
and duplicate ``(face, id)`` pairs, and never discovers, scans or imports a
pack.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from unigym._typing import JsonValue

from . import PackError
from .schema import FACE_NAMES, PackManifestV1, ProvidedItem, freeze_provider_params


@dataclass(frozen=True)
class CatalogEntry:
    """One provided item flattened into the cross-pack catalog.

    ``params`` is the scene-provider parameter schema, deep-frozen by the same
    boundary the manifest uses (read-only mappings, tuple sequences); it is
    ``None`` for every other item.
    """

    pack: str
    face: str
    id: str
    resource: str | None
    entrypoint: str | None
    kind: str | None
    params: Mapping[str, Mapping[str, JsonValue]] | None = None

    def __post_init__(self) -> None:
        """Deep-freeze the optional provider parameter schema (defense in depth)."""
        if self.params is not None:
            object.__setattr__(self, "params", freeze_provider_params(self.params))


def _entry(pack: str, face: str, item: ProvidedItem) -> CatalogEntry:
    """Flatten one provided item into a catalog entry."""
    return CatalogEntry(
        pack=pack,
        face=face,
        id=item.id,
        resource=item.resource,
        entrypoint=item.entrypoint,
        kind=item.kind,
        params=item.params,
    )


def _require_face(face: str) -> None:
    """Reject a name outside the declared face vocabulary."""
    if face not in FACE_NAMES:
        raise PackError(f"unknown pack face {face!r}; known faces: {list(FACE_NAMES)}")


def _label(origins: Mapping[str, str] | None, name: str) -> str:
    """The discovery label for *name*, or ``"unknown"`` when not recorded."""
    if origins is None:
        return "unknown"
    return origins.get(name, "unknown")


def validate_unique(
    manifests: Iterable[PackManifestV1], origins: Mapping[str, str] | None = None
) -> None:
    """Reject duplicate pack names and duplicate ``(face, id)`` pairs.

    The error names both packs, both pack roots and both discovery labels
    (``origins`` keys pack names; ``None`` reports ``"unknown"``).  There is no
    first-match resolution.  This is the one ambiguity check shared by
    :class:`PackCatalog` and :func:`unigym.packs.runtime.discover`.
    """
    manifests = tuple(manifests)
    seen_packs: dict[str, PackManifestV1] = {}
    for manifest in manifests:
        owner = seen_packs.get(manifest.name)
        if owner is not None:
            raise PackError(
                f"duplicate pack name {manifest.name!r} discovered at roots "
                f"{owner.root} and {manifest.root} (discovered_by "
                f"{_label(origins, owner.name)!r} and "
                f"{_label(origins, manifest.name)!r})"
            )
        seen_packs[manifest.name] = manifest
    seen_items: dict[tuple[str, str], tuple[str, PackManifestV1]] = {}
    for manifest in manifests:
        for face, items in manifest.provides.items():
            for item in items:
                key = (face, item.id)
                owner = seen_items.get(key)
                if owner is not None:
                    owner_pack, owner_manifest = owner
                    raise PackError(
                        f"duplicate {face!r} id {item.id!r} in catalog: provided by "
                        f"packs {owner_pack!r} (root {owner_manifest.root}, "
                        f"discovered_by {_label(origins, owner_pack)!r}) and "
                        f"{manifest.name!r} (root {manifest.root}, discovered_by "
                        f"{_label(origins, manifest.name)!r})"
                    )
                seen_items[key] = (manifest.name, manifest)


@dataclass(frozen=True)
class PackCatalog:
    """An ordered, duplicate-free view over validated manifests."""

    manifests: tuple[PackManifestV1, ...]

    def __post_init__(self) -> None:
        """Freeze the manifest tuple and validate catalog uniqueness."""
        object.__setattr__(self, "manifests", tuple(self.manifests))
        validate_unique(self.manifests)

    def packs(self) -> tuple[str, ...]:
        """Return pack names in declaration order."""
        return tuple(manifest.name for manifest in self.manifests)

    def get(self, name: str) -> PackManifestV1:
        """Return the named manifest or fail with the known pack names."""
        for manifest in self.manifests:
            if manifest.name == name:
                return manifest
        known = sorted(manifest.name for manifest in self.manifests)
        raise PackError(f"unknown pack {name!r}; known packs: {known}")

    def entries(self, face: str | None = None) -> tuple[CatalogEntry, ...]:
        """Return provided items, optionally restricted to one declared face."""
        if face is not None:
            _require_face(face)
        entries: list[CatalogEntry] = []
        for manifest in self.manifests:
            for item_face, items in manifest.provides.items():
                if face is not None and item_face != face:
                    continue
                for item in items:
                    entries.append(_entry(manifest.name, item_face, item))
        return tuple(entries)

    def entry(self, face: str, id: str) -> CatalogEntry:
        """Return the entry for an exact ``(face, id)`` pair.

        This is a pure lookup over already-loaded manifests: it validates
        ``face`` against the declared vocabulary and searches that face. It
        never discovers, scans, parses or imports a pack.

        Raises:
            PackError: The face is not declared, or no item on that face has
                the requested id. The message names the input and the known
                choices.
        """
        _require_face(face)
        items = self.entries(face)
        for entry in items:
            if entry.id == id:
                return entry
        known = sorted(entry.id for entry in items)
        raise PackError(f"unknown id {id!r} on pack face {face!r}; known ids: {known}")


__all__ = ["CatalogEntry", "PackCatalog", "validate_unique"]
