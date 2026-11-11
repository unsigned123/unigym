"""Public pack catalog over validated ``unigym.pack/1`` manifests.

The catalog is a pure, immutable view of what a set of manifests declares.
It preserves pack order and per-pack item order, rejects duplicate pack names
and duplicate ``(face, id)`` pairs, and never discovers, scans or imports a
pack.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import PackError
from .schema import FACE_NAMES, PackManifestV1, ProvidedItem


@dataclass(frozen=True)
class CatalogEntry:
    """One provided item flattened into the cross-pack catalog."""

    pack: str
    face: str
    id: str
    resource: str | None
    entrypoint: str | None
    kind: str | None


def _entry(pack: str, face: str, item: ProvidedItem) -> CatalogEntry:
    """Flatten one provided item into a catalog entry."""
    return CatalogEntry(
        pack=pack,
        face=face,
        id=item.id,
        resource=item.resource,
        entrypoint=item.entrypoint,
        kind=item.kind,
    )


def _require_face(face: str) -> None:
    """Reject a name outside the declared face vocabulary."""
    if face not in FACE_NAMES:
        raise PackError(f"unknown pack face {face!r}; known faces: {list(FACE_NAMES)}")


@dataclass(frozen=True)
class PackCatalog:
    """An ordered, duplicate-free view over validated manifests."""

    manifests: tuple[PackManifestV1, ...]

    def __post_init__(self) -> None:
        """Freeze the manifest tuple and validate catalog uniqueness."""
        object.__setattr__(self, "manifests", tuple(self.manifests))
        seen_packs: set[str] = set()
        seen_items: dict[tuple[str, str], str] = {}
        for manifest in self.manifests:
            if manifest.name in seen_packs:
                raise PackError(f"duplicate pack name {manifest.name!r} in catalog")
            seen_packs.add(manifest.name)
            for face, items in manifest.provides.items():
                for item in items:
                    key = (face, item.id)
                    owner = seen_items.get(key)
                    if owner is not None:
                        raise PackError(
                            f"duplicate {face!r} id {item.id!r} in catalog: "
                            f"provided by packs {owner!r} and {manifest.name!r}"
                        )
                    seen_items[key] = manifest.name

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


__all__ = ["CatalogEntry", "PackCatalog"]
