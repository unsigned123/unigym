"""Temporary internal catalog adapter over the stored pack model.

:class:`PackSet` is the public ordered index over :class:`~unigym.packs.schema.Pack`
values.  The runtime still exposes this small compatibility view because
existing consumers read ``runtime.catalog``; it holds references to the same
``Pack``/``PackItem`` objects and never copies or flattens item data.

This module is internal for the C1 transition.  C7 deletes it when the runtime
becomes the loader and consumers use :class:`~unigym.packs.packset.PackSet`
directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from . import PackError
from .schema import FACE_NAMES, Face, Pack, PackItem, coerce_face


def _label(origins: Mapping[str, str] | None, pack: Pack) -> str:
    """The discovery label for *pack*, or ``"unknown"`` when not recorded."""
    if origins is not None:
        return origins.get(pack.name, "unknown")
    return pack.origin.discovered_by or "unknown"


def validate_unique(packs: Iterable[Pack], origins: Mapping[str, str] | None = None) -> None:
    """Reject duplicate pack names and duplicate ``(face, name)`` pairs.

    The error names both packs, both pack roots and both discovery labels
    (``origins`` keys pack names; otherwise each pack's own origin is used).
    There is no first-match resolution.  This is the one ambiguity check
    shared by :class:`PackCatalog`,
    :class:`~unigym.packs.packset.PackSet` and discovery.
    """
    materialized = tuple(packs)
    seen_packs: dict[str, Pack] = {}
    for pack in materialized:
        owner = seen_packs.get(pack.name)
        if owner is not None:
            raise PackError(
                f"duplicate pack name {pack.name!r} discovered at roots "
                f"{owner.root} and {pack.root} (discovered_by "
                f"{_label(origins, owner)!r} and {_label(origins, pack)!r})"
            )
        seen_packs[pack.name] = pack
    seen_items: dict[tuple[Face, str], tuple[str, Pack]] = {}
    for pack in materialized:
        for item in pack.items:
            key = (item.face, item.name)
            owner = seen_items.get(key)
            if owner is not None:
                owner_pack, owner_record = owner
                raise PackError(
                    f"duplicate {item.face.value!r} id {item.name!r} in catalog: "
                    f"provided by packs {owner_pack!r} (root {owner_record.root}, "
                    f"discovered_by {_label(origins, owner_record)!r}) and "
                    f"{pack.name!r} (root {pack.root}, discovered_by "
                    f"{_label(origins, pack)!r})"
                )
            seen_items[key] = (pack.name, pack)


@dataclass(frozen=True)
class PackCatalog:
    """An ordered, duplicate-free compatibility view over stored packs."""

    manifests: tuple[Pack, ...]

    def __post_init__(self) -> None:
        """Freeze the pack tuple and validate catalog uniqueness."""
        object.__setattr__(self, "manifests", tuple(self.manifests))
        validate_unique(self.manifests)

    def packs(self) -> tuple[str, ...]:
        """Return pack names in declaration order."""
        return tuple(pack.name for pack in self.manifests)

    def get(self, name: str) -> Pack:
        """Return the named pack or fail with the known pack names."""
        for pack in self.manifests:
            if pack.name == name:
                return pack
        known = sorted(pack.name for pack in self.manifests)
        raise PackError(f"unknown pack {name!r}; known packs: {known}")

    def entries(self, face: str | Face | None = None) -> tuple[PackItem, ...]:
        """Return stored items, optionally restricted to one declared face."""
        face_value = None if face is None else coerce_face(face)
        return tuple(
            item
            for pack in self.manifests
            for item in pack.items
            if face_value is None or item.face == face_value
        )

    def entry(self, face: str | Face | None, id: str) -> PackItem:
        """Return the item for an exact ``(face, id)`` pair.

        This is a pure lookup over already-loaded packs: it validates ``face``
        against the declared vocabulary and searches that face.  It never
        discovers, scans, parses or imports a pack.

        Raises:
            PackError: The face is not declared, or no item on that face has
                the requested id. The message names the input and the known
                choices.
        """
        if face is None:
            raise PackError(f"unknown pack face None; known faces: {list(FACE_NAMES)}")
        face_value = coerce_face(face)
        items = self.entries(face_value)
        for entry in items:
            if entry.name == id:
                return entry
        known = sorted(entry.name for entry in items)
        raise PackError(f"unknown id {id!r} on pack face {face_value.value!r}; known ids: {known}")


__all__ = ["PackCatalog", "validate_unique"]
