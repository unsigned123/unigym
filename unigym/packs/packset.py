"""The ordered pack set: a pure index over stored packs and items.

:func:`unigym.packs.installed` enumerates the whole configured scope and
:func:`unigym.packs.require` returns the filtered view over exactly the named
packs.  Both return a :class:`PackSet`, the cross-pack query handle.  A pack's
scene presets and scene providers are acquired through the canonical
pack-scoped item form::

    packs = unigym.packs.require("acme")
    preset = packs.pack("acme").item(Face.SCENE_PRESETS, "acme.tabletop")  # PackItem
    provider = packs.pack("acme").provider("scene", "acme.table_top")      # PackItem

There is no attribute sugar: ``packs.acme`` never resolves a pack.

Construction is a pure index operation: it validates unique pack names and
unique ``(face, name)`` pairs and never activates, imports pack Python, resolves
a resource path or mutates a registry.  The typed refs
(:class:`AssetResourceRef`, :class:`ScenePresetRef`, :class:`SceneProviderRef`)
are frozen projections minted from one :class:`~unigym.packs.schema.PackItem`;
they carry that item's manifest provenance and read no filesystem.  Resource
path resolution and object loading belong to
:class:`~unigym.packs.loader.PackLoader`, not to this index.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from unigym.provenance import ProviderOrigin
from unigym.scenes.provider_params import (
    ProviderParamSpec,
    SceneProviderParamError,
    normalize_manifest_params,
)

from . import PackError
from .catalog import validate_unique
from .schema import (
    Face,
    ItemStatus,
    Pack,
    PackItem,
    ResourceLocator,
    coerce_face,
    provider_param_declarations,
)

__all__ = [
    "AssetResourceRef",
    "PackSet",
    "ScenePresetRef",
    "SceneProviderRef",
]


@dataclass(frozen=True)
class AssetResourceRef:
    """An inert handle for one resource-backed ``assets`` manifest item."""

    pack: str
    pack_version: str
    id: str
    kind: str | None
    resource: str
    origin: ProviderOrigin


@dataclass(frozen=True)
class ScenePresetRef:
    """An inert handle for one ``scene_presets`` manifest resource."""

    pack: str
    pack_version: str
    id: str
    resource: str
    origin: ProviderOrigin


@dataclass(frozen=True)
class SceneProviderRef:
    """An inert handle for one ``kind: scene`` manifest entrypoint.

    ``params_schema`` is the normalized manifest parameter declaration, or
    ``None`` when the provider declares no parameters.  Acquiring the handle
    does not import the entrypoint.
    """

    pack: str
    pack_version: str
    id: str
    entrypoint: str
    params_schema: Mapping[str, ProviderParamSpec] | None
    origin: ProviderOrigin

    def __post_init__(self) -> None:
        """Freeze the optional parameter schema mapping."""
        if self.params_schema is not None:
            object.__setattr__(self, "params_schema", MappingProxyType(dict(self.params_schema)))

    def __hash__(self) -> int:
        """Hash the handle identity; the schema is derived from that identity."""
        return hash((self.pack, self.pack_version, self.id, self.entrypoint, self.origin))


def _provider_schema(item: PackItem, *, pack: str) -> Mapping[str, ProviderParamSpec] | None:
    """Normalize one provider item's manifest parameter declaration."""
    if item.params is None:
        return None
    try:
        schema = normalize_manifest_params(provider_param_declarations(item.params))
    except SceneProviderParamError as err:
        raise PackError(
            f"pack {pack!r} scene provider id {item.name!r} params are invalid: {err}"
        ) from err
    return MappingProxyType(schema)


def _asset_ref(item: PackItem) -> AssetResourceRef:
    """Mint one asset handle from a stored asset item."""
    if not isinstance(item.locator, ResourceLocator):
        raise PackError(
            f"pack {item.pack!r} asset id {item.name!r} declares entrypoint "
            f"{item.entrypoint!r}, not a resource; assets are resource-backed"
        )
    return AssetResourceRef(
        pack=item.pack,
        pack_version=item.origin.pack_version or "",
        id=item.name,
        kind=item.kind,
        resource=item.locator.path,
        origin=item.origin,
    )


def _preset_ref(item: PackItem) -> ScenePresetRef:
    """Mint one preset handle from a stored preset item."""
    if not isinstance(item.locator, ResourceLocator):
        raise PackError(f"pack {item.pack!r} scene preset id {item.name!r} declares no resource")
    return ScenePresetRef(
        pack=item.pack,
        pack_version=item.origin.pack_version or "",
        id=item.name,
        resource=item.locator.path,
        origin=item.origin,
    )


def _provider_ref(item: PackItem) -> SceneProviderRef:
    """Mint one ``kind: scene`` provider handle from a stored provider item."""
    if item.kind != "scene":
        raise PackError(
            f"pack {item.pack!r} provider id {item.name!r} declares kind "
            f"{item.kind!r}, not 'scene'"
        )
    if item.entrypoint is None:
        raise PackError(
            f"pack {item.pack!r} scene provider id {item.name!r} declares no entrypoint"
        )
    return SceneProviderRef(
        pack=item.pack,
        pack_version=item.origin.pack_version or "",
        id=item.name,
        entrypoint=item.entrypoint,
        params_schema=_provider_schema(item, pack=item.pack),
        origin=item.origin,
    )


@dataclass(frozen=True)
class PackSet:
    """An ordered, duplicate-free index over discovered or hand-built packs.

    Equality is by the ordered pack tuple.  Construction performs no filesystem
    access, resolves no resource path and loads no pack object.
    """

    _packs: tuple[Pack, ...]

    def __post_init__(self) -> None:
        """Freeze the pack tuple and reject duplicate pack or item owners."""
        object.__setattr__(self, "_packs", tuple(self._packs))
        validate_unique(self._packs)

    def packs(self) -> tuple[Pack, ...]:
        """Every pack in this set, in discovery order."""
        return self._packs

    def pack(self, name: str) -> Pack:
        """Return the named pack or fail with the known pack names."""
        for pack in self._packs:
            if pack.name == name:
                return pack
        known = sorted(pack.name for pack in self._packs)
        raise PackError(f"unknown pack {name!r}; known packs: {known}")

    def get(self, name: str) -> Pack:
        """Alias for :meth:`pack`."""
        return self.pack(name)

    def origins(self, *, pack: str | None = None) -> tuple[ProviderOrigin, ...]:
        """One pack-level origin per pack in discovery order, or one named pack's."""
        if pack is not None:
            return (self.pack(pack).origin,)
        return tuple(entry.origin for entry in self._packs)

    def items(
        self,
        *,
        face: Face | None = None,
        kind: str | None = None,
        status: ItemStatus | None = None,
    ) -> tuple[PackItem, ...]:
        """Every stored item, preserving pack order then declaration order."""
        face_value = None if face is None else coerce_face(face)
        return tuple(
            item
            for pack in self._packs
            for item in pack.items_for(face=face_value, kind=kind, status=status)
        )

    def item(self, face: Face | str, name: str) -> PackItem:
        """Return the unique stored ``(face, name)`` item across the set."""
        face_value = coerce_face(face)
        for pack in self._packs:
            for item in pack.items:
                if item.face == face_value and item.name == name:
                    return item
        known = sorted(
            item.name for pack in self._packs for item in pack.items if item.face == face_value
        )
        raise PackError(
            f"unknown id {name!r} on pack face {face_value.value!r}; known ids: {known}"
        )

    def provider(self, kind: str, name: str) -> PackItem:
        """Return the unique ``providers`` item for an exact ``(kind, name)`` pair."""
        for pack in self._packs:
            for item in pack.items:
                if item.face is Face.PROVIDERS and item.kind == kind and item.name == name:
                    return item
        known = sorted(
            item.name
            for pack in self._packs
            for item in pack.items
            if item.face is Face.PROVIDERS and item.kind == kind
        )
        raise PackError(
            f"unknown {kind} id {name!r} on the installed providers face; "
            f"known {kind} ids: {known}"
        )

    def asset(self, id: str) -> AssetResourceRef:
        """Cross-pack edge lookup for one resource-backed ``assets`` item."""
        item = self.item(Face.ASSETS, id)
        return _asset_ref(item)

    def assets(self) -> tuple[AssetResourceRef, ...]:
        """Every resource-backed asset handle in this set, in discovery order."""
        return tuple(_asset_ref(item) for item in self._resources(Face.ASSETS))

    def providers(self, kind: str | None = None) -> tuple[PackItem, ...]:
        """Every ``providers`` item in this set, optionally filtered by kind."""
        return tuple(
            item for item in self.items(face=Face.PROVIDERS) if kind is None or item.kind == kind
        )

    def resources(self, face: Face | str | None = None) -> tuple[PackItem, ...]:
        """Every resource-backed item in this set, optionally filtered by face."""
        face_value = None if face is None else coerce_face(face)
        return self._resources(face_value)

    def _resources(self, face: Face | None = None) -> tuple[PackItem, ...]:
        """Resource-located items in pack order, optionally restricted to a face."""
        return tuple(
            item for item in self.items(face=face) if isinstance(item.locator, ResourceLocator)
        )

    def scene_preset(self, id: str) -> ScenePresetRef:
        """Cross-pack edge lookup for one ``scene_presets`` item id."""
        return _preset_ref(self.item(Face.SCENE_PRESETS, id))

    def scene_presets(self) -> tuple[ScenePresetRef, ...]:
        """Every preset handle in this set, pack order then manifest order."""
        return tuple(_preset_ref(item) for item in self._resources(Face.SCENE_PRESETS))

    def scene_provider(self, id: str) -> SceneProviderRef:
        """Cross-pack edge lookup for one ``kind: scene`` provider item id."""
        return _provider_ref(self.item(Face.PROVIDERS, id))

    def scene_providers(self) -> tuple[SceneProviderRef, ...]:
        """Every ``kind: scene`` provider handle in this set, in discovery order."""
        return tuple(_provider_ref(item) for item in self.providers("scene"))
