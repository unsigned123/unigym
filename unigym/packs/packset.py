"""Read-only pack-object handles over one manifest discovery scope.

:func:`installed` enumerates the whole configured scope and :func:`require`
returns the filtered view over exactly the named packs.  Both return a
:class:`PackSet`, the cross-pack query handle.  A pack's scene presets and scene
providers are acquired only through their canonical pack-scoped form::

    packs = unigym.packs.require("acme")
    preset = packs.pack("acme").scene_preset("acme.tabletop")
    provider = packs.pack("acme").scene_provider("acme.table_top")

There is no attribute sugar: ``packs.acme`` never resolves a pack.

Refs are frozen dataclasses minted from catalog data.  They are equal by value,
carry their manifest provenance and never activate a pack, import pack Python,
read the filesystem or mutate a registry.  :meth:`PackSet.resource_path` is the
only method here that reads a path, and it refuses entrypoint-backed handles
with the existing "activate the pack" teaching error.  Discovery itself reads
manifests only; installed entry points are never resolved in this scope.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from unigym.scenes.provider_params import (
    ProviderParamSpec,
    SceneProviderParamError,
    normalize_manifest_params,
)

from . import PackError
from .catalog import CatalogEntry, PackCatalog
from .runtime import PackRuntime, discover
from .schema import PackManifestV1

__all__ = [
    "PackInfo",
    "PackItemOrigin",
    "PackSet",
    "ScenePresetRef",
    "SceneProviderRef",
    "installed",
    "require",
]

_PRESET_FACE = "scene_presets"
_PROVIDER_FACE = "providers"


@dataclass(frozen=True)
class PackItemOrigin:
    """The pack-layer provenance carried by one object handle.

    ``kind`` is ``"manifest"``; ``locator`` is the declared resource for a
    preset and the declared entrypoint for a provider.  This is the pack-layer
    origin, not the richer runtime ``ProviderOrigin``; neither is serialized.
    """

    kind: str
    pack: str
    pack_version: str
    root: str
    face: str
    item_id: str
    locator: str


@dataclass(frozen=True)
class ScenePresetRef:
    """An inert handle for one ``scene_presets`` manifest resource."""

    pack: str
    pack_version: str
    id: str
    resource: str
    origin: PackItemOrigin


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
    origin: PackItemOrigin

    def __post_init__(self) -> None:
        """Freeze the optional parameter schema mapping."""
        if self.params_schema is not None:
            object.__setattr__(self, "params_schema", MappingProxyType(dict(self.params_schema)))

    def __hash__(self) -> int:
        """Hash the handle identity; the schema is derived from that identity."""
        return hash((self.pack, self.pack_version, self.id, self.entrypoint, self.origin))


def _origin(
    entry: CatalogEntry,
    *,
    pack: str,
    pack_version: str,
    root: str,
) -> PackItemOrigin:
    """Build one handle origin from a catalog entry."""
    locator = entry.resource if entry.face == _PRESET_FACE else entry.entrypoint
    assert locator is not None  # the manifest loader enforces one locator
    return PackItemOrigin(
        kind="manifest",
        pack=pack,
        pack_version=pack_version,
        root=root,
        face=entry.face,
        item_id=entry.id,
        locator=locator,
    )


def _provider_schema(
    entry: CatalogEntry,
    *,
    pack: str,
) -> Mapping[str, ProviderParamSpec] | None:
    """Normalize one provider item's manifest parameter declaration."""
    if entry.params is None:
        return None
    try:
        schema = normalize_manifest_params(entry.params)
    except SceneProviderParamError as err:
        raise PackError(
            f"pack {pack!r} scene provider id {entry.id!r} params are invalid: {err}"
        ) from err
    return MappingProxyType(schema)


@dataclass(frozen=True)
class PackInfo:
    """One installed pack and its pack-scoped object queries.

    ``origin`` is the discovery label (``env:UNIGYM_PACKS``, ``local-yaml``,
    ``explicit-root`` or ``unknown``).  ``_entries`` and ``_root`` are the
    per-pack catalog view used to mint handles; they carry no runtime token.
    """

    name: str
    version: str
    origin: str
    _entries: tuple[CatalogEntry, ...] = field(default=(), repr=False, compare=False)
    _root: str = field(default="", repr=False, compare=False)

    def _entry(self, face: str, item_id: str) -> CatalogEntry:
        """Return this pack's entry for an exact ``(face, id)`` pair."""
        for entry in self._entries:
            if entry.face == face and entry.id == item_id:
                return entry
        known = sorted(entry.id for entry in self._entries if entry.face == face)
        raise PackError(
            f"pack {self.name!r} declares no {face!r} id {item_id!r}; known ids: {known}"
        )

    def _preset_ref(self, entry: CatalogEntry) -> ScenePresetRef:
        """Mint one preset handle from this pack's catalog entry."""
        if entry.resource is None:
            raise PackError(f"pack {self.name!r} scene preset id {entry.id!r} declares no resource")
        return ScenePresetRef(
            pack=self.name,
            pack_version=self.version,
            id=entry.id,
            resource=entry.resource,
            origin=_origin(entry, pack=self.name, pack_version=self.version, root=self._root),
        )

    def _provider_ref(self, entry: CatalogEntry) -> SceneProviderRef:
        """Mint one ``kind: scene`` provider handle from this pack's entry."""
        if entry.kind != "scene":
            raise PackError(
                f"pack {self.name!r} provider id {entry.id!r} declares kind "
                f"{entry.kind!r}, not 'scene'"
            )
        if entry.entrypoint is None:
            raise PackError(
                f"pack {self.name!r} scene provider id {entry.id!r} declares no entrypoint"
            )
        return SceneProviderRef(
            pack=self.name,
            pack_version=self.version,
            id=entry.id,
            entrypoint=entry.entrypoint,
            params_schema=_provider_schema(entry, pack=self.name),
            origin=_origin(entry, pack=self.name, pack_version=self.version, root=self._root),
        )

    def scene_preset(self, id: str) -> ScenePresetRef:
        """Mint the handle for one preset declared by this pack.

        A preset owned by another pack is not resolvable here; use
        :meth:`PackSet.scene_preset` for the cross-pack edge lookup.
        """
        return self._preset_ref(self._entry(_PRESET_FACE, id))

    def scene_presets(self) -> tuple[ScenePresetRef, ...]:
        """Every preset handle this pack declares, in manifest order."""
        return tuple(
            self._preset_ref(entry) for entry in self._entries if entry.face == _PRESET_FACE
        )

    def scene_provider(self, id: str) -> SceneProviderRef:
        """Mint the handle for one ``kind: scene`` provider declared by this pack."""
        return self._provider_ref(self._entry(_PROVIDER_FACE, id))

    def scene_providers(self) -> tuple[SceneProviderRef, ...]:
        """Every ``kind: scene`` provider handle this pack declares, in manifest order."""
        return tuple(
            self._provider_ref(entry)
            for entry in self._entries
            if entry.face == _PROVIDER_FACE and entry.kind == "scene"
        )


def _pack_info(manifest: PackManifestV1, runtime: PackRuntime) -> PackInfo:
    """Build one :class:`PackInfo` from a runtime manifest."""
    entries = tuple(
        CatalogEntry(
            pack=manifest.name,
            face=face,
            id=item.id,
            resource=item.resource,
            entrypoint=item.entrypoint,
            kind=item.kind,
            params=item.params,
        )
        for face, items in manifest.provides.items()
        for item in items
    )
    return PackInfo(
        name=manifest.name,
        version=manifest.version,
        origin=str(runtime.origins.get(manifest.name, "unknown")),
        _entries=entries,
        _root=str(manifest.root),
    )


@dataclass(frozen=True)
class PackSet:
    """A read-only aggregate over one manifest discovery scope.

    The private runtime supplies the manifest catalog, pack roots and discovery
    labels.  Construction only reads already-validated manifests; no method
    activates, imports, mutates a registry or reads a resource path except
    :meth:`resource_path`.
    """

    _runtime: PackRuntime = field(repr=False, compare=False)
    _packs: tuple[PackInfo, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the immutable per-pack handles in discovery order."""
        manifests = self._runtime.catalog.manifests
        object.__setattr__(
            self,
            "_packs",
            tuple(_pack_info(manifest, self._runtime) for manifest in manifests),
        )

    def packs(self) -> tuple[PackInfo, ...]:
        """Every pack in this set, in discovery order."""
        return self._packs

    def pack(self, name: str) -> PackInfo:
        """Return the named pack handle or fail with the known pack names.

        This is the canonical, collision-free query.  :class:`PackSet` has no
        attribute sugar, so ``packs.pack("acme")`` is the only guaranteed form.
        """
        for pack in self._packs:
            if pack.name == name:
                return pack
        known = sorted(pack.name for pack in self._packs)
        raise PackError(f"unknown pack {name!r}; known packs: {known}")

    def _owners(self, face: str, item_id: str) -> tuple[PackInfo, ...]:
        """Every pack in this set that declares an exact ``(face, id)`` pair."""
        return tuple(
            pack
            for pack in self._packs
            if any(entry.face == face and entry.id == item_id for entry in pack._entries)
        )

    def scene_preset(self, id: str) -> ScenePresetRef:
        """Cross-pack edge lookup for one ``scene_presets`` catalog id."""
        owners = self._owners(_PRESET_FACE, id)
        if len(owners) == 1:
            return owners[0].scene_preset(id)
        if len(owners) > 1:
            names = sorted(pack.name for pack in owners)
            raise PackError(f"scene preset id {id!r} is declared by more than one pack: {names}")
        known = sorted(
            entry.id
            for pack in self._packs
            for entry in pack._entries
            if entry.face == _PRESET_FACE
        )
        raise PackError(f"unknown scene preset id {id!r}; known ids: {known}")

    def scene_presets(self) -> tuple[ScenePresetRef, ...]:
        """Every preset handle in this set, pack order then manifest order."""
        return tuple(ref for pack in self._packs for ref in pack.scene_presets())

    def scene_provider(self, id: str) -> SceneProviderRef:
        """Cross-pack edge lookup for one ``kind: scene`` provider id."""
        owners = self._owners(_PROVIDER_FACE, id)
        if len(owners) == 1:
            return owners[0].scene_provider(id)
        if len(owners) > 1:
            names = sorted(pack.name for pack in owners)
            raise PackError(f"scene provider id {id!r} is declared by more than one pack: {names}")
        known = sorted(
            entry.id
            for pack in self._packs
            for entry in pack._entries
            if entry.face == _PROVIDER_FACE and entry.kind == "scene"
        )
        raise PackError(f"unknown scene provider id {id!r}; known ids: {known}")

    def scene_providers(self) -> tuple[SceneProviderRef, ...]:
        """Every ``kind: scene`` provider handle in this set, in discovery order."""
        return tuple(ref for pack in self._packs for ref in pack.scene_providers())

    def resource_path(self, ref: ScenePresetRef | SceneProviderRef) -> Path:
        """Resolve a preset handle's declared resource under its pack root.

        A scene provider is entrypoint-backed and has no resource path, so it
        raises the existing "activate the pack" teaching error.  No other method
        on this class reads the filesystem.
        """
        if isinstance(ref, SceneProviderRef):
            raise PackError(
                f"pack {ref.pack!r} face {_PROVIDER_FACE!r} id {ref.id!r} declares "
                f"entrypoint {ref.entrypoint!r}, not a resource; use "
                f"runtime.activate({ref.pack!r}) to activate the pack"
            )
        if not isinstance(ref, ScenePresetRef):
            raise PackError(
                "resource_path requires a ScenePresetRef or SceneProviderRef, "
                f"got {type(ref).__name__}"
            )
        entry = self._runtime.catalog.entry(_PRESET_FACE, ref.id)
        if entry.pack != ref.pack:
            raise PackError(
                f"scene preset id {ref.id!r} is declared by pack {entry.pack!r}, "
                f"not {ref.pack!r}"
            )
        return self._runtime.resource_for(_PRESET_FACE, ref.id)

    def to_runtime(self) -> PackRuntime:
        """The activation-capable runtime over this same catalog; never activates."""
        return self._runtime


def _filtered_runtime(runtime: PackRuntime, names: set[str]) -> PackRuntime:
    """A runtime over exactly the named packs, preserving discovery order."""
    manifests = tuple(manifest for manifest in runtime.catalog.manifests if manifest.name in names)
    kept = {manifest.name for manifest in manifests}
    return PackRuntime(
        catalog=PackCatalog(manifests),
        root_by_pack={name: runtime.root_by_pack[name] for name in kept},
        origins={name: runtime.origins[name] for name in kept if name in runtime.origins},
    )


def installed() -> PackSet:
    """Enumerate every pack in the configured manifest-only discovery scope.

    Installed ``unigym.packs`` entry points are not resolved here: this scope
    never imports pack Python.  Use :meth:`PackSet.to_runtime` for an
    activation-capable runtime over the same catalog.
    """
    return PackSet(discover())


def require(*names: str) -> PackSet:
    """Return the filtered pack set over exactly *names* in the installed scope.

    Requirement resolution is not performed: each name must be installed in
    the same configured discovery scope.  A missing name fails before any
    handle exists and names the installed packs.
    """
    invalid = [name for name in names if not isinstance(name, str) or not name]
    if invalid:
        raise PackError(f"required pack names must be non-empty strings, got {invalid!r}")
    full = installed()
    available = {pack.name for pack in full.packs()}
    missing = [name for name in names if name not in available]
    if missing:
        raise PackError(
            f"pack {missing[0]!r} is not installed in this scope; "
            f"required/installed packs: {sorted(available)}"
        )
    return PackSet(_filtered_runtime(full._runtime, set(names)))
