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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from unigym.provenance import ProviderInfo, ProviderOrigin
from unigym.scenes.provider_params import (
    ProviderParamSpec,
    SceneProviderParamError,
    normalize_manifest_params,
)

from . import PackError
from .catalog import CatalogEntry, PackCatalog
from .runtime import PackRuntime, discover
from .schema import FACE_NAMES, PackManifestV1

__all__ = [
    "AssetResourceRef",
    "PackInfo",
    "PackSet",
    "ResourceInfo",
    "RobotRef",
    "ScenePresetRef",
    "SceneProviderRef",
    "discover_set",
    "installed",
    "require",
]

_PRESET_FACE = "scene_presets"
_PROVIDER_FACE = "providers"
_ASSET_FACE = "assets"


@dataclass(frozen=True)
class RobotRef:
    """An inert handle for one ``kind: robot`` manifest provider."""

    pack: str
    pack_version: str
    id: str
    kind: str | None
    resource: str | None
    entrypoint: str | None
    origin: ProviderOrigin

    @property
    def is_data(self) -> bool:
        """Whether the robot resolves from a declared document (no pack Python)."""
        return self.resource is not None


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
class ResourceInfo:
    """One resource-backed catalog item, its resolved path and provenance."""

    pack: str
    pack_version: str
    face: str
    id: str
    kind: str | None
    resource: str
    path: Path
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


def _require_face(face: str) -> None:
    """Reject an unknown catalog face with the scene replacement hint."""
    if face in FACE_NAMES:
        return
    hint = ""
    if face == "scenes":
        hint = (
            "; scene presets are declared under 'scene_presets'; scene providers "
            "use providers with kind: scene"
        )
    raise PackError(f"unknown face {face!r}{hint}; known faces: {list(FACE_NAMES)}")


def _pack_origin(manifest: PackManifestV1, runtime: PackRuntime) -> ProviderOrigin:
    """Build the pack-level origin from the runtime discovery label."""
    discovered_by = str(runtime.origins.get(manifest.name, "unknown"))
    return ProviderOrigin(
        kind="pack",
        pack=manifest.name,
        pack_version=manifest.version,
        root=str(manifest.root),
        discovered_by=discovered_by,
        location_imported=discovered_by == "entry-point",
    )


def _item_origin(
    entry: CatalogEntry,
    *,
    pack: str,
    pack_version: str,
    root: str,
    discovered_by: str | None,
    activation_entrypoint: str | None,
) -> ProviderOrigin:
    """Build the manifest origin carried by one minted handle."""
    return ProviderOrigin(
        kind="manifest",
        pack=pack,
        pack_version=pack_version,
        root=root,
        discovered_by=discovered_by,
        face=entry.face,
        item_id=entry.id,
        resource=entry.resource,
        entrypoint=entry.entrypoint,
        activation_entrypoint=activation_entrypoint,
        location_imported=discovered_by == "entry-point",
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

    Equality is by ``name``, ``version``, ``root`` and ``origin``; the private
    per-pack item views are catalog-derived data and do not participate.
    """

    name: str
    version: str
    api: int = field(compare=False)
    trust: str = field(compare=False)
    root: Path
    requires: Mapping[str, str] = field(compare=False)
    origin: ProviderOrigin
    _entries: tuple[CatalogEntry, ...] = field(default=(), repr=False, compare=False)
    _resources: tuple[ResourceInfo, ...] = field(default=(), repr=False, compare=False)
    _providers: tuple[ProviderInfo, ...] = field(default=(), repr=False, compare=False)
    _entrypoint: str | None = field(default=None, repr=False, compare=False)

    def _entry(self, face: str, item_id: str) -> CatalogEntry:
        """Return this pack's entry for an exact ``(face, id)`` pair."""
        for entry in self._entries:
            if entry.face == face and entry.id == item_id:
                return entry
        known = sorted(entry.id for entry in self._entries if entry.face == face)
        raise PackError(
            f"pack {self.name!r} declares no {face!r} id {item_id!r}; known ids: {known}"
        )

    def _provider_entry(self, kind: str, item_id: str) -> CatalogEntry:
        """Return this pack's ``providers`` entry for an exact ``(kind, id)`` pair."""
        for entry in self._entries:
            if entry.face == _PROVIDER_FACE and entry.kind == kind and entry.id == item_id:
                return entry
        known = sorted(
            entry.id
            for entry in self._entries
            if entry.face == _PROVIDER_FACE and entry.kind == kind
        )
        raise PackError(
            f"pack {self.name!r} has no {kind} id {item_id!r}; known {kind} ids "
            f"on {self.name!r}: {known}"
        )

    def _origin(self, entry: CatalogEntry) -> ProviderOrigin:
        """Build the manifest origin for one entry owned by this pack."""
        return _item_origin(
            entry,
            pack=self.name,
            pack_version=self.version,
            root=str(self.root),
            discovered_by=self.origin.discovered_by,
            activation_entrypoint=self._entrypoint,
        )

    def _robot_ref(self, entry: CatalogEntry) -> RobotRef:
        """Mint one robot handle from this pack's catalog entry."""
        return RobotRef(
            pack=self.name,
            pack_version=self.version,
            id=entry.id,
            kind=entry.kind,
            resource=entry.resource,
            entrypoint=entry.entrypoint,
            origin=self._origin(entry),
        )

    def _asset_ref(self, entry: CatalogEntry) -> AssetResourceRef:
        """Mint one asset handle from this pack's catalog entry."""
        if entry.resource is None:
            raise PackError(
                f"pack {self.name!r} asset id {entry.id!r} declares entrypoint "
                f"{entry.entrypoint!r}, not a resource; assets are resource-backed"
            )
        return AssetResourceRef(
            pack=self.name,
            pack_version=self.version,
            id=entry.id,
            kind=entry.kind,
            resource=entry.resource,
            origin=self._origin(entry),
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
            origin=self._origin(entry),
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
            origin=self._origin(entry),
        )

    def robot(self, id: str) -> RobotRef:
        """Mint the handle for one ``kind: robot`` provider declared by this pack."""
        return self._robot_ref(self._provider_entry("robot", id))

    def robots(self) -> tuple[RobotRef, ...]:
        """Every robot handle this pack declares, in manifest order."""
        return tuple(
            self._robot_ref(entry)
            for entry in self._entries
            if entry.face == _PROVIDER_FACE and entry.kind == "robot"
        )

    def asset(self, id: str) -> AssetResourceRef:
        """Mint the handle for one resource-backed asset declared by this pack."""
        return self._asset_ref(self._entry(_ASSET_FACE, id))

    def assets(self) -> tuple[AssetResourceRef, ...]:
        """Every resource-backed asset handle this pack declares, in manifest order."""
        return tuple(
            self._asset_ref(entry)
            for entry in self._entries
            if entry.face == _ASSET_FACE and entry.resource is not None
        )

    def providers(self, kind: str | None = None) -> tuple[ProviderInfo, ...]:
        """This pack's ``providers`` items, optionally restricted to one kind."""
        return tuple(info for info in self._providers if kind is None or info.kind == kind)

    def resources(self, face: str | None = None) -> tuple[ResourceInfo, ...]:
        """This pack's resource-backed items, optionally restricted to one face."""
        if face is not None:
            _require_face(face)
        return tuple(info for info in self._resources if face is None or info.face == face)

    @property
    def face_counts(self) -> Mapping[str, int]:
        """Declared item counts per face (declaration counts, not existence claims)."""
        counts: dict[str, int] = {}
        for entry in self._entries:
            counts[entry.face] = counts.get(entry.face, 0) + 1
        return MappingProxyType({face: counts[face] for face in FACE_NAMES if face in counts})

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


def _catalog_entries(manifest: PackManifestV1) -> tuple[CatalogEntry, ...]:
    """Flatten one manifest's provided items into catalog entries."""
    return tuple(
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


def _pack_info(manifest: PackManifestV1, runtime: PackRuntime) -> PackInfo:
    """Build one :class:`PackInfo` from a runtime manifest."""
    entries = _catalog_entries(manifest)
    origin = _pack_origin(manifest, runtime)
    resources = tuple(
        ResourceInfo(
            pack=manifest.name,
            pack_version=manifest.version,
            face=entry.face,
            id=entry.id,
            kind=entry.kind,
            resource=entry.resource,
            path=runtime.resource_for(entry.face, entry.id),
            origin=_item_origin(
                entry,
                pack=manifest.name,
                pack_version=manifest.version,
                root=str(manifest.root),
                discovered_by=origin.discovered_by,
                activation_entrypoint=manifest.entrypoint,
            ),
        )
        for entry in entries
        if entry.resource is not None
    )
    providers = tuple(
        ProviderInfo(
            name=entry.id,
            kind=entry.kind,
            origin=_item_origin(
                entry,
                pack=manifest.name,
                pack_version=manifest.version,
                root=str(manifest.root),
                discovered_by=origin.discovered_by,
                activation_entrypoint=manifest.entrypoint,
            ),
            active=None,
            pack=manifest.name,
            pack_version=manifest.version,
            face=entry.face,
            resource=entry.resource,
            entrypoint=entry.entrypoint,
        )
        for entry in entries
        if entry.face == _PROVIDER_FACE
    )
    return PackInfo(
        name=manifest.name,
        version=manifest.version,
        api=manifest.api,
        trust=manifest.trust,
        root=manifest.root,
        requires=manifest.requires,
        origin=origin,
        _entries=entries,
        _resources=resources,
        _providers=providers,
        _entrypoint=manifest.entrypoint,
    )


@dataclass(frozen=True)
class PackSet:
    """A read-only aggregate over one manifest discovery scope.

    The private runtime supplies the manifest catalog, pack roots and discovery
    labels.  Construction only reads already-validated manifests and the paths
    those manifests declare; no method activates, imports, mutates a registry
    or reads a resource path except :meth:`resource_path`.

    Equality is by the ordered :class:`PackInfo` tuple; the runtime is not part
    of the value.
    """

    _runtime: PackRuntime = field(repr=False, compare=False)
    _packs: tuple[PackInfo, ...] = field(init=False, repr=False)
    _resources: tuple[ResourceInfo, ...] = field(init=False, repr=False, compare=False)
    _providers: tuple[ProviderInfo, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build the immutable per-pack handles in discovery order."""
        manifests = self._runtime.catalog.manifests
        packs = tuple(_pack_info(manifest, self._runtime) for manifest in manifests)
        object.__setattr__(self, "_packs", packs)
        object.__setattr__(self, "_resources", tuple(r for pack in packs for r in pack._resources))
        object.__setattr__(self, "_providers", tuple(p for pack in packs for p in pack._providers))

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

    def origins(self, *, pack: str | None = None) -> tuple[ProviderOrigin, ...]:
        """One pack-level origin per pack in discovery order, or one named pack's."""
        if pack is not None:
            return (self.pack(pack).origin,)
        return tuple(entry.origin for entry in self._packs)

    def _owners(self, face: str, item_id: str) -> tuple[PackInfo, ...]:
        """Every pack in this set that declares an exact ``(face, id)`` pair."""
        return tuple(
            pack
            for pack in self._packs
            if any(entry.face == face and entry.id == item_id for entry in pack._entries)
        )

    def _provider_owners(self, kind: str, item_id: str) -> tuple[PackInfo, ...]:
        """Every pack in this set that declares an exact ``(providers, kind, id)`` item."""
        return tuple(
            pack
            for pack in self._packs
            if any(
                entry.face == _PROVIDER_FACE and entry.kind == kind and entry.id == item_id
                for entry in pack._entries
            )
        )

    def robot(self, id: str) -> RobotRef:
        """Cross-pack edge lookup for one ``kind: robot`` provider id."""
        owners = self._provider_owners("robot", id)
        if len(owners) == 1:
            return owners[0].robot(id)
        if len(owners) > 1:
            names = sorted(pack.name for pack in owners)
            raise PackError(f"robot id {id!r} is declared by more than one pack: {names}")
        known = sorted(info.name for info in self._providers if info.kind == "robot")
        packs = sorted({str(info.pack) for info in self._providers if info.kind == "robot"})
        raise PackError(
            f"unknown robot id {id!r} on the installed providers face; "
            f"known robot ids: {known} (packs: {packs})"
        )

    def robots(self) -> tuple[RobotRef, ...]:
        """Every robot handle in this set, pack order then manifest order."""
        return tuple(ref for pack in self._packs for ref in pack.robots())

    def asset(self, id: str) -> AssetResourceRef:
        """Cross-pack edge lookup for one resource-backed ``assets`` catalog id."""
        owners = tuple(
            pack
            for pack in self._packs
            if any(
                entry.face == _ASSET_FACE and entry.id == id and entry.resource is not None
                for entry in pack._entries
            )
        )
        if len(owners) == 1:
            return owners[0].asset(id)
        if len(owners) > 1:
            names = sorted(pack.name for pack in owners)
            raise PackError(f"asset id {id!r} is declared by more than one pack: {names}")
        known = sorted(info.id for info in self._resources if info.face == _ASSET_FACE)
        raise PackError(f"unknown asset id {id!r}; known asset ids: {known}")

    def assets(self) -> tuple[AssetResourceRef, ...]:
        """Every resource-backed asset handle in this set, in discovery order."""
        return tuple(ref for pack in self._packs for ref in pack.assets())

    def providers(self, kind: str | None = None) -> tuple[ProviderInfo, ...]:
        """Every ``providers`` item in this set, optionally filtered by kind."""
        return tuple(info for info in self._providers if kind is None or info.kind == kind)

    def resources(self, face: str | None = None) -> tuple[ResourceInfo, ...]:
        """Every resource-backed item in this set, optionally filtered by face."""
        if face is not None:
            _require_face(face)
        return tuple(info for info in self._resources if face is None or info.face == face)

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

    def resource_path(
        self,
        ref: RobotRef | ScenePresetRef | SceneProviderRef | AssetResourceRef,
    ) -> Path:
        """Resolve a resource-backed handle's declared resource under its pack root.

        Entrypoint-backed handles have no resource path, so they raise the
        existing "activate the pack" teaching error.  No other method on this
        class reads the filesystem.
        """
        if isinstance(ref, SceneProviderRef):
            raise PackError(
                f"pack {ref.pack!r} face {_PROVIDER_FACE!r} id {ref.id!r} declares "
                f"entrypoint {ref.entrypoint!r}, not a resource; use "
                f"runtime.activate({ref.pack!r}) to activate the pack"
            )
        if isinstance(ref, RobotRef):
            if ref.resource is None:
                raise PackError(
                    f"pack {ref.pack!r} provider id {ref.id!r} declares entrypoint "
                    f"{ref.entrypoint!r}, not a resource; use "
                    f"runtime.activate({ref.pack!r}) to activate the pack"
                )
            face = _PROVIDER_FACE
        elif isinstance(ref, ScenePresetRef):
            face = _PRESET_FACE
        elif isinstance(ref, AssetResourceRef):
            face = _ASSET_FACE
        else:
            raise PackError(
                "resource_path requires a RobotRef, ScenePresetRef, SceneProviderRef "
                f"or AssetResourceRef, got {type(ref).__name__}"
            )
        entry = self._runtime.catalog.entry(face, ref.id)
        if entry.pack != ref.pack:
            raise PackError(
                f"{face} id {ref.id!r} is declared by pack {entry.pack!r}, " f"not {ref.pack!r}"
            )
        return self._runtime.resource_for(face, ref.id)

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


def discover_set(
    roots: Iterable[str | Path] | None = None,
    *,
    config_path: str | Path | None = None,
    entry_points: bool = False,
) -> PackSet:
    """Build the read-only pack set over explicit roots or configured discovery."""
    return PackSet(discover(roots, config_path=config_path, entry_points=entry_points))


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
