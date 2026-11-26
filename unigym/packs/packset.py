"""The ordered pack set: a pure index over stored packs and items.

:func:`installed` enumerates the whole configured scope and :func:`require`
returns the filtered view over exactly the named packs.  Both return a
:class:`PackSet`, the cross-pack query handle.  A pack's scene presets and scene
providers are acquired through the canonical pack-scoped item form::

    packs = unigym.packs.require("acme")
    preset = packs.pack("acme").item(Face.SCENE_PRESETS, "acme.tabletop")  # PackItem
    provider = packs.pack("acme").provider("scene", "acme.table_top")      # PackItem

There is no attribute sugar: ``packs.acme`` never resolves a pack.

Construction is a pure index operation: it validates unique pack names and
unique ``(face, name)`` pairs and never activates, imports pack Python, resolves
a resource path or mutates a registry.  The typed refs
(:class:`AssetResourceRef`, :class:`ScenePresetRef`, :class:`SceneProviderRef`)
are frozen projections minted from one :class:`~unigym.packs.schema.PackItem`;
they carry that item's manifest provenance and read no filesystem.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from unigym.provenance import ProviderOrigin
from unigym.scenes.provider_params import (
    ProviderParamSpec,
    SceneProviderParamError,
    normalize_manifest_params,
)

from . import PackError
from .catalog import PackCatalog, validate_unique
from .runtime import PackRuntime, discover
from .schema import (
    Face,
    ItemStatus,
    Pack,
    PackItem,
    ResourceLocator,
    coerce_face,
    pack_origin,
    provider_param_declarations,
    with_pack_origin,
)

if TYPE_CHECKING:
    from unigym.robots.base import RobotProfile

__all__ = [
    "AssetResourceRef",
    "PackSet",
    "ScenePresetRef",
    "SceneProviderRef",
    "discover_set",
    "installed",
    "require",
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


def _normalize_origin_override(pack: Pack, value: ProviderOrigin | str) -> ProviderOrigin:
    """Normalize one ``from_runtime(origins=...)`` value to a pack origin."""
    if isinstance(value, str):
        return pack_origin(pack, value)
    if not isinstance(value, ProviderOrigin):
        raise PackError(
            f"origin for pack {pack.name!r} must be a ProviderOrigin or a "
            f"discovery label string, got {type(value).__name__}"
        )
    if value.kind == "pack":
        facts = (
            ("pack", value.pack, pack.name),
            ("pack_version", value.pack_version, pack.version),
            ("root", value.root, str(pack.root)),
        )
        for field_name, actual, expected in facts:
            if actual != expected:
                raise PackError(
                    f"origin for pack {pack.name!r} has {field_name} "
                    f"{actual!r}, expected {expected!r}"
                )
        return value
    if value.kind == "unknown":
        return pack_origin(pack, "unknown")
    raise PackError(
        f"origin for pack {pack.name!r} must be a pack-level ProviderOrigin "
        f"(kind='pack'), got kind {value.kind!r}"
    )


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

    Equality is by the ordered pack tuple; the optional C1 runtime adapter is
    not part of the value.  Construction performs no filesystem access.
    """

    _packs: tuple[Pack, ...]
    _runtime: PackRuntime | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Freeze the pack tuple and reject duplicate pack or item owners."""
        object.__setattr__(self, "_packs", tuple(self._packs))
        validate_unique(self._packs)

    @classmethod
    def from_runtime(
        cls,
        runtime: PackRuntime,
        *,
        origins: Mapping[str, ProviderOrigin | str] | None = None,
    ) -> PackSet:
        """Build a pack set over an existing runtime without rediscovering roots.

        ``origins`` overrides the pack-level discovery origin per pack name.
        A string is a discovery label (``discovered_by``); a
        :class:`~unigym.provenance.ProviderOrigin` must be pack-level
        (``kind="pack"``) or the explicit ``unknown()`` record.  Omitted packs
        keep the runtime's own label.

        Construction is pure: it reads the runtime catalog and relabels the
        already-loaded packs; it never resolves entry points, imports pack
        Python or activates a pack.
        """
        overrides: dict[str, ProviderOrigin] = {}
        packs = runtime.catalog.manifests
        if origins is not None:
            names = set(runtime.catalog.packs())
            unknown = sorted(set(origins) - names)
            if unknown:
                raise PackError(
                    f"origins name unknown packs {unknown}; known packs: {sorted(names)}"
                )
            by_name = {pack.name: pack for pack in packs}
            for name, value in origins.items():
                overrides[name] = _normalize_origin_override(by_name[name], value)
        relabeled: list[Pack] = []
        for pack in packs:
            desired = overrides.get(pack.name)
            if desired is None:
                label = runtime.origins.get(pack.name)
                desired = pack_origin(pack, "unknown" if label is None else str(label))
            relabeled.append(with_pack_origin(pack, desired))
        return cls(tuple(relabeled), _runtime=runtime)

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

    def _require_runtime(self) -> PackRuntime:
        """Return the C1 runtime adapter or explain that this set is data-only."""
        if self._runtime is None:
            raise PackError(
                "this PackSet has no loader runtime; build it with "
                "PackSet.from_runtime(...) for robot profile resolution"
            )
        return self._runtime

    def robot(self, id: str) -> RobotProfile:
        """Cross-pack edge lookup for one ``kind: robot`` provider id."""
        item = self.provider("robot", id)
        return self._require_runtime().robot_profile(item.pack, item.name)

    def robots(self) -> tuple[RobotProfile, ...]:
        """Every robot profile in this set, pack order then manifest order."""
        runtime = self._require_runtime()
        return tuple(
            runtime.robot_profile(item.pack, item.name) for item in self.providers("robot")
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

    def resource_path(
        self,
        ref: ScenePresetRef | SceneProviderRef | AssetResourceRef,
    ) -> Path:
        """Resolve a resource-backed handle's declared resource under its pack root.

        Entrypoint-backed handles have no resource path, so they raise the
        existing "activate the pack" teaching error.  This C1 adapter delegates
        to the runtime's resource rules; C2 moves the method to the loader.
        """
        if isinstance(ref, SceneProviderRef):
            raise PackError(
                f"pack {ref.pack!r} face 'providers' id {ref.id!r} declares "
                f"entrypoint {ref.entrypoint!r}, not a resource; use "
                f"runtime.activate({ref.pack!r}) to activate the pack"
            )
        if isinstance(ref, ScenePresetRef):
            face = Face.SCENE_PRESETS
        elif isinstance(ref, AssetResourceRef):
            face = Face.ASSETS
        else:
            raise PackError(
                "resource_path requires a ScenePresetRef, SceneProviderRef "
                f"or AssetResourceRef, got {type(ref).__name__}"
            )
        item = self.item(face, ref.id)
        if item.pack != ref.pack:
            raise PackError(
                f"{face.value} id {ref.id!r} is declared by pack {item.pack!r}, "
                f"not {ref.pack!r}"
            )
        if not isinstance(item.locator, ResourceLocator):
            raise PackError(
                f"pack {ref.pack!r} face {face.value!r} id {ref.id!r} declares an "
                "entrypoint, not a resource"
            )
        return self._require_runtime().resource(item.pack, item.locator.path)

    def to_runtime(self) -> PackRuntime:
        """The activation-capable C1 runtime over this same catalog; never activates."""
        return self._require_runtime()


def _filtered_runtime(runtime: PackRuntime, names: set[str]) -> PackRuntime:
    """A runtime over exactly the named packs, preserving discovery order."""
    packs = tuple(pack for pack in runtime.catalog.manifests if pack.name in names)
    kept = {pack.name for pack in packs}
    return PackRuntime(
        catalog=PackCatalog(packs),
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
    return PackSet.from_runtime(discover(roots, config_path=config_path, entry_points=entry_points))


def installed() -> PackSet:
    """Enumerate every pack in the configured manifest-only discovery scope.

    Installed ``unigym.packs`` entry points are not resolved here: this scope
    never imports pack Python.  Use :meth:`PackSet.to_runtime` for an
    activation-capable runtime over the same catalog.
    """
    return PackSet.from_runtime(discover())


def require(*names: str, entry_points: bool = True) -> PackSet:
    """Return the filtered pack set over exactly *names* in the installed scope.

    Requirement resolution is not performed: each name must be installed in
    the same configured discovery scope.  A missing name fails before any
    handle exists and names the installed packs.

    Installed ``unigym.packs`` entry points are resolved by default (the UX
    for ``pip install``-ed packs such as ``unigym-robots``); pass
    ``entry_points=False`` for the manifest-only scope :func:`installed`
    exposes.
    """
    invalid = [name for name in names if not isinstance(name, str) or not name]
    if invalid:
        raise PackError(f"required pack names must be non-empty strings, got {invalid!r}")
    runtime = discover(entry_points=entry_points)
    full = PackSet.from_runtime(runtime)
    available = {pack.name for pack in full.packs()}
    missing = [name for name in names if name not in available]
    if missing:
        raise PackError(
            f"pack {missing[0]!r} is not installed in this scope; "
            f"required/installed packs: {sorted(available)}"
        )
    return PackSet.from_runtime(_filtered_runtime(runtime, set(names)))
