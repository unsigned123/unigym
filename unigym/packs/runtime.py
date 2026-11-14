"""Static-root pack runtime: discover, resolve resources, activate.

The runtime is a Phase 3 contract over ``unigym.pack/1``.  :func:`discover`
reads explicit roots, or the configured sources when the caller names none,
loads their manifests without importing pack Python, and returns an immutable
catalog plus each pack's root.  Resources resolve under their own pack root.
A code pack activates through normal Python: its ``python/`` directory is the
explicit import boundary, the manifest entrypoint is imported and called with
one :class:`PackContext`, and the handle is cached per runtime.

This is the only pack loader.  There is no implicit repository mount, no
mount order and no separate code loader.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import yaml

from unigym.assets.refs import canonical_asset_path
from unigym.core.paths import repo_root

from . import PackError
from .catalog import PackCatalog
from .schema import PROVIDER_ID_RE, PackManifestV1, ProvidedItem, load_manifest_v1

if TYPE_CHECKING:
    from unigym.adapters.base import Adapter
    from unigym.robots import RobotProfile
    from unigym.scenes.provider_params import ProviderParamSpec
    from unigym.scenes.providers import (
        ProviderOrigin,
        SceneProvider,
        SceneProviderRegistrySnapshot,
    )

_ASSET_PACK_PREFIX = "pack:"

_PACKS_ENV_VAR = "UNIGYM_PACKS"
_DEPS_ENV_VAR = "UNIGYM_DEPS_ROOT"
_PACKS_YAML_KEY = "packs"
_PACKS_TOP_DIR = "unigym_packs"
_LOCAL_YAML_NAME = "unigym.local.yaml"
_ENTRY_POINT_GROUP = "unigym.packs"


@dataclass(frozen=True)
class PackContext:
    """The explicit handle a pack entrypoint receives on activation.

    It names the activated pack, its root and the runtime that owns the
    catalog, so pack code never infers ambient state.  :meth:`resource`
    delegates to :meth:`PackRuntime.resource` and therefore applies the same
    inside-root path rules.
    """

    pack: str
    root: Path
    runtime: PackRuntime

    def resource(self, rel: str) -> Path:
        """Resolve *rel* inside this pack's root through the owning runtime."""
        return self.runtime.resource(self.pack, rel)


@dataclass(frozen=True)
class ActivationHandle:
    """The cached result of activating one pack.

    ``entrypoint`` is the manifest-level entrypoint (``None`` for a content
    pack or a code pack that declares none).  ``adapters`` names the manifest
    ``kind: adapter`` items this activation registered, in declaration order.
    ``robots`` names the manifest ``kind: robot`` items this activation
    registered, in declaration order; robots registered by the manifest-level
    entrypoint itself are not recorded here.  ``scene_providers`` names the
    manifest ``kind: scene`` items this activation registered or adopted from
    a pack-code mirror, in declaration order.  Manifest providers are still
    activated when the manifest-level entrypoint is absent; this handle caches
    the whole activation, not one entrypoint.
    """

    pack: str
    entrypoint: str | None
    adapters: tuple[str, ...] = ()
    robots: tuple[str, ...] = ()
    scene_providers: tuple[str, ...] = ()


def _import_boundary(manifest: PackManifestV1, entrypoint: str | None = None) -> Path:
    """Return a code pack's explicit ``python/`` import boundary.

    The directory is required: importing the entrypoint from an ambient
    ``sys.path`` entry would make activation depend on unrelated state.
    """
    python_dir = manifest.root / "python"
    if not python_dir.is_dir():
        declared = entrypoint if entrypoint is not None else manifest.entrypoint
        raise PackError(
            f"pack {manifest.name!r} entrypoint {declared!r} needs the pack's "
            f"python/ import boundary, but it is missing: {python_dir}"
        )
    return python_dir.resolve()


def _add_to_sys_path(python_dir: Path) -> None:
    """Put *python_dir* at the front of ``sys.path`` exactly once.

    The boundary stays for the process lifetime: a pack entrypoint may import
    its own submodules after activation returns, so removing it immediately
    would break the pack.
    """
    for entry in sys.path:
        try:
            if Path(entry or ".").resolve() == python_dir:
                return
        except OSError:
            continue
    sys.path.insert(0, str(python_dir))


def _resolve_entrypoint(entrypoint: str) -> Callable[..., Any]:
    """Import ``module:attribute`` and return the callable target."""
    module_name, separator, attribute = entrypoint.partition(":")
    if not separator or not module_name or not attribute:
        raise PackError(f"entrypoint {entrypoint!r} must be 'module:attribute'")
    module = importlib.import_module(module_name)
    try:
        target = getattr(module, attribute)
    except AttributeError as err:
        raise PackError(
            f"entrypoint {entrypoint!r} resolves to missing attribute "
            f"{attribute!r} on module {module_name!r}"
        ) from err
    if not callable(target):
        raise PackError(f"entrypoint {entrypoint!r} target {target!r} is not callable")
    return target


def _activate_adapter_providers(manifest: PackManifestV1) -> tuple[str, ...]:
    """Import and register manifest-declared ``kind: adapter`` providers.

    The item id becomes the adapter name a run config selects.  Validation is
    complete before any registration: every item must carry an entrypoint,
    resolve to an ``Adapter`` subclass class and name a free registry slot.
    A later invalid item therefore leaves no earlier registration behind, and
    retrying the same pack fails on the original defect rather than on a
    leaked duplicate.  The adapter package imports torch, so it is imported
    lazily here and never at runtime-module import time.  Other provider kinds
    stay inert in this phase.
    """
    items = tuple(item for item in manifest.provides.get("providers", ()) if item.kind == "adapter")
    if not items:
        return ()

    from unigym.adapters import ADAPTER_REGISTRY, Adapter, register_adapter

    for item in items:
        if item.entrypoint is None:
            raise PackError(
                f"pack {manifest.name!r} provider id {item.id!r} declares "
                f"kind 'adapter' but no entrypoint; declare "
                f"'entrypoint: module:AdapterClass'"
            )

    python_dir = _import_boundary(manifest, items[0].entrypoint)
    _add_to_sys_path(python_dir)

    validated: list[tuple[ProvidedItem, type[Adapter]]] = []
    proposed: set[str] = set()
    for item in items:
        where = f"pack {manifest.name!r} provider id {item.id!r}"
        try:
            target = _resolve_entrypoint(item.entrypoint)
        except Exception as err:
            raise PackError(
                f"{where} entrypoint {item.entrypoint!r} failed to import: {err}"
            ) from err
        if not isinstance(target, type) or not issubclass(target, Adapter):
            raise PackError(
                f"{where} entrypoint {item.entrypoint!r} target {target!r} must be "
                f"an Adapter subclass class, not an instance or factory"
            )
        existing = ADAPTER_REGISTRY.get(item.id)
        if existing is not None:
            raise PackError(
                f"{where} entrypoint {item.entrypoint!r} cannot register adapter "
                f"name {item.id!r}: already registered to "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        if item.id in proposed:
            raise PackError(f"{where} appears more than once in the adapter provider list")
        proposed.add(item.id)
        validated.append((item, target))

    registered: list[str] = []
    try:
        for item, target in validated:
            register_adapter(item.id, target)
            registered.append(item.id)
    except Exception as err:
        for name in registered:
            ADAPTER_REGISTRY.pop(name, None)
        raise PackError(
            f"pack {manifest.name!r} provider id {item.id!r} entrypoint "
            f"{item.entrypoint!r} failed to register: {err}"
        ) from err
    return tuple(item.id for item in items)


def _constant_robot_factory(where: str, profile: RobotProfile) -> Callable[..., RobotProfile]:
    """Return the parameterless factory a resource robot registers under its id."""

    def factory(**params: object) -> RobotProfile:
        if params:
            raise PackError(
                f"{where} is a resource robot and takes no parameters, " f"got {sorted(params)}"
            )
        return profile

    return factory


def _activate_robot_providers(manifest: PackManifestV1, runtime: PackRuntime) -> tuple[str, ...]:
    """Load and register manifest-declared ``kind: robot`` providers.

    The item id becomes the robot name a run config selects.  An item is
    either an ``entrypoint`` factory (R1) or a ``resource`` pointing at one
    ``unigym.robot/1`` document.  A resource-only provider needs no ``python/``
    directory and never mutates ``sys.path``.  When both forms are mixed, the
    import boundary comes from the first item that declares an entrypoint.

    Validation is complete before any registration: every resource profile is
    loaded and checked against its item id, every entrypoint resolves to a
    callable factory and every item names a free registry slot (the policy
    owner is :mod:`unigym.robots`).  A later invalid item therefore leaves no
    earlier registration behind, and retrying the same pack fails on the
    original defect rather than on a leaked duplicate.  Return-type and
    profile-name legality for entrypoint factories stay in
    :func:`unigym.robots.build_robot_profile`; a resource profile's name is
    checked here because the document is already loaded.  Other registration
    helpers are imported lazily here.
    """
    items = tuple(item for item in manifest.provides.get("providers", ()) if item.kind == "robot")
    if not items:
        return ()

    from unigym.robots import (
        load_robot_profile,
        register_robot,
        validate_robot_registration,
    )

    first_entrypoint = next(
        (item.entrypoint for item in items if item.entrypoint is not None), None
    )
    if first_entrypoint is not None:
        python_dir = _import_boundary(manifest, first_entrypoint)
        _add_to_sys_path(python_dir)

    validated: list[tuple[ProvidedItem, Callable[..., RobotProfile]]] = []
    for item in items:
        where = f"pack {manifest.name!r} provider id {item.id!r}"
        if item.entrypoint is not None:
            entrypoint = item.entrypoint
            try:
                target = _resolve_entrypoint(entrypoint)
            except Exception as err:
                raise PackError(
                    f"{where} entrypoint {entrypoint!r} failed to import: {err}"
                ) from err
            try:
                validate_robot_registration(item.id, target)
            except ValueError as err:
                raise PackError(
                    f"{where} entrypoint {entrypoint!r} cannot register robot "
                    f"name {item.id!r}: {err}"
                ) from err
            alias = next((earlier.id for earlier, factory in validated if factory is target), None)
            if alias is not None:
                raise PackError(
                    f"{where} entrypoint {entrypoint!r} registers the same robot "
                    f"factory as provider id {alias!r} under a second name {item.id!r}; "
                    f"one factory owns exactly one name"
                )
            validated.append((item, target))
            continue

        resource = item.resource
        assert resource is not None  # schema guarantees exactly one locator
        try:
            resource_path = runtime.resource(manifest.name, resource)
        except PackError as err:
            raise PackError(f"{where} resource {resource!r} failed to load: {err}") from err
        try:
            profile = load_robot_profile(resource_path)
        except ValueError as err:
            raise PackError(f"{where} resource {resource!r} failed to load: {err}") from err
        if profile.name != item.id:
            raise PackError(
                f"{where} resource {resource!r} declares profile name "
                f"{profile.name!r}; the resource profile name must equal the "
                f"provider id {item.id!r}"
            )
        factory = _constant_robot_factory(where, profile)
        try:
            validate_robot_registration(item.id, factory)
        except ValueError as err:
            raise PackError(
                f"{where} resource {resource!r} cannot register robot " f"name {item.id!r}: {err}"
            ) from err
        validated.append((item, factory))

    for item, target in validated:
        locator = (
            f"entrypoint {item.entrypoint!r}"
            if item.entrypoint is not None
            else (f"resource {item.resource!r}")
        )
        try:
            register_robot(item.id, target)
        except Exception as err:
            raise PackError(
                f"pack {manifest.name!r} provider id {item.id!r} {locator} "
                f"failed to register: {err}"
            ) from err
    return tuple(item.id for item in items)


def _provider_schema_diff(
    left: Mapping[str, ProviderParamSpec] | None,
    right: Mapping[str, ProviderParamSpec] | None,
) -> list[str]:
    """Return the parameter names whose normalized declarations differ."""
    left_map = dict(left or {})
    right_map = dict(right or {})
    keys = sorted(set(left_map) | set(right_map))
    return [key for key in keys if left_map.get(key) != right_map.get(key)]


def _manifest_scene_provider_origin(
    manifest: PackManifestV1,
    runtime: PackRuntime,
    item: ProvidedItem,
) -> ProviderOrigin:
    """Build the exact provenance record for one manifest ``kind: scene`` item."""
    from unigym.scenes.providers import ProviderOrigin

    return ProviderOrigin(
        kind="manifest",
        pack=manifest.name,
        pack_version=manifest.version,
        root=str(manifest.root),
        discovered_by=runtime.origins.get(manifest.name, "unknown"),
        face="providers",
        item_id=item.id,
        entrypoint=item.entrypoint,
        activation_entrypoint=manifest.entrypoint,
    )


def _attribute_pack_code_scene_providers(
    manifest: PackManifestV1,
    runtime: PackRuntime,
    before: SceneProviderRegistrySnapshot,
) -> None:
    """Label scene providers the pack registered itself during activation."""
    from unigym.scenes.providers import (
        ProviderOrigin,
        attribute_scene_provider,
        registered_scene_providers,
        scene_provider_origin,
    )

    origin = ProviderOrigin(
        kind="pack-code",
        pack=manifest.name,
        pack_version=manifest.version,
        root=str(manifest.root),
        discovered_by=runtime.origins.get(manifest.name, "unknown"),
        item_id=None,
        activation_entrypoint=manifest.entrypoint,
    )
    for name in registered_scene_providers():
        if name in before.providers:
            continue
        # A manifest item adopted this name and already recorded the exact
        # manifest origin; only unattributed registrations are pack-code.
        if scene_provider_origin(name).kind != "unknown":
            continue
        attribute_scene_provider(name, origin)


def _activate_scene_provider_providers(
    manifest: PackManifestV1,
    runtime: PackRuntime,
    *,
    window_names: frozenset[str],
) -> tuple[str, ...]:
    """Import and register manifest-declared ``kind: scene`` providers.

    A scene provider item is entrypoint-backed: the item id becomes the
    provider name and the entrypoint resolves to the pure provider factory.
    Validation is complete before any registration: every entrypoint imports,
    every target is a documented callable that is not a ``Scene`` instance,
    every id uses the adapter/robot grammar and is free (unless this pack's
    entrypoint already registered that exact factory as its code mirror), and
    every manifest ``params`` declaration is a valid schema.  When a manifest
    item adopts a pack-code registration, the two parameter schemas must match
    exactly: a one-sided declaration (manifest only or code only) is forbidden
    just like a differing value, and the manifest schema is authoritative.

    Args:
        manifest: The pack being activated.
        runtime: The runtime that owns discovery provenance labels.
        window_names: Public provider names the pack's entrypoint registered
            during this activation window.  Those names may be adopted by a
            manifest item; any other pre-existing name is a duplicate.

    Returns:
        The manifest ``kind: scene`` item ids, in declaration order.

    Raises:
        PackError: A scene provider item or its parameter schema is invalid.
    """
    items = tuple(item for item in manifest.provides.get("providers", ()) if item.kind == "scene")
    if not items:
        return ()

    from unigym.scenes.provider_params import (
        SceneProviderParamError,
        normalize_manifest_params,
        schemas_equal,
    )
    from unigym.scenes.providers import (
        attribute_scene_provider,
        register_scene_provider,
        snapshot_scene_providers,
    )
    from unigym.scenes.scene_ir import Scene

    first_entrypoint = next(
        (item.entrypoint for item in items if item.entrypoint is not None), None
    )
    if first_entrypoint is not None:
        python_dir = _import_boundary(manifest, first_entrypoint)
        _add_to_sys_path(python_dir)

    snapshot = snapshot_scene_providers()
    validated: list[
        tuple[ProvidedItem, SceneProvider, Mapping[str, ProviderParamSpec] | None, bool]
    ] = []
    proposed: set[str] = set()
    for item in items:
        where = f"pack {manifest.name!r} scene provider id {item.id!r}"
        if item.id in proposed:
            raise PackError(f"{where} appears more than once in the scene provider list")
        proposed.add(item.id)
        if PROVIDER_ID_RE.fullmatch(item.id) is None:
            raise PackError(
                f"{where} id must use the adapter/robot grammar "
                "[a-z][a-z0-9_]*(?:\\.[a-z][a-z0-9_]*)*"
            )
        # The manifest schema already rejects a resource-backed scene item; this
        # re-check keeps activation honest for a hand-built manifest object.
        if item.resource is not None:
            raise PackError(
                f"{where} is entrypoint-backed and must not declare "
                f"resource {item.resource!r}; drop 'resource:'"
            )
        if item.entrypoint is None:
            raise PackError(f"{where} requires an entrypoint; declare 'entrypoint: module:factory'")
        try:
            target = _resolve_entrypoint(item.entrypoint)
        except Exception as err:
            raise PackError(
                f"{where} entrypoint {item.entrypoint!r} failed to import: {err}"
            ) from err
        if isinstance(target, Scene):
            raise PackError(
                f"{where} entrypoint {item.entrypoint!r} target is a Scene instance, "
                "not a provider factory"
            )
        if not (getattr(target, "__doc__", "") or "").strip():
            raise PackError(f"{where} entrypoint {item.entrypoint!r} target must have a docstring")
        try:
            manifest_schema = (
                normalize_manifest_params(item.params) if item.params is not None else None
            )
        except SceneProviderParamError as err:
            raise PackError(f"{where} params declaration is invalid: {err}") from err

        adopted = item.id in snapshot.providers
        if adopted:
            if item.id not in window_names:
                raise PackError(
                    f"{where} name {item.id!r} is already registered; a manifest "
                    "scene provider id must be free"
                )
            if snapshot.providers[item.id] is not target:
                raise PackError(
                    f"{where} entrypoint {item.entrypoint!r} names a different factory "
                    f"than the one pack code registered under {item.id!r}"
                )
            code_schema = snapshot.params.get(item.id)
            if not schemas_equal(manifest_schema, code_schema):
                differing = _provider_schema_diff(manifest_schema, code_schema)
                one_sided = (manifest_schema is None) != (code_schema is None)
                hint = (
                    "; declare the schema on both the manifest item and the pack-code "
                    "registration, or on neither"
                    if one_sided
                    else "; the manifest schema is authoritative"
                )
                raise PackError(
                    f"{where} manifest params {manifest_schema!r} disagree with the "
                    f"pack-code registration schema {code_schema!r} "
                    f"(differing keys: {differing}){hint}"
                )
        else:
            alias = next(
                (name for name, factory in snapshot.providers.items() if factory is target),
                None,
            )
            if alias is None:
                alias = next(
                    (name for name, factory in snapshot.inline.items() if factory is target),
                    None,
                )
            if alias is not None:
                raise PackError(
                    f"{where} entrypoint {item.entrypoint!r} registers the same factory "
                    f"as scene provider {alias!r} under a second name {item.id!r}; "
                    "one factory owns exactly one name"
                )
        validated.append((item, target, manifest_schema, adopted))

    for item, target, manifest_schema, adopted in validated:
        origin = _manifest_scene_provider_origin(manifest, runtime, item)
        if adopted:
            attribute_scene_provider(item.id, origin, params=manifest_schema)
        else:
            try:
                register_scene_provider(
                    item.id,
                    target,
                    params=item.params,
                    origin=origin,
                )
            except Exception as err:
                raise PackError(
                    f"pack {manifest.name!r} scene provider id {item.id!r} entrypoint "
                    f"{item.entrypoint!r} failed to register: {err}"
                ) from err
    return tuple(item.id for item in items)


@dataclass(frozen=True)
class PackRuntime:
    """An immutable catalog plus the resolved root of every discovered pack.

    ``root_by_pack`` must name exactly the catalog's packs; the constructor
    fails fast instead of letting a later resource or activation lookup miss.
    ``origins`` is the discovery label per pack (``explicit-root``,
    ``env:UNIGYM_PACKS``, ``local-yaml``, ``entry-point`` or ``deps-root``);
    a hand-built runtime without labels reports ``unknown`` provenance.
    ``_handles`` is the private per-runtime activation cache.
    """

    catalog: PackCatalog
    root_by_pack: Mapping[str, Path]
    origins: Mapping[str, str] = field(default_factory=dict)
    _handles: dict[str, ActivationHandle] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        """Freeze the root/origin mappings and verify they match the catalog."""
        names = self.catalog.packs()
        provided = set(self.root_by_pack)
        expected = set(names)
        if provided != expected:
            raise PackError(
                f"pack runtime roots {sorted(provided)} do not match "
                f"catalog packs {sorted(expected)}"
            )
        roots = {name: Path(self.root_by_pack[name]).resolve() for name in names}
        labels = {str(name): str(label) for name, label in self.origins.items()}
        unknown = sorted(set(labels) - expected)
        if unknown:
            raise PackError(
                f"pack runtime origin labels {unknown} do not match "
                f"catalog packs {sorted(expected)}"
            )
        object.__setattr__(self, "root_by_pack", MappingProxyType(roots))
        object.__setattr__(self, "origins", MappingProxyType(labels))
        object.__setattr__(self, "_handles", {})

    def resource(self, pack: str, rel: str) -> Path:
        """Resolve *rel* to an absolute path inside the named pack's root.

        The same relative/inside-root rules as the manifest schema apply:
        non-empty, not absolute, no ``..`` component, no symlink escape.

        Raises:
            PackError: The pack is unknown, the path is illegal, or it does
                not exist.  The message names both the pack and the path.
        """
        self.catalog.get(pack)
        root = self.root_by_pack[pack]
        where = f"pack {pack!r} resource {rel!r}"
        if not isinstance(rel, str) or not rel:
            raise PackError(f"{where} must be a non-empty relative path")
        relative = Path(rel)
        if not relative.parts:
            raise PackError(f"{where} must be a non-empty relative path")
        if relative.is_absolute():
            raise PackError(f"{where} must be relative to the pack root")
        if any(part == ".." for part in relative.parts):
            raise PackError(f"{where} must not contain '..'")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise PackError(f"{where} escapes the pack root {root}") from None
        if not candidate.exists():
            raise PackError(f"{where} does not exist under {root}")
        return candidate

    def asset_path(self, value: str) -> Path:
        """Resolve one declared asset reference to an absolute path.

        Accepted forms are a pack-qualified ``pack:<pack>/<rel>`` reference and
        an absolute or ``~`` filesystem path:

        - a pack reference is canonicalized with the pure texture-leaf rule
          (:func:`unigym.assets.refs.canonical_asset_path`) and then
          resolved through :meth:`resource`, inheriting the inside-root path
          checks and the existence requirement;
        - an absolute or ``~`` path is the run config's own asset: it is
          expanded and returned without an existence check, matching the legacy
          absolute pass-through;
        - a bare relative reference is rejected because it has no owning pack.

        Raises:
            PackError: The value is empty or not a string, the reference is a
                bare relative path, the pack is unknown, the path is illegal or
                the resource is absent.  Pack and reference are both named.
        """
        if not isinstance(value, str) or not value:
            raise PackError(f"asset reference {value!r} must be a non-empty string")

        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        if not value.startswith(_ASSET_PACK_PREFIX):
            raise PackError(
                f"asset reference {value!r} is not pack-qualified or absolute: "
                "write 'pack:<pack>/<rel>' or an absolute filesystem path; "
                "a bare relative asset reference has no owning pack"
            )

        canonical = canonical_asset_path(value)
        pack, separator, rel = canonical[len(_ASSET_PACK_PREFIX) :].partition("/")
        if not separator or not pack or not rel:
            raise PackError(
                f"asset reference {value!r} must be 'pack:<pack>/<rel>'; "
                "the pack name or relative resource path is missing"
            )
        try:
            return self.resource(pack, rel)
        except PackError as err:
            raise PackError(f"asset reference {value!r}: {err}") from err

    def resource_for(self, face: str, id: str) -> Path:
        """Resolve the declared resource of the ``(face, id)`` catalog item.

        The lookup is catalog-only: it never activates or imports pack code.
        An entrypoint-only item has no resource path, so it raises a teaching
        error that points at :meth:`activate` instead.

        Raises:
            PackError: The face/id is unknown, the item declares an entrypoint
                rather than a resource, or the declared resource fails the
                existing path checks.
        """
        entry = self.catalog.entry(face, id)
        if entry.resource is None:
            raise PackError(
                f"pack {entry.pack!r} face {entry.face!r} id {entry.id!r} "
                f"declares entrypoint {entry.entrypoint!r}, not a resource; "
                f"use runtime.activate({entry.pack!r}) to activate the pack"
            )
        return self.resource(entry.pack, entry.resource)

    def activate(self, pack: str) -> ActivationHandle:
        """Activate *pack* once and return the cached handle.

        A manifest-level entrypoint is imported normally and called with
        exactly one :class:`PackContext` argument; there is no zero-argument
        fallback.  After it (or without it), ``providers`` items with
        ``kind: adapter`` are imported and registered under their item id,
        then ``kind: robot`` items are registered through the robot registry,
        then ``kind: scene`` items are registered through the scene-provider
        registry.  A robot item is either an entrypoint factory or a resource
        pointing at one ``unigym.robot/1`` document; a resource-only robot list
        needs no ``python/`` directory and never mutates ``sys.path``, while a
        mixed list takes its import boundary from the first entrypoint item.
        Scene provider items are entrypoint-only; their manifest ``params``
        schema is authoritative and must not drift from a pack-code mirror.
        Every item set is validated before any registration.  Other provider
        kinds stay inert.  Any imported pack module can register through
        ``unigym.pack_api`` at import time, regardless of the kind that pulled
        it in, so the adapter, robot and scene-provider registries are all
        snapshotted before the first import or provider registration.  Failures
        are wrapped with the pack, item id and locator named; a failed
        activation restores all three registries to their prior contents,
        including names pack code registered itself.

        Raises:
            PackError: The pack is unknown or activation failed.
        """
        cached = self._handles.get(pack)
        if cached is not None:
            return cached

        manifest = self.catalog.get(pack)
        entrypoint = manifest.entrypoint
        adapter_items = tuple(
            item for item in manifest.provides.get("providers", ()) if item.kind == "adapter"
        )
        robot_items = tuple(
            item for item in manifest.provides.get("providers", ()) if item.kind == "robot"
        )
        scene_items = tuple(
            item for item in manifest.provides.get("providers", ()) if item.kind == "scene"
        )
        # Importing any pack module can register through ``unigym.pack_api``,
        # not only modules pulled in by their own provider kind: a scene item
        # import may register an adapter or robot.  Snapshot every registry
        # before the first import so a failed activation can restore all three.
        # The adapter registry lives in the torch-only adapters package, so
        # snapshot it only when activation can actually execute pack code
        # (an entrypoint, or an adapter / robot / scene provider entrypoint).
        # A resource-only robot pack imports nothing and stays engine-free.
        activation_surface = entrypoint is not None or bool(
            adapter_items or robot_items or scene_items
        )
        imports_pack_code = entrypoint is not None or any(
            item.entrypoint is not None for item in (*adapter_items, *robot_items, *scene_items)
        )
        adapter_registry = None
        adapters_before: set[str] | None = None
        robots_before: dict[str, Callable[..., RobotProfile]] | None = None
        scene_before: SceneProviderRegistrySnapshot | None = None
        if activation_surface:
            if imports_pack_code:
                from unigym.adapters import ADAPTER_REGISTRY as adapter_registry

                adapters_before = set(adapter_registry)
            from unigym.robots import snapshot_robots
            from unigym.scenes.providers import snapshot_scene_providers

            robots_before = snapshot_robots()
            scene_before = snapshot_scene_providers()

        def rollback_adapters() -> None:
            """Remove adapter names a failed activation added to the registry.

            ``register_adapter`` rejects an existing name, so activation can
            only add entries; removing the additions restores the registry.
            """
            if adapter_registry is None or adapters_before is None:
                return
            for name in set(adapter_registry) - adapters_before:
                adapter_registry.pop(name, None)

        def rollback_robots() -> None:
            """Restore the external robot registry a failed activation changed."""
            if robots_before is None:
                return
            from unigym.robots import restore_robots

            restore_robots(robots_before)

        def rollback_scene_providers() -> None:
            """Restore the scene-provider registries a failed activation changed."""
            if scene_before is None:
                return
            from unigym.scenes.providers import restore_scene_providers

            restore_scene_providers(scene_before)

        try:
            if entrypoint is not None:
                context = PackContext(pack=pack, root=manifest.root, runtime=self)
                python_dir = _import_boundary(manifest, entrypoint)
                _add_to_sys_path(python_dir)
                target = _resolve_entrypoint(entrypoint)
                target(context)

            window_names: frozenset[str] = frozenset()
            if scene_before is not None:
                from unigym.scenes.providers import registered_scene_providers

                window_names = frozenset(
                    set(registered_scene_providers()) - set(scene_before.providers)
                )

            adapters = _activate_adapter_providers(manifest)
            robots = _activate_robot_providers(manifest, self)
            scene_providers = _activate_scene_provider_providers(
                manifest, self, window_names=window_names
            )
            if scene_before is not None:
                _attribute_pack_code_scene_providers(manifest, self, scene_before)
        except PackError:
            rollback_adapters()
            rollback_robots()
            rollback_scene_providers()
            raise
        except Exception as err:
            rollback_adapters()
            rollback_robots()
            rollback_scene_providers()
            if entrypoint is None:
                raise PackError(f"pack {pack!r} provider activation failed: {err}") from err
            raise PackError(
                f"pack {pack!r} entrypoint {entrypoint!r} failed to activate: {err}"
            ) from err

        handle = ActivationHandle(
            pack=pack,
            entrypoint=entrypoint,
            adapters=adapters,
            robots=robots,
            scene_providers=scene_providers,
        )
        self._handles[pack] = handle
        return handle

    def activate_all(self) -> tuple[ActivationHandle, ...]:
        """Activate every catalog pack in catalog order and return the handles.

        Repeated calls return the same cached handles because each pack is
        activated at most once per runtime.

        Raises:
            PackError: Any pack fails to activate.
        """
        return tuple(self.activate(pack) for pack in self.catalog.packs())


@dataclass(frozen=True)
class AssetContext:
    """The explicit asset-path handle render and template code receives.

    It carries the runtime that owns asset resolution, so a consumer never
    infers a global asset face.  :meth:`resolve` delegates to
    :meth:`PackRuntime.asset_path` and owns no second resolution rule.
    """

    runtime: PackRuntime

    def resolve(self, value: str) -> str:
        """Resolve *value* to an absolute asset path string."""
        return str(self.runtime.asset_path(value))


def _env_pack_roots() -> list[Path]:
    """Roots named by ``UNIGYM_PACKS``, in declaration order.

    The variable is an ``os.pathsep``-separated list.  An empty entry is a
    declaration error, not a silent skip: it usually means a stray separator
    or an unset shell variable was substituted into the list.
    """
    value = os.environ.get(_PACKS_ENV_VAR)
    if value is None:
        return []
    roots: list[Path] = []
    for index, entry in enumerate(value.split(os.pathsep)):
        if not entry.strip():
            raise PackError(
                f"{_PACKS_ENV_VAR} entry {index} is empty; expected a "
                f"{os.pathsep!r}-separated list of pack roots"
            )
        roots.append(Path(entry))
    return roots


def _local_yaml_roots(path: Path) -> list[Path]:
    """Roots listed under ``packs:`` in one local overlay (or none).

    A missing file means the overlay declares nothing.  A file that exists but
    cannot be read, parsed or shaped fails: a malformed declaration must not
    silently disable configured discovery.  Relative entries resolve against
    the overlay's own directory.
    """
    if not path.is_file():
        return []
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError) as err:
        raise PackError(f"unreadable local pack config {path}: {err}") from err
    if not isinstance(document, Mapping):
        raise PackError(f"local pack config {path} must be a mapping")
    value = document.get(_PACKS_YAML_KEY)
    if value is None:
        return []
    if not isinstance(value, list):
        raise PackError(f"{path}: 'packs' must be a list of pack roots")
    roots: list[Path] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, str) or not entry.strip():
            raise PackError(f"{path}: 'packs' entry {index} must be a non-empty path")
        candidate = Path(entry).expanduser()
        roots.append(candidate if candidate.is_absolute() else path.parent / candidate)
    return roots


def _deps_pack_roots() -> list[Path]:
    """Every directory under ``<UNIGYM_DEPS_ROOT>/unigym_packs``, sorted.

    An unset or empty ``UNIGYM_DEPS_ROOT``, or a missing deps/top directory,
    declares no packs.  A directory that does exist is a root and therefore
    still has to carry a manifest; that check belongs to :func:`discover`.
    """
    value = os.environ.get(_DEPS_ENV_VAR)
    if value is None or not value.strip():
        return []
    top = Path(value).expanduser() / _PACKS_TOP_DIR
    if not top.is_dir():
        return []
    return [child for child in sorted(top.iterdir()) if child.is_dir()]


def _entry_point_target(entry: importlib.metadata.EntryPoint) -> Path:
    """Resolve one entry point to the pack root it declares.

    The target is either a path string (``str`` or ``os.PathLike``) or a
    zero-argument callable returning one.  A load failure, a raising callable
    or a wrong type fails naming the entry point.
    """
    name = entry.name
    try:
        target = entry.load()
    except Exception as err:
        raise PackError(f"pack entry point {name!r} failed to load its target: {err}") from err
    if callable(target):
        try:
            target = target()
        except Exception as err:
            raise PackError(f"pack entry point {name!r} callable target failed: {err}") from err
    if not isinstance(target, (str, os.PathLike)):
        raise PackError(
            f"pack entry point {name!r} target must be a path string or a "
            f"zero-argument callable returning one, got {type(target).__name__}"
        )
    return Path(target)


def _entry_point_roots() -> list[tuple[Path, str]]:
    """Installed ``unigym.packs`` entry-point roots, sorted by entry-point name.

    Resolving an entry point loads and may import its declared target, so this
    helper is only called for opt-in configured discovery.  Default discovery
    never reads installed entry points.
    """
    selected = importlib.metadata.entry_points().select(group=_ENTRY_POINT_GROUP)
    roots: list[tuple[Path, str]] = []
    for entry in sorted(selected, key=lambda candidate: candidate.name):
        roots.append((_entry_point_target(entry), entry.name))
    return roots


def _configured_roots(
    config_path: str | Path | None, *, entry_points: bool
) -> tuple[list[Path], dict[Path, str], dict[Path, str]]:
    """Configured roots in source order, with discovery labels and entry-point names.

    Order: ``UNIGYM_PACKS``, the local overlay, installed ``unigym.packs``
    entry points (name-sorted) when opted in, then ``<UNIGYM_DEPS_ROOT>``
    packs.  The label map keys canonical roots to their discovery label for
    :class:`~unigym.scenes.providers.ProviderOrigin.discovered_by`; the name
    map keys canonical entry-point roots to their entry-point name so a
    manifest failure can name the entry point.
    """
    local_yaml = (
        Path(config_path) if config_path is not None else Path(repo_root()) / _LOCAL_YAML_NAME
    )
    roots: list[Path] = []
    labels: dict[Path, str] = {}
    entry_names: dict[Path, str] = {}

    def add(root: Path, label: str) -> None:
        roots.append(root)
        labels.setdefault(root.resolve(), label)

    for root in _env_pack_roots():
        add(root, "env:UNIGYM_PACKS")
    for root in _local_yaml_roots(local_yaml):
        add(root, "local-yaml")
    if entry_points:
        for root, name in _entry_point_roots():
            add(root, "entry-point")
            entry_names.setdefault(root.resolve(), name)
    for root in _deps_pack_roots():
        add(root, "deps-root")
    return roots, labels, entry_names


def discover(
    roots: Iterable[str | Path] | None = None,
    *,
    config_path: str | Path | None = None,
    entry_points: bool = False,
) -> PackRuntime:
    """Discover packs from explicit roots or the configured sources.

    A non-``None`` *roots* iterable is authoritative: only those roots are
    read, in order, and configured sources are ignored.  ``discover([])`` is
    therefore an explicit empty discovery.

    ``roots=None`` reads, in order, the ``UNIGYM_PACKS`` entries, the
    ``packs:`` entries of the local overlay (``<repo>/unigym.local.yaml`` by
    default), installed ``unigym.packs`` entry points when *entry_points* is
    true, and the sorted ``<UNIGYM_DEPS_ROOT>/unigym_packs/*`` directories.

    *config_path* overrides the local-overlay path for embedding and tests;
    ``None`` keeps the repository overlay.

    *entry_points* is an explicit opt-in that only affects configured
    discovery.  Passing it with explicit roots is a teaching error rather than
    a silent ignore.  Resolving an entry point loads and may import the target;
    default discovery never reads installed entry points.

    Canonical duplicate roots are visited once.  Every root must carry a valid
    ``unigym.pack.yaml``; duplicate pack names fail.  Default discovery reads
    manifests only and never imports pack Python.

    Raises:
        PackError: A configured declaration is malformed, a root has no valid
            manifest, two roots name the same pack, or *entry_points* is
            combined with explicit roots.
    """
    if roots is not None and entry_points:
        raise PackError(
            "discover(entry_points=True) only extends configured discovery; "
            "explicit roots must already name every pack root"
        )

    if roots is None:
        selected, labels, entry_names = _configured_roots(config_path, entry_points=entry_points)
        default_label = "unknown"
    else:
        selected, labels, entry_names = roots, {}, {}
        default_label = "explicit-root"

    manifests: list[PackManifestV1] = []
    root_by_pack: dict[str, Path] = {}
    origin_by_pack: dict[str, str] = {}
    seen: set[Path] = set()
    for raw in selected:
        root = Path(raw).resolve()
        if root in seen:
            continue
        seen.add(root)
        try:
            manifest = load_manifest_v1(root)
        except PackError as err:
            entry_name = entry_names.get(root)
            if entry_name is None:
                raise
            raise PackError(f"pack entry point {entry_name!r} resolved to {root}: {err}") from err
        owner = root_by_pack.get(manifest.name)
        if owner is not None:
            raise PackError(
                f"duplicate pack name {manifest.name!r} discovered at "
                f"roots {owner} and {manifest.root}"
            )
        root_by_pack[manifest.name] = manifest.root
        origin_by_pack[manifest.name] = labels.get(root, default_label)
        manifests.append(manifest)
    return PackRuntime(
        catalog=PackCatalog(tuple(manifests)),
        root_by_pack=root_by_pack,
        origins=origin_by_pack,
    )


__all__ = ["ActivationHandle", "AssetContext", "PackContext", "PackRuntime", "discover"]
