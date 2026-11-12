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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import yaml

from unigym.assets.refs import canonical_asset_path
from unigym.core.paths import repo_root

from . import PackError
from .catalog import PackCatalog
from .schema import PackManifestV1, load_manifest_v1

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

    ``entrypoint`` is ``None`` for a content pack or a code pack that declares
    no manifest entrypoint; both are no-ops in this phase.
    """

    pack: str
    entrypoint: str | None


def _import_boundary(manifest: PackManifestV1) -> Path:
    """Return a code pack's explicit ``python/`` import boundary.

    The directory is required: importing the entrypoint from an ambient
    ``sys.path`` entry would make activation depend on unrelated state.
    """
    python_dir = manifest.root / "python"
    if not python_dir.is_dir():
        raise PackError(
            f"pack {manifest.name!r} has entrypoint {manifest.entrypoint!r} "
            f"but no python/ directory: {python_dir}"
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


def _resolve_entrypoint(entrypoint: str):
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


@dataclass(frozen=True)
class PackRuntime:
    """An immutable catalog plus the resolved root of every discovered pack.

    ``root_by_pack`` must name exactly the catalog's packs; the constructor
    fails fast instead of letting a later resource or activation lookup miss.
    ``_handles`` is the private per-runtime activation cache.
    """

    catalog: PackCatalog
    root_by_pack: Mapping[str, Path]
    _handles: dict[str, ActivationHandle] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        """Freeze the root mapping and verify it matches the catalog."""
        names = self.catalog.packs()
        provided = set(self.root_by_pack)
        expected = set(names)
        if provided != expected:
            raise PackError(
                f"pack runtime roots {sorted(provided)} do not match "
                f"catalog packs {sorted(expected)}"
            )
        roots = {name: Path(self.root_by_pack[name]).resolve() for name in names}
        object.__setattr__(self, "root_by_pack", MappingProxyType(roots))
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

        Content packs and code packs without a manifest entrypoint are no-ops.
        A code pack's entrypoint is imported normally and called with exactly
        one :class:`PackContext` argument; there is no zero-argument fallback.
        Failures are wrapped with the pack and entrypoint named.

        Raises:
            PackError: The pack is unknown or activation failed.
        """
        cached = self._handles.get(pack)
        if cached is not None:
            return cached

        manifest = self.catalog.get(pack)
        entrypoint = manifest.entrypoint
        if entrypoint is None:
            handle = ActivationHandle(pack=pack, entrypoint=None)
            self._handles[pack] = handle
            return handle

        context = PackContext(pack=pack, root=manifest.root, runtime=self)
        try:
            python_dir = _import_boundary(manifest)
            _add_to_sys_path(python_dir)
            target = _resolve_entrypoint(entrypoint)
            target(context)
        except Exception as err:
            raise PackError(
                f"pack {pack!r} entrypoint {entrypoint!r} failed to activate: {err}"
            ) from err

        handle = ActivationHandle(pack=pack, entrypoint=entrypoint)
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
) -> tuple[list[Path], dict[Path, str]]:
    """Configured roots in source order, plus each entry-point root's origin.

    Order: ``UNIGYM_PACKS``, the local overlay, installed ``unigym.packs``
    entry points (name-sorted) when opted in, then ``<UNIGYM_DEPS_ROOT>``
    packs.  The origin map keys canonical roots to their entry-point name so a
    manifest failure can name the entry point.
    """
    local_yaml = (
        Path(config_path) if config_path is not None else Path(repo_root()) / _LOCAL_YAML_NAME
    )
    roots = [*_env_pack_roots(), *_local_yaml_roots(local_yaml)]
    origins: dict[Path, str] = {}
    if entry_points:
        for root, name in _entry_point_roots():
            origins.setdefault(root.resolve(), name)
            roots.append(root)
    roots.extend(_deps_pack_roots())
    return roots, origins


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
        selected, origins = _configured_roots(config_path, entry_points=entry_points)
    else:
        selected, origins = roots, {}

    manifests: list[PackManifestV1] = []
    root_by_pack: dict[str, Path] = {}
    seen: set[Path] = set()
    for raw in selected:
        root = Path(raw).resolve()
        if root in seen:
            continue
        seen.add(root)
        try:
            manifest = load_manifest_v1(root)
        except PackError as err:
            origin = origins.get(root)
            if origin is None:
                raise
            raise PackError(f"pack entry point {origin!r} resolved to {root}: {err}") from err
        owner = root_by_pack.get(manifest.name)
        if owner is not None:
            raise PackError(
                f"duplicate pack name {manifest.name!r} discovered at "
                f"roots {owner} and {manifest.root}"
            )
        root_by_pack[manifest.name] = manifest.root
        manifests.append(manifest)
    return PackRuntime(catalog=PackCatalog(tuple(manifests)), root_by_pack=root_by_pack)


__all__ = ["ActivationHandle", "AssetContext", "PackContext", "PackRuntime", "discover"]
