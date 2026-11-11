"""Static-root pack runtime: discover, resolve resources, activate.

The runtime is a Phase 3 contract over ``unigym.pack/1``.  :func:`discover`
reads explicit roots, or the configured sources when the caller names none,
loads their manifests without importing pack Python, and returns an immutable
catalog plus each pack's root.  Resources resolve under their own pack root.
A code pack activates through normal Python: its ``python/`` directory is the
explicit import boundary, the manifest entrypoint is imported and called with
no arguments, and the handle is cached per runtime.

This is a parallel runtime.  The legacy ``unigym.packs`` mount/load path stays
the active runtime until a later migration phase.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import yaml

from unigym.core.asset_root import repo_root

from . import PackError
from .catalog import PackCatalog
from .schema import PackManifestV1, load_manifest_v1

_PACKS_ENV_VAR = "UNIGYM_PACKS"
_DEPS_ENV_VAR = "UNIGYM_DEPS_ROOT"
_PACKS_YAML_KEY = "packs"
_PACKS_TOP_DIR = "unigym_packs"
_LOCAL_YAML_NAME = "unigym.local.yaml"


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
        A code pack's entrypoint is imported normally and called with no
        arguments; failures are wrapped with the pack and entrypoint named.

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

        try:
            python_dir = _import_boundary(manifest)
            _add_to_sys_path(python_dir)
            target = _resolve_entrypoint(entrypoint)
            target()
        except Exception as err:
            raise PackError(
                f"pack {pack!r} entrypoint {entrypoint!r} failed to activate: {err}"
            ) from err

        handle = ActivationHandle(pack=pack, entrypoint=entrypoint)
        self._handles[pack] = handle
        return handle


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


def _configured_roots(config_path: str | Path | None) -> list[Path]:
    """Configured pack roots in source order: env, local overlay, deps."""
    local_yaml = (
        Path(config_path) if config_path is not None else Path(repo_root()) / _LOCAL_YAML_NAME
    )
    return [*_env_pack_roots(), *_local_yaml_roots(local_yaml), *_deps_pack_roots()]


def discover(
    roots: Iterable[str | Path] | None = None,
    *,
    config_path: str | Path | None = None,
) -> PackRuntime:
    """Discover packs from explicit roots or the configured sources.

    A non-``None`` *roots* iterable is authoritative: only those roots are
    read, in order, and configured sources are ignored.  ``discover([])`` is
    therefore an explicit empty discovery.

    ``roots=None`` reads, in order, the ``UNIGYM_PACKS`` entries, the
    ``packs:`` entries of the local overlay (``<repo>/unigym.local.yaml`` by
    default) and the sorted ``<UNIGYM_DEPS_ROOT>/unigym_packs/*`` directories.

    *config_path* overrides the local-overlay path for embedding and tests;
    ``None`` keeps the repository overlay.

    Canonical duplicate roots are visited once.  Every root must carry a valid
    ``unigym.pack.yaml``; duplicate pack names fail.  Discovery reads manifests
    only and never imports pack Python.

    Raises:
        PackError: A configured declaration is malformed, a root has no valid
            manifest, or two roots name the same pack.
    """
    selected = _configured_roots(config_path) if roots is None else roots
    manifests: list[PackManifestV1] = []
    root_by_pack: dict[str, Path] = {}
    seen: set[Path] = set()
    for raw in selected:
        root = Path(raw).resolve()
        if root in seen:
            continue
        seen.add(root)
        manifest = load_manifest_v1(root)
        owner = root_by_pack.get(manifest.name)
        if owner is not None:
            raise PackError(
                f"duplicate pack name {manifest.name!r} discovered at "
                f"roots {owner} and {manifest.root}"
            )
        root_by_pack[manifest.name] = manifest.root
        manifests.append(manifest)
    return PackRuntime(catalog=PackCatalog(tuple(manifests)), root_by_pack=root_by_pack)


__all__ = ["ActivationHandle", "PackRuntime", "discover"]
