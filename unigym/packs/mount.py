"""Mount discovery: environment, local overlay, entry points, deps, repository.

Mount order is fixed: roots named by ``UNIGYM_PACKS`` first, then roots
listed under ``packs:`` in ``unigym.local.yaml``, then installed packs
declared through the ``unigym.packs`` entry-point group (sorted by entry
point name), then every ``<deps_root>/unigym_packs/*`` directory that
carries a manifest.  The checkout itself is always the last, implicit
``repo`` mount.  Duplicate pack names fail closed unless the later
manifest declares ``shadow: true``.

Entry points are a *declared install*, not a scan: each one must resolve
to a filesystem path (``str`` / ``os.PathLike``, directly or through a
zero-argument callable) carrying ``unigym.pack.yaml``.  A broken entry
point (load error, wrong type, missing manifest) raises
:class:`~unigym.packs.PackError` naming the entry point and the reason;
it is never silently skipped.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import yaml

from unigym.core.asset_root import deps_root, repo_root

from . import MountedPack, PackError
from .manifest import FACE_NAMES, MANIFEST_NAME, _shadow_declared, load_manifest

PACKS_ENV_VAR = "UNIGYM_PACKS"
"""Colon-separated list of external pack roots, highest priority."""

PACKS_YAML_KEY = "packs"
"""``unigym.local.yaml`` key listing machine-local pack roots."""

PACKS_TOP_DIR = "unigym_packs"
"""Directory under the dependency root that holds installed packs."""

ENTRY_POINT_GROUP = "unigym.packs"
"""Entry-point group naming installed pack roots (``ep.load() -> path``)."""

REPO_FACES = {
    "runs": "config",
    "benchmarks": "benchmark_packs",
    "scenes": "scene_packs",
    "props": None,
    "terminations": None,
    "assets": "assets",
}
"""Repository face -> checkout directory (``None`` = no such face)."""


def repo_pack() -> MountedPack:
    """The implicit ``repo`` mount (root ``None``, faces from the checkout)."""
    base = Path(repo_root())
    faces: dict[str, Path | None] = {}
    for face, rel in REPO_FACES.items():
        path = None if rel is None else base / rel
        faces[face] = path if path is not None and path.is_dir() else None
    return MountedPack(
        name="repo",
        root=None,
        origin="repo",
        trust="in-package",
        faces=faces,
    )


def mount(root: Path) -> MountedPack:
    """Validate one external pack root and wrap it as a mounted pack."""
    manifest = load_manifest(root)
    faces = {face: manifest.faces.get(face) for face in FACE_NAMES}
    return MountedPack(
        name=manifest.name,
        root=manifest.root,
        origin="pack",
        trust=manifest.trust,
        faces=faces,
    )


def _local_yaml_roots() -> list[Path]:
    """Roots listed under ``packs:`` in ``unigym.local.yaml`` (or none)."""
    path = Path(repo_root()) / "unigym.local.yaml"
    if not path.is_file():
        return []
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError) as err:
        raise PackError(f"unreadable {path}: {err}") from err
    value = doc.get(PACKS_YAML_KEY) if isinstance(doc, Mapping) else None
    if value is None:
        return []
    if not isinstance(value, list):
        raise PackError(f"{path}: 'packs' must be a list of pack roots")
    roots: list[Path] = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise PackError(f"{path}: every 'packs' entry must be a non-empty path")
        candidate = Path(entry).expanduser()
        roots.append(candidate if candidate.is_absolute() else path.parent / candidate)
    return roots


def _entrypoint_target(entry_point) -> Path:
    """Resolve one entry point to a validated pack root (or fail naming it)."""
    name = entry_point.name
    try:
        value = entry_point.load()
        if callable(value):
            value = value()
    except Exception as err:
        raise PackError(
            f"unigym.packs entry point {name!r} failed to load: {err}"
        ) from err
    if not isinstance(value, (str, os.PathLike)):
        raise PackError(
            f"unigym.packs entry point {name!r} must resolve to a "
            f"filesystem path, got {type(value).__name__}"
        )
    root = Path(value).expanduser()
    if not (root / MANIFEST_NAME).is_file():
        raise PackError(
            f"unigym.packs entry point {name!r} root has no "
            f"{MANIFEST_NAME}: {root}"
        )
    return root


def _entrypoint_roots() -> list[Path]:
    """Installed pack roots from the ``unigym.packs`` group (name-sorted)."""
    from importlib.metadata import entry_points

    return [
        _entrypoint_target(entry_point)
        for entry_point in sorted(
            entry_points(group=ENTRY_POINT_GROUP), key=lambda ep: ep.name
        )
    ]


def _deps_roots() -> list[Path]:
    """Every installed pack under ``<deps_root>/unigym_packs`` (sorted)."""
    base = Path(deps_root()) / PACKS_TOP_DIR
    if not base.is_dir():
        return []
    return [
        child
        for child in sorted(base.iterdir())
        if child.is_dir() and (child / MANIFEST_NAME).is_file()
    ]


def _declared_roots() -> list[Path]:
    """All external roots in priority order, before deduplication."""
    roots: list[Path] = []
    env = os.environ.get(PACKS_ENV_VAR, "")
    roots.extend(Path(part) for part in env.split(os.pathsep) if part)
    roots.extend(_local_yaml_roots())
    roots.extend(_entrypoint_roots())
    roots.extend(_deps_roots())
    return roots


def _add_mount(mounts: list[MountedPack], pack: MountedPack, root: Path) -> None:
    """Append *pack* or replace a same-name mount when it declares shadow."""
    for index, existing in enumerate(mounts):
        if existing.name != pack.name:
            continue
        if _shadow_declared(root):
            mounts[index] = pack
            return
        raise PackError(
            f"duplicate pack name {pack.name!r}: {existing.root} and {root}"
        )
    mounts.append(pack)


def pack_roots() -> list[MountedPack]:
    """Every mounted pack in resolution order, the implicit repo last."""
    mounts: list[MountedPack] = []
    seen: set[Path] = set()
    for root in _declared_roots():
        resolved = Path(root).expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        _add_mount(mounts, mount(resolved), resolved)
    if any(entry.name == "repo" for entry in mounts):
        raise PackError("pack name 'repo' is reserved for the implicit checkout mount")
    mounts.append(repo_pack())
    return mounts
