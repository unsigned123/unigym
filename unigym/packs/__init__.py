"""Pluggable content packs: mount order, manifests and reference resolution.

A pack is a directory carrying ``unigym.pack.yaml``.  Its content faces
(``runs``, ``benchmarks``, ``scenes``, ``props``, ``terminations`` and
``assets``) mount under one name, while the checkout itself is always
mounted last as the implicit ``repo`` pack.  A bare reference therefore
keeps working during the migration window while external packs may shadow
the repository copy.

Packs reach the registries and the resolution surface through
:mod:`unigym.pack_api` only; the modules here are the core-side machinery.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PACK_API = 1
"""Manifest API version this checkout implements."""


class PackError(ValueError):
    """A pack manifest, mount or reference is invalid."""


@dataclass(frozen=True)
class MountedPack:
    """One mounted pack: its identity, root and face paths.

    ``root`` is ``None`` for the implicit repository mount; ``faces`` holds
    every content face name, with ``None`` for a face the pack does not
    provide.
    """

    name: str
    root: Path | None
    origin: str
    trust: str
    faces: Mapping[str, Path | None]


@dataclass(frozen=True)
class PackManifest:
    """A validated ``unigym.pack.yaml`` document."""

    name: str
    version: str
    api: int
    requires: Mapping[str, str]
    trust: str
    faces: Mapping[str, Path]
    root: Path


from .manifest import load_manifest  # noqa: E402
from .mount import mount, pack_roots, repo_pack  # noqa: E402
from .resolve import find, resolve  # noqa: E402

__all__ = [
    "PACK_API",
    "MountedPack",
    "PackError",
    "PackManifest",
    "find",
    "load_manifest",
    "mount",
    "pack_roots",
    "repo_pack",
    "resolve",
]
