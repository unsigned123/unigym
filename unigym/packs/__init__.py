"""Pluggable content packs: the shared error type and public package boundary.

The pack contract is :mod:`unigym.pack_api`; discovery, catalog lookup,
resource resolution and activation are owned by :mod:`unigym.packs.runtime`.
A pack root carries a ``unigym.pack.yaml`` document validated by
:mod:`unigym.packs.schema`.  There is no implicit repository mount and no
mount order: every resource belongs to an explicitly discovered pack.

:func:`require` and :func:`installed` are the read-only pack-object front door.
They return a :class:`PackSet`; robot, asset, scene preset and scene provider
handles are minted through ``packs.pack(name).robot(id)`` / ``.asset(id)`` /
``.scene_preset(id)`` / ``.scene_provider(id)``.
"""

from __future__ import annotations

from unigym.provenance import ProviderInfo, ProviderOrigin


class PackError(ValueError):
    """A pack manifest, resource or activation is invalid."""


# PackError must exist before packset imports it back through ``from . import``.
from .packset import (
    AssetResourceRef,
    PackInfo,
    PackSet,
    ResourceInfo,
    RobotRef,
    ScenePresetRef,
    SceneProviderRef,
    discover_set,
    installed,
    require,
)

__all__ = [
    "AssetResourceRef",
    "PackError",
    "PackInfo",
    "PackSet",
    "ProviderInfo",
    "ProviderOrigin",
    "ResourceInfo",
    "RobotRef",
    "ScenePresetRef",
    "SceneProviderRef",
    "discover_set",
    "installed",
    "require",
]
