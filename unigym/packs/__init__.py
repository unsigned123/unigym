"""Pluggable content packs: the shared error type and public package boundary.

The pack contract is :mod:`unigym.pack_api`; discovery, resource resolution and
activation are owned by :mod:`unigym.packs.loader`.  A pack root carries a
``unigym.pack.yaml`` document validated by :mod:`unigym.packs.schema`, which
returns the stored :class:`Pack` / :class:`PackItem` model.  There is no
implicit repository mount and no mount order: every resource belongs to an
explicitly discovered pack.

:func:`require` and :func:`installed` are the read-only pack-object front door.
They return a :class:`PackSet`; typed asset, scene preset and scene provider
handles are minted from one stored :class:`PackItem`.  Object loading and
resource path resolution belong to :class:`PackLoader`.
"""

from __future__ import annotations

from unigym.provenance import ProviderInfo, ProviderOrigin


class PackError(ValueError):
    """A pack manifest, resource or activation is invalid."""


# PackError must exist before packset imports it back through ``from . import``.
from .loader import (
    PackLoader,
    discover_set,
    installed,
    require,
)
from .packset import (
    AssetResourceRef,
    PackSet,
    ScenePresetRef,
    SceneProviderRef,
)
from .schema import (
    EntrypointLocator,
    Face,
    ItemStatus,
    Locator,
    Pack,
    PackItem,
    ResourceLocator,
)

__all__ = [
    "AssetResourceRef",
    "EntrypointLocator",
    "Face",
    "ItemStatus",
    "Locator",
    "Pack",
    "PackError",
    "PackItem",
    "PackLoader",
    "PackSet",
    "ProviderInfo",
    "ProviderOrigin",
    "ResourceLocator",
    "ScenePresetRef",
    "SceneProviderRef",
    "discover_set",
    "installed",
    "require",
]
