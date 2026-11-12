"""Asset manifests and the L0 asset primitive.

:mod:`.manifest` is the closed ``unigym.asset/1`` schema gate: one
manifest = one asset's stable id, carrier kind, shipped files,
capabilities, license and provenance.  It is shape-only -- the catalog
(the fact index) and the selection lock (the decision record) are
separate layers built on top of it.

The package is intentionally thin and dependency-light (stdlib + yaml):
it is the foundation every later asset card imports, so it never reaches
into backends, scenes or the pack loader.

The public manifest names resolve lazily (PEP 562) so importing a light
submodule (for example :mod:`.asset_ref`) does not import ``yaml``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FORBIDDEN_KEYS",
    "KINDS",
    "SCHEMA",
    "ManifestError",
    "is_asset_id",
    "load_manifest",
    "split_asset_id",
    "validate_manifest",
]

_MANIFEST_NAMES = frozenset(__all__)


def __getattr__(name: str) -> Any:
    if name in _MANIFEST_NAMES:
        from . import manifest

        return getattr(manifest, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
