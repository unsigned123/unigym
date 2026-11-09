"""Asset manifests and the L0 asset primitive.

:mod:`.manifest` is the closed ``unigym.asset/1`` schema gate: one
manifest = one asset's stable id, carrier kind, shipped files,
capabilities, license and provenance.  It is shape-only -- the catalog
(the fact index) and the selection lock (the decision record) are
separate layers built on top of it.

The package is intentionally thin and dependency-light (stdlib + yaml):
it is the foundation every later asset card imports, so it never reaches
into backends, scenes or the pack loader.
"""

from .manifest import (
    FORBIDDEN_KEYS,
    KINDS,
    SCHEMA,
    ManifestError,
    is_asset_id,
    load_manifest,
    split_asset_id,
    validate_manifest,
)

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
