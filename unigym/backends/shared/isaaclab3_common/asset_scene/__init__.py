"""Shared IsaacLab 3 asset-scene library (P1b A-class slice).

This package owns the engine-neutral asset-scene leaf modules used by both
IsaacLab 3 backend identities.  The A-class slice is import-safe:

* :mod:`.names` -- scene-field and prim-path naming (pure strings);
* :mod:`.inspect` -- URDF guard plus lazy OpenUSD authoring inspection;
* :mod:`.artifacts` -- deterministic artifact capture over the composed
  product, with the composed-product classes imported lazily.

The composed-product ``engine`` and cfg-builder ``fields`` modules belong to
the later cfg-builder slice; importing this package must not import them, and
must not import any engine (``isaaclab*`` / ``pxr`` / Kit).
"""

from __future__ import annotations

from .artifacts import write_asset_scene_artifacts
from .inspect import UsdAuthoring, inspect_usd_authoring, urdf_joint_names
from .names import (
    asset_field,
    asset_prim_path,
    static_field,
    static_prim_path,
    static_shape_field,
    static_shape_prim_path,
)

__all__ = [
    "UsdAuthoring",
    "asset_field",
    "asset_prim_path",
    "inspect_usd_authoring",
    "static_field",
    "static_prim_path",
    "static_shape_field",
    "static_shape_prim_path",
    "urdf_joint_names",
    "write_asset_scene_artifacts",
]
