"""Shared IsaacLab 3 asset-scene package.

The package exports the composed product and its field spec, plus the engine
(backend-parameterised: the concrete backend name and asset formats are
injected, because the shared library never names a backend and never
registers one).  The engine is a lazy export: importing the package, or one
of its leaf modules such as :mod:`~unigym.backends.shared.isaaclab3_common.
asset_scene.names`, must not import ``engine`` and must not import any engine
(``isaaclab*`` / ``pxr`` / Kit).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from .engine import AssetFieldSpec, AssetSceneEngine, ComposedAssetScene

_LAZY = {
    "AssetFieldSpec": ("engine", "AssetFieldSpec"),
    "AssetSceneEngine": ("engine", "AssetSceneEngine"),
    "ComposedAssetScene": ("engine", "ComposedAssetScene"),
}


def __getattr__(name: str) -> object:
    """Resolve one lazy engine export (PEP 562), caching it in globals.

    Args:
        name: The requested attribute.

    Returns:
        The lazily imported object.

    Raises:
        AttributeError: The name is not part of this package's surface.
    """
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = entry
    value = getattr(import_module(f".{module_name}", __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Advertise the eager globals plus every lazy export."""
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "AssetFieldSpec",
    "AssetSceneEngine",
    "ComposedAssetScene",
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
