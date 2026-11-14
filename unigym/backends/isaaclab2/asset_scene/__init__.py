"""IsaacLab 2 asset-scene engine package.

The package exports the composed product and its field spec, plus the
engine and the backend name it registers under.  The engine is a lazy
export: importing the package, or one of its leaf modules such as
:mod:`~unigym.backends.isaaclab2.asset_scene.names`, must not import
``engine`` and therefore must not register the ``isaaclab2`` engine.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

NAME = "isaaclab2"

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


__all__ = ["AssetFieldSpec", "AssetSceneEngine", "ComposedAssetScene", "NAME"]
