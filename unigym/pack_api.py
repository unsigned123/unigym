"""The only import surface pack code may use.

Pack Python modules receive a :class:`PackContext` from the runtime and use
this module to register providers through the existing closed registries.
Discovery, catalog lookup, resource resolution and activation live in
:mod:`unigym.packs.runtime`; this module only re-exports the public contract
and owns the registration helpers.

The ``unigym.pack/1`` manifest schema, its loaded types, the read-only pack
catalog and the explicit runtime are the public pack contract.  Discovery and
activation stay lazy: importing this module never imports pack Python.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from unigym.packs.catalog import CatalogEntry, PackCatalog
from unigym.packs.packset import (
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
from unigym.packs.runtime import (
    ActivationHandle,
    AssetContext,
    PackContext,
    PackRuntime,
    discover,
)
from unigym.packs.schema import PackManifestV1, ProvidedItem, load_manifest_v1
from unigym.provenance import ProviderInfo, ProviderOrigin
from unigym.scenes.providers import (
    SceneProvider,
    SceneProviderContext,
    build_scene_from_provider,
    registered_scene_providers,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from unigym.adapters.base import Adapter
    from unigym.robots import RobotProfile


def load_asset_manifest(path: str | Path) -> dict:
    """Load one asset ``asset.yaml`` through the core manifest gate.

    Pack code reads asset manifests through this module (the only
    sanctioned pack -> core surface); the loader validates the closed
    ``unigym.asset/1`` schema and fails closed with a teaching message.
    """
    from unigym.assets.manifest import load_manifest

    return load_manifest(path)


def register_scene_provider(
    name: str,
    factory: SceneProvider,
    *,
    params: Mapping[str, object] | None = None,
) -> SceneProvider:
    """Register a scene provider through the core provider registry."""
    from unigym.scenes.providers import register_scene_provider as register

    return register(name, factory, params=params)


def register_scene_content_filler(name: str, factory):
    """Register one content-filler factory through the core registry."""
    from unigym.scenes.pipeline import register_scene_content_filler as register

    return register(name, factory)


def register_prop_template(name: str, *, params: dict[str, dict]):
    """Register a prop-template builder through the core template registry."""
    from unigym.props.templates import register_prop_template as register

    return register(name, params=params)


def register_adapter(name: str, cls: type[Adapter]) -> type[Adapter]:
    """Register one adapter class in the core adapter registry.

    The adapter package imports torch, so this wrapper imports it lazily at
    call time; importing :mod:`unigym.pack_api` stays torch-free.
    """
    from unigym.adapters import register_adapter as register

    return register(name, cls)


def register_robot(name: str, factory: Callable[..., RobotProfile]) -> Callable[..., RobotProfile]:
    """Register one external robot profile factory in the core registry."""
    from unigym.robots import register_robot as register

    return register(name, factory)


def register_termination(name: str, cls):
    """Register a termination class in the core termination registry."""
    from unigym.terminations import register_termination as register

    return register(name, cls)


__all__ = [
    "ActivationHandle",
    "AssetContext",
    "AssetResourceRef",
    "CatalogEntry",
    "PackCatalog",
    "PackContext",
    "PackInfo",
    "PackManifestV1",
    "PackRuntime",
    "PackSet",
    "ProvidedItem",
    "ProviderInfo",
    "ProviderOrigin",
    "ResourceInfo",
    "RobotRef",
    "ScenePresetRef",
    "SceneProvider",
    "SceneProviderContext",
    "SceneProviderRef",
    "build_scene_from_provider",
    "discover",
    "discover_set",
    "installed",
    "load_asset_manifest",
    "load_manifest_v1",
    "register_adapter",
    "register_prop_template",
    "register_robot",
    "register_scene_content_filler",
    "register_scene_provider",
    "register_termination",
    "registered_scene_providers",
    "require",
]
