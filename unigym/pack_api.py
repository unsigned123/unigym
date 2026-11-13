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
from unigym.packs.runtime import (
    ActivationHandle,
    AssetContext,
    PackContext,
    PackRuntime,
    discover,
)
from unigym.packs.schema import PackManifestV1, ProvidedItem, load_manifest_v1

if TYPE_CHECKING:
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


def register_scene(name: str):
    """Register a scene provider through the core scene registry."""
    from unigym.scenes.registry import register_scene as register

    return register(name)


def register_content_provider(name: str, factory):
    """Register one content-provider factory through the core registry."""
    from unigym.scenes.pipeline import register_content_provider as register

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
    from unigym.terminations import TERMINATION_REGISTRY

    if (
        not isinstance(name, str)
        or not name
        or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name)
    ):
        raise ValueError(f"termination name {name!r} must be lowercase snake_case")
    if name in TERMINATION_REGISTRY:
        raise ValueError(f"termination {name!r} registered twice")
    if not callable(cls):
        raise ValueError(f"termination {name!r} must be callable")
    TERMINATION_REGISTRY[name] = cls
    return cls


__all__ = [
    "ActivationHandle",
    "AssetContext",
    "CatalogEntry",
    "PackCatalog",
    "PackContext",
    "PackManifestV1",
    "PackRuntime",
    "ProvidedItem",
    "discover",
    "load_asset_manifest",
    "load_manifest_v1",
    "register_adapter",
    "register_content_provider",
    "register_prop_template",
    "register_robot",
    "register_scene",
    "register_termination",
]
