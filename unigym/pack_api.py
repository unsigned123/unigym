"""The only import surface pack code may use.

Pack Python modules resolve content and register providers through this
module.  It stays thin: resolution is delegated to :mod:`unigym.packs`
and registration to the existing closed registries, which this module
does not modify.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from unigym.packs import find, resolve
from unigym.packs.load import PACK_MODULE_PREFIX


def code_pack_root() -> Path | None:
    """The calling pack's ``python/`` directory, or ``None`` in core code.

    The root is recovered from the imported pack module's ``__file__`` by
    walking the call stack to the first ``unigym_packs_<name>`` frame.
    """
    frame = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module.startswith(PACK_MODULE_PREFIX):
            file = frame.f_globals.get("__file__")
            if not file:
                return None
            return Path(file).resolve().parent
        frame = frame.f_back
    return None


def load_table(rel: str) -> object:
    """Load the YAML table resolved from *rel*."""
    return yaml.safe_load(resolve(rel).read_text(encoding="utf-8"))


def load_asset_manifest(path: str | Path) -> dict:
    """Load one asset ``asset.yaml`` through the core manifest gate.

    Pack code reads asset manifests through this module (the only
    sanctioned pack -> core surface); the loader validates the closed
    ``unigym.asset/1`` schema and fails closed with a teaching message.
    """
    from unigym.assets.manifest import load_manifest

    return load_manifest(path)


def asset_path(rel: str) -> Path:
    """Resolve *rel* to an asset path."""
    return resolve(rel)


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
    "asset_path",
    "code_pack_root",
    "find",
    "load_asset_manifest",
    "load_table",
    "register_content_provider",
    "register_prop_template",
    "register_scene",
    "register_termination",
    "resolve",
]
