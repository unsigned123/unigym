"""Shared IsaacLab 3 terrain library.

This package re-exports the engine-free preset contract, the four terrain
presets (flat / steps / native / generator) and the custom height-field
generators.  Every isaaclab import stays inside ``build()`` / resolver
functions, so importing this package never pulls an engine at module scope.

The preset registry is canonical-only: ``flat``, ``steps``, ``native`` and
``generator``.  Backend packages map their own user-facing terrain names onto
these presets; the shared library carries no backend-specific aliases.
"""

from __future__ import annotations

from . import custom
from .base import TerrainPreset
from .flat import FlatTerrain
from .generator import SUB_TERRAIN_TYPES, GeneratorTerrain
from .native import NativeTerrainPreset
from .steps import StepTerrain

TERRAIN_REGISTRY = {
    "flat": FlatTerrain,
    "steps": StepTerrain,
    "native": NativeTerrainPreset,
    "generator": GeneratorTerrain,
}


def build_terrain(
    section: dict, num_envs: int = 1, env_spacing: float = 4.0, classes: dict | None = None
) -> dict:
    """Build terrain scene components from a yaml ``terrain`` section.

    Args:
        section: ``{"name": str, "params": {...}}``.
        num_envs: number of parallel environments.
        env_spacing: grid spacing between env origins (m).
        classes: optional injected cfg-class table (unit-test only).
    """
    name = section["name"]
    if name not in TERRAIN_REGISTRY:
        raise ValueError(f"unknown terrain '{name}' (available: {sorted(TERRAIN_REGISTRY)})")
    params = dict(section.get("params", {}))
    preset = TERRAIN_REGISTRY[name](**params)
    return preset.build(num_envs=num_envs, env_spacing=env_spacing, classes=classes)


__all__ = [
    "TerrainPreset",
    "FlatTerrain",
    "StepTerrain",
    "GeneratorTerrain",
    "NativeTerrainPreset",
    "SUB_TERRAIN_TYPES",
    "TERRAIN_REGISTRY",
    "build_terrain",
    "custom",
]
