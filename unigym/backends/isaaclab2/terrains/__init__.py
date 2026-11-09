"""Terrain registry (isaaclab-dependent, lazy imports).

``build_terrain(section, num_envs, env_spacing)`` instantiates the preset
named by ``section["name"]`` with ``section["params"]`` and calls its
``build()`` to produce a ``{SceneCfg field: cfg}`` dict for IsaaclabEnv.
"""
from __future__ import annotations

from .base import TerrainPreset
from .flat import FlatTerrain
from .generator import GeneratorTerrain
from .native import NativeTerrainPreset
from .steps import StepTerrain

TERRAIN_REGISTRY = {
    "flat": FlatTerrain,
    "flat_plane": FlatTerrain,   # user-config key (asset continuity)
    "steps": StepTerrain,
    "generator": GeneratorTerrain,   # route A+C: Isaac Lab TerrainGenerator + custom:
}


def build_terrain(section: dict, num_envs: int = 1, env_spacing: float = 4.0,
                  classes: dict | None = None) -> dict:
    """Build terrain scene components from a yaml ``terrain`` section.

    Args:
        section: ``{"name": str, "params": {...}}``.
        num_envs: number of parallel environments.
        env_spacing: grid spacing between env origins (m).
        classes: optional injected cfg-class table (unit-test only).
    """
    name = section["name"]
    if name not in TERRAIN_REGISTRY:
        raise ValueError(
            f"unknown terrain '{name}' (available: {sorted(TERRAIN_REGISTRY)})")
    params = dict(section.get("params", {}))
    preset = TERRAIN_REGISTRY[name](**params)
    return preset.build(num_envs=num_envs, env_spacing=env_spacing, classes=classes)


__all__ = ["TerrainPreset", "FlatTerrain", "StepTerrain", "GeneratorTerrain",
           "NativeTerrainPreset", "TERRAIN_REGISTRY", "build_terrain"]
