"""Pluggable terrain preset interface (isaaclab-free ABC).

A terrain preset builds the SceneCfg components needed by IsaaclabEnv.
``build()`` returns a dict mapping SceneCfg field name -> cfg object; IsaaclabEnv
merges these into its scene config.

- Flat terrain returns ``{"ground": AssetBaseCfg(GroundPlaneCfg)}``.
- Mesh terrain (stairs etc.) returns ``{"terrain": TerrainImporterCfg(...)}``.

The ``classes`` injection point lets unit tests substitute fake cfg classes
without importing isaaclab, mirroring the actuator factory pattern.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TerrainPreset(ABC):
    """Abstract pluggable terrain preset."""

    @abstractmethod
    def build(self, num_envs: int, env_spacing: float, **kwargs) -> dict:
        """Return a ``{SceneCfg field name: cfg object}`` dict.

        Args:
            num_envs: number of parallel environments.
            env_spacing: grid spacing between env origins (m).
            classes: optional injected cfg-class table (testing); when None
                the preset lazily imports the real isaaclab classes.
            **kwargs: preset-specific extras; the abstract base ignores
                them (signature parity with concrete presets).
        """
