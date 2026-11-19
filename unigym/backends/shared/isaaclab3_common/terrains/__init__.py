"""Shared IsaacLab 3 terrain library (P1b A-class slice).

The A-class slice extracts the engine-free preset interface plus the custom
height-field generators:

* :mod:`.base` -- :class:`TerrainPreset`, the pluggable preset contract;
* :mod:`.custom` -- ``@height_field_to_mesh`` generators (lazy isaaclab use).

The preset registry and the ``flat`` / ``steps`` / ``native`` / ``generator``
presets belong to a later P1b slice; this package only re-exports the
engine-free base and must not import an engine at module scope.
"""

from __future__ import annotations

from .base import TerrainPreset

__all__ = ["TerrainPreset"]
