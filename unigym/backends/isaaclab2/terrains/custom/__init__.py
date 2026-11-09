"""Custom height-field sub-terrains (Route C).

Each module exposes a ``@height_field_to_mesh``-decorated generator function
plus (when isaaclab is importable) a matching ``HfTerrainBaseCfg`` subclass.

Locally the decorator degrades to the identity (see :mod:`._compat`), so the
functions stay plain height-field producers returning ``np.ndarray[int16]``
and are unit-testable without isaaclab/``pxr``.

Use them from a yaml ``generator`` terrain via the ``custom:`` escape hatch::

    sub_terrains:
      gap: {type: "custom:unigym.backends.isaaclab2.terrains.custom.gap.gap_terrain",
            proportion: 0.5, params: {gap_width: 0.6, gap_depth: -1.0}}
"""

from __future__ import annotations

from .gap import gap_terrain
from .pit import pit_terrain
from .slope_rough import slope_rough_terrain
from .unidirectional import (
    unidirectional_slope_terrain,
    unidirectional_stairs_terrain,
)

__all__ = [
    "gap_terrain",
    "pit_terrain",
    "slope_rough_terrain",
    "unidirectional_slope_terrain",
    "unidirectional_stairs_terrain",
]
