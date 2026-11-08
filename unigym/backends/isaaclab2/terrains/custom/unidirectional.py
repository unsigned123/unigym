"""Unidirectional slope & stairs (non-pyramid, monotonic along +x).

Unlike Isaac Lab's pyramid terrains (symmetric around a central platform),
these generate a *global* monotonic rise: every row increases along x — a
single ramp, or a staircase that climbs one way. Two generators are exported:

* :func:`unidirectional_slope_terrain` — smooth linear ramp.
* :func:`unidirectional_stairs_terrain` — equal-width step bands climbing +x.

Both are height-field generators decorated with :func:`height_field_to_mesh`.
"""
from __future__ import annotations

import numpy as np

from ._compat import height_field_to_mesh

try:
    from isaaclab.utils import configclass
    from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg

    _HAS_IL = True
except Exception:  # optional isaaclab import: skipped locally -> importable
    _HAS_IL = False


@height_field_to_mesh
def unidirectional_slope_terrain(difficulty: float, cfg) -> np.ndarray:
    """Global linear ramp rising along +x (monotonic, not symmetric)."""
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    height_max = int(slope * cfg.size[0] / cfg.vertical_scale)
    x = np.arange(width_pixels, dtype=np.float64)
    col = height_max * x / max(width_pixels - 1, 1)
    hf = np.tile(col.reshape(-1, 1), (1, length_pixels))
    return np.rint(hf).astype(np.int16)


@height_field_to_mesh
def unidirectional_stairs_terrain(difficulty: float, cfg) -> np.ndarray:
    """Equal-width step bands climbing along +x (each band strictly taller)."""
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    step_height = (
        cfg.step_height_range[0]
        + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    )
    step_width = max(1, int(cfg.step_width / cfg.horizontal_scale))
    step_height = int(step_height / cfg.vertical_scale)
    hf = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    x = 0
    h = 0
    while x + step_width <= width_pixels:
        hf[x : x + step_width, :] = h
        h += step_height
        x += step_width
    if x < width_pixels:  # fill trailing partial band with the next level
        hf[x:, :] = h
    return hf


if _HAS_IL:

    @configclass
    class UnidirectionalSlopeTerrainCfg(HfTerrainBaseCfg):
        """Cfg for :func:`unidirectional_slope_terrain`."""

        function = unidirectional_slope_terrain
        slope_range: tuple[float, float] = (0.1, 0.3)

    @configclass
    class UnidirectionalStairsTerrainCfg(HfTerrainBaseCfg):
        """Cfg for :func:`unidirectional_stairs_terrain`."""

        function = unidirectional_stairs_terrain
        step_height_range: tuple[float, float] = (0.05, 0.15)
        step_width: float = 0.3
