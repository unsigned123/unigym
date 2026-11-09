"""Overall pit terrain (a single bounded deep hole at the centre).

Flat ground with a square pit dug in the middle; the robot must skirt or
traverse it. Height-field generator decorated with :func:`height_field_to_mesh`.
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
def pit_terrain(difficulty: float, cfg) -> np.ndarray:
    """Square pit centred in the terrain; pit_depth is negative."""
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    pit_width = max(1, int(cfg.pit_width / cfg.horizontal_scale))
    pit_depth = int(cfg.pit_depth / cfg.vertical_scale)
    hf = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    x1 = max(0, (width_pixels - pit_width) // 2)
    x2 = min(width_pixels, (width_pixels + pit_width) // 2)
    y1 = max(0, (length_pixels - pit_width) // 2)
    y2 = min(length_pixels, (length_pixels + pit_width) // 2)
    hf[x1:x2, y1:y2] = pit_depth
    return hf


if _HAS_IL:

    @configclass
    class PitTerrainCfg(HfTerrainBaseCfg):
        """Cfg for :func:`pit_terrain`."""

        function = pit_terrain
        pit_width: float = 2.0
        pit_depth: float = -1.0
