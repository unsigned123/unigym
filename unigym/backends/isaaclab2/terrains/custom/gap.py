"""Overall gap / trench terrain (jump gap).

Flat ground (height 0) with a full-length trench cut across the middle along
*y*: the robot approaches along +x and must jump the gap. Height-field
generator decorated with :func:`height_field_to_mesh`.
"""

from __future__ import annotations

import numpy as np

from ._compat import height_field_to_mesh

try:  # isaaclab-only block (skipped locally -> keeps module importable)
    from isaaclab.utils import configclass
    from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg

    _HAS_IL = True
except Exception:  # optional isaaclab import: skipped locally -> importable
    _HAS_IL = False


@height_field_to_mesh
def gap_terrain(difficulty: float, cfg) -> np.ndarray:
    """Full-length trench across the middle (rows are constant along y)."""
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    gap_width = max(1, int(cfg.gap_width / cfg.horizontal_scale))
    gap_depth = int(cfg.gap_depth / cfg.vertical_scale)
    hf = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    cx = width_pixels // 2
    x1 = max(0, cx - gap_width // 2)
    x2 = min(width_pixels, cx + (gap_width + 1) // 2)
    hf[x1:x2, :] = gap_depth
    return hf


if _HAS_IL:

    @configclass
    class GapTerrainCfg(HfTerrainBaseCfg):
        """Cfg for :func:`gap_terrain` (height = gap_depth is negative => a trench)."""

        function = gap_terrain
        gap_width: float = 0.5
        gap_depth: float = -1.0
