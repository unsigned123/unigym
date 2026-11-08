"""Slope + roughness overlay (combo terrain).

A global linear ramp (like :func:`unidirectional_slope_terrain`) with additive
uniform height noise on top — the canonical "rough slope" used to
train robust foothold adaptation. Height-field generator decorated with
:func:`height_field_to_mesh`.
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
def slope_rough_terrain(difficulty: float, cfg) -> np.ndarray:
    """Linear ramp along +x with uniform additive roughness."""
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    height_max = int(slope * cfg.size[0] / cfg.vertical_scale)
    x = np.arange(width_pixels, dtype=np.float64)
    col = height_max * x / max(width_pixels - 1, 1)
    base = np.tile(col.reshape(-1, 1), (1, length_pixels))
    noise_lo = int(cfg.noise_range[0] / cfg.vertical_scale)
    noise_hi = int(cfg.noise_range[1] / cfg.vertical_scale)
    if noise_hi < noise_lo:
        noise_hi = noise_lo
    noise = np.random.randint(noise_lo, noise_hi + 1, size=(width_pixels, length_pixels))
    hf = base + noise
    return np.rint(hf).astype(np.int16)


if _HAS_IL:

    @configclass
    class SlopeRoughTerrainCfg(HfTerrainBaseCfg):
        """Cfg for :func:`slope_rough_terrain`."""

        function = slope_rough_terrain
        slope_range: tuple[float, float] = (0.1, 0.3)
        noise_range: tuple[float, float] = (0.02, 0.06)
