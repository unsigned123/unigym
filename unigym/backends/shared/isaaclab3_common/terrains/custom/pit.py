"""Overall pit terrain (a single bounded deep hole at the centre).

Flat ground with a square pit dug in the middle; the robot must skirt or
traverse it. Height-field generator decorated with :func:`height_field_to_mesh`.

The optional isaaclab-based ``PitTerrainCfg`` is exposed lazily (PEP 562), so
importing this module never imports isaaclab.
"""

from __future__ import annotations

import numpy as np

from ._compat import height_field_to_mesh


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


def _has_isaaclab() -> bool:
    """Return whether isaaclab is importable in this process."""
    try:  # pragma: no cover - only true on the server (isaaclab present)
        import isaaclab  # noqa: F401
    except Exception:  # optional isaaclab import: skipped locally
        return False
    return True


def _build_pit_terrain_cfg():
    """Build ``PitTerrainCfg`` against the real isaaclab base class."""
    from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
    from isaaclab.utils import configclass

    @configclass
    class PitTerrainCfg(HfTerrainBaseCfg):
        """Cfg for :func:`pit_terrain`."""

        function = pit_terrain
        pit_width: float = 2.0
        pit_depth: float = -1.0

    return PitTerrainCfg


def __getattr__(name: str) -> object:
    """Resolve the lazy isaaclab-only exports (PEP 562)."""
    if name == "_HAS_IL":
        value: object = _has_isaaclab()
    elif name == "PitTerrainCfg":
        value = _build_pit_terrain_cfg()
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
