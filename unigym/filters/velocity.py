"""Velocity range filter: keep episodes whose average forward velocity
falls within a configurable range.

Useful for collecting only high-speed or low-speed locomotion data.
"""
from __future__ import annotations

import numpy as np

from .base import EpisodeFilter


class VelocityRangeFilter(EpisodeFilter):
    """Keep episodes within a velocity window.

    Args:
        min_vx: minimum mean forward velocity (m/s). None = no lower bound.
        max_vx: maximum mean forward velocity (m/s). None = no upper bound.
        min_abs_vx: minimum absolute mean velocity (m/s); useful for
            excluding near-stationary episodes. None = no check.
    """

    def __init__(self, min_vx: float | None = None,
                 max_vx: float | None = None,
                 min_abs_vx: float | None = None):
        self.min_vx = min_vx
        self.max_vx = max_vx
        self.min_abs_vx = min_abs_vx

    def __call__(self, episode_data: dict) -> bool:
        vel = np.asarray(episode_data["base_lin_vel"])  # (T, 3)
        mean_vx = float(vel[:, 0].mean())

        if self.min_vx is not None and mean_vx < self.min_vx:
            return False
        if self.max_vx is not None and mean_vx > self.max_vx:
            return False
        if self.min_abs_vx is not None and abs(mean_vx) < self.min_abs_vx:
            return False
        return True
