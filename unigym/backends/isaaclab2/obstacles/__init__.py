"""URDF obstacle manager (Route D, no curriculum).

``build_obstacle_manager(config, num_envs, env_spacing, device, base_dir)``
returns an :class:`ObstacleManager` for a yaml ``obstacles:`` section, or
``None`` when the section is absent / disabled.

    obstacles:
      enabled: true
      types:
        - name: box
          urdf_path: obstacles/box.urdf
          proportion: 1.0
          num_joints: 0
      placement:
        mode: random
        x_range: [1.0, 5.0]
        y_range: [-1.0, 1.0]
        yaw_range: [-10, 10]
        height_offset: 0.0
"""
from __future__ import annotations

from .base import ObstacleManager, ObstacleSpec
from .placement import PlacementStrategy


def build_obstacle_manager(config: dict | None, num_envs: int,
                           env_spacing: float, device: str = "cpu",
                           base_dir: str | None = None) -> ObstacleManager | None:
    """Build an :class:`ObstacleManager` from a yaml ``obstacles`` section.

    Returns ``None`` when the section is missing, ``enabled`` is false, or it
    declares no types — so callers can treat the result uniformly.
    """
    if not config:
        return None
    if not config.get("enabled", True):
        return None
    if not config.get("types"):
        return None
    return ObstacleManager(
        config, num_envs=num_envs, env_spacing=env_spacing,
        device=device, base_dir=base_dir)


__all__ = [
    "ObstacleManager",
    "ObstacleSpec",
    "PlacementStrategy",
    "build_obstacle_manager",
]
