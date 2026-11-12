"""IsaacLab robot realizers: neutral ``RobotProfile`` -> ``ArticulationCfg``.

This package owns every isaaclab-specific robot construction detail (asset
choice, actuator tables, prim paths, spawn conversion).  The neutral
``unigym.robots`` package never imports it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from isaaclab.assets import ArticulationCfg

from unigym._typing import JsonValue
from unigym.robots.base import RobotProfile

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from unigym.scenes.articulation import ArticulationSpec

__all__ = ["articulation_cfg", "build_robot_realization"]


def articulation_cfg(realizations: Mapping[str, object] | None, name: str) -> ArticulationCfg:
    """The backend realization of one articulation (fail-fast when absent)."""
    value = dict(realizations or {}).get(name)
    if not isinstance(value, ArticulationCfg):
        raise ValueError(
            f"isaaclab backend: articulation {name!r} carries no "
            f"ArticulationCfg realization (the backend robot realizer must "
            f"build one)"
        )
    return value


def build_robot_realization(
    profile: RobotProfile,
    *,
    spec: ArticulationSpec,
    params: Mapping[str, JsonValue],
    asset_plan: object | None = None,
) -> object:
    """Build the isaaclab realization of one neutral robot profile.

    Args:
        profile: the neutral robot profile.
        spec: the run-level articulation declaration (the spawn authority).
        params: the robot section's factory params (gain/prim-path values not
            carried as neutral data).
        asset_plan: the composition boundary's resolved asset plan; no
            isaaclab realization consumes a pack asset yet, so it is
            accepted for the shared realizer seam and currently unused.

    Raises:
        NotImplementedError: The profile names a robot with no isaaclab
            realization.
    """
    del asset_plan
    if profile.name == "g1":
        from .g1 import realize_g1

        return realize_g1(profile, spec=spec, params=params)
    raise NotImplementedError(
        f"isaaclab2 backend: no robot realizer for profile {profile.name!r} | "
        f"hint: run this robot on a backend that consumes the neutral profile "
        f"(mujoco) or add its isaaclab realizer under "
        f"unigym/backends/isaaclab2/robots/"
    )
