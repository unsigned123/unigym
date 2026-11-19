"""Scene-field and prim-leaf naming for one IsaacLab articulation slot.

The first declared articulation keeps the historical scene field ``robot``
and prim leaf ``Robot``; every following articulation uses ``robot_<name>``
and ``Robot_<name>``.  This module is the single owner of that rule so the
articulation runtime, the contact backend and their tests cannot drift.
It is import-safe without Kit (pure strings).
"""

from __future__ import annotations

__all__ = ["articulation_prim_leaf", "articulation_scene_field"]


def articulation_scene_field(name: str, index: int) -> str:
    """Scene field of the ``index``-th declared articulation."""
    if index < 0:
        raise ValueError(f"articulation index must be >= 0, got {index}")
    return "robot" if index == 0 else f"robot_{name}"


def articulation_prim_leaf(name: str, index: int) -> str:
    """Robot prim leaf of the ``index``-th declared articulation."""
    if index < 0:
        raise ValueError(f"articulation index must be >= 0, got {index}")
    return "Robot" if index == 0 else f"Robot_{name}"
