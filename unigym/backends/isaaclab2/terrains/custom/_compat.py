"""Lazy import of isaaclab's ``height_field_to_mesh`` decorator.

Locally (no isaaclab installed) the decorator degrades to the identity, so the
custom terrain functions remain *plain height-field producers* returning a
``np.ndarray[int16]`` — directly callable and testable without isaaclab or
``pxr``. On the server (isaaclab present, after AppLauncher) the real decorator
wraps them into mesh-returning functions exactly as Isaac Lab's
``TerrainGenerator`` expects.
"""
from __future__ import annotations

try:  # pragma: no cover - exercised only on the server (isaaclab present)
    from isaaclab.terrains.height_field.utils import (
        height_field_to_mesh as height_field_to_mesh,
    )
except Exception:  # identity fallback keeps modules importable locally

    def height_field_to_mesh(func):
        """Identity decorator (no isaaclab): keep the function as-is."""
        return func
