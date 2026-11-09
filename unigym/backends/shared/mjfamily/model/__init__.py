"""Shared MuJoCo-family model walkers and provider.

Both ``backends/mujoco`` and ``backends/mjwarp`` consume the same
``MjSpec``/``MjModel``, so the IR walkers and the provider live here; the
backend packages only re-export them (``mujoco/model`` and ``mjwarp/model``).
"""

from __future__ import annotations

from .provider import MujocoModelProvider

__all__ = ["MujocoModelProvider"]
