"""Stdlib-only engine version probes.

Every probe is side-effect free at import time and uses only the standard
library: ``importlib.metadata`` first (a distribution lookup, no import), then
a guarded ``import`` as a fallback for exotic installs.  Neutral layers
(``unigym.core``) import this module instead of the backend composition graph.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re

__all__ = [
    "isaaclab_version",
    "mujoco_version",
    "mujoco_warp_version",
    "newton_version",
    "warp_version",
]


def _dist_version(dist: str) -> tuple[int, int] | None:
    """(major, minor) of an installed distribution, else None."""
    try:
        raw = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None
    nums = re.findall(r"\d+", raw)
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]))
    if len(nums) == 1:
        return (int(nums[0]), 0)
    return None


def _import_version(module: str) -> tuple[int, int] | None:
    """Fallback probe: guarded import + ``__version__``.

    Post-AppLauncher
    the metadata lookup may already have sufficed; this catches exotic
    installs where isaaclab is importable but not pip-registered.
    """
    try:
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001 — any failure means "not usable here"
        return None
    raw = getattr(mod, "__version__", None)
    if not raw:
        return None
    nums = re.findall(r"\d+", str(raw))
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]))
    return None


def isaaclab_version() -> tuple[int, int] | None:
    """Return the IsaacLab (major, minor) version probe.

    Metadata first, guarded import second.
    """
    return _dist_version("isaaclab") or _import_version("isaaclab")


def mujoco_version() -> tuple[int, int] | None:
    """Return the MuJoCo (major, minor) version probe."""
    return _dist_version("mujoco") or _import_version("mujoco")


def mujoco_warp_version() -> tuple[int, int] | None:
    """Return the ``mujoco-warp`` (major, minor) version probe."""
    return _dist_version("mujoco-warp") or _import_version("mujoco_warp")


def newton_version() -> tuple[int, int] | None:
    """Return the ``newton`` (major, minor) version probe."""
    return _dist_version("newton") or _import_version("newton")


def warp_version() -> tuple[int, int] | None:
    """Return the ``warp-lang`` (major, minor) version probe.

    The distribution name is ``warp-lang``; the import/module name is ``warp``.
    """
    return _dist_version("warp-lang") or _import_version("warp")
