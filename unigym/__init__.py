"""General robot locomotion data collection framework (Isaac Lab ManagerBasedEnv).

The package root stays light: importing ``unigym`` must not import torch,
mujoco or isaaclab.  Heavy users (``Run`` and the public error classes that
pull in backend code) resolve lazily through PEP 562 ``__getattr__``.

Quickstart (no doctest markers on purpose)::

    import unigym

    with unigym.Run.from_config("config/simple/simple_pick_scripted_mj.yaml") as run:
        print(run.model.joints.names())      # static IR (no runner needed)
        print(run.controls[0].name)          # the v104 control groups
        result = run.step()                  # builds the live env on demand
        print(result.t, result.obs.keys())

    env = unigym.make(run, kind="gym")       # Gymnasium single-env facade

Use ``run.model`` for the frozen model IR, ``run.live`` for zero-copy
dynamic proxies and ``run.step`` / ``run.steps`` for the default synchronous
runner.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

__all__ = [
    "Run",
    "load",
    "make",
    "World",
    "Camera",
    "Box",
    "Sphere",
    "Cylinder",
    "Capsule",
    "Plane",
    "Terrain",
    "UniGymError",
    "ConfigError",
    "BackendError",
    "CapabilityError",
    "ModelIRError",
    "StaleModelError",
    "UnknownExtensionError",
    "Control",
    "SimulationBuilder",
]

_LAZY = {
    "Run": ("unigym.run.api", "Run"),
    "load": ("unigym.run.api", "load"),
    "make": ("unigym.run.api", "make"),
    "World": ("unigym.world", "World"),
    "Camera": ("unigym.camera", "Camera"),
    "Box": ("unigym.world", "Box"),
    "Sphere": ("unigym.world", "Sphere"),
    "Cylinder": ("unigym.world", "Cylinder"),
    "Capsule": ("unigym.world", "Capsule"),
    "Plane": ("unigym.world", "Plane"),
    "Terrain": ("unigym.scenes.terrain", "Terrain"),
    "UniGymError": ("unigym.errors", "UniGymError"),
    "ConfigError": ("unigym.errors", "ConfigError"),
    "BackendError": ("unigym.errors", "BackendError"),
    "CapabilityError": ("unigym.errors", "CapabilityError"),
    "ModelIRError": ("unigym.model.errors", "ModelIRError"),
    "StaleModelError": ("unigym.model.errors", "StaleModelError"),
    "UnknownExtensionError": ("unigym.model.errors", "UnknownExtensionError"),
    "Control": ("unigym.sim", "Control"),
    "SimulationBuilder": ("unigym.sim", "SimulationBuilder"),
}


if TYPE_CHECKING:
    # Static view of the lazy surface above: pyright resolves ``unigym.Run``
    # etc. here while runtime importing stays PEP-562 lazy.
    from unigym.camera import Camera
    from unigym.errors import (
        BackendError,
        CapabilityError,
        ConfigError,
        UniGymError,
    )
    from unigym.model.errors import (
        ModelIRError,
        StaleModelError,
        UnknownExtensionError,
    )
    from unigym.run.api import Run, load, make
    from unigym.scenes.terrain import Terrain
    from unigym.sim import Control, SimulationBuilder
    from unigym.world import Box, Capsule, Cylinder, Plane, Sphere, World


def __getattr__(name):
    """Resolve the public API names lazily (keeps ``import unigym`` light).

    Args:
        name: The requested public attribute.

    Returns:
        The lazily imported object, cached in the module globals.

    Raises:
        AttributeError: The name is not part of the public API.
    """
    entry = _LAZY.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = entry
    from importlib import import_module

    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def _resolve_version() -> str:
    """The REAL version: the checkout's pyproject.toml first (the runs
    must stamp the code that actually executes -- a stale installed
    dist-info once stamped every run record with an ancient version,
    the cwd-resolves-the-package lesson generalized), the installed
    metadata second, "unknown" last."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("version"):
                _, _, rhs = s.partition("=")
                v = rhs.strip().strip('"').strip("'")
                if v:
                    return v
    except OSError:
        pass
    try:
        return metadata.version("unigym")
    except metadata.PackageNotFoundError:  # not installed (bare checkout)
        return "unknown"


__version__ = _resolve_version()
