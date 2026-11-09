"""Engine-neutral model IR and its collections (design v098 N0).

Importing this package must stay light: no mujoco, no torch.  Backend
providers import their native engine only inside their methods.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .diff import ModelDiff
from .errors import (
    ModelIRError,
    ModelSetError,
    StaleModelError,
    UnknownExtensionError,
)
from .ir import (
    Actuator,
    Body,
    Camera,
    EqSlot,
    Geom,
    GeomContact,
    GeomGroup,
    Joint,
    Material,
    MjExt,
    ModelExt,
    ModelIR,
    Site,
    Texture,
    UnparsedExt,
    UsdExt,
)
from .table import Table
from .set import ModelSet

if TYPE_CHECKING:
    from .live import LiveBody, LiveJoint, LiveModel, LiveTable

    class MujocoModelProvider:
        """Typing stub for the lazily imported MuJoCo model provider."""

        def __init__(self, backend: str | None = None) -> None:
            ...

        def build(self, mjspec: object) -> ModelIR:
            """Build the declared IR from a native spec."""
            ...

        def describe(self, mjmodel: object) -> ModelIR:
            """Fill the compiled IR from a native model."""
            ...


def __getattr__(name: str) -> object:
    """Resolve the MuJoCo provider lazily (keeps this import engine-free).

    Args:
        name: The requested public attribute.

    Returns:
        The lazily imported MuJoCo provider when *name* asks for it.

    Raises:
        AttributeError: The name is not exported by this package.
    """
    if name == "MujocoModelProvider":
        from importlib import import_module

        module = import_module(
            "unigym.backends.shared.mjfamily.model.provider")
        return module.MujocoModelProvider
    if name in {"LiveModel", "LiveTable", "LiveJoint", "LiveBody"}:
        from . import live

        return getattr(live, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Actuator",
    "Body",
    "Camera",
    "EqSlot",
    "Geom",
    "GeomContact",
    "GeomGroup",
    "Joint",
    "LiveBody",
    "LiveJoint",
    "LiveModel",
    "LiveTable",
    "Material",
    "MjExt",
    "ModelDiff",
    "ModelExt",
    "ModelIR",
    "ModelIRError",
    "ModelSet",
    "ModelSetError",
    "MujocoModelProvider",
    "Site",
    "StaleModelError",
    "Table",
    "Texture",
    "UnknownExtensionError",
    "UnparsedExt",
    "UsdExt",
]
