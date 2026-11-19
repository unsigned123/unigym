"""Shared IsaacLab 3 prop handlers.

The concrete handler classes live here for the ``isaaclab3_physx`` /
``isaaclab3_newton`` identities; the neutral ``unigym/props`` package keeps
the spec vocabulary, validation and the generated-URDF cache.  Handlers
never import a concrete backend.
"""

from __future__ import annotations

from .articulation import FreeArticulationProp, PropArticulation, UsdArticulationProp
from .ball import BallProp
from .box import BoxProp
from .generic import GenericProp
from .primitive import PrimitiveProp

__all__ = [
    "BallProp",
    "BoxProp",
    "FreeArticulationProp",
    "GenericProp",
    "PrimitiveProp",
    "PropArticulation",
    "UsdArticulationProp",
]
