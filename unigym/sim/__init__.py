"""Experimental simulation facade (``SIM_API_VERSION = 1``).

The facade is a thin, additive, declaration-first surface over the existing
``Run`` / runner / backend stack.  This namespace currently ships the
declaration half only: an explicit-pack builder, identity-scoped handles, the
``Control.position_pd`` preset and the facade error taxonomy.  ``Simulation``
and the typed ``Observation`` arrive in later slices; they are not exported
before they exist.

``import unigym.sim`` stays torch/mujoco/isaaclab-free: backend and action
modules are imported lazily at the function boundary.
"""

from __future__ import annotations

from typing import Final

from .builder import SimulationBuilder
from .control import Control, ControlFaceSpec, ControlSpec
from .errors import (
    BindingError,
    BuilderSealedError,
    ConfigModeError,
    DeferredFeatureError,
    ForeignHandleError,
    MissingActionError,
    NoControlError,
    NoWorldError,
    ObservationError,
    PendingActionError,
    SimulationClosedError,
    SimulationConfigError,
    SimulationError,
    SimulationFaultedError,
    StaleStateError,
)
from .handles import ActionBinding, ArticulationHandle, BodyHandle, CameraHandle

SIM_API_VERSION: Final[int] = 1
"""The experimental facade API version."""

__all__ = [
    "SIM_API_VERSION",
    "ActionBinding",
    "ArticulationHandle",
    "BodyHandle",
    "CameraHandle",
    "Control",
    "ControlFaceSpec",
    "ControlSpec",
    "SimulationBuilder",
    "SimulationError",
    "SimulationConfigError",
    "ForeignHandleError",
    "BindingError",
    "PendingActionError",
    "MissingActionError",
    "StaleStateError",
    "ObservationError",
    "SimulationClosedError",
    "SimulationFaultedError",
    "BuilderSealedError",
    "ConfigModeError",
    "DeferredFeatureError",
    "NoWorldError",
    "NoControlError",
]
