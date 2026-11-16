"""Experimental simulation facade (``SIM_API_VERSION = 4``).

The facade is a thin, additive surface over the existing ``Run`` / runner /
backend stack.  This namespace ships the explicit-pack builder, identity-scoped
handles, the ``Control.position_pd`` preset, the error taxonomy, the live
:class:`Simulation` and its typed :class:`Observation` projection (articulation
records, published camera frames and ``contact/`` tensors).

Actions are expressed with the ordered action template
(:class:`~unigym.runtime.actions.OrderedActionGroup`): each articulation handle
carries its action groups, and ``Simulation`` holds the global
:class:`~unigym.runtime.actions.OrderedActionSpec` they must match.

The tested coverage extends to ``num_envs`` in ``{1, 2, 3}`` on ``mujoco``
and targeted ``{1, 2}`` on ``isaaclab2`` (full dog lane unverified): explicit
``(N, dim)`` actions, per-env named values, ``reset(env_ids=...)`` and reported
due/terminated rows; ``step`` never resets a row.  Cameras at ``N > 1`` remain
deferred.

``import unigym.sim`` stays torch/mujoco/isaaclab-free: backend and action
modules are imported lazily at the function boundary.
"""

from __future__ import annotations

from typing import Final

from .builder import SimulationBuilder
from .control import Control, ControlGroupSpec, ControlSpec
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
from .handles import ArticulationHandle, BodyHandle, CameraHandle
from .observation import ArticulationObservation, CameraFrame, Observation
from .simulation import Simulation

SIM_API_VERSION: Final[int] = 4
"""The experimental facade API version."""

__all__ = [
    "SIM_API_VERSION",
    "ArticulationHandle",
    "ArticulationObservation",
    "BodyHandle",
    "CameraFrame",
    "CameraHandle",
    "Control",
    "ControlGroupSpec",
    "ControlSpec",
    "Observation",
    "Simulation",
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
