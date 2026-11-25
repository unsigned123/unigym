"""Pluggable Actuator seam top-level package: registry + built-in laws.

Semantics: the backend Actuator seam; a first-class package, sibling of
adapters.  The backend's per-substep ctrl write face converges onto
pluggable control laws; the action-group semantics are untouched (NamedAction
carries the control's declared target kind, routing stays with the backend).
The built-ins:

* ``standard_pd``       the explicit-PD law;
* ``computed_torque_pd`` tau = M_part @ (kp*e - kd*dq) + qfrc_bias with a
  fresh M/bias read per substep, per-part blocked (the reference body semantics);
* ``native_servo``      ctrl = clip(target, ctrlrange) passthrough for models
  whose actuators are affine position servos (the reference hand semantics);
* ``direct_torque``     ctrl = clip(target torque, ctrlrange) over compiled
  motor actuators;
* ``direct_velocity``   ctrl = clip(target velocity, ctrlrange) over compiled
  velocity servos.

The five core laws are declared as pure descriptors
(:mod:`unigym.actuators.descriptors`); their concrete classes remain in
:data:`ACTUATOR_REGISTRY` for compatibility.  Backend factories live in the
internal implementation registry (:mod:`unigym.actuators.implementations`).
"""

from __future__ import annotations

from . import descriptors as _descriptors  # noqa: F401  (registers core laws)
from .base import (
    ACTUATOR_REGISTRY,
    ActuatorBase,
    build_actuator,
    register,
    resolve_gain_vector,
)
from .standard_pd import StandardPdActuator
from .computed_torque_pd import ComputedTorquePdActuator
from .native_servo import NativeServoActuator
from .direct import DirectTorqueActuator, DirectVelocityActuator
from .ir import (
    BACKEND_LAW_CAPABILITIES,
    ActuationError,
    ActuationGroup,
    GainValue,
    ResolvedActuationGroup,
    normalize_actuation_group,
    normalize_actuation_groups,
    resolve_actuation_groups,
    resolve_group_joints,
    validate_actuation_group,
    validate_actuation_groups,
    validate_drive_law_matrix,
)
from .implementations import (
    ACTUATOR_IMPLEMENTATION_REGISTRY,
    ActuatorBuildContext,
    ActuatorImplementation,
    get_actuator_implementation,
    has_actuator_implementation,
    register_actuator_implementation,
    restore_actuator_implementations,
    snapshot_actuator_implementations,
)
from unigym.actuator_law import (
    ACTUATOR_LAW_REGISTRY,
    ActuatorLawDescriptor,
    actuator_law,
    register_actuator_law,
    restore_actuator_laws,
    snapshot_actuator_laws,
)

__all__ = [
    "ACTUATOR_IMPLEMENTATION_REGISTRY",
    "ACTUATOR_LAW_REGISTRY",
    "ACTUATOR_REGISTRY",
    "BACKEND_LAW_CAPABILITIES",
    "ActuationError",
    "ActuationGroup",
    "ActuatorBase",
    "ActuatorBuildContext",
    "ActuatorImplementation",
    "ActuatorLawDescriptor",
    "ComputedTorquePdActuator",
    "DirectTorqueActuator",
    "DirectVelocityActuator",
    "GainValue",
    "NativeServoActuator",
    "ResolvedActuationGroup",
    "StandardPdActuator",
    "actuator_law",
    "build_actuator",
    "get_actuator_implementation",
    "has_actuator_implementation",
    "normalize_actuation_group",
    "normalize_actuation_groups",
    "register",
    "register_actuator_implementation",
    "register_actuator_law",
    "resolve_actuation_groups",
    "resolve_gain_vector",
    "resolve_group_joints",
    "restore_actuator_implementations",
    "restore_actuator_laws",
    "snapshot_actuator_implementations",
    "snapshot_actuator_laws",
    "validate_actuation_group",
    "validate_actuation_groups",
    "validate_drive_law_matrix",
]
