"""Pluggable Actuator seam top-level package: registry + built-in laws.

Semantics: the backend Actuator seam; a first-class package, sibling of
adapters/benchmarks.  The backend's per-substep ctrl write face converges onto
pluggable control laws; the action-face semantics are untouched (NamedAction
carries the control's declared target kind, routing stays with the backend).
The built-ins:

* ``standard_pd``       the explicit-PD law;
* ``computed_torque_pd`` tau = M_part @ (kp*e - kd*dq) + qfrc_bias with a
  fresh M/bias read per substep, per-part blocked (the GR1 body semantics);
* ``native_servo``      ctrl = clip(target, ctrlrange) passthrough for models
  whose actuators are affine position servos (the GR1 hand semantics);
* ``direct_torque``     ctrl = clip(target torque, ctrlrange) over compiled
  motor actuators;
* ``direct_velocity``   ctrl = clip(target velocity, ctrlrange) over compiled
  velocity servos.

Authoring face: :class:`unigym.scenes.articulation.ArticulationSpec.actuation`
declares the neutral groups; the run layer resolves them and each backend
translates the law config.  The isaaclab backends translate only
``standard_pd`` natively and reject direct drive; the MuJoCo family serves
all five.
"""

from __future__ import annotations

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

__all__ = [
    "ACTUATOR_REGISTRY",
    "BACKEND_LAW_CAPABILITIES",
    "ActuationError",
    "ActuationGroup",
    "ActuatorBase",
    "ComputedTorquePdActuator",
    "DirectTorqueActuator",
    "DirectVelocityActuator",
    "GainValue",
    "NativeServoActuator",
    "ResolvedActuationGroup",
    "StandardPdActuator",
    "build_actuator",
    "normalize_actuation_group",
    "normalize_actuation_groups",
    "register",
    "resolve_actuation_groups",
    "resolve_gain_vector",
    "resolve_group_joints",
    "validate_actuation_group",
    "validate_actuation_groups",
    "validate_drive_law_matrix",
]
