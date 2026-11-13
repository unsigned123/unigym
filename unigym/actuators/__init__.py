"""Pluggable Actuator seam top-level package: registry + three built-ins.

Semantics: the backend Actuator seam; a first-class package, sibling of
adapters/benchmarks.  The backend's per-substep ctrl write face converges onto
pluggable control laws; the action-face semantics are untouched (NamedAction =
absolute position targets, routing stays with the backend).  The three
built-ins:

* ``standard_pd``       the explicit-PD law;
* ``computed_torque_pd`` tau = M_part @ (kp*e - kd*dq) + qfrc_bias with a
  fresh M/bias read per substep, per-part blocked (the GR1 body semantics);
* ``native_servo``      ctrl = clip(target, ctrlrange) passthrough for models
  whose actuators are affine position servos (the GR1 hand semantics).

Authoring face: :class:`unigym.scenes.articulation.ArticulationSpec.actuation`
declares the neutral groups; the run layer resolves them and each backend
translates the law config.  The isaaclab backends translate only
``standard_pd`` natively; the MuJoCo family serves all three.
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
from .ir import (
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
    "ActuationError",
    "ActuationGroup",
    "ActuatorBase",
    "ComputedTorquePdActuator",
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
