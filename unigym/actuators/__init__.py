"""Pluggable Actuator seam top-level package: registry + three built-ins.

Provenance and semantics: docs/designs/v063-gr00t-adapter.md D2/D9 (the
    backend Actuator seam; a first-class package, sibling of
adapters/benchmarks).  The backend's per-substep ctrl write face
converges onto pluggable control laws; the action-face semantics are
untouched (NamedAction = absolute position targets, routing stays with
the backend).  The three built-ins:

* ``standard_pd``       the default law, float-op identical to the
  mujoco backend's retired ``_apply_pd`` formula;
* ``computed_torque_pd`` tau = M_part @ (kp*e - kd*dq) + qfrc_bias with
  a fresh M/bias read per substep, per-part blocked (the GR1 body
  semantics);
* ``native_servo``      ctrl = clip(target, ctrlrange) passthrough for
  models whose actuators are affine position servos (the GR1 hand
  semantics).

Config face: env.params.actuators (group name -> {type,
regex|joint_names, ...}).  D9 fresh-forward protocol: no declared key =
the backend's original code path verbatim; a declaration containing a
non-standard law switches the decimation loop into the
mj_step1 -> actuator.step -> mj_step2 phase split.  The isaaclab
backend rejects non-standard declarations at the capability gate with
error+hint.
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

__all__ = [
    "ACTUATOR_REGISTRY",
    "ActuatorBase",
    "ComputedTorquePdActuator",
    "NativeServoActuator",
    "StandardPdActuator",
    "build_actuator",
    "register",
    "resolve_gain_vector",
]
