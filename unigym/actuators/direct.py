"""Direct-drive laws: write the drive target to the compiled ctrl slot.

Two explicit neutral laws, never a reuse of a position law:

* ``direct_torque``   -- ``ctrl = clip(target torque, ctrlrange)`` over
  compiled ``motor`` actuators (gain fixed, bias none);
* ``direct_velocity`` -- ``ctrl = clip(target velocity, ctrlrange)`` over
  compiled ``velocity_servo`` actuators (affine bias with a damping term).

The law owns the target write only; the compiled actuator owns the plant
(gear, ctrlrange and -- for velocity -- the servo gain).  A group with gains
or parts fails at IR validation: a direct law accepts no software gain, so
there is no silent approximation of a different physics model.  These are
ctrl-space writes: compiled force is ``gear * (gainprm[0] * ctrl + bias)``, so
``direct_torque`` means joint torque only when ``gear * gainprm[0] == 1`` (a
gain-2 general actuator still compiles as a motor but doubles the torque).
``direct_velocity`` requires ``gear == 1`` and the affine ``kv*(ctrl - qvel)``
law.  The MuJoCo-family seam checks both before the law binds; a mismatch is a
teaching error, not a fallback.
"""

from __future__ import annotations

import numpy as np

from .base import ActuatorBase, register


@register("direct_torque")
class DirectTorqueActuator(ActuatorBase):
    """``ctrl = clip(target torque, ctrlrange)`` over compiled motors."""

    def step(self, model, data) -> None:
        """One sim substep: write the clipped torque targets.

        Args:
            model: Live MjModel (supplies the actuators' ctrlrange).
            data: Live MjData; only the ``ctrl`` slots owned at bind are
                written.
        """
        self._require_bound()
        assert self._slots is not None and self._targets is not None
        cr = model.actuator_ctrlrange[self._slots]
        data.ctrl[self._slots] = np.clip(self._targets, cr[:, 0], cr[:, 1])


@register("direct_velocity")
class DirectVelocityActuator(ActuatorBase):
    """``ctrl = clip(target velocity, ctrlrange)`` over compiled velocity servos."""

    def step(self, model, data) -> None:
        """One sim substep: write the clipped velocity targets.

        Args:
            model: Live MjModel (supplies the actuators' ctrlrange).
            data: Live MjData; only the ``ctrl`` slots owned at bind are
                written.
        """
        self._require_bound()
        assert self._slots is not None and self._targets is not None
        cr = model.actuator_ctrlrange[self._slots]
        data.ctrl[self._slots] = np.clip(self._targets, cr[:, 0], cr[:, 1])
