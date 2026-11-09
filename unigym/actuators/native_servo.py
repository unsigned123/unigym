"""native_servo: ctrl = clip(target, ctrlrange) passthrough.

For models whose actuators are affine POSITION servos in the MJCF itself
(``force = gainprm[0]*ctrl + biasprm[1]*q = k*(ctrl - q)``): the software
law only relays the target angle -- the model-side servo closes the loop
(docs/audits/v063_gr1_control_chain_audit.md §5: the GR1 Fourier hands,
kp=2 / thumb-yaw kp=4, ctrlrange [0, ~1.74]; negative targets clamp to
the range floor 0, exactly the audited stack's clip semantics).

Group fields: none beyond addressing.  ``kp``/``kd`` are DELIBERATELY
rejected (fail-fast): the servo gain lives in the MJCF gainprm/biasprm --
a software gain here would silently mean a different plant.
"""

from __future__ import annotations

import numpy as np

from .base import ActuatorBase, register


@register("native_servo")
class NativeServoActuator(ActuatorBase):
    """``ctrl = clip(target, ctrlrange)`` -- the model closes the loop."""

    config_keys = ActuatorBase.config_keys | {"kp", "kd"}  # rejected in _on_bind

    def _on_bind(self, env, joint_group: dict) -> None:
        for banned in ("kp", "kd"):
            if banned in joint_group:
                raise ValueError(
                    f"native_servo actuator group got {banned!r}; the servo "
                    f"gain lives in the MJCF actuator (gainprm/biasprm) -- "
                    f"a software {banned} here would silently change the "
                    f"plant | hint: switch the group to standard_pd for a "
                    f"software PD"
                )

    def step(self, model, data) -> None:
        """One sim substep: relay the clipped targets to the owned slots.

        Args:
            model: Live MjModel (supplies the actuators' ctrlrange).
            data: Live MjData; only the ``ctrl`` slots owned at bind are
                written.
        """
        self._require_bound()
        assert self._slots is not None and self._targets is not None
        cr = model.actuator_ctrlrange[self._slots]
        data.ctrl[self._slots] = np.clip(self._targets, cr[:, 0], cr[:, 1])
