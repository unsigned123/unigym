"""standard_pd: the explicit-PD ctrl law.

Playground-authority form (playground_official_sonic.py::apply_pd):
``tau = kp*(q_target - q) - kd*dq``, clipped to the actuator ctrlrange,
recomputed from the CURRENT q/dq on every call.  This is the mujoco
backend's retired inline ``_apply_pd`` body, float-op for float-op
(the default law must keep the legacy formula/op order -- the MJ4/I2
guard tests lock the identity).
The damping term runs through the per-dof damping-stability scale when
the env provides one (``_damping_stability_scale``) -- a no-op (scale 1)
for every dof inside the explicit-integration stability bound.

Group fields: ``kp``/``kd`` as scalar | joint-name/regex dict | full
vector.  Both are REQUIRED: actuation is the only physical gain source,
so a group without explicit gains fails at bind (there is no env gain
fallback channel).

``live_ctrlrange`` (constructor arg, NOT a config field): read the
ctrlrange from the model on every step instead of the bind-time copy --
the legacy hand-PD path read it live; the written values are identical
either way, the switch only preserves the exact legacy read pattern.
"""

from __future__ import annotations

import numpy as np

from .base import ActuatorBase, register, resolve_gain_vector


@register("standard_pd")
class StandardPdActuator(ActuatorBase):
    """``tau = clip(kp*(q* - q) - kd_eff*dq, ctrlrange)`` per owned slot."""

    config_keys = ActuatorBase.config_keys | {"kp", "kd"}

    def __init__(self, live_ctrlrange: bool = False) -> None:
        super().__init__()
        self._live_ctrlrange = bool(live_ctrlrange)
        self._kp: np.ndarray | None = None
        self._kd: np.ndarray | None = None
        self._scale_fn = None

    def _on_bind(self, env, joint_group: dict) -> None:
        missing = [key for key in ("kp", "kd") if joint_group.get(key) is None]
        if missing:
            raise ValueError(
                f"standard_pd actuator group is missing {missing} | hint: set "
                f"the group's kp/kd (scalar or joint-name/regex dict); there "
                f"is no env gain fallback"
            )
        self._kp = resolve_gain_vector(joint_group["kp"], "kp", self._joint_names)
        self._kd = resolve_gain_vector(joint_group["kd"], "kd", self._joint_names)
        # the damping-stability guard is an env-side numerical concern
        # (mj_fullM of THIS model at THIS sim_dt): adopt the env's when it
        # provides one, exactly as the legacy _apply_pd path did
        self._scale_fn = getattr(env, "_damping_stability_scale", None)

    def step(self, model, data) -> None:
        """One sim substep: explicit-PD write over the owned slots.

        ``tau = kp*(q* - q) - kd_eff*dq`` from the CURRENT q/dq (recomputed
        per call); kd_eff runs through the env's damping-stability scale
        when provided, and the result is clipped to ctrlrange (live model
        read when ``live_ctrlrange``).

        Args:
            model: Live MjModel (supplies ctrlrange per read mode).
            data: Live MjData; only the ``ctrl`` slots owned at bind are
                written.
        """
        self._require_bound()
        assert (
            self._qadr is not None
            and self._dadr is not None
            and self._slots is not None
            and self._targets is not None
            and self._kp is not None
            and self._kd is not None
            and self._ctrlrange is not None
        )
        q = data.qpos[self._qadr]
        dq = data.qvel[self._dadr]
        if self._scale_fn is not None:
            kd_eff = self._kd * self._scale_fn(data, self._dadr, self._kp, self._kd)
        else:
            kd_eff = self._kd
        tau = self._kp * (self._targets - q) - kd_eff * dq
        cr = model.actuator_ctrlrange[self._slots] if self._live_ctrlrange else self._ctrlrange
        data.ctrl[self._slots] = np.clip(tau, cr[:, 0], cr[:, 1])
