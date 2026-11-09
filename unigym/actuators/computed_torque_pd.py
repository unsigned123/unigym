"""computed_torque_pd: tau = M_part @ (kp*(q* - q) - kd*dq) + qfrc_bias.

The GR1 (Fourier) stack's software computed-torque law, per-part blocked
(docs/audits/v063_gr1_control_chain_audit.md §3; design D2 -- the GR1 body
law): the mass matrix is read FRESH on every substep via ``mj_fullM``;
each part contributes its OWN diagonal block ``M[qvel_index][:, qvel_index]
@ desired_torque`` -- cross-part coupling is dropped by construction
(torso / right-arm / left-arm blocks in the audited stack, declared here
by the group's ``parts`` regex list), and ``qfrc_bias`` (gravity +
Coriolis) rides on top.  Targets are absolute joint positions held per
control step; tau is clipped to the owning actuators' ctrlrange at write.

Group fields: ``kp``/``kd`` REQUIRED (scalar | joint-name/regex dict -- a
computed-torque law without gains is meaningless; there is no silent
default; the audited stack runs kp=1000/kd=200 on all body joints).
``parts``: optional list of regexes PARTITIONING the group's joints into
independent sub-blocks (default: one block = the whole group).  Each part
regex must match >= 1 group joint, parts must not overlap, and together
they must cover the group -- all fail-fast.
"""

from __future__ import annotations

import re

import numpy as np

from .base import ActuatorBase, register, resolve_gain_vector


@register("computed_torque_pd")
class ComputedTorquePdActuator(ActuatorBase):
    """``tau = clip(M_part @ (kp*e - kd*dq) + qfrc_bias, ctrlrange)``."""

    config_keys = ActuatorBase.config_keys | {"kp", "kd", "parts"}

    def __init__(self) -> None:
        super().__init__()
        self._kp: np.ndarray | None = None
        self._kd: np.ndarray | None = None
        self._parts: list = []  # [(label, index array into the group)]

    def _on_bind(self, env, joint_group: dict) -> None:
        if joint_group.get("kp") is None or joint_group.get("kd") is None:
            missing = [k for k in ("kp", "kd") if joint_group.get(k) is None]
            raise ValueError(
                f"computed_torque_pd actuator group is missing {missing} | "
                f"hint: a computed-torque law needs explicit gains (the "
                f"audited GR1 stack runs kp=1000/kd=200 on all body "
                f"joints); there is no silent default"
            )
        self._kp = resolve_gain_vector(joint_group["kp"], "kp", self._joint_names)
        self._kd = resolve_gain_vector(joint_group["kd"], "kd", self._joint_names)
        self._parts = self._resolve_parts(joint_group.get("parts"))

    def _resolve_parts(self, parts_cfg) -> list:
        n = len(self._joint_names)
        if not parts_cfg:
            return [(r".*", np.arange(n, dtype=np.int64))]
        if not isinstance(parts_cfg, (list, tuple)):
            raise ValueError(
                f"computed_torque_pd 'parts' must be a list of regexes "
                f"partitioning the group, got {type(parts_cfg).__name__}"
            )
        covered = np.zeros(n, dtype=bool)
        blocks = []
        for expr in parts_cfg:
            expr = str(expr)
            sel = np.array([re.fullmatch(expr, nm) is not None for nm in self._joint_names])
            if not sel.any():
                raise ValueError(
                    f"computed_torque_pd parts entry {expr!r} matches none "
                    f"of the group joints {self._joint_names} | hint: part "
                    f"regexes fullmatch against the GROUP's own joints"
                )
            if (covered & sel).any():
                dup = [nm for nm, c, s in zip(self._joint_names, covered, sel) if c and s]
                raise ValueError(
                    f"computed_torque_pd parts overlap at joints {dup} "
                    f"(each group joint must live in exactly one part)"
                )
            covered |= sel
            blocks.append((expr, np.nonzero(sel)[0].astype(np.int64)))
        if not bool(covered.all()):
            missing = [nm for nm, c in zip(self._joint_names, covered) if not c]
            raise ValueError(
                f"computed_torque_pd parts leave group joints uncovered: "
                f"{missing} (parts: {[str(p) for p in parts_cfg]})"
            )
        return blocks

    def step(self, model, data) -> None:
        """One sim substep: write the per-part computed-torque ctrl slots.

        Reads the mass matrix fresh via ``mj_fullM``, applies each part's
        own diagonal block to ``kp*(q* - q) - kd*dq`` plus ``qfrc_bias``,
        and clips to the owning actuators' ctrlrange.

        Args:
            model: Live MjModel (read-only here).
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
        )
        import mujoco

        nv = int(model.nv)
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(model, data, M)
        q = data.qpos[self._qadr]
        dq = data.qvel[self._dadr]
        bias = data.qfrc_bias
        tau = np.zeros(len(self._joint_names), dtype=np.float64)
        for _, idx in self._parts:
            dofs = self._dadr[idx]
            e = self._kp[idx] * (self._targets[idx] - q[idx]) - self._kd[idx] * dq[idx]
            tau[idx] = M[np.ix_(dofs, dofs)] @ e + bias[dofs]
        cr = model.actuator_ctrlrange[self._slots]
        data.ctrl[self._slots] = np.clip(tau, cr[:, 0], cr[:, 1])
