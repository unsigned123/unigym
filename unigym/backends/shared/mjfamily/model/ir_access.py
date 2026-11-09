"""Resolve the compiled ModelIR a mujoco-family env assembled.

Production envs publish ``model_set`` from the compiled ``MjModel``; the
whole-scene IR is ``model_set.merged``.  Test doubles and older
stand-alone probes may bind a backend with only ``mj_model``/``mj_data``;
describing the model keeps those paths working without a second, divergent
parser in the backend code.
"""
from __future__ import annotations


def model_ir_of(env):
    """Return *env*'s assembled whole-scene ModelIR (describe when absent).

    The provider import stays lazy here as well: this helper is reached
    only from a bind method, never from module import.
    """
    model_set = getattr(env, "model_set", None)
    if model_set is not None and model_set.merged is not None:
        return model_set.merged
    ir = getattr(env, "_model_ir", None)
    if ir is not None:
        return ir
    from .provider import MujocoModelProvider

    return MujocoModelProvider().describe(env.mj_model)


def body_id(ir, name):
    """Return the native body id for *name*, or ``-1`` when absent."""
    row = ir.bodies[name] if name in ir.bodies else None
    return -1 if row is None or row.body_id is None else int(row.body_id)


def site_id(ir, name):
    """Return the native site id for *name*, or ``-1`` when absent."""
    row = ir.sites[name] if name in ir.sites else None
    return -1 if row is None or row.site_id is None else int(row.site_id)


def camera_id(ir, name):
    """Return the native camera id for *name*, or ``-1`` when absent."""
    row = ir.cameras[name] if name in ir.cameras else None
    return -1 if row is None or row.camera_id is None else int(row.camera_id)


def actuator_by_joint(ir):
    """Return ``{joint name: actuator slot}`` for joint actuators."""
    return {act.target: int(act.actuator_id)
            for act in ir.actuators
            if act.trntype in ("joint", "jointinparent")
            and act.actuator_id is not None}


def qpos_addresses(ir, names):
    """Return each named joint's qpos address (compiled IR order-free)."""
    return [int(ir.joints[name].qpos_adr) for name in names]


def dof_addresses(ir, names):
    """Return each named joint's dof address (compiled IR order-free)."""
    return [int(ir.joints[name].dof_adr) for name in names]
