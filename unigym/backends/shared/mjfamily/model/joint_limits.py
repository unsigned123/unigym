"""Shared MuJoCo-family joint-limit reader.

The write-joints range gate and the curated scene tree consult the same
number: the compiled ``MjModel`` limit table.  Living in the family library
lets both the mujoco scene face and the shared director ops read it without
a backend -> backend import.
"""

from __future__ import annotations


def _jnt_limit(m, jid: int) -> tuple[float, float] | None:
    """The MJ model limit of joint *jid*.

    ``((lo, hi))`` when the engine
    limits it, else None -- the honest no-gate face.
    """
    if not bool(m.jnt_limited[jid]):
        return None
    return float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1])


def model_joint_limits(m, names) -> dict:
    """``{name: (lo, hi) | None}`` straight from the MJ model.

    The same
    face the write-joints range gate consults -- zero new queries;
    unknown names face None like unlimited joints.
    """
    import mujoco

    out: dict = {}
    for n in names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        out[n] = _jnt_limit(m, jid) if jid >= 0 else None
    return out
