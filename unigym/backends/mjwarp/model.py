"""MuJoCo Warp backend model face.

The compiled model is the same ``MjModel`` the CPU backend reads, so the
shared family provider does the work; only the IR backend label differs.
"""
from __future__ import annotations

from unigym.backends.shared.mjfamily.model.ir_access import (
    actuator_by_joint,
    body_id,
    dof_addresses,
    qpos_addresses,
    site_id,
)
from unigym.backends.shared.mjfamily.model.provider import (
    MujocoModelProvider as _MujocoModelProvider,
)


class MjwarpModelProvider(_MujocoModelProvider):
    """The shared MuJoCo-family provider labelled ``mjwarp``."""

    backend = "mjwarp"


MujocoModelProvider = MjwarpModelProvider
"""The canonical provider name, re-exported with the mjwarp label."""

__all__ = [
    "MjwarpModelProvider",
    "MujocoModelProvider",
    "actuator_by_joint",
    "body_id",
    "dof_addresses",
    "qpos_addresses",
    "site_id",
]
