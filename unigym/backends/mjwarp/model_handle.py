"""mjwarp compiled-model handle face (neutral ``describe --compiled``).

mjwarp keeps one CPU-mirror ``MjModel`` backing its batched worlds; the
resolver is the mjwarp-labelled twin of the mujoco face.  The neutral CLI
asks the registry for this handle instead of sniffing env attributes.
"""

from __future__ import annotations


def model_handle(env):
    """Return the live compiled ``MjModel`` (or ``None`` when absent)."""
    return getattr(env, "mj_model", None)
