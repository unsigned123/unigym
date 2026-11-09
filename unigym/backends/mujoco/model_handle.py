"""MuJoCo compiled-model handle face (neutral ``describe --compiled``).

The handle resolver lives here so the neutral CLI never names an engine
attribute.  ``model_provider`` still owns the IR description;
this module owns only the raw compiled object.
"""
from __future__ import annotations


def model_handle(env):
    """Return the live compiled ``MjModel`` (or ``None`` when absent)."""
    return getattr(env, "mj_model", None)
