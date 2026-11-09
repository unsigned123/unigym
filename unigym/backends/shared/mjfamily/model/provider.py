"""MuJoCo model provider: pre-compile ``build`` and post-compile ``describe``.

The provider is the only engine-facing entry point of the MuJoCo family.
It imports no MuJoCo symbol at module import time; the spec/model walkers
are plain attribute readers and are loaded only when a provider method is
called.  ``mjwarp`` reuses this provider (with ``backend="mjwarp"``)
because it consumes the same ``MjModel``.
"""

from __future__ import annotations

__all__ = ["MujocoModelProvider"]


class MujocoModelProvider:
    """Build the declared IR from ``MjSpec`` and fill it from ``MjModel``."""

    backend = "mujoco"

    def __init__(self, backend=None):
        """Create a provider; *backend* overrides the label (e.g. mjwarp)."""
        if backend is not None:
            self.backend = backend

    def build(self, mjspec):
        """Walk an uncompiled ``MjSpec`` and return the declared model IR."""
        from .spec_walk import build_from_spec

        return build_from_spec(mjspec, backend=self.backend)

    def describe(self, mjmodel):
        """Read a compiled ``MjModel`` and return the filled model IR."""
        from .model_walk import describe_model

        return describe_model(mjmodel, backend=self.backend)
