"""IsaacLab 3.x Newton env class.

Thin subclass of the shared :class:`Isaaclab3EnvBase`: it pins the backend
identity and injects the Newton engine profile.  All lifecycle/step/reset
logic lives in the shared base.
"""

from __future__ import annotations

from unigym.backends.shared.isaaclab3_common.env import Isaaclab3EnvBase
from unigym.env.backends.base import BackendCapabilities

from .profile import NEWTON_PROFILE

__all__ = ["BACKEND_CAPABILITIES", "Isaaclab3NewtonEnv"]

_CAPABILITIES = BackendCapabilities(
    multi_env=True,
    init_pose=True,
    state_write=True,
    native_handle=True,
)

BACKEND_CAPABILITIES = _CAPABILITIES


class Isaaclab3NewtonEnv(Isaaclab3EnvBase):
    """The IsaacLab 3.x Newton backend identity."""

    backend_name = "isaaclab3_newton"
    engine_profile = NEWTON_PROFILE
    BACKEND_CAPABILITIES = _CAPABILITIES
