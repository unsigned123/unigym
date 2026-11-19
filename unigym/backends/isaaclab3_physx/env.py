"""IsaacLab 3.x PhysX env class.

Thin subclass of the shared :class:`Isaaclab3EnvBase`: it pins the backend
identity and injects the PhysX engine profile.  All lifecycle/step/reset logic
lives in the shared base.
"""

from __future__ import annotations

from unigym.backends.shared.isaaclab3_common.env import Isaaclab3EnvBase
from unigym.env.backends.base import BackendCapabilities

from .profile import PHYSX_PROFILE

__all__ = ["BACKEND_CAPABILITIES", "Isaaclab3PhysxEnv"]

_CAPABILITIES = BackendCapabilities(
    multi_env=True,
    init_pose=True,
    state_write=True,
    native_handle=True,
    events=True,
)

BACKEND_CAPABILITIES = _CAPABILITIES


class Isaaclab3PhysxEnv(Isaaclab3EnvBase):
    """The IsaacLab 3.x PhysX backend identity."""

    backend_name = "isaaclab3_physx"
    engine_profile = PHYSX_PROFILE
    BACKEND_CAPABILITIES = _CAPABILITIES
