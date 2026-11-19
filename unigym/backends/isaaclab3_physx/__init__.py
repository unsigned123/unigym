"""IsaacLab 3.x PhysX backend identity (P1 skeleton).

The two IsaacLab 3 identities share one implementation package
(``unigym/backends/shared/isaaclab3_common``) but own separate trait tables,
engine profiles and runtime probes.  This package is the PhysX identity; it
currently declares no capability, so construction fails fast.  The
implementation lands in P1.
"""

from __future__ import annotations

from unigym.backends.registry import ISAACLAB3_PHYSX_TRAITS, BackendInfo

BACKEND_INFO = BackendInfo(
    "isaaclab3_physx",
    (3, 0),
    ISAACLAB3_PHYSX_TRAITS,
    asset_formats=("usd", "urdf"),
)

__all__ = ["BACKEND_INFO", "Isaaclab3PhysxEnv"]


class Isaaclab3PhysxEnv:
    """P1 skeleton: construction fails fast (no capability implemented)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "backend 'isaaclab3_physx' is a P1 skeleton — no capability is "
            "implemented yet; the shared backend lands in P1"
        )
