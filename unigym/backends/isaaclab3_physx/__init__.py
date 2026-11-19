"""IsaacLab 3.x PhysX backend identity.

The two IsaacLab 3 identities share one implementation package
(``unigym/backends/shared/isaaclab3_common``) but own separate trait tables,
engine profiles and runtime probes.  This package is the PhysX identity; the
env class lives in :mod:`unigym.backends.isaaclab3_physx.env` and is imported
lazily so importing this package stays engine-free.
"""

from __future__ import annotations

from unigym.backends.registry import ISAACLAB3_PHYSX_TRAITS, BackendInfo

BACKEND_INFO = BackendInfo(
    "isaaclab3_physx",
    (3, 0),
    ISAACLAB3_PHYSX_TRAITS,
    asset_formats=("usd", "urdf"),
)

__all__ = ["BACKEND_INFO"]
