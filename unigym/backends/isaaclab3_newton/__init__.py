"""IsaacLab 3.x Newton backend identity.

The two IsaacLab 3 identities share one implementation package
(``unigym/backends/shared/isaaclab3_common``) but own separate trait tables,
engine profiles and runtime probes.  This package is the Newton identity; the
env class lives in :mod:`unigym.backends.isaaclab3_newton.env` and is imported
lazily so importing this package stays engine-free.
"""

from __future__ import annotations

from unigym.backends.registry import ISAACLAB3_NEWTON_TRAITS, BackendInfo

BACKEND_INFO = BackendInfo(
    "isaaclab3_newton",
    (3, 0),
    ISAACLAB3_NEWTON_TRAITS,
    asset_formats=("usd", "urdf"),
)

__all__ = ["BACKEND_INFO"]
