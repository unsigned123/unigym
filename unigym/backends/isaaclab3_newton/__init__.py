"""IsaacLab 3.x Newton backend identity (P1 skeleton).

The two IsaacLab 3 identities share one implementation package
(``unigym/backends/shared/isaaclab3_common``) but own separate trait tables,
engine profiles and runtime probes.  This package is the Newton identity; it
currently declares no capability, so construction fails fast.  The
implementation lands in P1/P3.
"""

from __future__ import annotations

from unigym.backends.registry import ISAACLAB3_NEWTON_TRAITS, BackendInfo

BACKEND_INFO = BackendInfo(
    "isaaclab3_newton",
    (3, 0),
    ISAACLAB3_NEWTON_TRAITS,
    asset_formats=("usd", "urdf"),
)

__all__ = ["BACKEND_INFO", "Isaaclab3NewtonEnv"]


class Isaaclab3NewtonEnv:
    """P1 skeleton: construction fails fast (no capability implemented)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "backend 'isaaclab3_newton' is a P1 skeleton — no capability is "
            "implemented yet; the shared backend and Newton profile land in P1"
        )
