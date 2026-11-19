"""The engine-divergent decisions of the IsaacLab 3 backend library.

The shared IsaacLab 3 implementation is engine-neutral; the only places where
PhysX and Newton must differ are collected in :class:`EngineProfile`.  Each
backend identity constructs its own frozen profile and passes it to the shared
env.  Shared code must call the profile, never branch on ``profile.name``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["EngineProfile"]


@dataclass(frozen=True)
class EngineProfile:
    """Engine-specific decisions consumed by the shared IsaacLab 3 library.

    Attributes:
        name: Engine identity for diagnostics (``"physx"`` / ``"newton"``).
        build_physics_cfg: Build the ``SimulationCfg.physics`` value.
        renderer_cfg: Build the camera ``renderer_cfg``; ``None`` keeps the
            backend default.  Newton must return a warp renderer here.
        depth_background: The engine's "no hit" depth sentinel.
        default_joint_ordering: ``ArticulationCfg.joint_ordering`` default;
            ``None`` keeps the upstream default.
        set_body_com: Write body CoM positions for the DR event; the call
            signature and payload differ between PhysX and Newton.
    """

    name: str
    build_physics_cfg: Callable[[], object]
    renderer_cfg: Callable[[], object] | None
    depth_background: float
    default_joint_ordering: str | None
    set_body_com: Callable[[object, object, object], None]
