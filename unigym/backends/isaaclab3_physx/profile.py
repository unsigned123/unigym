"""Engine profile for the IsaacLab 3 PhysX backend identity.

Module import is engine-free: the profile only imports
:class:`EngineProfile` and ``math``; every isaaclab import lives inside a
function body and runs when the shared env asks the profile for a value.
"""

from __future__ import annotations

import math

from unigym.backends.shared.isaaclab3_common.engines import EngineProfile

__all__ = ["PHYSX_PROFILE"]


def _build_physics_cfg():
    """Build the PhysX ``SimulationCfg.physics`` value."""
    from isaaclab_physx.physics import PhysxCfg

    return PhysxCfg()


def _renderer_cfg():
    """Build the PhysX (RTX) camera renderer cfg."""
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    return IsaacRtxRendererCfg()


def _set_body_com(asset: object, com: object, env_ids: object) -> None:
    """Write body CoM positions for the DR event (lands in C2)."""
    raise NotImplementedError("IsaacLab 3 PhysX set_body_com lands in C2 (set_coms_index)")


PHYSX_PROFILE = EngineProfile(
    name="physx",
    build_physics_cfg=_build_physics_cfg,
    renderer_cfg=_renderer_cfg,
    depth_background=math.inf,
    default_joint_ordering="physx",
    set_body_com=_set_body_com,
)
