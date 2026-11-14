"""Lazy IsaacLab cfg-class resolution and static asset field building.

The table is resolved only when a real field is built, so importing the
asset-scene package in the base environment never imports ``isaaclab``.
Offline tests inject their own class table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import AssetFieldSpec

__all__ = ["build_field"]


def _resolve(classes):
    """Return the cfg-class table (real isaaclab when ``classes`` is None)."""
    if classes is not None:
        return classes
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sim.spawners import UsdFileCfg

    return {
        "AssetBaseCfg": AssetBaseCfg,
        "UsdFileCfg": UsdFileCfg,
        "CollisionPropertiesCfg": sim_utils.CollisionPropertiesCfg,
    }


def build_field(spec: AssetFieldSpec, classes=None) -> object:
    """Build the IsaacLab scene field for one composed asset-field spec.

    C4 owns the static USD form only: ``AssetBaseCfg`` + ``UsdFileCfg`` with
    the collision toggle and NO ``RigidBodyPropertiesCfg`` (a static USD must
    not carry an authored rigid-body API; ``attach`` inspects that).

    Args:
        spec: The composed asset field.
        classes: Optional cfg-class table (offline unit tests).

    Returns:
        The cfg object assigned to the scene config field.

    Raises:
        ValueError: The cfg kind is not the C4 static USD form, or the
            class table is missing a required entry.
    """
    if spec.cfg_kind != "static_usd":
        raise ValueError(
            f"asset field {spec.field!r}: cfg kind {spec.cfg_kind!r} is not "
            f"supported by the C4 asset-scene scaffold (supported: ['static_usd'])"
        )
    c = _resolve(classes)
    missing = sorted(
        key for key in ("AssetBaseCfg", "UsdFileCfg", "CollisionPropertiesCfg") if key not in c
    )
    if missing:
        raise ValueError(
            f"asset field {spec.field!r}: isaaclab cfg class table is missing "
            f"{missing}; pass the complete class table for this lane"
        )
    collision_props = c["CollisionPropertiesCfg"](collision_enabled=spec.collision)
    spawn = c["UsdFileCfg"](usd_path=spec.resolved_path, collision_props=collision_props)
    init_state = c["AssetBaseCfg"].InitialStateCfg(pos=spec.pos, rot=spec.rot)
    return c["AssetBaseCfg"](prim_path=spec.prim_path, spawn=spawn, init_state=init_state)
