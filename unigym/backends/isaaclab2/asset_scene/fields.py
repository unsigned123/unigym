"""Lazy IsaacLab cfg-class resolution and asset field building.

The table is resolved only when a real field is built, so importing the
asset-scene package in the base environment never imports ``isaaclab``.
Offline tests inject their own class table.

Three field forms exist: the static USD ``AssetBaseCfg`` form and the dynamic
``RigidObjectCfg`` forms backed by a USD file or a jointless URDF.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import AssetFieldSpec

__all__ = ["build_field"]

#: The cfg classes each field form requires.  A missing entry is a version
#: mismatch against the pinned ``isaaclab2`` lane, never a silent fallback.
_REQUIRED_CLASSES = {
    "static_usd": ("AssetBaseCfg", "UsdFileCfg", "CollisionPropertiesCfg"),
    "dynamic_usd": (
        "RigidObjectCfg",
        "UsdFileCfg",
        "RigidBodyPropertiesCfg",
        "CollisionPropertiesCfg",
    ),
    "dynamic_urdf": ("RigidObjectCfg", "UrdfFileCfg", "CollisionPropertiesCfg"),
}


def _resolve(classes):
    """Return the cfg-class table (real isaaclab when ``classes`` is None)."""
    if classes is not None:
        return classes
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.sim import schemas
    from isaaclab.sim.spawners import UrdfFileCfg, UsdFileCfg

    return {
        "AssetBaseCfg": AssetBaseCfg,
        "RigidObjectCfg": RigidObjectCfg,
        "UsdFileCfg": UsdFileCfg,
        "UrdfFileCfg": UrdfFileCfg,
        "CollisionPropertiesCfg": sim_utils.CollisionPropertiesCfg,
        "RigidBodyPropertiesCfg": schemas.RigidBodyPropertiesCfg,
        "MassPropertiesCfg": schemas.MassPropertiesCfg,
        "ArticulationRootPropertiesCfg": schemas.ArticulationRootPropertiesCfg,
    }


def _require_classes(spec: AssetFieldSpec, c) -> None:
    """Reject a class table that cannot build the spec's cfg form."""
    missing = sorted(key for key in _REQUIRED_CLASSES[spec.cfg_kind] if key not in c)
    if missing:
        raise ValueError(
            f"asset field {spec.field!r}: isaaclab cfg class table is missing "
            f"{missing} for cfg kind {spec.cfg_kind!r}; the pinned isaaclab2 "
            f"lane must expose these classes -- check the isaaclab2 environment"
        )


def _build_static_usd(spec: AssetFieldSpec, c) -> object:
    """Build the static USD ``AssetBaseCfg`` field (no ``rigid_props``)."""
    collision_props = c["CollisionPropertiesCfg"](collision_enabled=spec.collision)
    spawn = c["UsdFileCfg"](usd_path=spec.resolved_path, collision_props=collision_props)
    init_state = c["AssetBaseCfg"].InitialStateCfg(pos=spec.pos, rot=spec.rot)
    return c["AssetBaseCfg"](prim_path=spec.prim_path, spawn=spawn, init_state=init_state)


def _build_dynamic_usd(spec: AssetFieldSpec, c) -> object:
    """Build the dynamic USD ``RigidObjectCfg`` field.

    ``rigid_props`` is always passed: SIMPLE-authored USDs carry their rigid
    and collision APIs on inner prims, so the reference root has no
    ``RigidBodyAPI`` and ``RigidObject`` initialization fails without it.
    """
    collision_props = c["CollisionPropertiesCfg"](collision_enabled=spec.collision)
    spawn = c["UsdFileCfg"](
        usd_path=spec.resolved_path,
        rigid_props=c["RigidBodyPropertiesCfg"](),
        collision_props=collision_props,
    )
    init_state = c["RigidObjectCfg"].InitialStateCfg(pos=spec.pos, rot=spec.rot)
    return c["RigidObjectCfg"](
        prim_path=spec.prim_path,
        spawn=spawn,
        init_state=init_state,
        collision_group=0,
    )


def _build_dynamic_urdf(spec: AssetFieldSpec, c) -> object:
    """Build the jointless-URDF ``RigidObjectCfg`` field.

    The explicit builder mirrors the prop spawner's non-anchored URDF form
    (``props/spawn.py``): free base, unmerged fixed joints, collision from
    visuals, instanceable, no joint drive, and the instance collision toggle.
    """
    spawn = c["UrdfFileCfg"](
        asset_path=spec.resolved_path,
        fix_base=False,
        merge_fixed_joints=False,
        collision_from_visuals=True,
        make_instanceable=True,
        joint_drive=None,
        collision_props=c["CollisionPropertiesCfg"](collision_enabled=spec.collision),
    )
    init_state = c["RigidObjectCfg"].InitialStateCfg(pos=spec.pos, rot=spec.rot)
    return c["RigidObjectCfg"](
        prim_path=spec.prim_path,
        spawn=spawn,
        init_state=init_state,
        collision_group=0,
    )


_BUILDERS = {
    "static_usd": _build_static_usd,
    "dynamic_usd": _build_dynamic_usd,
    "dynamic_urdf": _build_dynamic_urdf,
}


def build_field(spec: AssetFieldSpec, classes=None) -> object:
    """Build the IsaacLab scene field for one composed asset-field spec.

    Dispatches on ``spec.cfg_kind``:

    - ``static_usd``: ``AssetBaseCfg`` + ``UsdFileCfg`` with the collision
      toggle and NO ``RigidBodyPropertiesCfg`` (a static USD must not carry an
      authored rigid-body API; ``attach`` inspects that).
    - ``dynamic_usd``: ``RigidObjectCfg`` + ``UsdFileCfg`` with
      ``rigid_props`` and ``collision_group=0``.
    - ``dynamic_urdf``: ``RigidObjectCfg`` + ``UrdfFileCfg`` for a jointless
      URDF (the joint guard runs Kit-free in ``compose``).

    Args:
        spec: The composed asset field.
        classes: Optional cfg-class table (offline unit tests).

    Returns:
        The cfg object assigned to the scene config field.

    Raises:
        ValueError: The cfg kind has no field form, or the class table is
            missing a required entry for this lane.
    """
    builder = _BUILDERS.get(spec.cfg_kind)
    if builder is None:
        raise ValueError(
            f"asset field {spec.field!r}: cfg kind {spec.cfg_kind!r} has no "
            f"isaaclab2 asset-scene field form (supported: {sorted(_BUILDERS)})"
        )
    c = _resolve(classes)
    _require_classes(spec, c)
    return builder(spec, c)
