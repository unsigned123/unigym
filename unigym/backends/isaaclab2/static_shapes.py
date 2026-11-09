"""Static ``World.add`` primitives -> IsaacLab static colliders (S2b).

The inline mujoco world renders static primitives in the scene-IR payload.
IsaacLab has no scene-IR composer, but it DOES have a clean static-collider
path: ``AssetBaseCfg`` + a shape spawner carrying ``CollisionPropertiesCfg``
and NO ``RigidBodyPropertiesCfg``.  The spawned prim then owns only the USD
CollisionAPI, so PhysX treats it as a static (non-simulated) collider -- the
same role the inline static shapes play on mujoco.

Each declaration is cloned per environment at its declared world pose
relative to the env origin (the inline world frame is the env frame).  The
static shape vocabulary is the closed ``World.add`` one: box / sphere /
cylinder / capsule / plane.  A plane maps to IsaacLab's ``GroundPlaneCfg``
(an infinite collision plane; the neutral ``size`` is the visual grid half
extent, so the mapped plane uses ``2 * size`` as the finite visual extent --
physics stays an infinite plane).

isaaaclab stays a lazy import: callers inject ``classes=`` for offline tests.
"""

from __future__ import annotations

#: neutral primitive kind -> the cfg-class name that spawns it
_CFG_CLASS = {
    "box": "CuboidCfg",
    "sphere": "SphereCfg",
    "cylinder": "CylinderCfg",
    "capsule": "CapsuleCfg",
    "plane": "GroundPlaneCfg",
}

_DEFAULT_COLOR = (0.55, 0.55, 0.58)


def _resolve(classes):
    """Return the cfg-class table (real isaaclab when ``classes`` is None)."""
    if classes is not None:
        return classes
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return {
        "AssetBaseCfg": AssetBaseCfg,
        "CuboidCfg": sim_utils.CuboidCfg,
        "SphereCfg": sim_utils.SphereCfg,
        "CylinderCfg": sim_utils.CylinderCfg,
        "CapsuleCfg": sim_utils.CapsuleCfg,
        "GroundPlaneCfg": sim_utils.GroundPlaneCfg,
        "CollisionPropertiesCfg": sim_utils.CollisionPropertiesCfg,
        "PreviewSurfaceCfg": sim_utils.PreviewSurfaceCfg,
    }


def _spawn(kind: str, params: dict, c: dict, color):
    """Build the collision-only spawner cfg for one primitive kind."""
    collision = c["CollisionPropertiesCfg"](collision_enabled=True)
    if kind == "plane":
        size = params["size"]
        return c["GroundPlaneCfg"](
            size=(2.0 * float(size[0]), 2.0 * float(size[1])),
            color=color,
        )
    visual = c["PreviewSurfaceCfg"](diffuse_color=color)
    common = dict(collision_props=collision, visual_material=visual)
    if kind == "box":
        return c["CuboidCfg"](size=tuple(float(v) for v in params["size"]), **common)
    if kind == "sphere":
        return c["SphereCfg"](radius=float(params["radius"]), **common)
    if kind == "cylinder":
        return c["CylinderCfg"](
            radius=float(params["radius"]), height=2.0 * float(params["half_height"]), **common
        )
    if kind == "capsule":
        # USD capsule ``height`` is the cylindrical section (caps are
        # added on top): the neutral half_height is that half-length.
        return c["CapsuleCfg"](
            radius=float(params["radius"]), height=2.0 * float(params["half_height"]), **common
        )
    raise ValueError(
        f"static primitive kind {kind!r} has no isaaclab mapping " f"(known: {sorted(_CFG_CLASS)})"
    )


def build_static_scene_fields(declarations, *, classes=None) -> dict:
    """``{scene_field: AssetBaseCfg}`` for static ``World.add`` primitives.

    Args:
        declarations: The :meth:`unigym.world.World.static_declarations`
            entries (``{name, kind, params, pos, quat}``).
        classes: Optional cfg-class table (offline unit tests).

    Returns:
        Scene-config fields to merge into ``InteractiveSceneCfg``.

    Raises:
        ValueError: An unknown primitive kind or a colliding field name.
    """
    c = _resolve(classes)
    fields: dict = {}
    for decl in declarations:
        kind = decl["kind"]
        if kind not in _CFG_CLASS:
            raise ValueError(
                f"static primitive {decl.get('name')!r}: kind {kind!r} has "
                f"no isaaclab static mapping (known: {sorted(_CFG_CLASS)})"
            )
        field = f"static_{decl['name']}"
        if field in fields:
            raise ValueError(
                f"static primitive {decl['name']!r}: duplicate scene field " f"{field!r}"
            )
        spawn = _spawn(kind, dict(decl["params"]), c, _DEFAULT_COLOR)
        fields[field] = c["AssetBaseCfg"](
            prim_path=f"{{ENV_REGEX_NS}}/Static_{decl['name']}",
            spawn=spawn,
            init_state=c["AssetBaseCfg"].InitialStateCfg(
                pos=tuple(float(v) for v in decl["pos"]),
                rot=tuple(float(v) for v in decl["quat"]),
            ),
        )
    return fields


__all__ = ["build_static_scene_fields"]
