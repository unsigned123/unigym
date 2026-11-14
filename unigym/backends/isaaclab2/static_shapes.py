"""Static ``World.add`` primitives -> IsaacLab static colliders (S2b).

The inline mujoco world renders static primitives in the scene-IR payload.
IsaacLab's file-backed scene composer lives in
``unigym/backends/isaaclab2/asset_scene``; this module remains the primitive
mapper and keeps the static-collider path: ``AssetBaseCfg`` + a shape spawner
carrying ``CollisionPropertiesCfg`` and NO ``RigidBodyPropertiesCfg``.  The
spawned prim then owns only the USD CollisionAPI, so PhysX treats it as a
static (non-simulated) collider -- the same role the inline static shapes play
on mujoco.

Each declaration is cloned per environment at its declared world pose
relative to the env origin (the inline world frame is the env frame).  The
static shape vocabulary is the closed ``World.add`` one: box / sphere /
cylinder / capsule / plane.  A plane maps to IsaacLab's ``GroundPlaneCfg``
(an infinite collision plane; the neutral ``size`` is the visual grid half
extent, so the mapped plane uses ``2 * size`` as the finite visual extent --
physics stays an infinite plane).

:func:`build_static_scene_fields` keeps the ``World.add`` naming.  The
asset-scene engine uses :func:`build_static_payload_fields`, which maps a
neutral :class:`~unigym.scenes.primitives.ShapePayload` onto the same
static fields: one shape keeps ``static_<id>``, a multi-shape payload
(``room_shell``) gets ``static_<id>_<shape>`` under ``Static_<id>/<shape>``.
Primitive materials stay the documented fallback colour; the neutral
material table is not bound here.

isaaaclab stays a lazy import: callers inject ``classes=`` for offline tests.
"""

from __future__ import annotations

from unigym.scenes.primitives import (
    BoxShape,
    CapsuleShape,
    CylinderShape,
    PlaneShape,
    SphereShape,
)

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


def _require_kind(name, kind) -> None:
    """Reject a primitive kind outside the closed static mapping."""
    if kind not in _CFG_CLASS:
        raise ValueError(
            f"static primitive {name!r}: kind {kind!r} has "
            f"no isaaclab static mapping (known: {sorted(_CFG_CLASS)})"
        )


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


def build_static_field(
    name,
    kind,
    params,
    pos,
    quat,
    *,
    classes=None,
    prim_path=None,
):
    """Build one collision-only ``AssetBaseCfg`` static primitive field.

    Args:
        name: The primitive name used in the teaching errors.
        kind: The closed neutral primitive kind.
        params: The primitive dimension parameters.
        pos: The world position ``(x, y, z)``.
        quat: The world rotation ``(w, x, y, z)``.
        classes: Optional cfg-class table (offline unit tests).
        prim_path: Override the default ``{ENV_REGEX_NS}/Static_<name>``.

    Returns:
        The ``AssetBaseCfg`` scene field.

    Raises:
        ValueError: An unknown primitive kind.
    """
    from .asset_scene.names import static_prim_path

    c = _resolve(classes)
    _require_kind(name, kind)
    prim = static_prim_path(name) if prim_path is None else prim_path
    spawn = _spawn(kind, dict(params), c, _DEFAULT_COLOR)
    return c["AssetBaseCfg"](
        prim_path=prim,
        spawn=spawn,
        init_state=c["AssetBaseCfg"].InitialStateCfg(
            pos=tuple(float(v) for v in pos),
            rot=tuple(float(v) for v in quat),
        ),
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
    from .asset_scene.names import static_field

    c = _resolve(classes)
    fields: dict = {}
    for decl in declarations:
        name = decl["name"]
        _require_kind(name, decl["kind"])
        field = static_field(name)
        if field in fields:
            raise ValueError(f"static primitive {name!r}: duplicate scene field {field!r}")
        fields[field] = build_static_field(
            name,
            decl["kind"],
            dict(decl["params"]),
            decl["pos"],
            decl["quat"],
            classes=c,
        )
    return fields


def _shape_kind_params(shape) -> tuple[str, dict]:
    """Map one neutral shape to its primitive kind and dimension params."""
    if isinstance(shape, BoxShape):
        return "box", {"size": list(shape.size)}
    if isinstance(shape, SphereShape):
        return "sphere", {"radius": shape.radius}
    if isinstance(shape, CylinderShape):
        return "cylinder", {"radius": shape.radius, "half_height": shape.half_height}
    if isinstance(shape, CapsuleShape):
        return "capsule", {"radius": shape.radius, "half_height": shape.half_height}
    if isinstance(shape, PlaneShape):
        return "plane", {"size": list(shape.size)}
    raise TypeError(
        f"static payload shape {type(shape).__name__} has no isaaclab static "
        f"mapping (known: {sorted(_CFG_CLASS)})"
    )


def build_static_payload_fields(payload, *, instance_id, classes=None) -> dict:
    """``{scene_field: AssetBaseCfg}`` for one neutral primitive payload.

    A single-shape payload keeps the ``World.add`` naming (``static_<id>`` /
    ``{ENV_REGEX_NS}/Static_<id>``).  A multi-shape payload addresses each
    shape as ``static_<id>_<shape.name>`` under
    ``{ENV_REGEX_NS}/Static_<id>/<shape.name>``.  Every field uses the
    shape's own resolved ``pos``/``quat_wxyz`` and dimensions; the primitive
    material stays the documented fallback colour.  The caller folds the
    instance pose into the generator ``pos``/``quat`` parameters, so the
    payload shapes already carry the world pose.

    Args:
        payload: The generator's neutral
            :class:`~unigym.scenes.primitives.ShapePayload`.
        instance_id: The scene instance id anchoring the payload.
        classes: Optional cfg-class table (offline unit tests).

    Returns:
        Scene-config fields for the payload.

    Raises:
        ValueError: The payload is empty or two shapes share a field name.
        TypeError: A payload entry is not a closed neutral shape class.
    """
    from .asset_scene.names import (
        static_field,
        static_prim_path,
        static_shape_field,
        static_shape_prim_path,
    )

    c = _resolve(classes)
    shapes = payload.shapes
    if not shapes:
        raise ValueError(
            f"static payload for instance {instance_id!r} has no shapes; a "
            f"primitive instance must generate at least one shape"
        )
    fields: dict = {}
    if len(shapes) == 1:
        shape = shapes[0]
        field = static_field(instance_id)
        kind, params = _shape_kind_params(shape)
        fields[field] = build_static_field(
            instance_id,
            kind,
            params,
            shape.pos,
            shape.quat_wxyz,
            classes=c,
            prim_path=static_prim_path(instance_id),
        )
        return fields
    for shape in shapes:
        field = static_shape_field(instance_id, shape.name)
        if field in fields:
            raise ValueError(
                f"static payload for instance {instance_id!r}: duplicate scene "
                f"field {field!r} -- shape names must be unique"
            )
        kind, params = _shape_kind_params(shape)
        fields[field] = build_static_field(
            instance_id,
            kind,
            params,
            shape.pos,
            shape.quat_wxyz,
            classes=c,
            prim_path=static_shape_prim_path(instance_id, shape.name),
        )
    return fields


__all__ = [
    "build_static_field",
    "build_static_payload_fields",
    "build_static_scene_fields",
]
