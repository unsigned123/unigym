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

import math
from collections.abc import Mapping, Sequence

from unigym._typing import JsonValue, is_json_object
from unigym.backends.isaaclab2.cfg_classes import IsaacLabCfgClasses, cfg_class
from unigym.scenes.primitives import (
    BoxShape,
    CapsuleShape,
    CylinderShape,
    PlaneShape,
    Shape,
    ShapePayload,
    SphereShape,
)

#: neutral primitive kind -> the cfg-class name that spawns it
_CFG_CLASS: Mapping[str, str] = {
    "box": "CuboidCfg",
    "sphere": "SphereCfg",
    "cylinder": "CylinderCfg",
    "capsule": "CapsuleCfg",
    "plane": "GroundPlaneCfg",
}

_DEFAULT_COLOR = (0.55, 0.55, 0.58)


def _resolve(classes: IsaacLabCfgClasses | None) -> IsaacLabCfgClasses:
    """Return the cfg-class table (real isaaclab when ``classes`` is None).

    The module handles are concrete; the opaque class attributes are read at
    this external class boundary because IsaacLab ships no usable stubs.
    """
    if classes is not None:
        return classes
    import isaaclab.assets as isaaclab_assets
    import isaaclab.sim as sim_utils

    table: IsaacLabCfgClasses = {
        "AssetBaseCfg": getattr(isaaclab_assets, "AssetBaseCfg"),
        "CuboidCfg": getattr(sim_utils, "CuboidCfg"),
        "SphereCfg": getattr(sim_utils, "SphereCfg"),
        "CylinderCfg": getattr(sim_utils, "CylinderCfg"),
        "CapsuleCfg": getattr(sim_utils, "CapsuleCfg"),
        "GroundPlaneCfg": getattr(sim_utils, "GroundPlaneCfg"),
        "CollisionPropertiesCfg": getattr(sim_utils, "CollisionPropertiesCfg"),
        "PreviewSurfaceCfg": getattr(sim_utils, "PreviewSurfaceCfg"),
    }
    return table


def _text(owner: str, name: str, value: JsonValue) -> str:
    """Narrow a JSON value to a string with a teaching error."""
    if not isinstance(value, str):
        raise ValueError(f"{owner}: {name} must be a string, got {value!r}")
    return value


def _float(owner: str, name: str, value: JsonValue) -> float:
    """Narrow a JSON value to a finite float with a teaching error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{owner}: {name} must be a finite number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{owner}: {name} must be a finite number, got {value!r}")
    return float(value)


def _float_vec(owner: str, name: str, value: JsonValue, length: int) -> tuple[float, ...]:
    """Narrow a JSON vector to *length* finite floats with a teaching error."""
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{owner}: {name} must be a {length}-vector of numbers, got {value!r}")
    return tuple(_float(owner, name, item) for item in value)


def _json_object(owner: str, name: str, value: JsonValue) -> dict[str, JsonValue]:
    """Narrow a JSON value to a string-keyed object with a teaching error."""
    if not is_json_object(value):
        raise ValueError(f"{owner}: {name} must be a JSON object, got {value!r}")
    return value


def _require_kind(name: str, kind: str) -> None:
    """Reject a primitive kind outside the closed static mapping."""
    if kind not in _CFG_CLASS:
        raise ValueError(
            f"static primitive {name!r}: kind {kind!r} has "
            f"no isaaclab static mapping (known: {sorted(_CFG_CLASS)})"
        )


def _spawn(
    kind: str,
    params: Mapping[str, JsonValue],
    c: IsaacLabCfgClasses,
    color: tuple[float, float, float],
) -> object:
    """Build the collision-only spawner cfg for one primitive kind."""
    owner = f"static primitive kind {kind!r}"
    collision = cfg_class(c, "CollisionPropertiesCfg")(collision_enabled=True)
    if kind == "plane":
        size = _float_vec(owner, "size", params["size"], 2)
        return cfg_class(c, "GroundPlaneCfg")(
            size=(2.0 * size[0], 2.0 * size[1]),
            color=color,
        )
    visual = cfg_class(c, "PreviewSurfaceCfg")(diffuse_color=color)
    common = dict(collision_props=collision, visual_material=visual)
    if kind == "box":
        return cfg_class(c, "CuboidCfg")(
            size=_float_vec(owner, "size", params["size"], 3), **common
        )
    if kind == "sphere":
        return cfg_class(c, "SphereCfg")(radius=_float(owner, "radius", params["radius"]), **common)
    if kind == "cylinder":
        return cfg_class(c, "CylinderCfg")(
            radius=_float(owner, "radius", params["radius"]),
            height=2.0 * _float(owner, "half_height", params["half_height"]),
            **common,
        )
    if kind == "capsule":
        # USD capsule ``height`` is the cylindrical section (caps are
        # added on top): the neutral half_height is that half-length.
        return cfg_class(c, "CapsuleCfg")(
            radius=_float(owner, "radius", params["radius"]),
            height=2.0 * _float(owner, "half_height", params["half_height"]),
            **common,
        )
    raise ValueError(
        f"static primitive kind {kind!r} has no isaaclab mapping " f"(known: {sorted(_CFG_CLASS)})"
    )


def build_static_field(
    name: str,
    kind: str,
    params: Mapping[str, JsonValue],
    pos: Sequence[float],
    quat: Sequence[float],
    *,
    classes: IsaacLabCfgClasses | None = None,
    prim_path: str | None = None,
) -> object:
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
    spawn = _spawn(kind, params, c, _DEFAULT_COLOR)
    asset_base = cfg_class(c, "AssetBaseCfg")
    return asset_base(
        prim_path=prim,
        spawn=spawn,
        init_state=asset_base.InitialStateCfg(
            pos=tuple(float(v) for v in pos),
            rot=tuple(float(v) for v in quat),
        ),
    )


def build_static_scene_fields(
    declarations: Sequence[Mapping[str, JsonValue]],
    *,
    classes: IsaacLabCfgClasses | None = None,
) -> dict[str, object]:
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
    fields: dict[str, object] = {}
    for decl in declarations:
        name = _text("static primitive", "name", decl["name"])
        owner = f"static primitive {name!r}"
        kind = _text(owner, "kind", decl["kind"])
        _require_kind(name, kind)
        params = _json_object(owner, "params", decl["params"])
        pos = _float_vec(owner, "pos", decl["pos"], 3)
        quat = _float_vec(owner, "quat", decl["quat"], 4)
        field = static_field(name)
        if field in fields:
            raise ValueError(f"static primitive {name!r}: duplicate scene field {field!r}")
        fields[field] = build_static_field(
            name,
            kind,
            params,
            pos,
            quat,
            classes=c,
        )
    return fields


def _shape_kind_params(shape: Shape) -> tuple[str, dict[str, JsonValue]]:
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


def build_static_payload_fields(
    payload: ShapePayload,
    *,
    instance_id: str,
    classes: IsaacLabCfgClasses | None = None,
) -> dict[str, object]:
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
    fields: dict[str, object] = {}
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
