"""Neutral built-in shape generators: the closed payload registry.

Standard library plus the sibling neutral modules (``.scene_ir`` and
``.primitives``) only -- no engine import, no content import.  A
generator maps an opaque parameter mapping to a
:class:`~unigym.scenes.primitives.ShapePayload`: the neutral geometry
carrier the engine side turns into its own elements.  Nothing here names
an engine element; the payload is boxes with OUTER envelopes, logical
material names and a neutral layer hint.

The registry is closed and validated at the seam:

* :func:`register_generator` rejects duplicate or non-snake_case names;
* :func:`build_payload` validates every parameter against the registered
  schema and fails closed with a teaching message (unknown keys name the
  legal set, missing keys are named, types and finiteness are enforced).

The built-ins register at import: ``room_shell`` (six slabs: floor,
ceiling and four walls), ``box`` (one slab from an OUTER envelope) and
the single-shape ``sphere`` / ``cylinder`` / ``capsule`` / ``plane``
generators (static scene geometry, same placement/material/contact/layer
keys as ``box``).  All are primitives, so the reserved ``verbatim`` key
(a mounting policy that belongs to non-primitive generators) is rejected
here.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType

from .primitives import (
    SHAPE_LAYERS,
    BoxShape,
    CapsuleShape,
    ContactSpec,
    CylinderShape,
    PlaneShape,
    ShapePayload,
    SphereShape,
)
from .scene_ir import require_neutral_name

_ROOM_SHELL_SCHEMA = MappingProxyType(
    {
        "name": {"type": "name", "required": True},
        "width": {"type": "positive", "required": True},
        "depth": {"type": "positive", "required": True},
        "height": {"type": "positive", "required": True},
        "thickness": {"type": "positive", "default": 0.1},
        "wall_material": {"type": "name", "required": True},
        "floor_material": {"type": "name", "required": True},
        "ceiling_material": {"type": "name", "default": None, "nullable": True},
    }
)

_BOX_SCHEMA = MappingProxyType(
    {
        "name": {"type": "name", "required": True},
        "size": {"type": "envelope", "required": True},
        "material": {"type": "name", "required": True},
        "pos": {"type": "vector3", "default": (0.0, 0.0, 0.0)},
        "quat": {"type": "rotation", "default": (1.0, 0.0, 0.0, 0.0)},
        "contact": {"type": "contact", "default": None},
        "layer": {"type": "layer", "default": "visual"},
    }
)


def _shape_schema(*dim_rules: tuple[str, Mapping]) -> MappingProxyType:
    """Build one single-shape schema mirroring ``box``.

    Args:
        *dim_rules: ``(param name, rule)`` pairs for the shape's own
            dimensions (they follow ``name`` and precede the common
            placement/material/contact/layer keys).

    Returns:
        The read-only schema mapping.
    """
    schema = {"name": {"type": "name", "required": True}}
    for key, rule in dim_rules:
        schema[key] = rule
    schema.update({
        "material": {"type": "name", "required": True},
        "pos": {"type": "vector3", "default": (0.0, 0.0, 0.0)},
        "quat": {"type": "rotation", "default": (1.0, 0.0, 0.0, 0.0)},
        "contact": {"type": "contact", "default": None},
        "layer": {"type": "layer", "default": "visual"},
    })
    return MappingProxyType(schema)


_SPHERE_SCHEMA = _shape_schema(
    ("radius", {"type": "positive", "required": True}))
_CYLINDER_SCHEMA = _shape_schema(
    ("radius", {"type": "positive", "required": True}),
    ("half_height", {"type": "positive", "required": True}),
)
_CAPSULE_SCHEMA = _shape_schema(
    ("radius", {"type": "positive", "required": True}),
    ("half_height", {"type": "positive", "required": True}),
)
_PLANE_SCHEMA = _shape_schema(
    ("size", {"type": "extent2", "required": True}))

_REGISTRY: dict[str, tuple[Mapping, Callable]] = {}


def _finite_vector(where: str, value, length: int) -> tuple:
    """Validate a finite float vector of exactly *length* components."""
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in value
        )
    ):
        raise ValueError(
            f"{where} must be a {length}-vector of finite numbers, got {value!r}"
        )
    return tuple(float(v) for v in value)


def _coerce(gen_name: str, key: str, rule: Mapping, value):
    """Validate one parameter against its schema rule (fail closed)."""
    where = f"generator {gen_name!r} param {key!r}"
    kind = rule["type"]
    if kind == "name":
        if value is None and rule.get("nullable"):
            return None
        return require_neutral_name(where, value)
    if kind == "positive":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{where} must be a finite number > 0, got {value!r}")
        return float(value)
    if kind == "vector3":
        return _finite_vector(where, value, 3)
    if kind == "envelope":
        size = _finite_vector(where, value, 3)
        if any(side <= 0 for side in size):
            raise ValueError(f"{where} components must be > 0, got {value!r}")
        return size
    if kind == "extent2":
        size = _finite_vector(where, value, 2)
        if any(side <= 0 for side in size):
            raise ValueError(f"{where} components must be > 0, got {value!r}")
        return size
    if kind == "rotation":
        quat = _finite_vector(where, value, 4)
        if sum(v * v for v in quat) < 1e-24:
            raise ValueError(f"{where} must be a non-zero rotation, got {value!r}")
        return quat
    if kind == "contact":
        if value is not None and not isinstance(value, ContactSpec):
            raise ValueError(
                f"{where} must be a ContactSpec or None, got {type(value).__name__}"
            )
        return value
    if kind == "layer":
        if value not in SHAPE_LAYERS:
            raise ValueError(f"{where} must be one of {SHAPE_LAYERS}, got {value!r}")
        return value
    raise ValueError(f"generator {gen_name!r}: schema type {kind!r} is not defined")


def _apply_schema(gen_name: str, schema: Mapping, params: Mapping) -> dict:
    """Validate + normalize *params* against *schema*; fill defaults."""
    legal = sorted(schema)
    unknown = sorted(set(params) - set(legal))
    if unknown:
        raise ValueError(
            f"generator {gen_name!r}: unknown params {unknown} (legal: {legal})"
        )
    missing = sorted(
        key for key, rule in schema.items() if rule.get("required") and key not in params
    )
    if missing:
        raise ValueError(f"generator {gen_name!r}: missing required params {missing}")
    return {
        key: (
            _coerce(gen_name, key, rule, params[key])
            if key in params
            else rule.get("default")
        )
        for key, rule in schema.items()
    }


def register_generator(name: str, schema: Mapping, builder: Callable) -> None:
    """Register one built-in generator in the closed registry.

    Args:
        name: The generator name (lowercase snake_case, unique).
        schema: The parameter schema (parameter name -> rule mapping).
        builder: The pure ``normalized params -> ShapePayload`` function.

    Raises:
        ValueError: The name is not snake_case, is already registered,
            the schema is not a mapping, or the builder is not callable.
    """
    require_neutral_name("generator", name)
    if name in _REGISTRY:
        raise ValueError(
            f"generator {name!r} is already registered "
            f"(registered: {registered_generators()})"
        )
    if not isinstance(schema, Mapping):
        raise ValueError(
            f"generator {name!r}: schema must be a mapping, got {type(schema).__name__}"
        )
    if not callable(builder):
        raise ValueError(f"generator {name!r}: builder must be callable")
    _REGISTRY[name] = (MappingProxyType(dict(schema)), builder)


def registered_generators() -> list[str]:
    """Return the registered generator names in registration order."""
    return list(_REGISTRY)


def generator_schema(name: str) -> Mapping:
    """Return the declared parameter schema of one registered generator.

    The engine side reads it to decide which declaration fields a given
    payload consumes (the scene seam folds an instance pose into the
    ``pos``/``quat`` params the schema declares).

    Args:
        name: A registered generator name.

    Returns:
        The generator's parameter schema (read-only mapping).

    Raises:
        ValueError: The generator is unknown.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"generator {name!r} is not registered "
            f"(registered: {registered_generators()})"
        )
    return _REGISTRY[name][0]


def build_payload(name: str, params: Mapping) -> ShapePayload:
    """Validate *params* against the named schema and build its payload.

    Args:
        name: A registered generator name.
        params: The instance parameters (validated, never interpreted by
            the scene IR).

    Returns:
        The generator's :class:`~unigym.scenes.primitives.ShapePayload`.

    Raises:
        ValueError: The generator is unknown, the params are not a
            mapping, carry the reserved ``verbatim`` key, miss a required
            key, or fail a type/finiteness rule.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"generator {name!r} is not registered "
            f"(registered: {registered_generators()})"
        )
    if not isinstance(params, Mapping):
        raise ValueError(
            f"generator {name!r}: params must be a mapping, got "
            f"{type(params).__name__}"
        )
    if "verbatim" in params:
        raise ValueError(
            f"generator {name!r}: the reserved key 'verbatim' belongs to "
            f"non-primitive generators, not the built-in payload generators"
        )
    schema, builder = _REGISTRY[name]
    payload = builder(_apply_schema(name, schema, params))
    if not isinstance(payload, ShapePayload):
        raise ValueError(
            f"generator {name!r}: builder returned {type(payload).__name__}, "
            f"expected ShapePayload"
        )
    return payload


def _build_room_shell(params: Mapping) -> ShapePayload:
    """Build the six-slab room envelope from interior dimensions.

    Args:
        params: The normalized ``room_shell`` parameters.

    Returns:
        Six box shapes in the fixed floor/ceiling/wall order; every slab
        sits fully outside the interior and carries half of its OUTER
        envelope per axis (the payload never pre-halves).

    The layer is ``collision`` because the pre-refactor composer set no
    group on these slabs, so the engine default (the collision class) is
    the legacy truth this refactor must preserve.  Whether a room belongs
    in the visual class is a rendering decision, not a refactor: moving it
    would change every room-shell world outside the frozen baselines'
    coverage.
    """
    name = params["name"]
    thickness = params["thickness"]
    w_half = params["width"] / 2.0
    d_half = params["depth"] / 2.0
    h_half = params["height"] / 2.0
    big_w = w_half + thickness
    big_d = d_half + thickness
    big_h = h_half + thickness
    wall = params["wall_material"]
    ceiling = params["ceiling_material"] or wall
    rows = (
        ("floor", (2.0 * big_w, 2.0 * big_d, thickness),
         (0.0, 0.0, -thickness / 2.0), params["floor_material"]),
        ("ceiling", (2.0 * big_w, 2.0 * big_d, thickness),
         (0.0, 0.0, params["height"] + thickness / 2.0), ceiling),
        ("wall_xp", (thickness, 2.0 * big_d, 2.0 * big_h),
         (w_half + thickness / 2.0, 0.0, h_half), wall),
        ("wall_xm", (thickness, 2.0 * big_d, 2.0 * big_h),
         (-w_half - thickness / 2.0, 0.0, h_half), wall),
        ("wall_yp", (2.0 * big_w, thickness, 2.0 * big_h),
         (0.0, d_half + thickness / 2.0, h_half), wall),
        ("wall_ym", (2.0 * big_w, thickness, 2.0 * big_h),
         (0.0, -d_half - thickness / 2.0, h_half), wall),
    )
    shapes = tuple(
        BoxShape(name=f"{name}_{slab}", size=size, material=material, pos=pos,
                 layer="collision")
        for slab, size, pos, material in rows
    )
    return ShapePayload(shapes)


def _build_box(params: Mapping) -> ShapePayload:
    """Build one box slab from an OUTER envelope.

    Args:
        params: The normalized ``box`` parameters.

    Returns:
        A one-shape payload; the envelope is carried unchanged (the
        engine side halves it, never this generator).
    """
    shape = BoxShape(
        name=f"{params['name']}_geom",
        size=params["size"],
        material=params["material"],
        pos=params["pos"],
        quat_wxyz=params["quat"],
        contact=params["contact"],
        layer=params["layer"],
    )
    return ShapePayload((shape,))


def _shape_kwargs(params: Mapping) -> dict:
    """Common constructor kwargs for one single-shape payload entry."""
    return {
        "name": f"{params['name']}_geom",
        "material": params["material"],
        "pos": params["pos"],
        "quat_wxyz": params["quat"],
        "contact": params["contact"],
        "layer": params["layer"],
    }


def _build_sphere(params: Mapping) -> ShapePayload:
    """Build one sphere shape from its normalized parameters."""
    return ShapePayload((SphereShape(radius=params["radius"],
                                      **_shape_kwargs(params)),))


def _build_cylinder(params: Mapping) -> ShapePayload:
    """Build one cylinder shape from its normalized parameters."""
    return ShapePayload((CylinderShape(radius=params["radius"],
                                       half_height=params["half_height"],
                                       **_shape_kwargs(params)),))


def _build_capsule(params: Mapping) -> ShapePayload:
    """Build one capsule shape from its normalized parameters."""
    return ShapePayload((CapsuleShape(radius=params["radius"],
                                      half_height=params["half_height"],
                                      **_shape_kwargs(params)),))


def _build_plane(params: Mapping) -> ShapePayload:
    """Build one plane shape from its normalized parameters."""
    return ShapePayload((PlaneShape(size=params["size"],
                                    **_shape_kwargs(params)),))


register_generator("room_shell", _ROOM_SHELL_SCHEMA, _build_room_shell)
register_generator("box", _BOX_SCHEMA, _build_box)
register_generator("sphere", _SPHERE_SCHEMA, _build_sphere)
register_generator("cylinder", _CYLINDER_SCHEMA, _build_cylinder)
register_generator("capsule", _CAPSULE_SCHEMA, _build_capsule)
register_generator("plane", _PLANE_SCHEMA, _build_plane)

__all__ = [
    "build_payload",
    "generator_schema",
    "register_generator",
    "registered_generators",
]
