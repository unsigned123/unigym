"""Programmatic primitive declarations for :class:`unigym.World` (S2a/S2c).

The primitive vocabulary and its validation live here so ``unigym/world.py``
stays the world-composition module.  A primitive is first a declaration:
``World.add`` either turns it into a static scene-IR instance (the default)
or, with ``dynamic=True``, into a content/props free-body declaration.

The content/props declaration vocabulary is the existing ``objects.types``
schema (sphere / cylinder / capsule / box).  Prop origin conventions differ:
sphere / cylinder / capsule are centre-origin while the box family is
bottom-face-origin, so a dynamic box shifts its placement by ``sz / 2`` and
keeps ``pos`` meaning the geometric centre, matching the static primitive.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypedDict, TypeGuard

from unigym._typing import JsonValue
from unigym.scenes.scene_ir import (
    Instance,
    Pose,
    require_neutral_name,
)

if TYPE_CHECKING:
    from unigym.world import World

__all__ = [
    "Box", "Sphere", "Cylinder", "Capsule", "Plane",
    "_Primitive", "_SHAPE_TYPES", "_spawn_dict", "_validate_dynamic_entry",
    "_vec2", "_vec3",
]


class _DynamicDecl(TypedDict, total=False):
    """One dynamic ``objects.types`` declaration (the props schema shape)."""

    name: str
    type: str
    size: list[float]
    aabb: dict[str, list[float]]
    placement: dict[str, JsonValue]
    spawn: dict[str, JsonValue]
    radius: float
    half_height: float
    material: str


def _is_dynamic_decl(value: Mapping[str, JsonValue]) -> TypeGuard[_DynamicDecl]:
    """Narrow a built declaration to the props schema (name/type required)."""
    name = value.get("name")
    return isinstance(name, str) and isinstance(value.get("type"), str)


def _vec3(value: Sequence[object], what: str) -> tuple[float, float, float]:
    """Validate a finite 3-vector and return it as floats."""
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in value)):
        raise ValueError(f"{what} must be [x, y, z] finite numbers, "
                         f"got {value!r}")
    nums = tuple(float(v) for v in value if isinstance(v, (int, float)))
    return (nums[0], nums[1], nums[2])


def _quat(value: Sequence[object]) -> tuple[float, float, float, float]:
    """Validate a non-zero ``(w, x, y, z)`` quaternion."""
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in value)):
        raise ValueError(f"quat must be [w, x, y, z] finite numbers, "
                         f"got {value!r}")
    nums = tuple(float(v) for v in value if isinstance(v, (int, float)))
    quat = (nums[0], nums[1], nums[2], nums[3])
    if sum(v * v for v in quat) < 1e-24:
        raise ValueError(f"quat must be non-zero, got {value!r}")
    return quat


def _yaw_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    """Convert a yaw (degrees, world +z) to a wxyz quaternion."""
    half = math.radians(float(yaw_deg)) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _positive(value: object, what: str) -> float:
    """Validate a finite strictly-positive scalar and return it as float."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{what} must be a finite number > 0, got {value!r}")
    return float(value)


def _spawn_dict(value: Mapping[str, JsonValue] | None) -> dict[str, JsonValue]:
    """Validate and copy an optional dynamic-object spawn mapping."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"spawn must be a mapping, got {value!r}")
    spawn: dict[str, JsonValue] = {}
    for key, val in value.items():
        spawn[str(key)] = val
    return spawn


def _vec2(value: Sequence[object], what: str) -> tuple[float, float]:
    """Validate a finite 2-vector of positive numbers and return floats."""
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in value)):
        raise ValueError(f"{what} must be [x, y] finite numbers, got {value!r}")
    nums = tuple(float(v) for v in value if isinstance(v, (int, float)))
    vec = (nums[0], nums[1])
    if any(v <= 0 for v in vec):
        raise ValueError(f"{what} components must be > 0, got {value!r}")
    return vec


def _yaw_degrees(quat: Sequence[float], what: str) -> float:
    """Extract a pure-yaw angle (degrees) from a ``(w, x, y, z)`` quat.

    Dynamic props are placed by the existing ``PropPlacement`` rule, which
    supports yaw only.  A roll/pitch rotation has no declaration slot, so
    it fails fast instead of being silently flattened.
    """
    norm = math.sqrt(sum(v * v for v in quat))
    w, x, y, z = (v / norm for v in quat)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    ref = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    close = all(abs(a - b) <= 1e-6 for a, b in zip((w, x, y, z), ref))
    flipped = all(abs(a + b) <= 1e-6 for a, b in zip((w, x, y, z), ref))
    if not (close or flipped):
        raise ValueError(
            f"{what}: a dynamic object supports YAW-only rotation (the "
            f"PropPlacement rule); got quat {quat!r} with roll/pitch, "
            f"use rotate(yaw=<deg>) or keep the rotation about world +z")
    return math.degrees(yaw)


class _Primitive:
    """Shared placement/rotation face of the programmatic primitives."""

    kind = "primitive"
    #: subclass shape fields (type-only declarations; each subclass sets its own)
    _envelope: tuple[float, float, float]
    _plane_size: tuple[float, float]
    radius: float
    half_height: float

    def __init__(
        self,
        name: str | None,
        pos: Sequence[object],
        quat: Sequence[object],
        *,
        dynamic: bool = False,
        spawn: Mapping[str, JsonValue] | None = None,
    ):
        self._name = name
        self._pos: tuple[float, float, float] = _vec3(pos, f"{self.kind} pos")
        self._quat: tuple[float, float, float, float] = _quat(quat)
        self._world: World | None = None
        self.material: str | None = None
        self.dynamic = bool(dynamic)
        """Declared as a dynamic free body (content/props path)."""
        self.spawn: dict[str, JsonValue] = _spawn_dict(spawn)
        """Spawn overrides for a dynamic object (mass / density / friction /
        rgba / restitution)."""

    # -- identity ----------------------------------------------------- #
    @property
    def name(self) -> str | None:
        """The instance id (``None`` until added to a :class:`World`)."""
        return self._name

    @property
    def pos(self) -> tuple[float, float, float]:
        """Current local translation."""
        return self._pos

    @property
    def quat(self) -> tuple[float, float, float, float]:
        """Current local ``(w, x, y, z)`` rotation."""
        return self._quat

    # -- mutation (pre-build only) ------------------------------------ #
    def _check_mutable(self) -> None:
        if self._world is not None and self._world.built:
            raise RuntimeError(
                f"{self.kind} {self._name!r}: the world is already built; "
                f"placement/rotation mutations are pre-build only (create a "
                f"new World or rebuild with a fresh Run)")

    def at(self, pos: Sequence[object]) -> _Primitive:
        """Move this primitive to *pos* (world/instance frame metres)."""
        self._check_mutable()
        self._pos = _vec3(pos, f"{self.kind} pos")
        return self

    def rotate(
        self, quat: Sequence[object] | None = None, *,
        yaw: float | None = None,
    ) -> _Primitive:
        """Rotate in place (``quat`` wxyz, or a ``yaw`` in degrees)."""
        self._check_mutable()
        if (quat is None) == (yaw is None):
            raise ValueError(
                "rotate() takes exactly one of quat=[w,x,y,z] or yaw=<deg>")
        if quat is not None:
            self._quat = _quat(quat)
        else:
            assert yaw is not None
            self._quat = _yaw_quat(yaw)
        return self

    def _instance(self, material: str) -> Instance:
        """Build the scene-IR instance for this primitive."""
        assert self._name is not None
        params: dict[str, JsonValue] = {
            "name": self._name,
            **self._shape_params(),
            "material": self.material or material,
            "layer": "visual",
        }
        return Instance(
            id=require_neutral_name(self.kind, self._name),
            generator=self.kind,
            params=params,
            pose=Pose(frame="world", xyz=self._pos, quat_wxyz=self._quat),
            source="world:add",
        )

    def _shape_params(self) -> dict[str, JsonValue]:
        """The generator dimension params for this primitive."""
        raise NotImplementedError

    def _dynamic_declaration(self) -> dict[str, JsonValue]:
        """The content/props declaration for this dynamic free body."""
        assert self._name is not None
        if self.kind == "plane":
            raise ValueError(
                f"World.add: plane {self._name!r} cannot be dynamic -- a "
                f"plane is infinite static scene geometry (use Box/Sphere/"
                f"Cylinder/Capsule for free bodies)")
        x, y, z = self._pos
        entry: dict[str, JsonValue] = {"name": self._name, "type": self.kind}
        if self.kind == "box":
            sx, sy, sz = self._envelope
            entry["size"] = [sx, sy, sz]
            entry["aabb"] = {
                "center": [0.0, 0.0, sz / 2.0],
                "half_extents": [sx / 2.0, sy / 2.0, sz / 2.0],
            }
            z = z - sz / 2.0
        elif self.kind == "sphere":
            entry["radius"] = self.radius
        else:  # cylinder / capsule
            entry["radius"] = self.radius
            entry["half_height"] = self.half_height
        yaw = _yaw_degrees(self._quat, f"{self.kind} {self._name!r}")
        entry["placement"] = {
            "mode": "fixed",
            "x_range": [x, x],
            "y_range": [y, y],
            "z_range": [z, z],
            "yaw_range": [yaw, yaw],
        }
        if self.spawn:
            entry["spawn"] = dict(self.spawn)
        if self.material is not None:
            entry["material"] = self.material
        return entry


class Box(_Primitive):
    """One box primitive (OUTER envelope ``[sx, sy, sz]`` in metres)."""

    kind = "box"

    def __init__(
        self, size: Sequence[object], *, name: str | None = None,
        pos: Sequence[object] = (0.0, 0.0, 0.0),
        quat: Sequence[object] = (1.0, 0.0, 0.0, 0.0),
        material: str | None = None, dynamic: bool = False,
        spawn: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(name, pos, quat, dynamic=dynamic, spawn=spawn)
        self._envelope = _vec3(size, "box size")
        if any(side <= 0 for side in self._envelope):
            raise ValueError(f"box size components must be > 0, "
                             f"got {size!r}")
        self.material = material

    @property
    def size(self) -> tuple[float, float, float]:
        """The OUTER ``[sx, sy, sz]`` envelope."""
        return self._envelope

    def _shape_params(self) -> dict[str, JsonValue]:
        return {"size": list(self._envelope)}


class Sphere(_Primitive):
    """One static sphere primitive (radius in metres)."""

    kind = "sphere"

    def __init__(
        self, radius: float, *, name: str | None = None,
        pos: Sequence[object] = (0.0, 0.0, 0.0),
        quat: Sequence[object] = (1.0, 0.0, 0.0, 0.0),
        material: str | None = None, dynamic: bool = False,
        spawn: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(name, pos, quat, dynamic=dynamic, spawn=spawn)
        self.radius = _positive(radius, "sphere radius")
        self.material = material

    def _shape_params(self) -> dict[str, JsonValue]:
        return {"radius": self.radius}


class Cylinder(_Primitive):
    """One static cylinder primitive, length along local z."""

    kind = "cylinder"

    def __init__(
        self, radius: float, half_height: float, *,
        name: str | None = None,
        pos: Sequence[object] = (0.0, 0.0, 0.0),
        quat: Sequence[object] = (1.0, 0.0, 0.0, 0.0),
        material: str | None = None, dynamic: bool = False,
        spawn: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(name, pos, quat, dynamic=dynamic, spawn=spawn)
        self.radius = _positive(radius, "cylinder radius")
        self.half_height = _positive(half_height, "cylinder half_height")
        self.material = material

    def _shape_params(self) -> dict[str, JsonValue]:
        return {"radius": self.radius, "half_height": self.half_height}


class Capsule(_Primitive):
    """One static capsule primitive, length along local z."""

    kind = "capsule"

    def __init__(
        self, radius: float, half_height: float, *,
        name: str | None = None,
        pos: Sequence[object] = (0.0, 0.0, 0.0),
        quat: Sequence[object] = (1.0, 0.0, 0.0, 0.0),
        material: str | None = None, dynamic: bool = False,
        spawn: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(name, pos, quat, dynamic=dynamic, spawn=spawn)
        self.radius = _positive(radius, "capsule radius")
        self.half_height = _positive(half_height, "capsule half_height")
        self.material = material

    def _shape_params(self) -> dict[str, JsonValue]:
        return {"radius": self.radius, "half_height": self.half_height}


class Plane(_Primitive):
    """One static infinite plane primitive (``[sx, sy]`` grid half extents)."""

    kind = "plane"

    def __init__(
        self, size: Sequence[object], *, name: str | None = None,
        pos: Sequence[object] = (0.0, 0.0, 0.0),
        quat: Sequence[object] = (1.0, 0.0, 0.0, 0.0),
        material: str | None = None, dynamic: bool = False,
        spawn: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(name, pos, quat, dynamic=dynamic, spawn=spawn)
        self._plane_size = _vec2(size, "plane size")
        self.material = material

    @property
    def size(self) -> tuple[float, float]:
        """The visual grid ``[sx, sy]`` half extents."""
        return self._plane_size

    def _shape_params(self) -> dict[str, JsonValue]:
        return {"size": list(self._plane_size)}


_SHAPE_TYPES = (Box, Sphere, Cylinder, Capsule, Plane)
"""The closed primitive vocabulary ``World.add`` / ``World.add_many`` take."""


def _validate_dynamic_entry(entry: Mapping[str, JsonValue]) -> None:
    """Fail-fast validation of one dynamic ``objects.types`` declaration.

    Runs the same spec-level checks the props manager will apply, without
    constructing a handler (no temp URDF side effect).  The heavy import is
    local so ``import unigym`` stays light.
    """
    from unigym.props import _dispatch_handler
    from unigym.props.base import PropPlacement, PropSpec
    from unigym.props.primitive import resolve_mass, validate_primitive_spec

    if not _is_dynamic_decl(entry):
        raise ValueError(
            f"dynamic declaration must carry a string 'name' and 'type', "
            f"got {entry!r}"
        )
    spec = PropSpec(**entry)
    _dispatch_handler(spec)             # unknown type -> teaching error
    PropPlacement(entry.get("placement"))  # malformed placement -> early error
    if spec.type in ("cylinder", "capsule"):
        validate_primitive_spec(spec)
        return
    if spec.type == "sphere":
        spawn = entry.get("spawn")
        radius = (spec.radius if spec.radius is not None
                  else (spawn.get("radius") if spawn is not None else None))
        if radius is None:
            raise ValueError(
                f"Sphere {spec.name!r}: radius is required for a dynamic "
                f"object")
        resolve_mass(spec, default=0.1)
        return
    if spec.type == "box":
        from unigym.props.container import validate_box_spec
        validate_box_spec(spec)
        resolve_mass(spec, default=0.1)
