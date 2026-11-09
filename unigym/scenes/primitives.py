"""Neutral payload vocabulary for the built-in geometry generators.

Standard library + :mod:`.scene_ir` only: no engine import, no content
import.  A generator returns a :class:`ShapePayload` -- an ordered tuple
of neutral shape entries (:class:`BoxShape` / :class:`SphereShape` /
:class:`CylinderShape` / :class:`CapsuleShape` / :class:`PlaneShape`) in
the instance frame -- so the scene IR keeps ONE geometry carrier
(``Instance.generator`` + ``params``) and the built-in generators speak
this small payload instead of engine elements.

:class:`ContactSpec` mirrors the asset-IR contact-hint vocabulary
(``friction`` / ``condim`` / ``priority`` / ``solver_ref`` /
``solver_impedance`` / ``margin`` / ``gap``) as NEUTRAL solver intent;
the fields are redeclared here so this module imports nothing from the
asset-IR package.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from .scene_ir import require_neutral_name

#: legal contact condition dimensions (engine-independent physics facts).
CONDIMS = (1, 3, 4, 6)

#: the neutral shape-layer vocabulary: ``visual`` (shown by default) vs
#: ``collision`` (hidden by the visual camera class; the engine maps the
#: hint onto its own group numbering).
SHAPE_LAYERS = ("visual", "collision")


def _require_vec(
    owner: str,
    name: str,
    value: Sequence[object],
    lengths: int | tuple[int, ...],
) -> tuple[float, ...]:
    """Validate a finite float vector whose length is one of *lengths*."""
    allowed = (lengths,) if isinstance(lengths, int) else lengths
    if not isinstance(value, (list, tuple)):
        wanted = " or ".join(str(n) for n in allowed)
        raise ValueError(
            f"{owner}: {name} must be a {wanted}-vector of finite numbers, " f"got {value!r}"
        )
    if len(value) not in allowed or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in value
    ):
        want = " or ".join(str(n) for n in allowed)
        raise ValueError(
            f"{owner}: {name} must be a {want}-vector of finite numbers, " f"got {value!r}"
        )
    return tuple(float(v) for v in value if isinstance(v, (int, float)))


def _require_positive(owner: str, name: str, value: object) -> float:
    """Validate one finite strictly-positive scalar."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{owner}: {name} must be a finite number > 0, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class ContactSpec:
    """Per-shape contact-solve intent (neutral field names, engine-mapped).

    Every field is optional; a silent field keeps the engine default.
    """

    friction: tuple[float, float, float] | None = None
    condim: int | None = None
    priority: int | None = None
    solver_ref: tuple[float, float] | None = None
    solver_impedance: tuple[float, ...] | None = None
    margin: float | None = None
    gap: float | None = None

    def __post_init__(self):
        """Validate every present hint against its closed rule."""
        owner = "contact spec"
        if self.friction is not None:
            object.__setattr__(self, "friction", _require_vec(owner, "friction", self.friction, 3))
        if self.condim is not None:
            if self.condim not in CONDIMS:
                raise ValueError(f"{owner}: condim {self.condim!r} must be one of {CONDIMS}")
            object.__setattr__(self, "condim", int(self.condim))
        if self.priority is not None:
            if isinstance(self.priority, bool) or not isinstance(self.priority, int):
                raise ValueError(f"{owner}: priority must be an int, got {self.priority!r}")
        if self.solver_ref is not None:
            object.__setattr__(
                self,
                "solver_ref",
                _require_vec(owner, "solver_ref", self.solver_ref, 2),
            )
        if self.solver_impedance is not None:
            object.__setattr__(
                self,
                "solver_impedance",
                _require_vec(owner, "solver_impedance", self.solver_impedance, (3, 5)),
            )
        for field_name in ("margin", "gap"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _require_positive(owner, field_name, value))


class _ShapeLike(Protocol):
    """The common fields every neutral shape declares (validated once)."""

    @property
    def name(self) -> str: ...

    @property
    def material(self) -> str: ...

    @property
    def pos(self) -> tuple[float, float, float]: ...

    @property
    def quat_wxyz(self) -> tuple[float, float, float, float]: ...

    @property
    def contact(self) -> ContactSpec | None: ...

    @property
    def layer(self) -> str: ...


def _validate_shape_fields(shape: _ShapeLike, kind: str) -> str:
    """Validate the common fields shared by every neutral shape.

    Args:
        shape: The frozen shape instance being validated.
        kind: The neutral shape kind (``sphere`` / ``cylinder`` / ...).

    Returns:
        The error-owner string (``"<kind> shape '<name>'"``) for the
        shape-specific dimension checks.
    """
    name = require_neutral_name(f"{kind} shape", shape.name)
    object.__setattr__(shape, "name", name)
    owner = f"{kind} shape {name!r}"
    object.__setattr__(shape, "material", require_neutral_name(f"{kind} material", shape.material))
    object.__setattr__(shape, "pos", _require_vec(owner, "pos", shape.pos, 3))
    quat = _require_vec(owner, "quat_wxyz", shape.quat_wxyz, 4)
    if sum(v * v for v in quat) < 1e-24:
        raise ValueError(f"{owner}: quat_wxyz must be non-zero, got {shape.quat_wxyz!r}")
    object.__setattr__(shape, "quat_wxyz", quat)
    if shape.contact is not None and not isinstance(shape.contact, ContactSpec):
        raise ValueError(
            f"{owner}: contact must be a ContactSpec, got " f"{type(shape.contact).__name__}"
        )
    if shape.layer not in SHAPE_LAYERS:
        raise ValueError(f"{owner}: layer {shape.layer!r} must be one of {SHAPE_LAYERS}")
    return owner


@dataclass(frozen=True)
class SphereShape:
    """One sphere in the instance frame.

    ``radius`` is the sphere radius in metres; the remaining fields are
    the neutral shape convention shared with :class:`BoxShape`.
    """

    name: str
    radius: float
    material: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    contact: ContactSpec | None = None
    layer: str = "visual"

    def __post_init__(self):
        """Validate the radius and the shared shape fields."""
        owner = _validate_shape_fields(self, "sphere")
        object.__setattr__(self, "radius", _require_positive(owner, "radius", self.radius))


@dataclass(frozen=True)
class CylinderShape:
    """One cylinder in the instance frame, length along local z."""

    name: str
    radius: float
    half_height: float
    material: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    contact: ContactSpec | None = None
    layer: str = "visual"

    def __post_init__(self):
        """Validate the radius / half-height and the shared shape fields."""
        owner = _validate_shape_fields(self, "cylinder")
        object.__setattr__(self, "radius", _require_positive(owner, "radius", self.radius))
        object.__setattr__(
            self, "half_height", _require_positive(owner, "half_height", self.half_height)
        )


@dataclass(frozen=True)
class CapsuleShape:
    """One capsule in the instance frame, length along local z."""

    name: str
    radius: float
    half_height: float
    material: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    contact: ContactSpec | None = None
    layer: str = "visual"

    def __post_init__(self):
        """Validate the radius / half-height and the shared shape fields."""
        owner = _validate_shape_fields(self, "capsule")
        object.__setattr__(self, "radius", _require_positive(owner, "radius", self.radius))
        object.__setattr__(
            self, "half_height", _require_positive(owner, "half_height", self.half_height)
        )


@dataclass(frozen=True)
class PlaneShape:
    """One infinite plane in the instance frame.

    ``size`` is the ``[sx, sy]`` visual grid half extent in metres; the
    plane itself is infinite, so the size only controls the rendered grid.
    """

    name: str
    size: tuple[float, float]
    material: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    contact: ContactSpec | None = None
    layer: str = "visual"

    def __post_init__(self):
        """Validate the half extents and the shared shape fields."""
        owner = _validate_shape_fields(self, "plane")
        size = _require_vec(owner, "size", self.size, 2)
        if any(side <= 0 for side in size):
            raise ValueError(f"{owner}: size components must be > 0, got {self.size!r}")
        object.__setattr__(self, "size", size)


@dataclass(frozen=True)
class BoxShape:
    """One box in the instance frame (outer envelope + logical material).

    ``name`` is the box's own neutral identity; ``size`` is the OUTER
    ``[sx, sy, sz]`` envelope in metres (the props box-family
    convention); ``material`` cites a logical
    :class:`~unigym.scenes.scene_ir.Material` name.  ``pos`` /
    ``quat_wxyz`` place the box relative to the instance pose.
    ``layer`` is the neutral :data:`SHAPE_LAYERS` hint -- this payload
    never numbers an engine group.
    """

    name: str
    size: tuple[float, float, float]
    material: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    contact: ContactSpec | None = None
    layer: str = "visual"

    def __post_init__(self):
        """Validate the name, envelope, material, pose, contact and layer."""
        object.__setattr__(self, "name", require_neutral_name("box shape", self.name))
        owner = f"box shape {self.name!r}"
        object.__setattr__(self, "size", _require_vec(owner, "size", self.size, 3))
        for side in self.size:
            if side <= 0:
                raise ValueError(f"{owner}: size components must be > 0, got {self.size!r}")
        object.__setattr__(self, "material", require_neutral_name("box material", self.material))
        object.__setattr__(self, "pos", _require_vec(owner, "pos", self.pos, 3))
        quat = _require_vec(owner, "quat_wxyz", self.quat_wxyz, 4)
        if sum(v * v for v in quat) < 1e-24:
            raise ValueError(f"{owner}: quat_wxyz must be non-zero, got {self.quat_wxyz!r}")
        object.__setattr__(self, "quat_wxyz", quat)
        if self.contact is not None and not isinstance(self.contact, ContactSpec):
            raise ValueError(
                f"{owner}: contact must be a ContactSpec, got " f"{type(self.contact).__name__}"
            )
        if self.layer not in SHAPE_LAYERS:
            raise ValueError(f"{owner}: layer {self.layer!r} must be one of {SHAPE_LAYERS}")


#: the closed neutral shape vocabulary carried by a :class:`ShapePayload`.
SHAPE_CLASSES = (BoxShape, SphereShape, CylinderShape, CapsuleShape, PlaneShape)

Shape: TypeAlias = BoxShape | SphereShape | CylinderShape | CapsuleShape | PlaneShape


@dataclass(frozen=True)
class ShapePayload:
    """The built-in generators' payload: an ordered tuple of neutral shapes."""

    shapes: tuple[Shape, ...] = ()

    def __post_init__(self):
        """Validate the shape tuple."""
        if not isinstance(self.shapes, tuple):
            raise ValueError(
                f"shape payload: shapes must be a tuple of "
                f"{[cls.__name__ for cls in SHAPE_CLASSES]}, got "
                f"{type(self.shapes).__name__}"
            )
        for shape in self.shapes:
            if not isinstance(shape, SHAPE_CLASSES):
                raise ValueError(
                    f"shape payload: entries must be one of "
                    f"{[cls.__name__ for cls in SHAPE_CLASSES]}, got "
                    f"{type(shape).__name__}"
                )


__all__ = [
    "CONDIMS",
    "SHAPE_CLASSES",
    "SHAPE_LAYERS",
    "BoxShape",
    "CapsuleShape",
    "ContactSpec",
    "CylinderShape",
    "PlaneShape",
    "ShapePayload",
    "SphereShape",
]
