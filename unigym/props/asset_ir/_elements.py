"""The frozen tree-element dataclasses of the neutral asset IR.

One payload body tree speaks this vocabulary: bodies carry geoms /
joints / sites / children; every constructor runs its closed-vocabulary
and numeric guards in ``__post_init__`` (frozen dataclasses -- the
guards can never be bypassed).  The per-geom/joint hint blocks are
NEUTRALLY named solver intent; the renderer is their sole interpreter.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

from ._guards import (
    _require_bool,
    _require_nonneg,
    _require_quat,
    _require_range,
    _require_rgba,
    _require_vec,
)
from ._vocab import (
    CONDIMS,
    GEOM_SIZE_LEN,
    GEOM_TYPES,
    JOINT_TYPES,
    MESH_GEOM_TYPE,
    SITE_TYPES,
    _CONSTRUCTOR_SEGMENT_RE,
    _ENGINE_ELEMENT_NAMES,
    _require_name,
)


@dataclass(frozen=True)
class MatRef:
    """Logical material reference (a name; NEVER path-resolved).

    The name gate is the reference-namespace face: most references are
    snake material names, but a material's TEXTURE reference cites the
    robocasa texture namespace (segment-shaped -- ``flat/gray``; the
    verbatim bake reproduces it), so each ``/`` segment follows
    the lowercase snake rule instead of the whole string.
    """

    name: str

    def __post_init__(self):
        """Validate the reference name."""
        segments = str(self.name).split("/")
        if (
            not self.name
            or self.name in _ENGINE_ELEMENT_NAMES
            or any(not _CONSTRUCTOR_SEGMENT_RE.fullmatch(segment) for segment in segments)
        ):
            raise ValueError(
                f"material reference {self.name!r} must be constructor-"
                f"namespace segments joined by '/' and never an engine "
                f"element name"
            )
        object.__setattr__(self, "name", self.name)


@dataclass(frozen=True)
class Joint:
    """One generated hinge/slide/ball joint (closed set; no freejoint).

    ``springdamper`` (the composite pilot) is the
    optional mass-spring parametrization ``(timeconst, damping_ratio)``
    of the official composite chains: when authored, the engine DERIVES
    the joint stiffness and damping from it at compile time (the
    authored values are mass-dependent per joint, so the payload
    carries the parametrization, never the derived numbers. ``stiffness``
    is an optional explicit passive stiffness, independent of that
    parametrization. A ball
    joint ignores ``axis`` (quaternion dofs); its ``range`` is
    ``[0, max_angle]`` in radians.
    """

    name: str
    type: str
    axis: tuple = (0.0, 0.0, 1.0)
    pos: tuple = (0.0, 0.0, 0.0)
    range: tuple | None = None
    damping: float = 0.0
    armature: float = 0.0
    frictionloss: float = 0.0
    springdamper: tuple | None = None
    stiffness: float | None = None

    def __post_init__(self):
        """Validate the closed vocabulary + numeric guards."""
        object.__setattr__(self, "name", _require_name("joint", self.name))
        if self.type not in JOINT_TYPES:
            raise ValueError(
                f"joint {self.name!r}: type {self.type!r} must be one of "
                f"{JOINT_TYPES} (engine joint spellings -- e.g. 'free' -- "
                f"are not writable through the IR)"
            )
        owner = f"joint {self.name!r}"
        object.__setattr__(self, "axis", _require_vec(owner, "axis", self.axis, 3))
        object.__setattr__(self, "pos", _require_vec(owner, "pos", self.pos, 3))
        if self.range is not None:
            object.__setattr__(self, "range", _require_range(owner, self.range))
        for f in ("damping", "armature", "frictionloss"):
            object.__setattr__(self, f, _require_nonneg(owner, f, getattr(self, f)))
        if self.stiffness is not None:
            object.__setattr__(
                self, "stiffness", _require_nonneg(owner, "stiffness", self.stiffness)
            )
        if self.springdamper is not None:
            object.__setattr__(
                self, "springdamper", _require_vec(owner, "springdamper", self.springdamper, 2)
            )


@dataclass(frozen=True)
class Site:
    """One named position marker on a generated body.

    ``size`` is the marker's radius scalar unless ``size_vector`` carries
    an authored multi-component engine size (thin cylinder fixture
    markers). The renderer assigns scalar sizes component-wise, keeping
    unspecified trailing defaults intact.
    ``type`` is the optional marker projection (a
    :data:`SITE_TYPES` name); ``None`` keeps the engine's sphere
    default.
    """

    name: str
    pos: tuple = (0.0, 0.0, 0.0)
    size: float | None = None
    rgba: tuple | None = None
    type: str | None = None
    size_vector: tuple | None = None

    def __post_init__(self):
        """Validate the name + position/size/color vectors."""
        object.__setattr__(self, "name", _require_name("site", self.name))
        owner = f"site {self.name!r}"
        object.__setattr__(self, "pos", _require_vec(owner, "pos", self.pos, 3))
        if self.size is not None:
            size = _require_nonneg(owner, "size", self.size)
            if size <= 0.0:
                raise ValueError(f"{owner}: size must be > 0, got {size!r}")
            object.__setattr__(self, "size", size)
        if self.size_vector is not None:
            vector = self.size_vector
            if (
                not isinstance(vector, (list, tuple))
                or not 1 <= len(vector) <= 3
                or any(
                    isinstance(v, bool)
                    or not isinstance(v, (int, float))
                    or not math.isfinite(v)
                    or v <= 0
                    for v in vector
                )
            ):
                raise ValueError(f"{owner}: size_vector must contain 1-3 positive finite numbers")
            if self.size is not None:
                raise ValueError(f"{owner}: size and size_vector are mutually exclusive")
            object.__setattr__(self, "size_vector", tuple(float(v) for v in vector))
        if self.rgba is not None:
            rgba = _require_vec(owner, "rgba", self.rgba, 4)
            bad_rgb = any(v < 0.0 or v > 1.0 for v in rgba[:3])
            if bad_rgb or rgba[3] < -1.0 or rgba[3] > 1.0:
                raise ValueError(
                    f"{owner}: rgba rgb components must be within [0, 1] "
                    f"and alpha within [-1, 1] (the MJCF default-alpha "
                    f"marker), got {self.rgba!r}"
                )
            object.__setattr__(self, "rgba", rgba)
        if self.type is not None and self.type not in SITE_TYPES:
            raise ValueError(f"{owner}: type {self.type!r} must be one of " f"{SITE_TYPES}")


@dataclass(frozen=True)
class Geom:
    """One generated shape (closed type set; meshes via MeshDef refs).

    ``mass`` and ``density`` are mutually exclusive mass sources (an
    explicit mass pins the scalar; a density masses the shape from its
    volume).  ``collidable=False`` is the pure-visual twin marker: the
    renderer zeroes both contact-activation bitmasks; ``group`` is the
    engine's render/filter group index (visual twins conventionally
    land in group 1, collision geoms in group 0).

    A ``type="mesh"`` geom (S2.2) is the file-referencing form: ``mesh``
    names a :class:`MeshDef` of the SAME payload (the declaration owns
    file/scale; the geom owns only the reference); ``size`` must be the
    empty tuple (meshes have no shape vector).  Shape geoms must NOT
    carry a mesh name.  ``hints`` are optional per-geom contact
    hints (a :class:`ContactHints`, S2.2) -- per-geom authored solver
    facts beside the payload-wide ``solver_hints`` block.
    """

    name: str | None
    type: str
    size: tuple
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    material: MatRef | None = None
    mass: float | None = None
    density: float | None = None
    rgba: tuple | None = None
    group: int = 0
    collidable: bool = True
    mesh: str | None = None
    hints: ContactHints | None = None

    def __post_init__(self):
        """Validate the closed type set + the per-type size table."""
        if self.type == MESH_GEOM_TYPE:
            if self.size:
                raise ValueError(f"geom {self.name!r}: mesh geoms carry no size vector")
            if self.mesh is None:
                raise ValueError(
                    f"geom {self.name!r}: type 'mesh' requires a mesh "
                    f"name referencing the payload's MeshDef table"
                )
            object.__setattr__(self, "mesh", _require_name("mesh", self.mesh))
        elif self.type in GEOM_SIZE_LEN:
            if self.mesh is not None:
                raise ValueError(
                    f"geom {self.name!r}: a {self.type!r} geom cannot " f"carry a mesh name"
                )
        else:
            raise ValueError(
                f"geom {self.name!r}: type {self.type!r} must be one of "
                f"{GEOM_TYPES + (MESH_GEOM_TYPE,)} (sdf/hfield engine types "
                f"are not IR vocabulary)"
            )
        if self.name is not None:
            object.__setattr__(self, "name", _require_name("geom", self.name))
        owner = f"geom {self.name!r}" if self.name is not None else f"unnamed {self.type} geom"
        if self.type != MESH_GEOM_TYPE:
            object.__setattr__(
                self, "size", _require_vec(owner, "size", self.size, GEOM_SIZE_LEN[self.type])
            )
        object.__setattr__(self, "pos", _require_vec(owner, "pos", self.pos, 3))
        object.__setattr__(self, "quat", _require_quat(owner, self.quat))
        if self.mass is not None:
            object.__setattr__(self, "mass", _require_nonneg(owner, "mass", self.mass))
        if self.density is not None:
            object.__setattr__(self, "density", _require_nonneg(owner, "density", self.density))
        if self.mass is not None and self.density is not None:
            raise ValueError(f"{owner}: mass and density are mutually exclusive mass " f"sources")
        if self.rgba is not None:
            object.__setattr__(self, "rgba", _require_rgba(owner, self.rgba))
        if isinstance(self.group, bool) or not isinstance(self.group, int) or self.group < 0:
            raise ValueError(f"{owner}: group must be a non-negative int, " f"got {self.group!r}")
        object.__setattr__(self, "collidable", _require_bool(owner, "collidable", self.collidable))
        if self.hints is not None and not isinstance(self.hints, ContactHints):
            raise ValueError(f"{owner}: hints must be a ContactHints")


@dataclass(frozen=True)
class Inertial:
    """One explicit body inertial (the dressing-dump face).

    The SplitA dressing dumps pin explicit ``<inertial>`` blocks whose
    values embed the original compile-time state (the visual mesh
    geoms' density-100 contribution shows up in the pinned COM at
    ~1e-5 m) and are NOT recomputable from any geom normalization.
    The payload carries the dump's full-precision row verbatim; the
    renderer authors it onto the body (the engine's
    ``explicitinertial`` semantics -- the compiler skips the geom mass
    aggregation for that body, exactly like the XML spelling).  The
    kitchen bake does NOT use this face: its bodies' inertials recompute
    from their geoms bit-identically.
    """

    pos: tuple
    quat: tuple
    mass: float
    diaginertia: tuple

    def __post_init__(self):
        """Validate the frame vectors and the positive scalars."""
        owner = "inertial"
        object.__setattr__(self, "pos", _require_vec(owner, "pos", self.pos, 3))
        object.__setattr__(self, "quat", _require_quat(owner, self.quat))
        object.__setattr__(self, "mass", _require_nonneg(owner, "mass", self.mass))
        object.__setattr__(
            self, "diaginertia", _require_vec(owner, "diaginertia", self.diaginertia, 3)
        )
        if any(v <= 0.0 for v in self.diaginertia):
            raise ValueError(f"{owner}: diaginertia must be positive, got " f"{self.diaginertia!r}")


@dataclass(frozen=True)
class Body:
    """One generated (possibly nested) rigid body of the payload tree."""

    name: str
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    geoms: tuple = ()
    joints: tuple = ()
    sites: tuple = ()
    children: tuple = ()
    gravcomp: float = 0.0
    inertial: Inertial | None = None
    """Gravity compensation (v074: the zoo laptop lid carries 1.0 -- the
    official hinge face keeps the lid weightless so its damping alone
    governs; the engine default 0.0 stays the IR default)."""

    def __post_init__(self):
        """Validate the transform + the typed element tuples."""
        object.__setattr__(self, "name", _require_name("body", self.name))
        owner = f"body {self.name!r}"
        object.__setattr__(self, "pos", _require_vec(owner, "pos", self.pos, 3))
        object.__setattr__(self, "quat", _require_quat(owner, self.quat))
        if self.inertial is not None and not isinstance(self.inertial, Inertial):
            raise ValueError(f"{owner}: inertial must be an Inertial row")
        for f, etype in (("geoms", Geom), ("joints", Joint), ("sites", Site), ("children", Body)):
            entries = getattr(self, f)
            if not isinstance(entries, tuple):
                raise ValueError(f"{owner}: {f} must be a tuple of {etype.__name__}")
            for entry in entries:
                if not isinstance(entry, etype):
                    raise ValueError(
                        f"{owner}: {f} entries must be {etype.__name__}, "
                        f"got {type(entry).__name__}"
                    )
        object.__setattr__(self, "gravcomp", _require_nonneg(owner, "gravcomp", self.gravcomp))


@dataclass(frozen=True)
class ContactHints:
    """Per-geom contact-solve hints (NEUTRALLY named; ruling §12.4-1).

    Field names describe solver INTENT, never an engine spelling: the
    renderer maps ``solver_ref`` / ``solver_impedance`` onto its engine's
    contact solve parameters; other backends ignore + warn.
    """

    friction: tuple | None = None
    """(slide, spin, roll) friction triple."""
    condim: int | None = None
    """Contact condition dimension (one of :data:`CONDIMS`)."""
    priority: int | None = None
    """Contact priority sign override."""
    solver_ref: tuple | None = None
    """(timeconst, damping-ratio) contact solver reference pair."""
    solver_impedance: tuple | None = None
    """Solver impedance curve (3 or 5 values, engine-interpreted)."""
    margin: float | None = None
    """Contact detection margin."""
    gap: float | None = None
    """Inclusion gap."""

    def __post_init__(self):
        """Validate every present hint against its closed rule."""
        owner = "contact hints"
        if self.friction is not None:
            object.__setattr__(self, "friction", _require_vec(owner, "friction", self.friction, 3))
        if self.condim is not None:
            if self.condim not in CONDIMS:
                raise ValueError(f"{owner}: condim {self.condim!r} must be one of " f"{CONDIMS}")
            object.__setattr__(self, "condim", int(self.condim))
        if self.priority is not None:
            if isinstance(self.priority, bool) or not isinstance(self.priority, int):
                raise ValueError(f"{owner}: priority must be an int")
        if self.solver_ref is not None:
            object.__setattr__(
                self, "solver_ref", _require_vec(owner, "solver_ref", self.solver_ref, 2)
            )
        if self.solver_impedance is not None:
            length = (
                len(self.solver_impedance)
                if isinstance(self.solver_impedance, (list, tuple))
                else -1
            )
            # the impedance curve may be authored prefix-form (the MJCF
            # convention): the renderer pads the engine-default tail
            if length not in (2, 3, 5):
                raise ValueError(
                    f"{owner}: solver_impedance must be 2, 3 or 5 finite "
                    f"numbers, got {self.solver_impedance!r}"
                )
            object.__setattr__(
                self,
                "solver_impedance",
                _require_vec(owner, "solver_impedance", self.solver_impedance, length),
            )
        for f in ("margin", "gap"):
            if getattr(self, f) is not None:
                object.__setattr__(self, f, _require_nonneg(owner, f, getattr(self, f)))


@dataclass(frozen=True)
class JointHints:
    """Per-joint limit-solve hints (neutral names; renderer-interpreted).

    Both faces carry the MJCF prefix-form convention (v074: the zoo
    laptop's ``solreflimit="0.001 1"`` / ``solimplimit="0.95 0.99
    0.001"``): the renderer pads the engine-default tail, so the IR
    keeps the authored prefix (ref 1-2 numbers, impedance 2/3/5).
    """

    limit_solver_ref: tuple | None = None
    """(timeconst, damping-ratio) reference pair for the joint limits."""
    limit_solver_impedance: tuple | None = None
    """Solver impedance curve for the joint limits (2/3/5 values)."""

    def __post_init__(self):
        """Validate every present hint against its closed rule."""
        owner = "joint hints"
        if self.limit_solver_ref is not None:
            n = (
                len(self.limit_solver_ref)
                if isinstance(self.limit_solver_ref, (list, tuple))
                else -1
            )
            if n not in (1, 2):
                raise ValueError(
                    f"{owner}: limit_solver_ref must be 1 or 2 finite "
                    f"numbers, got {self.limit_solver_ref!r}"
                )
            object.__setattr__(
                self,
                "limit_solver_ref",
                _require_vec(owner, "limit_solver_ref", self.limit_solver_ref, n),
            )
        if self.limit_solver_impedance is not None:
            n = (
                len(self.limit_solver_impedance)
                if isinstance(self.limit_solver_impedance, (list, tuple))
                else -1
            )
            if n not in (2, 3, 5):
                raise ValueError(
                    f"{owner}: limit_solver_impedance must be 2, 3 or 5 "
                    f"finite numbers, got {self.limit_solver_impedance!r}"
                )
            object.__setattr__(
                self,
                "limit_solver_impedance",
                _require_vec(owner, "limit_solver_impedance", self.limit_solver_impedance, n),
            )


@dataclass(frozen=True)
class SolverHints:
    """The neutral solver-hints block of one payload (ruling §12.4-1).

    A provider writes INTENT here (never an engine spelling); the
    renderer interprets it, other backends ignore + warn.
    """

    contact: ContactHints | None = None
    """Contact-solve hints applied to every geom of the payload."""
    joint: JointHints | None = None
    """Limit-solve hints applied to every joint of the payload."""

    def __post_init__(self):
        """Validate the block members' types."""
        if self.contact is not None and not isinstance(self.contact, ContactHints):
            raise ValueError("solver_hints.contact must be a ContactHints")
        if self.joint is not None and not isinstance(self.joint, JointHints):
            raise ValueError("solver_hints.joint must be a JointHints")


def iter_bodies(root: Body) -> Iterator[Body]:
    """Yield a generated body subtree depth-first (declaration order)."""
    yield root
    for child in root.children:
        yield from iter_bodies(child)
