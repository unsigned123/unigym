"""Neutral scene IR: the closed five-type vocabulary and its container.

Pure data + validation -- standard library only (``dataclasses``,
``typing``, ``math``): no engine import, no content import, no IO.  The
vocabulary is the closed IR of design v085 section 3:

* :class:`Instance` -- the ONLY geometry carrier (``generator`` +
  ``params`` + ``pose``), with a mandatory provenance ``source``;
* :class:`Region` -- a named owner-local extent;
* :class:`Slot` -- the only scene/content shared object (id + frame +
  extent + free tags); the core stores it and never interprets the tags;
* :class:`Material` -- a logical name plus an opaque binding table;
* :class:`Scene` -- the container.

:class:`Pose` is the shared transform value.  ``Scene.engine_hints`` is
the ONE explicitly NON-neutral member: it is the engine's escape hatch
(solver/gravity/... knobs) and no other member may carry engine policy.

The neutral-name guard mirrors the asset-IR neutral-name guard without
importing it: one mechanical rule, a local copy, so this module stays
standard-library only (design v075 section 3 shares the rule between the
asset and scene vocabularies).
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

from unigym._typing import JsonValue

if TYPE_CHECKING:
    from .terrain import Terrain

#: one instance parameter value: JSON data plus the tuple form the engine seam
#: forwards (pose vectors).  The IR stores it verbatim and never reads it.
ParamValue: TypeAlias = JsonValue | tuple[float, ...]

_T = TypeVar("_T")


def _empty_slots() -> dict[str, JsonValue]:
    """An empty material binding table (typed dataclass default)."""
    return {}


def _empty_hints() -> dict[str, Any]:
    """An empty engine-hints table (typed dataclass default)."""
    return {}


#: engine knobs are an opaque, engine-owned payload: the core never reads
#: or interprets a hint value (only the backend does), so ``Any`` is honest
#: here rather than a placeholder for an unknown type.
_EngineHints: TypeAlias = Mapping[str, Any]

#: the closed instance-role vocabulary (a hint, never a dispatch branch).
INSTANCE_ROLES = ("rigid", "articulated", "terrain", "sensor")

CORE_SLOT_TAGS = ("support", "interior", "opening")
"""The slot tags the CORE itself understands (v091 #2).

Tags stay FREE by design: a Slot may carry any string, and a pack adds its
own vocabulary (``front``, ``shelf`` ...) without registering anything --
the core never rejects one.  These three are only the words the core itself
emits (they mirror the region kinds of the reference content) and that a
generic consumer may count on; treat the list as a convention, not a gate.
"""

#: engine element names an IR element may never be NAMED after (the
#: mechanical anti-leak rule shared with the asset IR, copied here so the
#: scenes package imports nothing from the asset-IR package).
_ENGINE_ELEMENT_NAMES = frozenset(
    {
        "body",
        "geom",
        "joint",
        "site",
        "freejoint",
        "frame",
        "include",
        "asset",
        "default",
        "worldbody",
        "world",
        "actuator",
        "motor",
        "position",
        "velocity",
        "intvelocity",
        "general",
        "damper",
        "muscle",
        "adhesion",
        "contact",
        "pair",
        "equality",
        "tendon",
        "sensor",
        "keyframe",
        "numeric",
        "text",
        "tuple",
        "flex",
        "skin",
        "hfield",
        "texture",
        "material",
        "mesh",
        "skybox",
    }
)

_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")


def require_neutral_name(kind: str, name: str) -> str:
    """Validate one IR element name against the mechanical ban.

    Args:
        kind: The element kind used in the teaching error.
        name: The candidate identifier.

    Returns:
        The validated name.

    Raises:
        ValueError: The name is not lowercase snake_case or it spells an
            engine element name; the message names the rule and the fix.
    """
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(
            f"{kind} name {name!r} must be a lowercase snake_case identifier "
            f"(path-like spellings are rejected)"
        )
    if name in _ENGINE_ELEMENT_NAMES:
        raise ValueError(
            f"{kind} name {name!r} is an engine element name -- scene and "
            f"content speak the neutral IR vocabulary only"
        )
    return name


def _require_text(owner: str, name: str, value: object) -> str:
    """Validate one non-empty string field."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner}: {name} must be a non-empty string, got {value!r}")
    return value


def _require_vec(
    owner: str, name: str, value: Sequence[object], lengths: int | tuple[int, ...],
) -> tuple[float, ...]:
    """Validate a fixed-length finite float vector (one of *lengths*)."""
    allowed = (lengths,) if isinstance(lengths, int) else lengths
    if not isinstance(value, (list, tuple)):
        wanted = " or ".join(str(n) for n in allowed)
        raise ValueError(
            f"{owner}: {name} must be a {wanted}-vector of finite numbers, "
            f"got {value!r}"
        )
    if len(value) not in allowed or any(
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        for v in value
    ):
        want = " or ".join(str(n) for n in allowed)
        raise ValueError(
            f"{owner}: {name} must be a {want}-vector of finite numbers, got {value!r}"
        )
    return tuple(
        float(v) for v in value if isinstance(v, (int, float))
    )


def _require_quat(owner: str, value: Sequence[object]) -> tuple[float, ...]:
    """Validate a non-zero ``(w, x, y, z)`` quaternion."""
    quat = _require_vec(owner, "quat_wxyz", value, (4,))
    if sum(v * v for v in quat) < 1e-24:
        raise ValueError(f"{owner}: quat_wxyz must be non-zero, got {value!r}")
    return quat


def _require_source(owner: str, value: object, optional: bool = False) -> str:
    """Validate a ``file:line`` provenance string (design v091 invariant 3)."""
    if optional and value == "":
        return ""
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{owner}: source must be a non-empty 'file:line' string, got "
            f"{value!r} -- every hop must carry its provenance"
        )
    if ":" not in value:
        raise ValueError(
            f"{owner}: source {value!r} must name 'file:line' provenance "
            f"(e.g. 'scene.yaml:42' or 'content:cup#1')"
        )
    return value


@dataclass(frozen=True)
class Pose:
    """One transform: frame name + translation + ``(w, x, y, z)`` rotation.

    ``frame`` is an opaque string ("world" or a slot/region frame); the
    core validates that it is non-empty and never parses it.  A silent
    rotation is the identity.
    """

    frame: str = "world"
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        """Validate the frame name, the translation and the rotation."""
        object.__setattr__(self, "frame", _require_text("pose", "frame", self.frame))
        object.__setattr__(self, "xyz", _require_vec("pose", "xyz", self.xyz, (3,)))
        object.__setattr__(self, "quat_wxyz", _require_quat("pose", self.quat_wxyz))


@dataclass(frozen=True)
class Instance:
    """One instance: a registered generator + opaque params at a pose.

    ``generator`` names a generator in the engine's registry (the scene
    core never resolves it); ``params`` is the generator's own payload,
    stored as a mapping and never interpreted here.  ``source`` is the
    mandatory ``file:line`` provenance of the declaration.  ``role`` is a
    closed hint (see :data:`INSTANCE_ROLES`), never a dispatch branch.
    """

    id: str
    generator: str
    params: Mapping[str, ParamValue]
    pose: Pose
    source: str
    role: str = "rigid"

    def __post_init__(self):
        """Validate the id, generator, params, pose, source and role."""
        object.__setattr__(self, "id", require_neutral_name("instance", self.id))
        object.__setattr__(
            self, "generator", _require_text(f"instance {self.id!r}", "generator", self.generator)
        )
        if not isinstance(self.params, Mapping):
            raise ValueError(
                f"instance {self.id!r}: params must be a mapping, got "
                f"{type(self.params).__name__}"
            )
        object.__setattr__(self, "params", dict(self.params))
        if not isinstance(self.pose, Pose):
            raise ValueError(
                f"instance {self.id!r}: pose must be a Pose, got {type(self.pose).__name__}"
            )
        object.__setattr__(
            self, "source", _require_source(f"instance {self.id!r}", self.source)
        )
        if self.role not in INSTANCE_ROLES:
            raise ValueError(
                f"instance {self.id!r}: role {self.role!r} must be one of "
                f"{INSTANCE_ROLES}"
            )


@dataclass(frozen=True)
class Region:
    """One named owner-local extent (a support, interior or opening fact).

    ``owner`` cites the instance (or other declaration) the extent is
    local to; ``frame`` is the opaque frame name; ``extent`` is a 2- or
    3-vector in metres.  ``source`` is mandatory provenance.
    """

    name: str
    owner: str
    frame: str
    extent: tuple[float, float] | tuple[float, float, float]
    source: str

    def __post_init__(self):
        """Validate the name, owner, frame, extent and source."""
        object.__setattr__(self, "name", require_neutral_name("region", self.name))
        object.__setattr__(
            self, "owner", require_neutral_name("region owner", self.owner)
        )
        object.__setattr__(
            self, "frame", _require_text(f"region {self.name!r}", "frame", self.frame)
        )
        object.__setattr__(
            self,
            "extent",
            _require_vec(f"region {self.name!r}", "extent", self.extent, (2, 3)),
        )
        object.__setattr__(
            self, "source", _require_source(f"region {self.name!r}", self.source)
        )


@dataclass(frozen=True)
class Slot:
    """The only scene/content shared object: id + frame + extent + tags.

    The core stores a slot and never parses its id or tag vocabulary --
    the naming convention is the engine's.  ``owner`` optionally cites
    the owning instance; ``source`` is optional provenance (empty when
    undeclared).
    """

    id: str
    frame: str
    extent: tuple[float, float] | tuple[float, float, float]
    owner: str = ""
    tags: tuple[str, ...] = ()
    source: str = ""

    def __post_init__(self):
        """Validate the id, frame, extent, optional owner, tags and source."""
        object.__setattr__(self, "id", require_neutral_name("slot", self.id))
        object.__setattr__(
            self, "frame", _require_text(f"slot {self.id!r}", "frame", self.frame)
        )
        object.__setattr__(
            self, "extent", _require_vec(f"slot {self.id!r}", "extent", self.extent, (2, 3))
        )
        if self.owner:
            object.__setattr__(
                self, "owner", require_neutral_name("slot owner", self.owner)
            )
        if not isinstance(self.tags, tuple) or any(
            not isinstance(tag, str) or not tag for tag in self.tags
        ):
            raise ValueError(
                f"slot {self.id!r}: tags must be a tuple of non-empty strings, "
                f"got {self.tags!r}"
            )
        object.__setattr__(
            self, "source", _require_source(f"slot {self.id!r}", self.source, optional=True)
        )


@dataclass(frozen=True)
class Material:
    """One logical material name plus an opaque binding table.

    ``slots`` maps the engine's binding keys to values; the core copies
    the mapping and never interprets it (a texture declaration, a color,
    ... all travel verbatim).
    """

    name: str
    slots: Mapping[str, JsonValue] = field(default_factory=_empty_slots)

    def __post_init__(self):
        """Validate the name and the binding mapping."""
        object.__setattr__(self, "name", require_neutral_name("material", self.name))
        if not isinstance(self.slots, Mapping):
            raise ValueError(
                f"material {self.name!r}: slots must be a mapping, got "
                f"{type(self.slots).__name__}"
            )
        object.__setattr__(self, "slots", dict(self.slots))


@dataclass(frozen=True)
class Scene:
    """The scene container: instances + regions + slots + materials.

    Sequence members are tuples of their typed entry.  ``engine_hints``
    is deliberately NOT neutral: it is the engine's escape hatch for
    solver/gravity/... knobs, kept out of every other member so the
    closed IR stays portable.
    """

    name: str
    instances: tuple[Instance, ...] = ()
    regions: tuple[Region, ...] = ()
    slots: tuple[Slot, ...] = ()
    materials: tuple[Material, ...] = ()
    engine_hints: _EngineHints = field(default_factory=_empty_hints)
    terrain: Terrain | None = None

    def __post_init__(self):
        """Validate the name, the typed sequences, hints, terrain and uniqueness."""
        object.__setattr__(self, "name", require_neutral_name("scene", self.name))
        _require_entries("instances", self.instances, Instance)
        _require_entries("regions", self.regions, Region)
        _require_entries("slots", self.slots, Slot)
        _require_entries("materials", self.materials, Material)
        if not isinstance(self.engine_hints, Mapping):
            raise ValueError(
                f"scene engine_hints must be a mapping, got "
                f"{type(self.engine_hints).__name__}"
            )
        object.__setattr__(self, "engine_hints", dict(self.engine_hints))
        if self.terrain is not None:
            from .terrain import Terrain

            if not isinstance(self.terrain, Terrain):
                raise ValueError(
                    f"scene terrain must be a Terrain or None, got "
                    f"{type(self.terrain).__name__}"
                )
        _validate_uniqueness(self)


def _require_entries(
    field_name: str, entries: tuple[object, ...], entry_type: type[_T],
) -> None:
    """Validate one typed tuple member (a sequence of *entry_type*)."""
    if not isinstance(entries, tuple):
        raise ValueError(
            f"scene {field_name} must be a tuple of {entry_type.__name__}, "
            f"got {type(entries).__name__}"
        )
    for entry in entries:
        if not isinstance(entry, entry_type):
            raise ValueError(
                f"scene {field_name} entries must be {entry_type.__name__}, "
                f"got {type(entry).__name__}"
            )


def _check_unique(
    kind: str, key: str, names: Iterable[str], scene_name: str,
) -> None:
    """Reject a duplicate declaration name within one kind."""
    seen: set[str] = set()
    for value in names:
        if value in seen:
            raise ValueError(
                f"duplicate {kind} {key} {value!r} in scene {scene_name!r} -- "
                f"every declaration must have a unique {key}"
            )
        seen.add(value)


def _validate_uniqueness(scene: Scene) -> None:
    """Reject duplicate instance ids, region names, slot ids and materials."""
    _check_unique(
        "instance", "id", (entry.id for entry in scene.instances), scene.name
    )
    _check_unique(
        "region", "name", (entry.name for entry in scene.regions), scene.name
    )
    _check_unique("slot", "id", (entry.id for entry in scene.slots), scene.name)
    _check_unique(
        "material", "name", (entry.name for entry in scene.materials), scene.name
    )


def validate_scene(scene: Scene) -> None:
    """Re-run the tree-level checks at a consumer boundary.

    Construction already validated every entry; this walk re-asserts the
    type and the uniqueness rules so a consumer can fail closed at its
    own seam (the asset-IR discipline).

    Args:
        scene: The scene to validate.

    Raises:
        ValueError: The value is not a Scene or a uniqueness rule fails.
    """
    if not isinstance(scene, Scene):
        raise ValueError(
            f"scene expected, got {type(scene).__name__}"
        )
    if scene.terrain is not None:
        from .terrain import Terrain

        if not isinstance(scene.terrain, Terrain):
            raise ValueError(
                f"scene terrain must be a Terrain or None, got "
                f"{type(scene.terrain).__name__}"
            )
    _validate_uniqueness(scene)


__all__ = [
    "CORE_SLOT_TAGS",
    "INSTANCE_ROLES",
    "Instance",
    "Material",
    "Pose",
    "Region",
    "Scene",
    "Slot",
    "require_neutral_name",
    "validate_scene",
]
