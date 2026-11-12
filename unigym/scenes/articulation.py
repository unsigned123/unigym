"""Neutral articulation IR (authoring model).

An articulation is ``root + model + joints + spawn``.  There is no
articulation-level control mode: *how* a joint is driven (and which adapter
owns it) is declared explicitly in the run-level ``controls[]`` block
(:mod:`unigym.runtime.controls`), and joints that are not driven at all are
listed in :attr:`ArticulationSpec.passive_joints`.  The IR is pure data and
imports only the standard library plus the neutral robot vocabulary -- no
engine and no backend name is allowed here (attribute neutrality).

v104 removed ``ControlSpec`` / ``JointSpec.mode`` / ``policy_articulation``
/ ``policy_section``: the drive declaration lives in ``controls[]`` and the
robot profile only supplies the joint order / q0 / body parts.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from unigym._typing import JsonValue, is_json_object, is_object_sequence
from unigym.assets.asset_ref import AssetRef, AssetRefError
from unigym.robots.base import (
    BodyBinding,
    BodyBindingError,
    CanonicalBodyId,
    normalize_canonical_body_id,
)

__all__ = [
    "ARTICULATION_PREFIX",
    "ArticulationError",
    "ArticulationSpec",
    "AssetRef",
    "ContactSensorGroup",
    "ContactSensorInventory",
    "JointSpec",
    "ROOT_MODES",
    "articulation_key",
    "normalize_articulation",
    "normalize_articulations",
    "normalize_contact_sensor_inventory",
    "validate_contact_sensor_inventory",
]

ARTICULATION_PREFIX = "articulation"
"""The observation namespace prefix for per-articulation keys (v101)."""

ROOT_MODES = ("free", "fixed", "welded")
"""Root semantics: free joint / fixed to world / welded to a host body."""


def articulation_key(name: str, key: str) -> str:
    """Namespace one observation key under an articulation.

    Args:
        name: articulation name (e.g. ``"g1"`` / ``"door"``).
        key: bare observation key (e.g. ``"joint/pos"``).

    Returns:
        ``"articulation/<name>/<key>"``.
    """
    if not name:
        raise ArticulationError("articulation name must be non-empty")
    if not key:
        raise ArticulationError("articulation observation key must be " "non-empty")
    return f"{ARTICULATION_PREFIX}/{name}/{key}"


class ArticulationError(ValueError):
    """A malformed articulation declaration (fail-fast, no alias)."""


def _finite(value: object, what: str) -> float:
    if not isinstance(value, (int, float, str)):
        raise ArticulationError(f"{what} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ArticulationError(f"{what} must be a number, got {value!r}")
    if not math.isfinite(number):
        raise ArticulationError(f"{what} must be finite, got {value!r}")
    return number


def _optional_float(value: JsonValue, what: str) -> float | None:
    if value is None:
        return None
    return _finite(value, what)


def _range_value(value: JsonValue, what: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ArticulationError(f"{what} must be [lo, hi], got {value!r}")
    return (_finite(value[0], f"{what}[0]"), _finite(value[1], f"{what}[1]"))


def _empty_json_map() -> dict[str, JsonValue]:
    """Typed empty default for mapping fields."""
    return {}


def _unknown_keys(
    mapping: Mapping[str, JsonValue],
    allowed: tuple[str, ...],
    what: str,
) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ArticulationError(
            f"{what} declares unknown keys {unknown}; allowed keys are " f"{list(allowed)}"
        )


_CONTACT_MODES = ("declared", "all", "none")
"""The closed contact-sensor inventory modes."""

_CONTACT_GROUP_NAME = re.compile(r"[A-Za-z0-9_-]+")
_RESERVED_CONTACT_GROUP_NAMES = frozenset({"all", "none"})
_CONTACT_GROUP_KEYS = ("name", "bodies", "body_set")


def _contact_bodies(value: object, what: str) -> tuple[CanonicalBodyId, ...] | None:
    if value is None:
        return None
    if not is_object_sequence(value):
        raise ArticulationError(f"{what} must be a list of body ids, got {value!r}")
    entries = tuple(value)
    if not entries:
        return None
    bodies: list[CanonicalBodyId] = []
    for index, entry in enumerate(entries):
        try:
            bodies.append(normalize_canonical_body_id(entry, f"{what}[{index}]"))
        except BodyBindingError as exc:
            raise ArticulationError(str(exc)) from exc
    duplicates = sorted({str(body) for body in bodies if bodies.count(body) > 1})
    if duplicates:
        raise ArticulationError(f"{what} contains duplicate bodies {duplicates}")
    return tuple(bodies)


def _contact_body_set(value: object, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ArticulationError(f"{what} must be a non-empty set name, got {value!r}")
    return value.strip()


@dataclass(frozen=True)
class ContactSensorGroup:
    """One named group of robot bodies that carry contact sensors.

    A group addresses its bodies through exactly one of an explicit
    ``bodies`` tuple or a ``body_set`` reference resolved by
    :func:`validate_contact_sensor_inventory`.
    """

    name: str
    bodies: tuple[CanonicalBodyId, ...] | None = None
    body_set: str | None = None

    def __post_init__(self) -> None:
        name = self.name
        if not isinstance(name, str) or not name.strip():
            raise ArticulationError(
                f"contact sensor group name must be a non-empty string, got {name!r}"
            )
        name = name.strip()
        if _CONTACT_GROUP_NAME.fullmatch(name) is None:
            raise ArticulationError(f"contact sensor group name {name!r} must match [A-Za-z0-9_-]+")
        if name in _RESERVED_CONTACT_GROUP_NAMES:
            raise ArticulationError(
                f"contact sensor group name {name!r} is reserved; "
                f"{sorted(_RESERVED_CONTACT_GROUP_NAMES)} are mode/wildcard names"
            )
        bodies = _contact_bodies(self.bodies, f"contact sensor group {name!r} bodies")
        body_set = _contact_body_set(self.body_set, f"contact sensor group {name!r} body_set")
        if (bodies is None) == (body_set is None):
            raise ArticulationError(
                f"contact sensor group {name!r} must declare exactly one of "
                f"non-empty `bodies` or `body_set` (got bodies={bodies!r}, "
                f"body_set={body_set!r})"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "bodies", bodies)
        object.__setattr__(self, "body_set", body_set)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize one inventory group (``bodies`` or ``body_set``)."""
        if self.body_set is not None:
            return {"name": self.name, "body_set": self.body_set}
        return {"name": self.name, "bodies": [str(body) for body in (self.bodies or ())]}


@dataclass(frozen=True)
class ContactSensorInventory:
    """One articulation's contact-sensor body inventory.

    ``mode`` is ``declared`` (named groups), ``all`` (every rigid body, not
    enumerated in this layer) or ``none`` (explicit opt-out).
    """

    mode: str
    groups: tuple[ContactSensorGroup, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in _CONTACT_MODES:
            raise ArticulationError(
                f"contact sensor inventory mode must be one of "
                f"{list(_CONTACT_MODES)}, got {self.mode!r}"
            )
        groups = self.groups
        if groups is None:
            groups = ()
        if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
            raise ArticulationError(
                f"contact sensor inventory groups must be a list, got {groups!r}"
            )
        entries = tuple(groups)
        for entry in entries:
            if not isinstance(entry, ContactSensorGroup):
                raise ArticulationError(
                    f"contact sensor inventory groups must be ContactSensorGroup "
                    f"records, got {entry!r}"
                )
        names = [group.name for group in entries]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ArticulationError(
                f"contact sensor inventory declares duplicate group names {duplicates}"
            )
        if self.mode == "declared" and not entries:
            raise ArticulationError(
                "contact sensor inventory mode 'declared' requires at least one group"
            )
        if self.mode in ("all", "none") and entries:
            raise ArticulationError(
                f"contact sensor inventory mode {self.mode!r} must not declare "
                f"groups, got {names!r}"
            )
        object.__setattr__(self, "groups", entries)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serialize the inventory for run metadata (mode + groups)."""
        return {
            "mode": self.mode,
            "groups": [group.to_dict() for group in self.groups],
        }


def normalize_contact_sensor_inventory(value: object) -> ContactSensorInventory:
    """Normalize ``all`` / ``none`` / a group list into the inventory IR.

    ``contacts: all`` and the wildcard spelling
    ``[{name: all, bodies: ["*"]}]`` normalize to the same ``mode="all"``
    inventory.  An already normalized inventory is returned unchanged.
    """
    if isinstance(value, ContactSensorInventory):
        return value
    if isinstance(value, str):
        mode = value.strip()
        if mode in ("all", "none"):
            return ContactSensorInventory(mode=mode)
        raise ArticulationError(
            f"contact sensor inventory string must be 'all' or 'none', got {value!r}"
        )
    if isinstance(value, Mapping):
        raise ArticulationError(
            f"contact sensor inventory must be 'all', 'none' or a list of groups, " f"got {value!r}"
        )
    if not is_object_sequence(value):
        raise ArticulationError(
            f"contact sensor inventory must be 'all', 'none' or a list of groups, " f"got {value!r}"
        )
    entries = tuple(value)
    if not entries:
        raise ArticulationError(
            "contact sensor inventory declared mode requires at least one group"
        )
    groups: list[ContactSensorGroup] = []
    for entry in entries:
        if isinstance(entry, ContactSensorGroup):
            groups.append(entry)
            continue
        if not is_json_object(entry):
            raise ArticulationError(
                f"contact sensor inventory groups must be mappings, got {entry!r}"
            )
        unknown = sorted(set(entry) - set(_CONTACT_GROUP_KEYS))
        if unknown:
            raise ArticulationError(
                f"contact sensor group declares unknown keys {unknown}; allowed "
                f"keys are {list(_CONTACT_GROUP_KEYS)}"
            )
        raw_name = entry.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ArticulationError(
                f"contact sensor group name must be a non-empty string, got {raw_name!r}"
            )
        name = raw_name.strip()
        raw_bodies = entry.get("bodies")
        if is_object_sequence(raw_bodies):
            body_values = tuple(raw_bodies)
            wildcards = [body for body in body_values if body == "*"]
            if wildcards:
                if (
                    name != "all"
                    or len(body_values) != 1
                    or entry.get("body_set") is not None
                    or len(entries) != 1
                ):
                    raise ArticulationError(
                        f"contact sensor wildcard '*' must be the only body of a "
                        f"single group named 'all', got name={name!r}, "
                        f"bodies={list(body_values)!r}"
                    )
                return ContactSensorInventory(mode="all")
        bodies = _contact_bodies(raw_bodies, f"contact sensor group {name!r} bodies")
        body_set = _contact_body_set(
            entry.get("body_set"), f"contact sensor group {name!r} body_set"
        )
        groups.append(ContactSensorGroup(name=name, bodies=bodies, body_set=body_set))
    return ContactSensorInventory(mode="declared", groups=tuple(groups))


def _format_names(formats: Iterable[str] | None) -> tuple[str, ...] | None:
    if formats is None:
        return None
    return tuple(str(asset_format) for asset_format in formats)


def validate_contact_sensor_inventory(
    inventory: ContactSensorInventory,
    *,
    body_sets: Mapping[str, Sequence[str]] | None = None,
    bindings: Mapping[CanonicalBodyId, BodyBinding] | None = None,
    formats: Iterable[str] | None = None,
) -> None:
    """Validate one contact-sensor inventory against sets, bindings and formats.

    ``all`` and ``none`` need no body lookup.  Declared groups resolve their
    body sets, reject unknown bodies/formats, and require every canonical body
    to appear in at most one group.
    """
    if not isinstance(inventory, ContactSensorInventory):
        raise ArticulationError(
            f"contact sensor inventory must be a ContactSensorInventory, " f"got {inventory!r}"
        )
    if inventory.mode in ("all", "none"):
        return
    declared_sets = (
        {} if body_sets is None else {str(key): value for key, value in body_sets.items()}
    )
    known_sets = sorted(declared_sets)
    format_names = _format_names(formats)
    if format_names is not None and bindings is None:
        raise ArticulationError("contact sensor inventory format validation requires body bindings")
    owner: dict[CanonicalBodyId, str] = {}
    for group in inventory.groups:
        if group.body_set is not None:
            if group.body_set not in declared_sets:
                raise ArticulationError(
                    f"contact sensor group {group.name!r} body_set "
                    f"{group.body_set!r} is unknown; declared body sets are "
                    f"{known_sets}"
                )
            refs = _contact_bodies(
                declared_sets[group.body_set],
                f"contact sensor group {group.name!r} body_set " f"{group.body_set!r} bodies",
            )
        else:
            refs = group.bodies
        referenced = refs or ()
        for body in referenced:
            if body in owner:
                raise ArticulationError(
                    f"body {str(body)!r} appears in multiple contact sensor groups "
                    f"{owner[body]!r} and {group.name!r}; one sensor per body"
                )
        if bindings is not None:
            unknown_bodies = sorted(str(body) for body in referenced if body not in bindings)
            if unknown_bodies:
                known = sorted(str(key) for key in bindings)
                raise ArticulationError(
                    f"contact sensor group {group.name!r} names unknown bodies "
                    f"{unknown_bodies}; bound bodies are {known}"
                )
        if format_names is not None and bindings is not None:
            missing: list[tuple[CanonicalBodyId, list[str]]] = []
            for body in referenced:
                absent = [fmt for fmt in format_names if fmt not in bindings[body].names]
                if absent:
                    missing.append((body, absent))
            if missing:
                details = ", ".join(f"{str(body)!r} missing {absent}" for body, absent in missing)
                raise ArticulationError(
                    f"contact sensor group {group.name!r} bodies lack required "
                    f"formats: {details}; declared formats are {list(format_names)}"
                )
        for body in referenced:
            owner[body] = group.name


@dataclass(frozen=True)
class JointSpec:
    """One declared joint of an articulation (authoring IR).

    A joint no longer carries a capability ``mode``: every declared joint is
    actuated unless it is named in :attr:`ArticulationSpec.passive_joints`,
    and the drive (``position_pd`` / ``position_servo`` / ...) is declared by
    the owning group in the run-level ``controls[]`` block.
    """

    name: str
    init: float = 0.0
    range: tuple[float, float] | None = None
    effort_limit: float | None = None
    damping: float | None = None

    def __post_init__(self):
        if not self.name:
            raise ArticulationError("joint name must be non-empty")
        object.__setattr__(self, "name", str(self.name))
        for key in ("effort_limit", "damping"):
            object.__setattr__(
                self, key, _optional_float(getattr(self, key), f"joint {self.name!r} {key}")
            )
        rng = self.range
        if rng is not None:
            if len(rng) != 2:
                raise ArticulationError(f"joint {self.name!r} range must be [lo, hi], got {rng!r}")
            rng = (
                _finite(rng[0], f"joint {self.name!r} range[0]"),
                _finite(rng[1], f"joint {self.name!r} range[1]"),
            )
            if rng[0] > rng[1]:
                raise ArticulationError(f"joint {self.name!r} range {list(rng)} has lo > hi")
            object.__setattr__(self, "range", rng)
        init = _finite(self.init, f"joint {self.name!r} init")
        if rng is not None and not (rng[0] <= init <= rng[1]):
            raise ArticulationError(
                f"joint {self.name!r} init {init!r} is outside the declared " f"range {list(rng)}"
            )
        object.__setattr__(self, "init", init)

    @classmethod
    def from_mapping(
        cls,
        name: str,
        params: Mapping[str, JsonValue] | None = None,
    ) -> JointSpec:
        """Normalize one joint declaration (``{init, range, ...}``)."""
        params = params or {}
        if "mode" in params:
            raise ArticulationError(
                f"joint {name!r} declares the removed v104 key 'mode'; joints "
                f"are actuated unless listed in the articulation's "
                f"`passive_joints`, and the drive belongs to a top-level "
                f"`controls[]` group (drive: position_pd | position_servo | "
                f"torque | velocity | effort | current)"
            )
        if "stiffness" in params or "latch" in params:
            raise ArticulationError(
                f"joint {name!r} declares a removed v104 key "
                f"({sorted(set(params) & {'stiffness', 'latch'})}); declare "
                f"gains on the owning top-level `controls[]` group instead"
            )
        _unknown_keys(params, ("init", "range", "effort_limit", "damping"), f"joint {name!r}")
        return cls(
            name=name,
            init=_finite(params.get("init", 0.0), f"joint {name!r} init"),
            range=_range_value(params.get("range"), f"joint {name!r} range"),
            effort_limit=_optional_float(
                params.get("effort_limit"), f"joint {name!r} effort_limit"
            ),
            damping=_optional_float(params.get("damping"), f"joint {name!r} damping"),
        )


_SPAWN_KEYS = ("pos", "yaw")
"""The closed run-level robot spawn vocabulary."""


def _spawn_value(value: object, name: str) -> dict[str, JsonValue]:
    """Validate one articulation spawn mapping (pos required when present)."""
    if not is_json_object(value):
        raise ArticulationError(f"articulation {name!r} spawn must be a mapping")
    spawn = dict(value)
    if not spawn:
        return {}
    unknown = sorted(set(spawn) - set(_SPAWN_KEYS))
    if unknown:
        raise ArticulationError(
            f"articulation {name!r} spawn declares unknown keys {unknown}; "
            f"allowed keys are {list(_SPAWN_KEYS)}"
        )
    raw_pos = spawn.get("pos")
    if raw_pos is None:
        raise ArticulationError(f"articulation {name!r} spawn requires `pos` (a finite 3-vector)")
    if not is_object_sequence(raw_pos) or len(raw_pos) != 3:
        raise ArticulationError(
            f"articulation {name!r} spawn.pos must be a finite 3-vector, got {raw_pos!r}"
        )
    pos = tuple(
        _finite(entry, f"articulation {name!r} spawn.pos[{index}]")
        for index, entry in enumerate(raw_pos)
    )
    yaw = _finite(spawn.get("yaw", 0.0), f"articulation {name!r} spawn.yaw")
    return {"pos": list(pos), "yaw": yaw}


def _optional_body_ids(value: object, what: str) -> tuple[CanonicalBodyId, ...] | None:
    """Normalize an optional canonical body-id tuple (None = inherit)."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not is_object_sequence(value):
        raise ArticulationError(f"{what} must be a list of body ids, got {value!r}")
    bodies: list[CanonicalBodyId] = []
    for index, entry in enumerate(value):
        try:
            bodies.append(normalize_canonical_body_id(entry, f"{what}[{index}]"))
        except BodyBindingError as exc:
            raise ArticulationError(str(exc)) from exc
    return tuple(bodies)


def _optional_body_sets(value: object, what: str) -> dict[str, tuple[CanonicalBodyId, ...]] | None:
    """Normalize optional named canonical body sets (None = inherit)."""
    if value is None:
        return None
    if not is_json_object(value):
        raise ArticulationError(f"{what} must be a mapping of set name -> bodies")
    sets: dict[str, tuple[CanonicalBodyId, ...]] = {}
    for raw_name, raw_bodies in value.items():
        name = str(raw_name)
        if not name:
            raise ArticulationError(f"{what} names must be non-empty")
        members = _optional_body_ids(raw_bodies, f"{what}[{name!r}]")
        if not members:
            raise ArticulationError(f"{what}[{name!r}] must declare at least one body id")
        if len(set(members)) != len(members):
            raise ArticulationError(f"{what}[{name!r}] contains duplicate body ids")
        sets[name] = members
    return sets


def _optional_body_id(value: object, what: str) -> CanonicalBodyId | None:
    """Normalize an optional canonical body id (None = inherit)."""
    if value is None:
        return None
    try:
        return normalize_canonical_body_id(value, what)
    except BodyBindingError as exc:
        raise ArticulationError(str(exc)) from exc


@dataclass(frozen=True)
class ArticulationSpec:
    """One articulation: root + model + joints + spawn (neutral IR).

    ``joints`` may be omitted for profile-derived articulations (the robot
    profile supplies the joint order); when it is declared, every joint is
    actuated unless listed in ``passive_joints``.  Control coverage is
    validated at the run level against ``controls[]``
    (:func:`unigym.runtime.controls.validate_controls`).

    The body fields are run-level overrides over the robot profile: ``None``
    means inherit the profile, an explicit empty tuple/mapping means empty.
    ``contacts`` is the per-articulation contact sensor inventory; ``None``
    means omitted.  ``spawn`` is the only run-level robot-root spawn source.
    """

    name: str
    model: AssetRef | None = None
    root: str = "free"
    spawn: Mapping[str, JsonValue] = field(default_factory=_empty_json_map)
    joints: tuple[JointSpec, ...] = ()
    base_body: CanonicalBodyId | None = None
    foot_bodies: tuple[CanonicalBodyId, ...] | None = None
    body_sets: Mapping[str, tuple[CanonicalBodyId, ...]] | None = None
    contacts: ContactSensorInventory | None = None
    passive_joints: tuple[str, ...] = ()
    params: Mapping[str, JsonValue] = field(default_factory=_empty_json_map)

    def __post_init__(self):
        if not self.name:
            raise ArticulationError("articulation name must be non-empty")
        object.__setattr__(self, "name", str(self.name))
        if self.root not in ROOT_MODES:
            raise ArticulationError(
                f"articulation {self.name!r} root must be one of "
                f"{ROOT_MODES}, got {self.root!r}"
            )
        if self.model is not None:
            try:
                model = AssetRef.from_mapping(self.model, f"articulation {self.name!r} model")
            except AssetRefError as exc:
                raise ArticulationError(str(exc)) from exc
            object.__setattr__(self, "model", model)
        object.__setattr__(self, "spawn", _spawn_value(self.spawn, self.name))
        object.__setattr__(self, "joints", _joints_tuple(self.name, self.joints))
        joint_names = [joint.name for joint in self.joints]
        duplicated = sorted({name for name in joint_names if joint_names.count(name) > 1})
        if duplicated:
            raise ArticulationError(
                f"articulation {self.name!r} declares duplicate joint names {duplicated!r}"
            )
        object.__setattr__(
            self,
            "base_body",
            _optional_body_id(self.base_body, f"articulation {self.name!r} base_body"),
        )
        object.__setattr__(
            self,
            "foot_bodies",
            _optional_body_ids(self.foot_bodies, f"articulation {self.name!r} foot_bodies"),
        )
        object.__setattr__(
            self,
            "body_sets",
            _optional_body_sets(self.body_sets, f"articulation {self.name!r} body_sets"),
        )
        if self.contacts is not None and not isinstance(self.contacts, ContactSensorInventory):
            object.__setattr__(self, "contacts", normalize_contact_sensor_inventory(self.contacts))
        passive = tuple(str(name) for name in (self.passive_joints or ()))
        if len(set(passive)) != len(passive):
            raise ArticulationError(
                f"articulation {self.name!r} passive_joints contains " f"duplicates {passive!r}"
            )
        declared = {joint.name for joint in self.joints}
        if declared:
            unknown = sorted(set(passive) - declared)
            if unknown:
                raise ArticulationError(
                    f"articulation {self.name!r} passive_joints {unknown} are "
                    f"not declared joints"
                )
        object.__setattr__(self, "passive_joints", passive)
        if not isinstance(self.params, Mapping):
            raise ArticulationError(f"articulation {self.name!r} params must be a mapping")
        object.__setattr__(self, "params", dict(self.params))


def _joints_tuple(
    name: str,
    joints: JsonValue | Sequence[JointSpec],
) -> tuple[JointSpec, ...]:
    if not joints:
        return ()
    if isinstance(joints, Mapping):
        entries: list[JointSpec] = []
        for jname, params in joints.items():
            if not isinstance(params, Mapping):
                raise ArticulationError(
                    f"articulation {name!r} joint {jname!r} must be a " f"mapping, got {params!r}"
                )
            entries.append(JointSpec.from_mapping(str(jname), params))
        return tuple(entries)
    if not isinstance(joints, (list, tuple)):
        raise ArticulationError(f"articulation {name!r} joints must be a list, got {joints!r}")
    out: list[JointSpec] = []
    for entry in joints:
        if isinstance(entry, JointSpec):
            out.append(entry)
            continue
        if not isinstance(entry, Mapping):
            raise ArticulationError(
                f"articulation {name!r} joints entries must be mappings, " f"got {entry!r}"
            )
        entry_map = dict(entry)
        jname = entry_map.pop("name", None)
        if jname is None:
            raise ArticulationError(f"articulation {name!r} joint entry is missing 'name'")
        out.append(JointSpec.from_mapping(str(jname), entry_map))
    return tuple(out)


def _articulation_name(entry: Mapping[str, JsonValue]) -> str:
    name = entry["name"]
    if not name:
        raise ArticulationError("articulation name must be non-empty")
    return str(name)


def _root_value(value: JsonValue, name: str) -> str:
    if not isinstance(value, str) or value not in ROOT_MODES:
        raise ArticulationError(
            f"articulation {name!r} root must be one of " f"{ROOT_MODES}, got {value!r}"
        )
    return value


def _mapping_field(value: JsonValue, what: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ArticulationError(f"{what} must be a mapping")
    return value


def _str_tuple(value: JsonValue, what: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ArticulationError(f"{what} must be a list of names, got {value!r}")
    return tuple(str(item) for item in value)


def _model_field(value: JsonValue, name: str) -> AssetRef | None:
    if value is None:
        return None
    if not isinstance(value, (AssetRef, Mapping)):
        raise ArticulationError(
            f"articulation {name!r} model must be a mapping "
            f"{{mjcf|urdf|usd: <path>}}, got {value!r}"
        )
    try:
        return AssetRef.from_mapping(value, f"articulation {name!r} model")
    except AssetRefError as exc:
        raise ArticulationError(str(exc)) from exc


def normalize_articulation(entry: Mapping[str, JsonValue]) -> ArticulationSpec:
    """Normalize one ``articulations[]`` entry into an :class:`ArticulationSpec`.

    The config surface is a mapping with the IR fields
    (``name/model/root/spawn/joints/base_body/foot_bodies/body_sets/
    contacts/passive_joints/params``); unknown keys fail fast (there is no
    shorthand and no alias).  The articulation-level ``control:`` block is
    rejected with a teaching error (drives are declared in the run-level
    ``controls[]``).
    """
    if not isinstance(entry, Mapping):
        raise ArticulationError(f"articulations entries must be mappings, got {entry!r}")
    if "control" in entry:
        raise ArticulationError(
            f"articulation {entry.get('name')!r} declares the removed v104 "
            f"key `control:`; articulation-level control modes "
            f"(policy/named/passive) no longer exist -- declare the drive "
            f"explicitly in the run-level `controls:` list "
            f"(name/adapter/articulation/drive/target) and list undriven "
            f"joints in `passive_joints`"
        )
    _unknown_keys(
        entry,
        (
            "name",
            "model",
            "root",
            "spawn",
            "joints",
            "base_body",
            "foot_bodies",
            "body_sets",
            "contacts",
            "passive_joints",
            "params",
        ),
        "articulation",
    )
    if "name" not in entry:
        raise ArticulationError("articulation entry is missing 'name'")
    name = _articulation_name(entry)
    raw_body_sets = entry.get("body_sets")
    raw_contacts = entry.get("contacts")
    return ArticulationSpec(
        name=name,
        model=_model_field(entry.get("model"), name),
        root=_root_value(entry.get("root", "free"), name),
        spawn=_mapping_field(entry.get("spawn", {}), f"articulation {name!r} spawn"),
        joints=_joints_tuple(name, entry.get("joints")),
        base_body=_optional_body_id(
            entry.get("base_body") if "base_body" in entry else None,
            f"articulation {name!r} base_body",
        ),
        foot_bodies=_optional_body_ids(
            entry.get("foot_bodies") if "foot_bodies" in entry else None,
            f"articulation {name!r} foot_bodies",
        ),
        body_sets=_optional_body_sets(raw_body_sets, f"articulation {name!r} body_sets"),
        contacts=(
            None if raw_contacts is None else normalize_contact_sensor_inventory(raw_contacts)
        ),
        passive_joints=_str_tuple(
            entry.get("passive_joints"), f"articulation {name!r} passive_joints"
        ),
        params=_mapping_field(entry.get("params", {}), f"articulation {name!r} params"),
    )


def normalize_articulations(entries: JsonValue) -> tuple[ArticulationSpec, ...]:
    """Normalize the run's ``articulations:`` list.

    The list is declaration data only: every articulation is ``root + model
    + joints + spawn``.  Which joints are driven is decided by the run-level
    ``controls[]`` block, not by a per-articulation mode.
    """
    if entries is None:
        raise ArticulationError(
            "run config declares no articulations; the neutral shape is "
            "`articulations: [{name: <robot>, params: ...}]` plus a "
            "top-level `controls:` list (the old top-level `robot:` block "
            "was removed -- no alias)"
        )
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ArticulationError("articulations must be a non-empty list of mappings")
    specs_list: list[ArticulationSpec] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ArticulationError(f"articulations entries must be mappings, got {entry!r}")
        specs_list.append(normalize_articulation(entry))
    specs = tuple(specs_list)
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ArticulationError(f"articulations names must be unique, got {names!r}")
    return specs
