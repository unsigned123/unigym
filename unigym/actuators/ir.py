"""Neutral actuation IR: declarative groups, resolution and law validation.

An :class:`ActuationGroup` declares how one group of joints is physically
driven.  It is authoring data only: ``law`` must name a registered
:data:`unigym.actuators.base.ACTUATOR_REGISTRY` law, but this layer does not
translate gains, limits or parts into a law-specific config (that capability
work belongs to the backend translation step).

``controls[]`` (action faces / target ownership) and actuation groups are
independent; :func:`validate_actuation_groups` only proves group addressing
and one-group-per-joint coverage, while :func:`validate_drive_law_matrix`
proves that each declared control drive is served by a compatible law.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

from unigym._typing import JsonValue

from .base import ACTUATOR_REGISTRY

__all__ = [
    "ActuationError",
    "ActuationGroup",
    "GainValue",
    "ResolvedActuationGroup",
    "normalize_actuation_group",
    "normalize_actuation_groups",
    "resolve_actuation_groups",
    "resolve_group_joints",
    "validate_actuation_group",
    "validate_actuation_groups",
    "validate_drive_law_matrix",
]


class ActuationError(ValueError):
    """A malformed actuation group or coverage declaration (fail-fast)."""


GainValue = float | Mapping[str, float]
"""A scalar gain or a non-empty joint-keyed (name/regex) gain map."""

_GROUP_KEYS = (
    "name",
    "law",
    "joints",
    "joint_set",
    "gains",
    "effort_limit",
    "velocity_limit",
    "command",
    "parts",
)

_LAW_GAIN_KEYS: dict[str, frozenset[str]] = {
    "standard_pd": frozenset({"kp", "kd"}),
    "computed_torque_pd": frozenset({"kp", "kd"}),
    "native_servo": frozenset(),
}

_POSITION_PD_LAWS = frozenset({"standard_pd", "computed_torque_pd"})
_DIRECT_DRIVES = frozenset({"torque", "velocity", "effort", "current"})

_BACKEND_LAWS: dict[str, frozenset[str]] = {
    "isaaclab2": frozenset({"standard_pd"}),
    "isaaclab3": frozenset({"standard_pd"}),
    "mujoco": frozenset({"standard_pd", "computed_torque_pd", "native_servo"}),
    "mjwarp": frozenset({"standard_pd", "computed_torque_pd", "native_servo"}),
}


class _DriveControl(Protocol):
    """The drive face :func:`validate_drive_law_matrix` consumes."""

    name: str
    articulation: str
    drive: str
    joint_names: tuple[str, ...]
    prop: bool


def _empty_gain_map() -> dict[str, GainValue]:
    """Typed empty default for the gains field."""
    return {}


def _empty_command_map() -> dict[str, JsonValue]:
    """Typed empty default for the command field."""
    return {}


def _finite_number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActuationError(f"{what} must be a finite number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ActuationError(f"{what} must be finite, got {value!r}")
    return number


def _non_negative_number(value: object, what: str) -> float:
    number = _finite_number(value, what)
    if number < 0:
        raise ActuationError(f"{what} must be non-negative, got {value!r}")
    return number


def _string_keys(value: Mapping, what: str) -> None:
    for key in value:
        if not isinstance(key, str) or not key.strip():
            raise ActuationError(f"{what} keys must be non-empty strings, got {key!r}")


def _gain_value(value: object, what: str) -> GainValue:
    if isinstance(value, Mapping):
        _string_keys(value, what)
        if not value:
            raise ActuationError(f"{what} must be a non-empty joint-keyed map")
        return {str(key): _finite_number(item, f"{what}[{key!r}]") for key, item in value.items()}
    return _finite_number(value, what)


def _gain_map(value: object, what: str) -> dict[str, GainValue]:
    if not isinstance(value, Mapping):
        raise ActuationError(f"{what} must be a mapping, got {value!r}")
    _string_keys(value, what)
    return {str(key): _gain_value(item, f"{what}[{key!r}]") for key, item in value.items()}


def _limit(value: object, what: str) -> float | dict[str, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        _string_keys(value, what)
        return {
            str(key): _non_negative_number(item, f"{what}[{key!r}]") for key, item in value.items()
        }
    return _non_negative_number(value, what)


def _names(value: object, what: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ActuationError(f"{what} must be a list of names, got {value!r}")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ActuationError(f"{what} must contain strings, got {item!r}")
        if not item.strip():
            raise ActuationError(f"{what} must not contain empty names, got {value!r}")
        if item != item.strip():
            raise ActuationError(
                f"{what} names must not carry surrounding whitespace, got {item!r}"
            )
        names.append(item)
    names_tuple = tuple(names)
    duplicates = sorted({name for name in names_tuple if names_tuple.count(name) > 1})
    if duplicates:
        raise ActuationError(f"{what} contains duplicate names {duplicates}")
    return names_tuple or None


@dataclass(frozen=True)
class ActuationGroup:
    """One declarative actuation group (neutral payload).

    Exactly one of non-empty ``joints`` or ``joint_set`` addresses the
    group.  ``gains``/``effort_limit``/``velocity_limit``/``command``/``parts``
    are the generic payload consumed by a later law translation; this record
    only validates their neutral shape.
    """

    name: str
    law: str
    joints: tuple[str, ...] | None = None
    joint_set: str | None = None
    gains: Mapping[str, GainValue] = field(default_factory=_empty_gain_map)
    effort_limit: float | Mapping[str, float] | None = None
    velocity_limit: float | Mapping[str, float] | None = None
    command: Mapping[str, JsonValue] = field(default_factory=_empty_command_map)
    parts: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        name = self.name
        if not isinstance(name, str) or not name.strip():
            raise ActuationError(f"actuation group name must be a non-empty string, got {name!r}")
        law = self.law
        if not isinstance(law, str) or not law:
            raise ActuationError(
                f"actuation group {name!r} law must be a non-empty string, got {law!r}"
            )
        if law not in ACTUATOR_REGISTRY:
            raise ActuationError(
                f"actuation group {name!r} law {law!r} is not registered; "
                f"registered laws are {sorted(ACTUATOR_REGISTRY)}"
            )
        joint_set = self.joint_set
        if joint_set is not None:
            if not isinstance(joint_set, str) or not joint_set.strip():
                raise ActuationError(
                    f"actuation group {name!r} joint_set must be a non-empty set "
                    f"name, got {joint_set!r}"
                )
            joint_set = joint_set.strip()
        joints = _names(self.joints, f"actuation group {name!r} joints")
        if (joints is None) == (joint_set is None):
            raise ActuationError(
                f"actuation group {name!r} must declare exactly one of non-empty "
                f"`joints` or `joint_set` (got joints={joints!r}, "
                f"joint_set={joint_set!r})"
            )
        gains = _gain_map(self.gains, f"actuation group {name!r} gains")
        effort_limit = _limit(self.effort_limit, f"actuation group {name!r} effort_limit")
        velocity_limit = _limit(self.velocity_limit, f"actuation group {name!r} velocity_limit")
        if not isinstance(self.command, Mapping):
            raise ActuationError(
                f"actuation group {name!r} command must be a mapping, got " f"{self.command!r}"
            )
        parts = _names(self.parts, f"actuation group {name!r} parts")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "law", law)
        object.__setattr__(self, "joints", joints)
        object.__setattr__(self, "joint_set", joint_set)
        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "effort_limit", effort_limit)
        object.__setattr__(self, "velocity_limit", velocity_limit)
        object.__setattr__(self, "command", dict(self.command))
        object.__setattr__(self, "parts", parts)


@dataclass(frozen=True)
class ResolvedActuationGroup:
    """One actuation group after ``joint_set`` resolution."""

    articulation: str
    group: ActuationGroup
    joint_names: tuple[str, ...]


def normalize_actuation_group(
    entry: ActuationGroup | Mapping[str, JsonValue],
    name: str | None = None,
) -> ActuationGroup:
    """Normalize one group declaration (an :class:`ActuationGroup` passes)."""
    if isinstance(entry, ActuationGroup):
        if name is not None and entry.name != name:
            raise ActuationError(
                f"actuation group {entry.name!r} does not match the declared "
                f"group name {name!r}"
            )
        return entry
    if not isinstance(entry, Mapping):
        raise ActuationError(
            f"actuation groups entries must be mappings, got {type(entry).__name__}"
        )
    values = dict(entry)
    unknown = sorted(set(values) - set(_GROUP_KEYS))
    if unknown:
        raise ActuationError(
            f"actuation group {values.get('name')!r} declares unknown keys "
            f"{unknown}; allowed keys are {list(_GROUP_KEYS)}"
        )
    declared = values.get("name")
    if name is not None:
        if declared is None:
            values["name"] = name
        elif declared != name:
            raise ActuationError(
                f"actuation group has declared name {declared!r} but was " f"normalized as {name!r}"
            )
    elif declared is None:
        raise ActuationError("actuation group entry is missing 'name'")
    # The dataclass __post_init__ validates every field; this boundary cast
    # lets the raw JSON mapping reach that single validator.
    return cast("Callable[..., ActuationGroup]", ActuationGroup)(**values)


def normalize_actuation_groups(
    entries: Iterable[ActuationGroup | Mapping[str, JsonValue]] | None,
) -> tuple[ActuationGroup, ...]:
    """Normalize a group list (unknown keys and duplicate names fail)."""
    if entries is None:
        return ()
    if isinstance(entries, (str, bytes, Mapping)):
        raise ActuationError(
            f"actuation groups must be a list of group records, got {type(entries).__name__}"
        )
    try:
        raw = tuple(entries)
    except TypeError:
        raise ActuationError(
            f"actuation groups must be a list of group records, got {entries!r}"
        ) from None
    groups = tuple(normalize_actuation_group(entry) for entry in raw)
    names = [group.name for group in groups]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ActuationError(f"actuation group names must be unique, got {duplicates}")
    return groups


def validate_actuation_groups(
    groups: Sequence[ActuationGroup],
    *,
    joints: Sequence[str],
    joint_sets: Mapping[str, Sequence[str]] | None = None,
    passive_joints: Iterable[str] = (),
) -> None:
    """Validate group uniqueness and exactly-once joint coverage.

    Every non-passive joint in ``joints`` must be covered by exactly one
    group.  Joint sets resolve through ``joint_sets``; a passive joint may not
    be covered.  An empty group list passes only when there are no driven
    joints.
    """
    entries = tuple(groups)
    for entry in entries:
        if not isinstance(entry, ActuationGroup):
            raise ActuationError(f"actuation groups must be ActuationGroup records, got {entry!r}")
    names = [group.name for group in entries]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ActuationError(f"actuation group names must be unique, got {duplicates}")
    universe = tuple(str(joint) for joint in joints)
    universe_set = set(universe)
    declared_sets: dict[str, tuple[str, ...]] = (
        {}
        if joint_sets is None
        else {
            str(set_name): tuple(str(joint) for joint in members)
            for set_name, members in joint_sets.items()
        }
    )
    known_sets = sorted(declared_sets)
    passive = {str(joint) for joint in passive_joints}
    owner: dict[str, str] = {}
    for group in entries:
        if group.joint_set is not None:
            if group.joint_set not in declared_sets:
                raise ActuationError(
                    f"actuation group {group.name!r} joint_set "
                    f"{group.joint_set!r} is unknown; declared joint sets are "
                    f"{known_sets}"
                )
            resolved = declared_sets[group.joint_set]
            source = f"joint_set {group.joint_set!r}"
        else:
            resolved = tuple(group.joints or ())
            source = "joints"
        unknown = sorted(set(resolved) - universe_set)
        if unknown:
            raise ActuationError(
                f"actuation group {group.name!r} {source} names unknown joints "
                f"{unknown}; known joints are {list(universe)}"
            )
        duplicated_joints = sorted({joint for joint in resolved if resolved.count(joint) > 1})
        if duplicated_joints:
            raise ActuationError(
                f"actuation group {group.name!r} {source} contains duplicate "
                f"joints {duplicated_joints}"
            )
        overlap = sorted(set(resolved) & set(owner))
        if overlap:
            details = ", ".join(f"{joint!r} (also in {owner[joint]!r})" for joint in overlap)
            raise ActuationError(
                f"actuation group {group.name!r} overlaps other groups on "
                f"joints {details}; every joint may be covered by only one group"
            )
        for joint in resolved:
            owner[joint] = group.name
    covered = set(owner)
    passive_covered = sorted(covered & passive)
    if passive_covered:
        raise ActuationError(
            f"actuation groups cover passive joints {passive_covered}; a passive "
            f"joint may not be covered"
        )
    missing = sorted(universe_set - passive - covered)
    if missing:
        raise ActuationError(
            f"actuation groups do not cover joints {missing}; every non-passive "
            f"joint must be covered exactly once"
        )


def resolve_actuation_groups(
    groups: Sequence[ActuationGroup | Mapping[str, JsonValue]] | None,
    *,
    articulation: str,
    joints: Sequence[str],
    joint_sets: Mapping[str, Sequence[str]] | None = None,
    passive_joints: Iterable[str] = (),
) -> tuple[ResolvedActuationGroup, ...]:
    """Resolve ``joint_set`` groups and validate exactly-once coverage."""
    entries = normalize_actuation_groups(groups)
    validate_actuation_groups(
        entries,
        joints=joints,
        joint_sets=joint_sets,
        passive_joints=passive_joints,
    )
    resolved: list[ResolvedActuationGroup] = []
    for group in entries:
        if group.joint_set is not None:
            assert joint_sets is not None
            joint_names = tuple(str(joint) for joint in joint_sets[group.joint_set])
        else:
            joint_names = tuple(str(joint) for joint in (group.joints or ()))
        resolved.append(
            ResolvedActuationGroup(
                articulation=str(articulation),
                group=group,
                joint_names=joint_names,
            )
        )
    return tuple(resolved)


def resolve_group_joints(
    groups: Sequence[ActuationGroup],
    *,
    articulation: str,
    joint_sets: Mapping[str, Sequence[str]] | None = None,
) -> tuple[ResolvedActuationGroup, ...]:
    """Resolve every ``joint_set`` against a vocabulary (no coverage check).

    This is the backend-realizer seam for an articulation whose actuation
    groups cover a subset of the profile's joints (the G1 IsaacLab path drives
    the 29 policy joints of a 43-joint asset).  Unlike
    :func:`resolve_actuation_groups`, it performs no exactly-once coverage
    validation; the run layer owns driven-joint coverage.  A ``joint_set``
    must be declared in ``joint_sets``; explicit ``joints`` pass through.

    Raises:
        ActuationError: A non-group entry or an unknown ``joint_set``.
    """
    entries = tuple(groups)
    for group in entries:
        if not isinstance(group, ActuationGroup):
            raise ActuationError(f"actuation groups must be ActuationGroup records, got {group!r}")
    declared_sets: dict[str, tuple[str, ...]] = (
        {}
        if joint_sets is None
        else {
            str(set_name): tuple(str(joint) for joint in members)
            for set_name, members in joint_sets.items()
        }
    )
    known_sets = sorted(declared_sets)
    resolved: list[ResolvedActuationGroup] = []
    for group in entries:
        if group.joint_set is not None:
            if group.joint_set not in declared_sets:
                raise ActuationError(
                    f"actuation group {group.name!r} joint_set "
                    f"{group.joint_set!r} is unknown; declared joint sets are "
                    f"{known_sets}"
                )
            joint_names = declared_sets[group.joint_set]
        else:
            joint_names = tuple(str(joint) for joint in (group.joints or ()))
        resolved.append(
            ResolvedActuationGroup(
                articulation=str(articulation),
                group=group,
                joint_names=joint_names,
            )
        )
    return tuple(resolved)


def validate_actuation_group(group: ActuationGroup, *, backend: str) -> None:
    """Validate one law payload against a backend's law capability.

    This is pure IR: it never inspects compiled actuator types.
    """
    supported = _BACKEND_LAWS.get(backend)
    if supported is None:
        raise ActuationError(
            f"backend {backend!r} declares no actuation law support; supported "
            f"backends are {sorted(_BACKEND_LAWS)}"
        )
    if group.law not in supported:
        raise ActuationError(
            f"actuation group {group.name!r} law {group.law!r} is not supported "
            f"by backend {backend!r}; supported laws are {sorted(supported)}"
        )
    if group.command:
        raise ActuationError(
            f"actuation group {group.name!r} law {group.law!r} declares a "
            f"non-empty `command`; no current law accepts command parameters"
        )
    allowed = _LAW_GAIN_KEYS[group.law]
    unknown = sorted(set(group.gains) - allowed)
    if unknown:
        raise ActuationError(
            f"actuation group {group.name!r} law {group.law!r} declares unknown "
            f"gains {unknown}; allowed gains are {sorted(allowed)}"
        )
    if group.law in ("standard_pd", "computed_torque_pd"):
        missing = [key for key in ("kp", "kd") if key not in group.gains]
        if missing:
            raise ActuationError(
                f"actuation group {group.name!r} law {group.law!r} is missing "
                f"gains {missing}; declare both kp and kd explicitly"
            )
    if group.law == "standard_pd" and group.parts:
        raise ActuationError(
            f"actuation group {group.name!r} law 'standard_pd' declares `parts` "
            f"{list(group.parts)}; parts are only valid for computed_torque_pd"
        )
    if group.law == "native_servo":
        if group.gains:
            raise ActuationError(
                f"actuation group {group.name!r} law 'native_servo' declares "
                f"gains {sorted(group.gains)}; the servo gain lives in the "
                f"compiled actuator"
            )
        if group.parts:
            raise ActuationError(
                f"actuation group {group.name!r} law 'native_servo' declares "
                f"parts {list(group.parts)}; native_servo has no parts"
            )
    if backend in ("mujoco", "mjwarp"):
        limits = [
            name for name in ("effort_limit", "velocity_limit") if getattr(group, name) is not None
        ]
        if limits:
            raise ActuationError(
                f"actuation group {group.name!r} declares backend-owned limits "
                f"{limits}; the compiled actuator ctrlrange owns limits for "
                f"{backend!r}"
            )


def validate_drive_law_matrix(
    groups: Sequence[ResolvedActuationGroup],
    controls: Iterable[_DriveControl],
) -> None:
    """Validate every control drive against the resolved actuation plan.

    ``position_pd`` needs ``standard_pd`` or ``computed_torque_pd``;
    ``position_servo`` needs ``native_servo``; the direct-drive kinds
    (``torque`` / ``velocity`` / ``effort`` / ``current``) have no D4 law and
    fail with a teaching error.
    """
    law_by_joint: dict[tuple[str, str], str] = {}
    for resolved in groups:
        for joint in resolved.joint_names:
            law_by_joint[(resolved.articulation, str(joint))] = resolved.group.law
    for control in controls:
        if getattr(control, "prop", False):
            continue
        drive = str(control.drive)
        name = str(getattr(control, "name", "<unnamed>"))
        articulation = str(control.articulation)
        if drive in _DIRECT_DRIVES:
            raise ActuationError(
                f"control {name!r} requests drive {drive!r}, which has no D4 "
                f"actuation law; direct torque/velocity/effort/current drives "
                f"are unsupported | hint: declare drive position_pd "
                f"(standard_pd/computed_torque_pd) or position_servo "
                f"(native_servo)"
            )
        if drive not in ("position_pd", "position_servo"):
            raise ActuationError(
                f"control {name!r} requests unknown drive {drive!r}; legal D4 "
                f"drives are position_pd and position_servo"
            )
        for joint in control.joint_names:
            law = law_by_joint.get((articulation, str(joint)))
            if law is None:
                raise ActuationError(
                    f"control {name!r} drives joint {joint!r} of articulation "
                    f"{articulation!r}, which no actuation group covers"
                )
            if drive == "position_pd" and law not in _POSITION_PD_LAWS:
                raise ActuationError(
                    f"control {name!r} requests drive 'position_pd' for joint "
                    f"{joint!r}, but its actuation law is {law!r}; position_pd "
                    f"requires standard_pd or computed_torque_pd"
                )
            if drive == "position_servo" and law != "native_servo":
                raise ActuationError(
                    f"control {name!r} requests drive 'position_servo' for joint "
                    f"{joint!r}, but its actuation law is {law!r}; position_servo "
                    f"requires native_servo"
                )
