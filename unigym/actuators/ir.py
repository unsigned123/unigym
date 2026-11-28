"""Neutral actuation IR: declarative groups, resolution and law validation.

An :class:`ActuationGroup` declares how one group of joints is physically
driven.  It is authoring data only: ``law`` must name a registered
:data:`unigym.actuators.base.ACTUATOR_REGISTRY` law, but this layer does not
translate gains, limits or parts into a law-specific config (that capability
work belongs to the backend translation step).

``action_spec`` groups (action target ownership) and actuation groups are
independent; :func:`validate_actuation_groups` only proves group addressing
and one-group-per-joint coverage, while :func:`validate_drive_law_matrix`
proves that each derived control drive is served by a compatible law.

The neutral law/drive matrix is explicit: ``position_pd`` -> ``standard_pd``
or ``computed_torque_pd``, ``position_servo`` -> ``native_servo``,
``torque`` -> ``direct_torque`` and ``velocity`` -> ``direct_velocity``.
``effort`` and ``current`` have no direct-drive law and fail with a teaching
error.  :data:`BACKEND_LAW_CAPABILITIES` is the backend capability matrix;
a law absent from a backend's set is rejected, never approximated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, cast

from unigym._typing import JsonValue
from unigym.actuator_law import ACTUATOR_LAW_REGISTRY, ActuatorLawDescriptor

from . import descriptors as _descriptors  # noqa: F401  (registers core laws)

__all__ = [
    "BACKEND_LAW_CAPABILITIES",
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
    "params",
)

_UNSERVED_DRIVES = frozenset({"effort", "current"})


def _canonical_backends() -> frozenset[str]:
    """The backend vocabulary used to derive the compatibility capability view."""
    from unigym.capabilities.traits import CANONICAL_BACKENDS

    return frozenset(CANONICAL_BACKENDS)


def _derive_backend_capabilities() -> dict[str, frozenset[str]]:
    """Derive ``backend -> laws`` from the law descriptors."""
    capabilities: dict[str, frozenset[str]] = {}
    for backend in _canonical_backends():
        capabilities[backend] = frozenset(
            name
            for name, descriptor in ACTUATOR_LAW_REGISTRY.items()
            if descriptor.backends is None or backend in descriptor.backends
        )
    return capabilities


def _derive_drive_requirements() -> dict[str, tuple[str, ...]]:
    """Derive ``drive -> laws`` from the law descriptors (registration order)."""
    requirements: dict[str, set[str]] = {}
    for name, descriptor in ACTUATOR_LAW_REGISTRY.items():
        for drive in descriptor.drives:
            requirements.setdefault(drive, set()).add(name)
    index = {name: position for position, name in enumerate(ACTUATOR_LAW_REGISTRY)}
    return {
        drive: tuple(sorted(laws, key=index.__getitem__)) for drive, laws in requirements.items()
    }


#: Backend -> the actuation laws that backend implements natively.  Derived
#: from the law descriptors; kept as the compatibility view IR consumers use.
BACKEND_LAW_CAPABILITIES: dict[str, frozenset[str]] = _derive_backend_capabilities()


def _backend_supported(backend: str) -> frozenset[str] | None:
    """The live law set for *backend* (pack laws included)."""
    if backend not in BACKEND_LAW_CAPABILITIES:
        return None
    return frozenset(
        name
        for name, descriptor in ACTUATOR_LAW_REGISTRY.items()
        if descriptor.backends is None or backend in descriptor.backends
    )


def _descriptor(law: str) -> ActuatorLawDescriptor:
    try:
        return ACTUATOR_LAW_REGISTRY[law]
    except KeyError:
        raise ActuationError(
            f"actuation law {law!r} is not registered; registered laws are "
            f"{sorted(ACTUATOR_LAW_REGISTRY)}"
        ) from None


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


def _freeze_param_value(value: object, what: str) -> JsonValue:
    """Freeze one JSON-like law parameter value (no arbitrary objects)."""
    if isinstance(value, Mapping):
        _string_keys(value, what)
        return MappingProxyType(
            {str(key): _freeze_param_value(item, f"{what}.{key}") for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_param_value(item, what) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ActuationError(
        f"{what} must be JSON data (scalar, list or string-keyed mapping), "
        f"got {type(value).__name__}"
    )


def _param_map(value: object, what: str) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ActuationError(f"{what} must be a mapping, got {value!r}")
    _string_keys(value, what)
    return {str(key): _freeze_param_value(item, f"{what}[{key!r}]") for key, item in value.items()}


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
    params: Mapping[str, JsonValue] = field(default_factory=_empty_command_map)

    def __post_init__(self) -> None:
        name = self.name
        if not isinstance(name, str) or not name.strip():
            raise ActuationError(f"actuation group name must be a non-empty string, got {name!r}")
        law = self.law
        if not isinstance(law, str) or not law:
            raise ActuationError(
                f"actuation group {name!r} law must be a non-empty string, got {law!r}"
            )
        if law not in ACTUATOR_LAW_REGISTRY:
            raise ActuationError(
                f"actuation group {name!r} law {law!r} is not registered; "
                f"registered laws are {sorted(ACTUATOR_LAW_REGISTRY)}"
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
        params = _param_map(self.params, f"actuation group {name!r} params")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "law", law)
        object.__setattr__(self, "joints", joints)
        object.__setattr__(self, "joint_set", joint_set)
        object.__setattr__(self, "gains", gains)
        object.__setattr__(self, "effort_limit", effort_limit)
        object.__setattr__(self, "velocity_limit", velocity_limit)
        object.__setattr__(self, "command", dict(self.command))
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "params", MappingProxyType(params))


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
    supported = _backend_supported(backend)
    if supported is None:
        raise ActuationError(
            f"backend {backend!r} declares no actuation law support; supported "
            f"backends are {sorted(BACKEND_LAW_CAPABILITIES)}"
        )
    if group.law not in supported:
        raise ActuationError(
            f"actuation group {group.name!r} law {group.law!r} is not supported "
            f"by backend {backend!r}; supported laws are {sorted(supported)}"
        )
    descriptor = _descriptor(group.law)
    unknown_params = sorted(set(group.params) - set(descriptor.config_keys))
    if unknown_params:
        raise ActuationError(
            f"actuation group {group.name!r} law {group.law!r} declares unknown "
            f"params {unknown_params}; allowed params are "
            f"{sorted(descriptor.config_keys)}"
        )
    if group.command and not descriptor.allows_command:
        raise ActuationError(
            f"actuation group {group.name!r} law {group.law!r} declares a "
            f"non-empty `command`; no current law accepts command parameters"
        )
    allowed = descriptor.gain_keys
    unknown = sorted(set(group.gains) - allowed)
    if unknown:
        raise ActuationError(
            f"actuation group {group.name!r} law {group.law!r} declares unknown "
            f"gains {unknown}; allowed gains are {sorted(allowed)}"
        )
    if descriptor.required_gains:
        missing = [
            key
            for key in descriptor.required_gains
            if key not in group.gains and key not in group.params
        ]
        if missing:
            raise ActuationError(
                f"actuation group {group.name!r} law {group.law!r} is missing "
                f"gains {missing}; declare them explicitly in `gains` or `params`"
            )
    if group.parts and not descriptor.allows_parts:
        if group.law in ("direct_torque", "direct_velocity"):
            raise ActuationError(
                f"actuation group {group.name!r} law {group.law!r} declares "
                f"parts {list(group.parts)}; a direct-drive law has no parts"
            )
        if group.law == "native_servo":
            raise ActuationError(
                f"actuation group {group.name!r} law 'native_servo' declares "
                f"parts {list(group.parts)}; native_servo has no parts"
            )
        raise ActuationError(
            f"actuation group {group.name!r} law {group.law!r} declares `parts` "
            f"{list(group.parts)}; parts are only valid for computed_torque_pd"
        )
    if group.law == "native_servo" and group.gains:
        raise ActuationError(
            f"actuation group {group.name!r} law 'native_servo' declares "
            f"gains {sorted(group.gains)}; the servo gain lives in the "
            f"compiled actuator"
        )
    if backend in descriptor.backend_owned_limits:
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

    The matrix is explicit: ``position_pd`` needs ``standard_pd`` or
    ``computed_torque_pd``; ``position_servo`` needs ``native_servo``;
    ``torque`` needs ``direct_torque``; ``velocity`` needs
    ``direct_velocity``.  ``effort`` and ``current`` have no native
    direct-drive law on any registered backend and fail with a teaching
    error.
    """
    drive_laws = _derive_drive_requirements()
    known_drives = frozenset(drive_laws) | _UNSERVED_DRIVES
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
        if drive not in known_drives:
            raise ActuationError(
                f"control {name!r} requests unknown drive {drive!r}; legal "
                f"drives are {sorted(known_drives)}"
            )
        if drive in _UNSERVED_DRIVES:
            raise ActuationError(
                f"control {name!r} requests drive {drive!r}, which has no native "
                f"direct-drive actuation law on any registered backend; "
                f"supported direct drives are torque (direct_torque) and "
                f"velocity (direct_velocity)"
            )
        required = drive_laws[drive]
        label = " or ".join(required)
        for joint in control.joint_names:
            law = law_by_joint.get((articulation, str(joint)))
            if law is None:
                raise ActuationError(
                    f"control {name!r} drives joint {joint!r} of articulation "
                    f"{articulation!r}, which no actuation group covers"
                )
            if law not in required:
                raise ActuationError(
                    f"control {name!r} requests drive {drive!r} for joint "
                    f"{joint!r}, but its actuation law is {law!r}; drive "
                    f"{drive!r} requires {label}"
                )
