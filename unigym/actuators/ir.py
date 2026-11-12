"""Neutral actuation IR: declarative groups and pure coverage validation.

An :class:`ActuationGroup` declares how one group of joints is physically
driven.  It is authoring data only: ``law`` must name a registered
:data:`unigym.actuators.base.ACTUATOR_REGISTRY` law, but this layer does not
translate gains, limits or parts into a law-specific config (that capability
work belongs to the backend translation step).

``controls[]`` (action faces / target ownership) and actuation groups are
independent; :func:`validate_actuation_groups` only proves group addressing
and one-group-per-joint coverage.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from unigym._typing import JsonValue

from .base import ACTUATOR_REGISTRY

__all__ = ["ActuationError", "ActuationGroup", "validate_actuation_groups"]


class ActuationError(ValueError):
    """A malformed actuation group or coverage declaration (fail-fast)."""


def _empty_float_map() -> dict[str, float]:
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


def _float_map(value: object, what: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ActuationError(f"{what} must be a mapping, got {value!r}")
    _string_keys(value, what)
    return {str(key): _finite_number(item, f"{what}[{key!r}]") for key, item in value.items()}


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
    gains: Mapping[str, float] = field(default_factory=_empty_float_map)
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
        gains = _float_map(self.gains, f"actuation group {name!r} gains")
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
