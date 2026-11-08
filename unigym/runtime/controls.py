"""Explicit control declarations (v104).

A run declares *how* each joint is driven in a top-level ``controls:`` list.
Every entry becomes one :class:`ControlGroup`: the action face (its ``name``),
the owning adapter, the articulation it belongs to, the joints it owns and an
explicit ``drive`` / ``target`` pair.

This module is pure data plus validation: it imports only the standard
library and the neutral articulation IR.  No engine or backend module is
named here (the runtime/backends consume these declarations in V104.2).

Drive semantics (owner ruling §3.2):

* ``position_pd``      -- software PD; ``target = position`` (absolute angle)
* ``position_servo``   -- engine position actuator; ``target = position``
* ``torque``           -- torque target; ``target = torque``
* ``velocity``         -- velocity target; ``target = velocity``
* ``effort``           -- effort/force magnitude; ``target = effort``
* ``current``          -- current target; ``target = current``

Joints that are not driven must be listed in the owning articulation's
``passive_joints``; coverage is validated here at construction time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "CONTROL_DRIVES",
    "CONTROL_TARGETS",
    "ControlError",
    "ControlGroup",
    "DRIVE_TARGET",
    "normalize_control",
    "normalize_controls",
    "project_group",
    "target_for_drive",
    "validate_controls",
]

CONTROL_DRIVES = (
    "position_pd",
    "position_servo",
    "torque",
    "velocity",
    "effort",
    "current",
)
"""The closed drive set (``passive`` is not a group -- use passive_joints)."""

CONTROL_TARGETS = ("position", "velocity", "torque", "effort", "current")
"""The closed ``NamedAction.kind`` / ``ControlGroup.target`` set."""

DRIVE_TARGET = {
    "position_pd": "position",
    "position_servo": "position",
    "torque": "torque",
    "velocity": "velocity",
    "effort": "effort",
    "current": "current",
}
"""The one legal target kind per drive (explicit, never inferred)."""

_CONTROL_KEYS = ("name", "adapter", "articulation", "joint_set", "joints",
                 "drive", "target", "q0", "gains", "command", "control_dt")


class ControlError(ValueError):
    """A malformed control declaration (fail-fast, no alias)."""


def target_for_drive(drive: str) -> str:
    """The explicit target kind mapped from *drive* (fail-fast)."""
    try:
        return DRIVE_TARGET[drive]
    except KeyError:
        raise ControlError(
            f"unknown control drive {drive!r}; legal drives are "
            f"{list(CONTROL_DRIVES)} (passive joints are declared in the "
            f"articulation's `passive_joints`, not as a group)") from None


def _finite(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ControlError(f"{what} must be a number, got {value!r}")
    if not math.isfinite(number):
        raise ControlError(f"{what} must be finite, got {value!r}")
    return number


def _strings(value: Any, what: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or isinstance(value, Mapping):
        raise ControlError(
            f"{what} must be a list of joint names, got {value!r}")
    try:
        names = tuple(str(name) for name in value)
    except TypeError:
        raise ControlError(
            f"{what} must be a list of joint names, got {value!r}") from None
    if len(set(names)) != len(names):
        raise ControlError(f"{what} contains duplicate names {names!r}")
    return names


@dataclass(frozen=True)
class ControlGroup:
    """One explicit drive face over one articulation (v105 §2.2).

    Exactly one of ``joint_set`` / non-empty ``joints`` must be declared.

    Attributes:
        name: the action face and ``NamedAction`` key.
        adapter: the adapter instance that drives this group.
        articulation: the articulation this group belongs to (one only, MVP).
        joint_set: a named joint set of the articulation's ``RobotRecipe``
            (e.g. ``body`` / ``hand``).  Requires a recipe-backed
            articulation; build resolves it exactly once.
        joints: explicit joint names (recipe-backed OR recipe-less
            articulation).  Exactly one of ``joint_set`` / ``joints``.
        drive: one of :data:`CONTROL_DRIVES`.
        target: one of :data:`CONTROL_TARGETS`; must match ``drive``.
        q0: optional per-joint default positions (overrides ``JointSpec.init``
            and the recipe defaults).
        gains: optional per-joint / group gains.
        command: optional execution parameters (e.g. servo parameters).
        control_dt: optional control period (s); all groups of one adapter
            must agree (absent values inherit).
    """

    name: str
    adapter: str
    articulation: str
    drive: str
    target: str
    joint_set: str | None = None
    joints: tuple[str, ...] | None = None
    q0: Mapping[str, float] = field(default_factory=dict)
    gains: Mapping[str, float] = field(default_factory=dict)
    command: Mapping | None = None
    control_dt: float | None = None

    def __post_init__(self):
        for key in ("name", "adapter", "articulation"):
            value = getattr(self, key)
            if not value:
                raise ControlError(
                    f"control group {self.name!r} {key} must be non-empty")
            object.__setattr__(self, key, str(value))
        if self.drive not in CONTROL_DRIVES:
            raise ControlError(
                f"control group {self.name!r} drive must be one of "
                f"{list(CONTROL_DRIVES)}, got {self.drive!r}")
        if self.target not in CONTROL_TARGETS:
            raise ControlError(
                f"control group {self.name!r} target must be one of "
                f"{list(CONTROL_TARGETS)}, got {self.target!r}")
        expected = target_for_drive(self.drive)
        if self.target != expected:
            raise ControlError(
                f"control group {self.name!r} declares drive {self.drive!r} "
                f"with target {self.target!r}; drive {self.drive!r} requires "
                f"target {expected!r}")
        joint_set = self.joint_set
        if joint_set is not None:
            if not isinstance(joint_set, str) or not joint_set.strip():
                raise ControlError(
                    f"control group {self.name!r} joint_set must be a "
                    f"non-empty set name, got {joint_set!r}")
            joint_set = joint_set.strip()
        object.__setattr__(self, "joint_set", joint_set)
        joints = (None if self.joints is None else _strings(
            self.joints, f"control group {self.name!r} joints"))
        if joints == ():
            joints = None
        if (joint_set is None) == (joints is None):
            raise ControlError(
                f"control group {self.name!r} must declare exactly one of "
                f"`joint_set` or non-empty `joints` (got "
                f"joint_set={joint_set!r}, joints={joints!r}); both set, "
                f"both empty and `joints: []` are rejected")
        object.__setattr__(self, "joints", joints)
        object.__setattr__(self, "q0", {
            str(name): _finite(value, f"control group {self.name!r} "
                                      f"q0[{name!r}]")
            for name, value in dict(self.q0 or {}).items()})
        object.__setattr__(self, "gains", {
            str(name): _finite(value, f"control group {self.name!r} "
                                      f"gains[{name!r}]")
            for name, value in dict(self.gains or {}).items()})
        if self.command is not None:
            if not isinstance(self.command, Mapping):
                raise ControlError(
                    f"control group {self.name!r} command must be a mapping, "
                    f"got {self.command!r}")
            object.__setattr__(self, "command", dict(self.command))
        if self.control_dt is not None:
            dt = _finite(self.control_dt,
                         f"control group {self.name!r} control_dt")
            if dt <= 0:
                raise ControlError(
                    f"control group {self.name!r} control_dt must be > 0, "
                    f"got {dt!r}")
            object.__setattr__(self, "control_dt", dt)

    @classmethod
    def from_mapping(cls, entry: Mapping) -> "ControlGroup":
        """Normalize one ``controls[]`` entry."""
        if isinstance(entry, cls):
            return entry
        if not isinstance(entry, Mapping):
            raise ControlError(
                f"controls entries must be mappings, got {entry!r}")
        unknown = sorted(set(entry) - set(_CONTROL_KEYS))
        if unknown:
            raise ControlError(
                f"control group {entry.get('name')!r} declares unknown keys "
                f"{unknown}; allowed keys are {list(_CONTROL_KEYS)}")
        for required in ("name", "adapter", "articulation", "drive", "target"):
            if required not in entry:
                raise ControlError(
                    f"control group {entry.get('name')!r} is missing the "
                    f"required key {required!r}; every group declares "
                    f"name/adapter/articulation/drive/target")
        return cls(
            name=entry["name"],
            adapter=entry["adapter"],
            articulation=entry["articulation"],
            drive=entry["drive"],
            target=entry["target"],
            joint_set=entry.get("joint_set"),
            joints=entry.get("joints"),
            q0=dict(entry.get("q0") or {}),
            gains=dict(entry.get("gains") or {}),
            command=(dict(entry["command"])
                     if entry.get("command") is not None else None),
            control_dt=entry.get("control_dt"),
        )


def normalize_control(entry: Any) -> ControlGroup:
    """Normalize one ``controls[]`` entry (a :class:`ControlGroup` passes)."""
    return ControlGroup.from_mapping(entry)


def normalize_controls(entries: Any) -> tuple[ControlGroup, ...]:
    """Normalize the run-level ``controls:`` list (unique group names)."""
    if entries is None:
        return ()
    if isinstance(entries, Mapping) or isinstance(entries, (str, bytes)):
        raise ControlError(
            "controls must be a list of group mappings, got "
            f"{type(entries).__name__}")
    try:
        raw = tuple(entries)
    except TypeError:
        raise ControlError(
            "controls must be a list of group mappings, got "
            f"{entries!r}") from None
    groups = tuple(normalize_control(entry) for entry in raw)
    names = [group.name for group in groups]
    if len(set(names)) != len(names):
        raise ControlError(
            f"control group names must be unique, got {names!r}")
    return groups


def _passive_map(articulations, passive_joints_by_articulation) -> dict:
    if passive_joints_by_articulation is not None:
        return {str(name): tuple(str(j) for j in (joints or ()))
                for name, joints in dict(
                    passive_joints_by_articulation).items()}
    return {art.name: tuple(getattr(art, "passive_joints", ()) or ())
            for art in articulations or ()}


def validate_controls(articulations, controls,
                      passive_joints_by_articulation=None) -> None:
    """Validate the explicit control coverage of one run (v104 §3.2/§5).

    Rules:
      * a group's articulation must be declared;
      * when the articulation declares its joints, every group joint must
        exist on it;
      * groups of the same articulation must not overlap;
      * one adapter may own several groups, one group owns one articulation;
      * every actuated joint of a declared articulation is covered by exactly
        one group or listed in its ``passive_joints``;
      * all groups of one adapter agree on ``control_dt`` (absent inherits).

    ``joint_set`` groups are resolved by the recipe at build time, so their
    concrete joints are not known here; joint-level checks skip them (the
    build-time resolution validates the set name and coverage in V105.2).
    Recipe-derived articulations (no declared ``joints``) cannot be checked
    for coverage here either.

    Raises:
        ControlError: any rule above is violated.
    """
    controls = tuple(controls or ())
    arts = {art.name: art for art in (articulations or ())}

    names = [group.name for group in controls]
    if len(set(names)) != len(names):
        raise ControlError(
            f"control group names must be unique, got {names!r}")

    driven: dict[str, list[str]] = {}
    unresolved: set[str] = set()
    for group in controls:
        art = arts.get(group.articulation)
        if art is None:
            raise ControlError(
                f"control group {group.name!r} names articulation "
                f"{group.articulation!r}, which is not declared; declared "
                f"articulations: {sorted(arts)}")
        declared = {joint.name for joint in getattr(art, "joints", ()) or ()}
        group_joints = tuple(group.joints or ())
        if group.joints is None and group.joint_set is not None:
            # expanded once at build time; joint-level checks cannot run here
            unresolved.add(art.name)
        if declared and group_joints:
            unknown = sorted(set(group_joints) - declared)
            if unknown:
                raise ControlError(
                    f"control group {group.name!r} names joints {unknown} "
                    f"that are not declared on articulation "
                    f"{art.name!r}; declared joints: {sorted(declared)}")
        owned = driven.setdefault(art.name, [])
        if group_joints:
            overlap = sorted(set(owned) & set(group_joints))
            if overlap:
                raise ControlError(
                    f"control groups of articulation {art.name!r} overlap on "
                    f"joints {overlap} (group {group.name!r}); every driven "
                    f"joint must belong to exactly one group")
            owned.extend(group_joints)

    # control_dt consistency per adapter (MVP)
    dts: dict[str, tuple[str, float]] = {}
    for group in controls:
        if group.control_dt is None:
            continue
        previous = dts.get(group.adapter)
        if previous is not None and not math.isclose(
                previous[1], group.control_dt, rel_tol=0.0, abs_tol=1e-9):
            raise ControlError(
                f"adapter {group.adapter!r} declares inconsistent "
                f"control_dt: group {previous[0]!r} = {previous[1]!r}s vs "
                f"group {group.name!r} = {group.control_dt!r}s; all groups "
                f"of one adapter must agree (absent values inherit)")
        dts.setdefault(group.adapter, (group.name, group.control_dt))

    passive = _passive_map(articulations, passive_joints_by_articulation)
    for art in arts.values():
        declared = {joint.name for joint in getattr(art, "joints", ()) or ()}
        if not declared:
            continue
        art_passive = set(passive.get(art.name, ()))
        unknown_passive = sorted(art_passive - declared)
        if unknown_passive:
            raise ControlError(
                f"articulation {art.name!r} passive_joints "
                f"{unknown_passive} are not declared joints")
        if art.name in unresolved:
            # a joint_set group is resolved by the recipe at build time;
            # coverage is validated there (V105.2), not against a partial
            # driven set
            continue
        actuated = declared - art_passive
        owned = set(driven.get(art.name, ()))
        missing = sorted(actuated - owned)
        extra = sorted(owned - actuated)
        if missing:
            raise ControlError(
                f"articulation {art.name!r} actuated joints {missing} are "
                f"neither covered by a controls[] group nor listed in "
                f"passive_joints; declare the drive explicitly")
        if extra:
            raise ControlError(
                f"articulation {art.name!r} control groups drive joints "
                f"{extra} that are declared passive -- a joint cannot be "
                f"both driven and passive")


def project_group(group: ControlGroup, *, defaults: Mapping | None = None,
                  prefix: str | None = None) -> dict:
    """Project a group into the runtime-facing order/q0 (engine-neutral).

    Args:
        group: the frozen :class:`ControlGroup`.
        defaults: optional recipe defaults (``{joint: q0}``) used when the
            group does not override a joint.
        prefix: optional composition prefix applied to the joint names.

    Returns:
        A plain dict ``{name, action_face, adapter, articulation,
        joint_names, q0, gains, drive, target}``; ``q0`` is a tuple aligned
        with ``joint_names``.

    ``joint_set`` groups must be resolved upstream (build expands the set
    against the recipe before projecting); this neutral helper only projects
    explicit ``joints``.
    """
    if group.joints is None:
        raise ControlError(
            f"control group {group.name!r} selects joint_set "
            f"{group.joint_set!r}; resolve it against the articulation's "
            f"RobotRecipe before projecting (joints are not known here)")
    base = dict(defaults or {})
    base.update(group.q0)
    q0 = []
    for joint in group.joints:
        if joint not in base:
            raise ControlError(
                f"control group {group.name!r} has no q0 for joint "
                f"{joint!r}; declare control.q0 or extend the recipe")
        q0.append(_finite(base[joint], f"control group {group.name!r} "
                                       f"q0[{joint!r}]"))
    names = tuple(f"{prefix}{joint}" if prefix and not joint.startswith(prefix)
                  else joint for joint in group.joints)
    return {
        "name": group.name,
        "action_face": group.name,
        "adapter": group.adapter,
        "articulation": group.articulation,
        "joint_names": names,
        "q0": tuple(q0),
        "gains": dict(group.gains),
        "drive": group.drive,
        "target": group.target,
    }
