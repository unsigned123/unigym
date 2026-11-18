"""Neutral runtime descriptions of articulations and explicit controls.

An *articulation* is ``root + model + joints + spawn``; a *control* is one
explicit drive group over an articulation (the owning adapter, its joints,
the drive/target pair and the backend-specific addressing/strategy).  There
is no "policy" runtime type: a policy is just ``position_pd`` (or
``position_servo``) plus an adapter.

Rules:

* every engine publishes ``env.articulations`` and ``env.controls`` and
  iterates them -- there is no single/multi mode switch;
* drives are explicit (:data:`unigym.runtime.controls.CONTROL_DRIVES`);
  a backend that cannot honour a drive fails fast at construction;
* the legacy policy/multi runtime types and their per-controller surfaces
  are removed; every engine consumes the neutral articulation/control
  runtimes described here.

This module is runtime data plus duck-typed resolution.  It names no engine.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping as AbcMapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import torch

from unigym._typing import JsonValue
from unigym.actuators.ir import ResolvedActuationGroup

from unigym.robots.base import CanonicalBodyId
from unigym.runtime.actions import OrderedActionGroup, OrderedActionSpec, PolicyBinding
from unigym.runtime.controls import (
    CONTROL_DRIVES,
    DRIVE_TARGET,
    ControlError,
    target_for_drive,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from unigym.env.backends.base import EnvBackend
    from unigym.robots.base import RobotProfile
    from unigym.scenes.articulation import (
        ArticulationSpec,
        AssetRef,
        ContactSensorInventory,
    )

__all__ = [
    "ArticulationRuntime",
    "ControlKey",
    "ControlRuntime",
    "DRIVE_STRATEGIES",
    "DriveStrategy",
    "LEGAL_CONTROL_ROLES",
    "ResolvedControl",
    "ResolvedControlPlan",
    "articulation_facts",
    "build_control_plan",
    "build_control_runtimes",
    "control_key",
    "controls_by_key",
    "drive_for_target_laws",
    "single_articulation_runtime",
    "strategy_for_drive",
    "validate_action_coverage",
    "validate_policy_bindings",
]

#: legal explicit body/aux role values on a resolved control row
LEGAL_CONTROL_ROLES = ("body", "aux")

#: The authoritative control identity: ``(articulation, control name)``.
ControlKey = tuple[str, str]


def control_key(control: ControlRuntime) -> ControlKey:
    """The control's authoritative table identity ``(articulation, name)``.

    Two articulations may legitimately declare the same group name (e.g.
    ``go1/legs`` and ``go2/legs``), so a bare name is never an identity.
    """
    return (str(control.articulation), str(control.name))


def controls_by_key(
    controls: Iterable[ControlRuntime],
) -> dict[ControlKey, ControlRuntime]:
    """Index control runtimes by :func:`control_key` (duplicates fail loud)."""
    table: dict[ControlKey, ControlRuntime] = {}
    for control in controls:
        key = control_key(control)
        if key in table:
            raise ControlError(
                f"control identity {key!r} is declared more than once; every "
                f"(articulation, name) pair must be unique"
            )
        table[key] = control
    return table


@dataclass(frozen=True)
class DriveStrategy:
    """Neutral descriptor of one drive law (dispatch metadata only).

    The backend owns the executable implementation; this descriptor is the
    one place the drive -> target mapping and the strategy name live, so the
    run layer never guesses a target kind.
    """

    drive: str
    target: str
    strategy: str


DRIVE_STRATEGIES: dict[str, DriveStrategy] = {
    drive: DriveStrategy(drive=drive, target=target, strategy=drive)
    for drive, target in DRIVE_TARGET.items()
}
"""Every legal drive -> its neutral strategy descriptor."""


def strategy_for_drive(drive: str) -> DriveStrategy:
    """Resolve the neutral strategy of *drive* (fail-fast)."""
    try:
        return DRIVE_STRATEGIES[drive]
    except KeyError:
        raise ControlError(
            f"unknown control drive {drive!r}; legal drives are " f"{list(CONTROL_DRIVES)}"
        ) from None


def _as_q0(values: torch.Tensor | Sequence[float] | None, count: int, what: str) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        q0 = values.to(dtype=torch.float32)
    else:
        # numpy arrays have no single-element truthiness; never use `values or`.
        seq = () if values is None else values
        try:
            q0 = torch.tensor([float(v) for v in seq], dtype=torch.float32)
        except (TypeError, ValueError):
            raise ControlError(f"{what} q0 must be numeric") from None
    if tuple(q0.shape) != (count,):
        raise ControlError(f"{what} q0 has {tuple(q0.shape)} entries for {count} joints")
    if not bool(torch.isfinite(q0).all().item()):
        raise ControlError(f"{what} q0 must be finite, got {q0}")
    return q0


def _empty_body_sets() -> dict[str, tuple[CanonicalBodyId, ...]]:
    """Typed empty default for the runtime body-set mapping."""
    return {}


@dataclass
class ArticulationRuntime:
    """One declared articulation's runtime facts (not a policy).

    Attributes:
        name: articulation name (the ``env.articulations`` key).
        model: the declared :class:`~unigym.scenes.articulation.AssetRef`
            (or ``None`` for profile-backed assets).
        root: ``free`` / ``fixed`` / ``welded``.
        joint_names: the asset hinge joint names (asset order).
        q0: default positions aligned with ``joint_names`` (torch float32).
        foot_bodies: canonical foot body ids (semantic metadata only).
        spawn: root spawn declaration (``{pos, yaw}``; empty when absent).
        base_body: the backend-native base/root body name (model lookups).
        base_body_id: the canonical base body id (public/meta fact).
        body_sets: named canonical body groups.
        contacts: the effective contact sensor inventory (or ``None``).
        addresses: backend-specific address bookkeeping (free joint, base
            body id, ...).
        profile: the :class:`~unigym.robots.base.RobotProfile` when the
            articulation is profile-backed.
    """

    name: str
    model: AssetRef | None = None
    root: str = "free"
    joint_names: tuple[str, ...] = ()
    #: normalized to a torch.Tensor in __post_init__ (accepts torch/numpy/
    #: sequence); kept arbitrary at the dataclass boundary so backends may
    #: hand over their native array without a conversion dance.
    q0: Any = None
    foot_bodies: tuple[str, ...] = ()
    spawn: Mapping = field(default_factory=dict)
    base_body: str = ""
    base_body_id: CanonicalBodyId | None = None
    body_sets: Mapping[str, tuple[CanonicalBodyId, ...]] = field(default_factory=_empty_body_sets)
    contacts: ContactSensorInventory | None = None
    addresses: dict = field(default_factory=dict)
    profile: RobotProfile | None = None
    # Backend-populated MuJoCo-family runtime state (compiled address tables +
    # per-articulation obs catalog).  Declared here so the step/obs faces can
    # read the contract without scattered ``setattr`` unknown-attribute errors;
    # the neutral backend (isaaclab) leaves them at their defaults.
    robot_joint_names: tuple[str, ...] = ()
    jadr_pol: np.ndarray | None = None
    jdr_pol: np.ndarray | None = None
    jadr_art: np.ndarray | None = None
    jdr_art: np.ndarray | None = None
    q0_mj: np.ndarray | None = None
    free_qpos_adr: int | None = None
    free_dof_adr: int | None = None

    def __post_init__(self):
        if not self.name:
            raise ControlError("articulation runtime name must be non-empty")
        self.name = str(self.name)
        self.joint_names = tuple(str(n) for n in (self.joint_names or ()))
        self.foot_bodies = tuple(str(n) for n in (self.foot_bodies or ()))
        spawn: Mapping[str, JsonValue] = self.spawn or {}
        self.spawn = dict(spawn)
        self.body_sets = {
            str(name): tuple(CanonicalBodyId(str(body)) for body in (bodies or ()))
            for name, bodies in (self.body_sets or {}).items()
        }
        addresses: Mapping[str, JsonValue] = self.addresses or {}
        self.addresses = dict(addresses)
        self.q0 = _as_q0(self.q0, len(self.joint_names), f"articulation {self.name!r}")

    def q0_for(self, joint_names: Sequence[str]) -> torch.Tensor:
        """Default positions for *joint_names* (subset order preserved)."""
        assert isinstance(self.q0, torch.Tensor)
        lut = dict(zip(self.joint_names, self.q0.tolist()))
        try:
            return torch.tensor([lut[name] for name in joint_names], dtype=torch.float32)
        except KeyError as exc:
            raise ControlError(
                f"articulation {self.name!r} has no q0 for joint " f"{exc.args[0]!r}"
            ) from None


def articulation_facts(art: ArticulationRuntime, control: ControlRuntime) -> dict[str, JsonValue]:
    """The v5 per-articulation neutral facts shared by meta and summary."""
    return {
        "joint_names": list(control.joint_names),
        "asset_joint_names": list(art.joint_names),
        "action_group": control.name,
        "base_body": None if art.base_body_id is None else str(art.base_body_id),
        "foot_bodies": [str(body) for body in art.foot_bodies],
        "body_sets": {
            set_name: [str(body) for body in bodies] for set_name, bodies in art.body_sets.items()
        },
        "contacts": None if art.contacts is None else art.contacts.to_dict(),
    }


@dataclass
class ControlRuntime:
    """One explicit drive group over an articulation (§3.3).

    Attributes:
        name: the physical control row label within its articulation; with
            :attr:`articulation` it is the ``env.controls`` physical-table
            key.  It is NOT an action identity: runtime action columns are
            addressed by ``(articulation, joint_name)``.
        role: explicit body/aux selection marker (``body`` | ``aux``),
            copied from the owning :class:`OrderedActionGroup`.
        adapter: the adapter instance name that drives this group.
        articulation: the owning articulation name.
        joint_names: joints owned by this group, in action order.
        q0: default positions aligned with ``joint_names`` (torch float32).
        drive: one of :data:`unigym.runtime.controls.CONTROL_DRIVES`, derived
            from the action target plus the resolved actuation law.
        target: the explicit target kind (must match ``drive``).
        control_dt: control period (s) or ``None``.  A derived MIRROR of the
            owning policy binding's ``control_dt`` (the adapter fallback when
            the binding omits it); the binding, never this field, is the
            timing authority.
        command: optional execution parameters / backend hints.
        strategy: the neutral :class:`DriveStrategy` descriptor.
        addresses: backend-specific addressing/strategy state.
        asset_joint_names: the owning articulation's full asset joints.
        required: True when every joint must be addressed by an action entry.
        source: provenance of the resolved joints.
        joint_set: always ``None`` (the action surface declares explicit
            joints; the field remains for the plan projection).
        prop: True for an articulated-prop servo group.
    """

    name: str
    adapter: str
    articulation: str
    joint_names: tuple[str, ...]
    #: normalized to a torch.Tensor in __post_init__ (see ArticulationRuntime)
    q0: Any = None
    drive: str = "position_pd"
    target: str = "position"
    control_dt: float | None = None
    command: Mapping[str, JsonValue] | None = None
    strategy: DriveStrategy | None = None
    addresses: dict = field(default_factory=dict)
    asset_joint_names: tuple[str, ...] = ()
    required: bool = True
    source: str = ""
    joint_set: str | None = None
    prop: bool = False
    role: str = "aux"

    def __post_init__(self):
        for key in ("name", "adapter", "articulation"):
            value = getattr(self, key)
            if not value:
                raise ControlError(f"control runtime {self.name!r} {key} must be non-empty")
            setattr(self, key, str(value))
        if self.role not in LEGAL_CONTROL_ROLES:
            raise ControlError(
                f"control runtime {self.name!r} role must be one of "
                f"{list(LEGAL_CONTROL_ROLES)}, got {self.role!r}"
            )
        if self.drive not in CONTROL_DRIVES:
            raise ControlError(
                f"control runtime {self.name!r} drive must be one of "
                f"{list(CONTROL_DRIVES)}, got {self.drive!r}"
            )
        expected = target_for_drive(self.drive)
        if self.target != expected:
            raise ControlError(
                f"control runtime {self.name!r} drive {self.drive!r} "
                f"requires target {expected!r}, got {self.target!r}"
            )
        self.joint_names = tuple(str(n) for n in (self.joint_names or ()))
        if not self.joint_names:
            raise ControlError(f"control runtime {self.name!r} owns no joints")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ControlError(f"control runtime {self.name!r} joint_names contains " f"duplicates")
        self.q0 = _as_q0(self.q0, len(self.joint_names), f"control {self.name!r}")
        self.asset_joint_names = tuple(str(n) for n in (self.asset_joint_names or ()))
        addresses: Mapping[str, JsonValue] = self.addresses or {}
        self.addresses = dict(addresses)
        self.source = str(self.source or "")
        if self.joint_set is not None:
            self.joint_set = str(self.joint_set)
        if self.strategy is None:
            self.strategy = strategy_for_drive(self.drive)
        if self.command is not None:
            self.command = dict(self.command)

    @property
    def strategy_name(self) -> str:
        """The backend strategy key (== the drive)."""
        if self.strategy is None:
            raise ControlError(f"control runtime {self.name!r} has no resolved strategy")
        return self.strategy.strategy


_POSITION_DRIVES = ("position_pd", "position_servo")


def _profile_defaults(profile: RobotProfile | None) -> dict[str, float]:
    return dict(getattr(profile, "default_joint_pos", {}) or {})


def drive_for_target_laws(
    target: str,
    laws: Iterable[str],
    *,
    what: str,
) -> str:
    """Derive the runtime drive from an action target and actuation laws.

    The action group owns the target kind; the actuation groups own the
    physical law.  The combination is explicit: ``position`` becomes
    ``position_pd`` for ``standard_pd``/``computed_torque_pd`` or
    ``position_servo`` for ``native_servo``; ``velocity`` requires
    ``direct_velocity`` and ``torque`` requires ``direct_torque``.  A group
    spanning incompatible laws (or a target with no native law) fails with a
    teaching error instead of guessing a drive.
    """
    unique = {str(law) for law in laws}
    if not unique:
        raise ControlError(f"{what} has no resolved actuation law")
    if target == "position":
        if unique <= {"standard_pd", "computed_torque_pd"}:
            return "position_pd"
        if unique <= {"native_servo"}:
            return "position_servo"
        raise ControlError(
            f"{what} target 'position' spans incompatible actuation laws "
            f"{sorted(unique)}; a position group must use only "
            f"standard_pd/computed_torque_pd (position_pd) or native_servo "
            f"(position_servo)"
        )
    if target == "velocity":
        if unique <= {"direct_velocity"}:
            return "velocity"
        raise ControlError(
            f"{what} target 'velocity' requires the direct_velocity actuation "
            f"law, got {sorted(unique)}"
        )
    if target == "torque":
        if unique <= {"direct_torque"}:
            return "torque"
        raise ControlError(
            f"{what} target 'torque' requires the direct_torque actuation "
            f"law, got {sorted(unique)}"
        )
    raise ControlError(
        f"{what} target {target!r} has no native direct-drive actuation law; "
        f"supported targets are position, velocity and torque"
    )


def _control_q0(
    joint_names: Sequence[str],
    profile: RobotProfile | None,
    art: ArticulationSpec,
    drive: str,
    what: str,
) -> tuple[float, ...]:
    """Default positions from joint inits, profile defaults, then zero.

    Action groups no longer carry a ``q0`` override: a joint's standby
    position is a joint/profile fact.  An explicit joint declaration wins over
    the profile default.  Only position drives may fall back to zero when
    neither source supplies one.
    """
    defaults = _profile_defaults(profile)
    joint_inits = {
        str(joint.name): float(joint.init) for joint in (getattr(art, "joints", ()) or ())
    }
    values: list[float] = []
    for joint in joint_names:
        if joint in joint_inits:
            values.append(joint_inits[joint])
        elif joint in defaults:
            values.append(float(defaults[joint]))
        elif drive in _POSITION_DRIVES:
            values.append(0.0)
        else:
            raise ControlError(
                f"{what} has no q0 for joint {joint!r} (only position " f"drives may default to 0)"
            )
    return tuple(values)


def _declared_control_runtime(
    group: OrderedActionGroup,
    adapter: str,
    control_dt: float | None,
    arts: Mapping[str, ArticulationSpec],
    profiles: Mapping[str, RobotProfile],
    law_by_joint: Mapping[tuple[str, str], str],
) -> ControlRuntime:
    """Derive one control runtime from action group + actuation law."""
    art = arts.get(group.articulation)
    if art is None:
        raise ControlError(
            f"action group {group.name!r} names articulation "
            f"{group.articulation!r}, which is not declared"
        )
    profile = profiles.get(group.articulation)
    laws: list[str] = []
    for joint in group.joints:
        law = law_by_joint.get((group.articulation, str(joint)))
        if law is None:
            raise ControlError(
                f"action group {group.name!r} drives joint {joint!r} of "
                f"articulation {group.articulation!r}, which no actuation "
                f"group covers"
            )
        laws.append(law)
    what = f"action group {group.name!r} of articulation {group.articulation!r}"
    drive = drive_for_target_laws(group.target, laws, what=what)
    return ControlRuntime(
        name=group.name,
        adapter=adapter,
        articulation=group.articulation,
        joint_names=tuple(group.joints),
        q0=_control_q0(group.joints, profile, art, drive, what),
        drive=drive,
        target=group.target,
        control_dt=control_dt,
        required=True,
        source="action_spec",
        role=group.role,
    )


def _law_by_joint(
    actuation_plans: Mapping[str, Sequence[ResolvedActuationGroup]],
) -> dict[tuple[str, str], str]:
    """``(articulation, joint) -> law`` over the resolved plans."""
    laws: dict[tuple[str, str], str] = {}
    for articulation, resolved in actuation_plans.items():
        for row in resolved:
            for joint in row.joint_names:
                laws[(str(articulation), str(joint))] = row.group.law
    return laws


@dataclass(frozen=True)
class ResolvedControl:
    """One build-time resolved control group (the read-only plan row).

    Every joint name here is a neutral ``unigym_name`` (§2.4): the
    backend name map never appears on this surface.
    """

    name: str
    articulation: str
    adapter: str
    joint_set: str | None
    source: str
    joint_names: tuple[str, ...]
    drive: str
    target: str
    q0: tuple[float, ...]
    role: str = "aux"

    def to_dict(self) -> dict[str, JsonValue]:
        """The row as plain data (inspection / metadata friendly)."""
        return {
            "name": self.name,
            "articulation": self.articulation,
            "adapter": self.adapter,
            "joint_set": self.joint_set,
            "source": self.source,
            "joint_names": list(self.joint_names),
            "drive": self.drive,
            "target": self.target,
            "q0": list(self.q0),
            "role": self.role,
        }


@dataclass(frozen=True)
class ResolvedControlPlan:
    """The frozen build-time control plan (derived from resolved controls)."""

    controls: tuple[ResolvedControl, ...] = ()

    def __iter__(self):
        return iter(self.controls)

    def __len__(self) -> int:
        return len(self.controls)

    def control(self, name: str, articulation: str | None = None) -> ResolvedControl:
        """One resolved control by name (+ optional articulation).

        A bare name is only unambiguous when exactly one declared control
        carries it; two articulations may share a group name, so a multi-match
        bare lookup fails loud instead of returning the first row.
        """
        matches = [
            control
            for control in self.controls
            if control.name == name
            and (articulation is None or control.articulation == articulation)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise KeyError(
                f"resolved plan control name {name!r} is ambiguous; pass "
                f"articulation= (declared identities: "
                f"{[(c.articulation, c.name) for c in matches]})"
            )
        raise KeyError(
            f"resolved plan has no control {name!r}"
            f"{'' if articulation is None else f' of articulation {articulation!r}'}; "
            f"declared: {[(c.articulation, c.name) for c in self.controls]}"
        )

    def for_articulation(self, articulation: str) -> tuple[ResolvedControl, ...]:
        """Every resolved control of one articulation (declaration order)."""
        return tuple(c for c in self.controls if c.articulation == articulation)

    def to_table(self) -> list[dict[str, JsonValue]]:
        """The plan as a list of plain-data rows (read-only projection)."""
        return [control.to_dict() for control in self.controls]


def build_control_plan(
    runtimes: Sequence[ControlRuntime] | None = None,
) -> ResolvedControlPlan:
    """Freeze the resolved-control result into a read-only plan.

    The joints/drive/target/q0/provenance are read from the already resolved
    :class:`ControlRuntime` entries -- the plan is never re-inferred later.
    """
    rows: list[ResolvedControl] = []
    for runtime in tuple(runtimes or ()):
        rows.append(
            ResolvedControl(
                name=runtime.name,
                articulation=runtime.articulation,
                adapter=runtime.adapter,
                joint_set=runtime.joint_set,
                source=runtime.source,
                joint_names=tuple(runtime.joint_names),
                drive=runtime.drive,
                target=runtime.target,
                q0=(
                    tuple(float(v) for v in runtime.q0.tolist())
                    if isinstance(runtime.q0, torch.Tensor)
                    else tuple(float(v) for v in (runtime.q0 or ()))
                ),
                role=runtime.role,
            )
        )
    return ResolvedControlPlan(tuple(rows))


def _actuated_joints(art: ArticulationSpec, profile: RobotProfile | None) -> tuple[str, ...]:
    """The articulation's full actuated joint vocabulary (§4)."""
    if profile is not None:
        sets: Mapping[str, Sequence[str]] = dict(getattr(profile, "joint_sets", None) or {})
        union: list[str] = []
        for joints in sets.values():
            for joint in joints:
                name = str(joint)
                if name not in union:
                    union.append(name)
        return tuple(union)
    return tuple(str(joint.name) for joint in (getattr(art, "joints", ()) or ()))


def validate_policy_bindings(
    groups: Sequence[OrderedActionGroup],
    policies: Sequence[PolicyBinding],
) -> None:
    """Validate policy ownership against the declared action groups.

    Every binding must reference a declared ``(articulation, group)``
    identity, a group may be owned by only one binding, and every declared
    group must be owned by exactly one binding.
    """
    declared = {(group.articulation, group.name) for group in groups}
    owners: dict[tuple[str, str], str] = {}
    for binding in policies:
        for identity in binding.groups:
            if identity not in declared:
                raise ControlError(
                    f"policy binding adapter {binding.adapter!r} references "
                    f"undeclared action group {identity!r}; declared groups: "
                    f"{sorted(declared)}"
                )
            previous = owners.get(identity)
            if previous is not None:
                raise ControlError(
                    f"action group {identity!r} is owned by two policies: "
                    f"{previous!r} and {binding.adapter!r}"
                )
            owners[identity] = binding.adapter
    missing = sorted(declared - set(owners))
    if missing:
        raise ControlError(
            f"action group(s) {missing} are not owned by any policy binding; "
            f"declare a `policies:` entry naming every action group"
        )


def validate_action_coverage(
    articulations: Iterable[ArticulationSpec] | None,
    profiles: AbcMapping[str, RobotProfile] | None,
    action_spec: OrderedActionSpec,
) -> None:
    """Validate the joint-authoritative action coverage of every articulation (§4).

    Driven joints come from the authored :class:`OrderedActionSpec` groups,
    never from a control row label used as an action identity.  For EVERY
    declared articulation (including ones with no action group):

    * ``actuated`` = union of the profile ``joint_sets`` (profile-backed)
      or ``ArticulationSpec.joints`` (profile-less);
    * ``driven`` = union of the authored action groups' joints;
    * ``passive`` = ``ArticulationSpec.passive_joints``;
    * every action group of one articulation must be disjoint;
    * ``driven`` and ``passive`` must be disjoint;
    * ``driven | passive`` must equal ``actuated``.

    An articulation with zero action groups is legal only when every actuated
    joint is listed in ``passive_joints``; an omitted profile-backed
    articulation therefore fails instead of building silently unactuated.

    Group names appear only as provenance ("source group"); the failing
    identity is always the articulation plus joint names.

    Raises:
        ControlError: with the articulation name and the missing / overlap /
            passive-conflict / unknown joint names.
    """
    arts = {art.name: art for art in tuple(articulations or ())}
    profiles = dict(profiles or {})

    driven: dict[str, list[str]] = {}
    provenance: dict[tuple[str, str], str] = {}
    for group in action_spec.groups:
        for joint in group.joints:
            key = (group.articulation, joint)
            previous = provenance.get(key)
            if previous is not None:
                raise ControlError(
                    f"articulation {group.articulation!r}: joint {joint!r} is "
                    f"driven by more than one action group "
                    f"(source groups: {previous!r}, {group.name!r})"
                )
            provenance[key] = group.name
        driven.setdefault(group.articulation, []).extend(group.joints)

    for art in arts.values():
        name = art.name
        profile = profiles.get(name)
        actuated = set(_actuated_joints(art, profile))
        passive = set(getattr(art, "passive_joints", ()) or ())
        driven_set = set(driven.get(name, ()))
        conflict = sorted(driven_set & passive)
        if conflict:
            raise ControlError(
                f"articulation {name!r}: joints {conflict} are both driven "
                f"by an action group and listed in passive_joints"
            )
        if not actuated:
            # profile-less articulation without declared joints: the actuated
            # vocabulary is not knowable here; overlap is still enforced.
            continue
        covered = driven_set | passive
        missing = sorted(actuated - covered)
        if missing:
            suffix = (
                ""
                if driven_set
                else (
                    f"; articulation {name!r} declares no action group, so "
                    f"list them in `passive_joints` or bind an action group"
                )
            )
            raise ControlError(
                f"articulation {name!r}: actuated joints {missing} are "
                f"neither driven by an action group nor listed in "
                f"passive_joints; declare the drive explicitly{suffix}"
            )
        unknown = sorted(covered - actuated)
        if unknown:
            raise ControlError(
                f"articulation {name!r}: joints {unknown} are not in the "
                f"articulation's actuated joint vocabulary (profile "
                f"joint_sets or ArticulationSpec.joints)"
            )


def build_control_runtimes(
    action_spec: OrderedActionSpec,
    policies: Sequence[PolicyBinding],
    articulations: Sequence[ArticulationSpec] | None,
    profiles: Mapping[str, RobotProfile] | None,
    actuation_plans: Mapping[str, Sequence[ResolvedActuationGroup]],
    *,
    default_control_dt: float | None = None,
) -> tuple[ControlRuntime, ...]:
    """Derive neutral :class:`ControlRuntime` rows for one action spec.

    Each :class:`OrderedActionGroup` contributes one physical control row; its
    table key is ``(articulation, control name)``.  The action identity is the
    joint, not the row label.  The adapter comes from the owning
    :class:`PolicyBinding`; the drive is derived from the group target plus
    the resolved actuation law, and ``q0`` comes from joint/profile defaults.
    ``control_dt`` mirrors the binding's declared period, falling back to
    ``default_control_dt`` (the adapter's period) when the binding omits it.
    There is no group-name inference and no implicit extra group.

    Args:
        action_spec: the run's ordered action groups (``RunSpec.action_spec``).
        policies: the run's policy bindings (``RunSpec.policies``).
        articulations: normalized ``ArticulationSpec`` entries.
        profiles: ``{articulation name: RobotProfile}`` for profile-backed
            (robot) articulations.
        actuation_plans: resolved actuation groups per articulation; their
            laws make the target -> drive derivation explicit.
        default_control_dt: fallback period (the adapter ``control_dt``) for
            bindings that declare none.

    Returns:
        The control runtimes in action-spec declaration order.
    """
    arts = {art.name: art for art in tuple(articulations or ())}
    profiles = dict(profiles or {})
    tables = {str(name): tuple(rows) for name, rows in actuation_plans.items()}
    validate_policy_bindings(action_spec.groups, policies)
    owners: dict[tuple[str, str], PolicyBinding] = {}
    for binding in policies:
        for identity in binding.groups:
            owners[identity] = binding
    laws = _law_by_joint(tables)
    rows: list[ControlRuntime] = []
    for group in action_spec.groups:
        binding = owners[(group.articulation, group.name)]
        control_dt = binding.control_dt if binding.control_dt is not None else default_control_dt
        rows.append(
            _declared_control_runtime(
                group,
                binding.adapter,
                control_dt,
                arts,
                profiles,
                laws,
            )
        )
    out = tuple(rows)
    validate_action_coverage(arts.values(), profiles, action_spec)
    return out


def single_articulation_runtime(env: EnvBackend) -> ArticulationRuntime:
    """The one articulation runtime (a cardinality fact, no primary rank)."""
    arts: Mapping[str, ArticulationRuntime] = getattr(env, "articulations", None) or {}
    if len(arts) != 1:
        raise RuntimeError(f"expected exactly one articulation, found {sorted(arts)}")
    return next(iter(arts.values()))
