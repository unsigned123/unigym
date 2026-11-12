"""Neutral runtime descriptions of articulations and explicit controls.

An *articulation* is ``root + model + joints + spawn``; a *control* is one
explicit drive face over an articulation (the owning adapter, its joints,
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

from unigym.runtime.controls import (
    CONTROL_DRIVES,
    DRIVE_TARGET,
    ControlError,
    target_for_drive,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from unigym.adapters.base import Adapter
    from unigym.env.backends.base import EnvBackend
    from unigym.env.obs_catalog import ObsCatalog
    from unigym.robots.base import RobotProfile
    from unigym.runtime.controls import ControlGroup
    from unigym.scenes.articulation import ArticulationSpec, AssetRef

__all__ = [
    "ArticulationRuntime",
    "ControlRuntime",
    "DRIVE_STRATEGIES",
    "DriveStrategy",
    "ResolvedControl",
    "ResolvedControlPlan",
    "build_control_plan",
    "build_control_runtimes",
    "control_of",
    "single_articulation_runtime",
    "strategy_for_drive",
    "validate_resolved_coverage",
]


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
        feet: foot body names (articulation order).
        body_parts: authoring metadata (feet/base/...).
        spawn: root spawn declaration (``{pos, yaw}``).
        base_body: base/root body name (unigym_name).
        addresses: backend-specific address bookkeeping (free joint, base
            body id, foot ids, ...).
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
    feet: tuple[str, ...] = ()
    body_parts: Mapping = field(default_factory=dict)
    spawn: Mapping = field(default_factory=dict)
    base_body: str = ""
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
    foot_body_ids: list[int] = field(default_factory=list)
    foot_geom_sets: list[set[int]] = field(default_factory=list)
    catalog: ObsCatalog | None = None

    def __post_init__(self):
        if not self.name:
            raise ControlError("articulation runtime name must be non-empty")
        self.name = str(self.name)
        self.joint_names = tuple(str(n) for n in (self.joint_names or ()))
        self.feet = tuple(str(n) for n in (self.feet or ()))
        body_parts: Mapping[str, JsonValue] = self.body_parts or {}
        self.body_parts = dict(body_parts)
        spawn: Mapping[str, JsonValue] = self.spawn or {}
        self.spawn = dict(spawn)
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


@dataclass
class ControlRuntime:
    """One explicit drive face over an articulation (§3.3).

    Attributes:
        name: the control name (the ``env.controls`` key).
        action_face: the accepted ``dict[str, NamedAction]`` key (== name).
        adapter: the adapter instance name that drives this face.
        articulation: the owning articulation name.
        joint_names: joints owned by this face, in action order.
        q0: default positions aligned with ``joint_names`` (torch float32).
        gains: declared gain mapping (backend fills defaults).
        drive: one of :data:`unigym.runtime.controls.CONTROL_DRIVES`.
        target: the explicit target kind (must match ``drive``).
        control_dt: control period (s) or ``None`` (inherits).
        command: optional execution parameters / backend hints.
        strategy: the neutral :class:`DriveStrategy` descriptor.
        addresses: backend-specific addressing/strategy state.
        asset_joint_names: the owning articulation's full asset joints.
        feet: owning articulation's foot body names.
        required: True when every joint must be addressed by an action entry.
        source: provenance of the resolved joints (``profile:<art>.<set>``
            when expanded from a profile joint set, ``config.joints`` when
            declared verbatim).  Neutral: no backend names.
        joint_set: the profile joint-set name selected by the declaration
            (``None`` for explicit ``joints``).
        prop: True for an articulated-prop servo face.
    """

    name: str
    action_face: str
    adapter: str
    articulation: str
    joint_names: tuple[str, ...]
    #: normalized to a torch.Tensor in __post_init__ (see ArticulationRuntime)
    q0: Any = None
    gains: Mapping[str, float] = field(default_factory=dict)
    drive: str = "position_pd"
    target: str = "position"
    control_dt: float | None = None
    command: Mapping[str, JsonValue] | None = None
    strategy: DriveStrategy | None = None
    addresses: dict = field(default_factory=dict)
    asset_joint_names: tuple[str, ...] = ()
    feet: tuple[str, ...] = ()
    required: bool = True
    source: str = ""
    joint_set: str | None = None
    prop: bool = False

    def __post_init__(self):
        for key in ("name", "action_face", "adapter", "articulation"):
            value = getattr(self, key)
            if not value:
                raise ControlError(f"control runtime {self.name!r} {key} must be non-empty")
            setattr(self, key, str(value))
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
        gains: Mapping[str, float] = self.gains or {}
        self.gains = {str(k): float(v) for k, v in dict(gains).items()}
        self.asset_joint_names = tuple(str(n) for n in (self.asset_joint_names or ()))
        self.feet = tuple(str(n) for n in (self.feet or ()))
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


def _profile_defaults(profile: RobotProfile | None) -> dict[str, float]:
    return dict(getattr(profile, "default_joint_pos", {}) or {})


def _control_q0(
    group: ControlGroup, joint_names: Sequence[str], profile: RobotProfile | None, what: str
) -> tuple[float, ...]:
    defaults = _profile_defaults(profile)
    values: list[float] = []
    for joint in joint_names:
        if joint in group.q0:
            values.append(float(group.q0[joint]))
        elif joint in defaults:
            values.append(float(defaults[joint]))
        elif group.drive in ("position_pd", "position_servo"):
            values.append(0.0)
        else:
            raise ControlError(
                f"{what} has no q0 for joint {joint!r} (only position " f"drives may default to 0)"
            )
    return tuple(values)


def _explicit_joint_set(group: ControlGroup, profile: RobotProfile | None) -> tuple[str, ...]:
    """Resolve ``group.joint_set`` against the articulation's profile.

    Fail-fast when the articulation has no profile or the set is unknown; no
    group-name / ``hand`` inference is involved.
    """
    name = str(group.joint_set)
    if profile is None:
        raise ControlError(
            f"control group {group.name!r} selects joint_set {name!r} on "
            f"articulation {group.articulation!r}, which has no RobotProfile; "
            f"`joint_set` requires a profile-backed articulation (list "
            f"`joints` explicitly for profile-less articulations)"
        )
    sets: Mapping[str, tuple[str, ...]] = dict(getattr(profile, "joint_sets", None) or {})
    try:
        joints = tuple(str(j) for j in sets[name])
    except KeyError:
        raise ControlError(
            f"control group {group.name!r} selects unknown joint_set "
            f"{name!r} on articulation {group.articulation!r}; the profile "
            f"declares {sorted(sets)}"
        ) from None
    if not joints:
        raise ControlError(
            f"control group {group.name!r} joint_set {name!r} resolves no "
            f"joints on articulation {group.articulation!r}"
        )
    return joints


def _declared_control_runtime(
    group: ControlGroup,
    arts: Mapping[str, ArticulationSpec],
    profiles: Mapping[str, RobotProfile],
) -> ControlRuntime:
    """Build one declared ``controls[]`` runtime (fail-fast resolution)."""
    art = arts.get(group.articulation)
    if art is None:
        raise ControlError(
            f"control group {group.name!r} names articulation "
            f"{group.articulation!r}, which is not declared"
        )
    profile = profiles.get(group.articulation)
    # explicit resolution: explicit joints win; otherwise the named
    # joint_set is expanded once from the profile; no group-name inference.
    if group.joints:
        bare = tuple(group.joints)
        source = "config.joints"
    elif group.joint_set:
        bare = _explicit_joint_set(group, profile)
        source = f"profile:{group.articulation}.{group.joint_set}"
    else:
        raise ControlError(
            f"control group {group.name!r} declares neither `joints` nor "
            f"`joint_set`; every group must select its joints explicitly"
        )
    if not bare:
        raise ControlError(
            f"control group {group.name!r} resolves no joints (declare "
            f"`joints` or use a profile-backed articulation)"
        )
    return ControlRuntime(
        name=group.name,
        action_face=group.name,
        adapter=group.adapter,
        articulation=group.articulation,
        joint_names=tuple(bare),
        q0=_control_q0(group, bare, profile, f"control group {group.name!r}"),
        gains=dict(group.gains),
        drive=group.drive,
        target=group.target,
        control_dt=group.control_dt,
        command=group.command,
        required=True,
        source=source,
        joint_set=group.joint_set,
    )


@dataclass(frozen=True)
class ResolvedControl:
    """One build-time resolved control face (the read-only plan row).

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
        }


@dataclass(frozen=True)
class ResolvedControlPlan:
    """The frozen build-time control plan (derived from resolved controls)."""

    controls: tuple[ResolvedControl, ...] = ()

    def __iter__(self):
        return iter(self.controls)

    def __len__(self) -> int:
        return len(self.controls)

    def control(self, name: str) -> ResolvedControl:
        """One resolved control by name (fail-fast)."""
        for control in self.controls:
            if control.name == name:
                return control
        raise KeyError(
            f"resolved plan has no control {name!r}; declared: "
            f"{[c.name for c in self.controls]}"
        )

    def for_articulation(self, articulation: str) -> tuple[ResolvedControl, ...]:
        """Every resolved control of one articulation (declaration order)."""
        return tuple(c for c in self.controls if c.articulation == articulation)

    def to_table(self) -> list[dict[str, JsonValue]]:
        """The plan as a list of plain-data rows (read-only projection)."""
        return [control.to_dict() for control in self.controls]


def build_control_plan(
    controls: Sequence[ControlGroup] | None = None,
    runtimes: Sequence[ControlRuntime] | None = None,
) -> ResolvedControlPlan:
    """Freeze the resolved-control result into a read-only plan.

    The joints/drive/target/q0/provenance are read from the already resolved
    :class:`ControlRuntime` entries -- the plan is never re-inferred later.
    """
    groups = {group.name: group for group in (controls or ())}
    rows: list[ResolvedControl] = []
    for runtime in tuple(runtimes or ()):
        group = groups.get(runtime.name)
        joint_set = group.joint_set if group is not None else runtime.joint_set
        rows.append(
            ResolvedControl(
                name=runtime.name,
                articulation=runtime.articulation,
                adapter=runtime.adapter,
                joint_set=joint_set,
                source=runtime.source,
                joint_names=tuple(runtime.joint_names),
                drive=runtime.drive,
                target=runtime.target,
                q0=(
                    tuple(float(v) for v in runtime.q0.tolist())
                    if isinstance(runtime.q0, torch.Tensor)
                    else tuple(float(v) for v in (runtime.q0 or ()))
                ),
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


def validate_resolved_coverage(
    articulations: Iterable[ArticulationSpec] | None,
    profiles: AbcMapping[str, RobotProfile] | None,
    controls: Iterable[ControlGroup] | None,
    runtimes: Iterable[ControlRuntime] | None,
) -> None:
    """Validate the resolved control coverage of every articulation (§4).

    For each articulation with declared controls:

    * ``actuated`` = union of the profile ``joint_sets`` (profile-backed)
      or ``ArticulationSpec.joints`` (profile-less);
    * ``driven`` = union of the resolved control joints;
    * ``passive`` = ``ArticulationSpec.passive_joints``;
    * every control group of one articulation must be disjoint;
    * ``driven`` and ``passive`` must be disjoint;
    * ``driven | passive`` must equal ``actuated``.

    Raises:
        ControlError: with the articulation name and the missing / overlap /
            passive-conflict / unknown joint names.
    """
    arts = {art.name: art for art in tuple(articulations or ())}
    profiles = dict(profiles or {})
    declared = {group.name for group in (controls or ())}

    driven: dict[str, list[str]] = {}
    for runtime in tuple(runtimes or ()):
        if runtime.name not in declared:
            continue
        art = arts.get(runtime.articulation)
        if art is None:
            raise ControlError(
                f"control {runtime.name!r} names articulation "
                f"{runtime.articulation!r}, which is not declared"
            )
        bare = list(runtime.joint_names)
        owned = driven.setdefault(art.name, [])
        overlap = sorted(set(owned) & set(bare))
        if overlap:
            raise ControlError(
                f"articulation {art.name!r}: control groups overlap on "
                f"joints {overlap} (group {runtime.name!r}); every driven "
                f"joint must belong to exactly one group"
            )
        owned.extend(bare)

    for name, owned in driven.items():
        art = arts[name]
        profile = profiles.get(name)
        actuated = set(_actuated_joints(art, profile))
        passive = set(getattr(art, "passive_joints", ()) or ())
        driven_set = set(owned)
        conflict = sorted(driven_set & passive)
        if conflict:
            raise ControlError(
                f"articulation {name!r}: joints {conflict} are both driven "
                f"by a control group and listed in passive_joints"
            )
        if not actuated:
            # profile-less articulation without declared joints: the actuated
            # vocabulary is not knowable here; overlap is still enforced.
            continue
        covered = driven_set | passive
        missing = sorted(actuated - covered)
        if missing:
            raise ControlError(
                f"articulation {name!r}: actuated joints {missing} are "
                f"neither driven by a control group nor listed in "
                f"passive_joints; declare the drive explicitly"
            )
        unknown = sorted(covered - actuated)
        if unknown:
            raise ControlError(
                f"articulation {name!r}: joints {unknown} are not in the "
                f"articulation's actuated joint vocabulary (profile "
                f"joint_sets or ArticulationSpec.joints)"
            )


def build_control_runtimes(
    controls: Sequence[ControlGroup] | None,
    articulations: Sequence[ArticulationSpec] | None,
    profiles: Mapping[str, RobotProfile] | None,
) -> tuple[ControlRuntime, ...]:
    """Build neutral :class:`ControlRuntime` entries from the run declaration.

    Only declared :class:`~unigym.runtime.controls.ControlGroup` entries are
    consumed: each selects its joints explicitly (``joints``) or through a
    profile ``joint_set``.  There is no face-name / ``hand`` inference and no
    implicit extra face -- an adapter that emits a face must have it
    declared in ``controls[]``.

    Args:
        controls: normalized :class:`~unigym.runtime.controls.ControlGroup`
            entries (``RunSpec.controls`` is authoritative).
        articulations: normalized :class:`~unigym.scenes.articulation.
            ArticulationSpec` entries.
        profiles: ``{articulation name: RobotProfile}`` for profile-backed
            (robot) articulations.

    Returns:
        The control runtimes in declaration order (one per control group).
    """
    arts = {art.name: art for art in tuple(articulations or ())}
    profiles = dict(profiles or {})
    declared = tuple(controls or ())
    out = tuple(_declared_control_runtime(group, arts, profiles) for group in declared)
    validate_resolved_coverage(arts.values(), profiles, declared, out)
    return out


def _iter_controls(
    controls: Mapping[str, ControlRuntime] | Sequence[ControlRuntime] | None,
) -> tuple[ControlRuntime, ...]:
    if controls is None:
        return ()
    if isinstance(controls, Mapping):
        return tuple(controls.values())
    return tuple(controls)


def control_of(env: EnvBackend, adapter: Adapter) -> ControlRuntime:
    """Resolve the BODY control group an adapter drives on *env*.

    This is a name-partition lookup, not face/joint inference: the adapter
    carries the registry name it was built under (``_unigym_name``) and the
    run's build-time ``controls[]`` partition names the groups it owns.
    Among those groups a declared (non-prop) group whose
    action face equals its articulation is the body group; a single
    declared group is returned directly.

    Raises:
        ControlError: no adapter name is available, or the ownership
            partition does not identify exactly one body group.
    """
    name = getattr(adapter, "_unigym_name", None) or getattr(adapter, "name", None)
    runtimes = _iter_controls(getattr(env, "controls", None))
    if not runtimes:
        raise ControlError("env exposes no controls")
    if not name:
        # test doubles / direct construction: a single declared control is a
        # cardinality fact, not a guess (multi runs always carry a name).
        declared = [c for c in runtimes if not c.prop]
        if len(declared) == 1:
            return declared[0]
        raise ControlError(
            f"adapter {type(adapter).__name__} carries no registry name "
            f"(_unigym_name) and the run has {len(declared)} declared "
            f"control groups; cannot resolve its control group without a "
            f"name"
        )
    declared = [c for c in runtimes if c.adapter == name and not c.prop]
    if not declared:
        raise ControlError(
            f"adapter {name!r} owns no declared control group; controls: "
            f"{[(c.name, c.adapter) for c in runtimes]}"
        )
    if len(declared) == 1:
        return declared[0]
    body = [c for c in declared if c.action_face == c.articulation]
    if len(body) == 1:
        return body[0]
    raise ControlError(
        f"adapter {name!r} owns several control groups and none is a "
        f"unique body face: {[c.name for c in declared]}; pass the group "
        f"explicitly instead of guessing"
    )


def single_articulation_runtime(env: EnvBackend) -> ArticulationRuntime:
    """The one articulation runtime (a cardinality fact, no primary rank)."""
    arts: Mapping[str, ArticulationRuntime] = getattr(env, "articulations", None) or {}
    if len(arts) != 1:
        raise RuntimeError(f"expected exactly one articulation, found {sorted(arts)}")
    return next(iter(arts.values()))
