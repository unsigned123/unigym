"""NamedAction: the self-describing action contract (feat/named-actions).

The action boundary is a SEQUENCE of self-describing values whose container
order is NOT semantic: an adapter returns ``tuple[NamedAction, ...]`` with one
entry per action group (e.g. the body group plus a hand group), and every
entry is addressed by its owning ``(articulation, group)`` identity.  Routing
therefore never depends on where an entry sits in the tuple.  What IS an
explicit per-entry contract is each ``NamedAction.joint_names`` tuple: its
order fixes that entry's column order.  Every entry carries its values and the
joint name for each column.  Consumers (backends) resolve the names to
simulator addresses ONCE per name-tuple (cached) and scatter the values; users
never touch simulator joint order (same philosophy as the name-based
obs/contact/foot layers — actions are the last piece).

Backends:
  * isaaclab — each action term is built with ``preserve_order=True`` and an
    EXPLICIT joint-name list, so term input dim ``i`` == ``joint_names[i]``;
    :class:`IsaaclabJointResolver` maps an entry's names to positions in the
    term's flat input slice.
  * mujoco — :class:`MujocoJointResolver` maps names to actuator/qpos/dof
    addresses (reusing the same name tables the PD path already uses,
    extended to the finger actuators).

This module must stay torch-only (importable by adapters/base.py without
isaaclab/mujoco; the mujoco resolver imports mujoco lazily).

Stage A0 adds declaration-ordered action templates
(:class:`ActionDim` / :class:`OrderedActionGroup` / :class:`OrderedActionSpec`):
a spec fixes the group order and each group's joint order, so a flattened
action vector is stable and self-describing (``idx articulation group joint
target``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    import numpy as np
    from mujoco import MjModel

    from unigym.model.ir import ModelIR

#: legal NamedAction target kinds (must equal the owning ControlRuntime.target)
LEGAL_ACTION_KINDS = ("position", "velocity", "torque", "effort", "current")

__all__ = [
    "LEGAL_ACTION_KINDS",
    "ActionDim",
    "IsaaclabJointResolver",
    "JointResolver",
    "MujocoJointResolver",
    "NamedAction",
    "OrderedActionGroup",
    "OrderedActionSpec",
    "ResolvedJointTargets",
]


def _require_text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value


# --------------------------------------------------------------------------- #
# declaration-ordered action templates (Stage A0)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ActionDim:
    """One flattened action dimension of an :class:`OrderedActionSpec`.

    Attributes:
        index: zero-based position in the flattened action vector.
        articulation: owning articulation name.
        group: owning action-group name within that articulation.
        joint: joint name addressed by this dimension.
        target: legal action kind (see :data:`LEGAL_ACTION_KINDS`).
    """

    index: int
    articulation: str
    group: str
    joint: str
    target: str

    def __post_init__(self):
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError(f"ActionDim.index must be a non-negative integer, got {self.index!r}")
        _require_text(self.articulation, "ActionDim.articulation")
        _require_text(self.group, "ActionDim.group")
        _require_text(self.joint, "ActionDim.joint")
        _require_text(self.target, "ActionDim.target")
        if self.target not in LEGAL_ACTION_KINDS:
            raise ValueError(
                f"ActionDim.target must be one of {list(LEGAL_ACTION_KINDS)}, "
                f"got {self.target!r}"
            )


@dataclass(frozen=True)
class OrderedActionGroup:
    """One named, declaration-ordered action group of an articulation.

    Attributes:
        name: action-face name (unique per articulation).
        articulation: owning articulation name.
        joints: joint names in declaration order (stored as a tuple).
        target: legal action kind (see :data:`LEGAL_ACTION_KINDS`).
    """

    name: str
    articulation: str
    joints: tuple[str, ...]
    target: str

    def __post_init__(self):
        _require_text(self.name, "OrderedActionGroup.name")
        _require_text(self.articulation, "OrderedActionGroup.articulation")
        if isinstance(self.joints, (str, bytes)) or not isinstance(self.joints, Sequence):
            raise ValueError(
                "OrderedActionGroup.joints must be a sequence of joint names, "
                f"got {type(self.joints).__name__}"
            )
        joints = tuple(self.joints)
        if not joints:
            raise ValueError(f"OrderedActionGroup {self.name!r}: joints must be non-empty")
        for joint in joints:
            _require_text(joint, "OrderedActionGroup joint name")
        dupes = sorted({joint for joint in joints if joints.count(joint) > 1})
        if dupes:
            raise ValueError(f"OrderedActionGroup {self.name!r}: duplicate joints {dupes}")
        if self.target not in LEGAL_ACTION_KINDS:
            raise ValueError(
                f"OrderedActionGroup.target must be one of {list(LEGAL_ACTION_KINDS)}, "
                f"got {self.target!r}"
            )
        object.__setattr__(self, "joints", joints)


@dataclass(frozen=True)
class OrderedActionSpec:
    """A declaration-ordered sequence of action groups.

    Groups must be non-empty, their ``(articulation, name)`` identity unique,
    and no two groups of one articulation may own the same joint.  Order is
    preserved exactly as declared; :meth:`dim_table` flattens it into
    ``ActionDim`` rows.
    """

    groups: tuple[OrderedActionGroup, ...]

    def __post_init__(self):
        if isinstance(self.groups, (str, bytes)) or not isinstance(self.groups, Sequence):
            raise ValueError(
                "OrderedActionSpec.groups must be a sequence of OrderedActionGroup, "
                f"got {type(self.groups).__name__}"
            )
        groups = tuple(self.groups)
        if not groups:
            raise ValueError("OrderedActionSpec.groups must contain at least one group")
        identities: set[tuple[str, str]] = set()
        joint_owner: dict[tuple[str, str], str] = {}
        for group in groups:
            if not isinstance(group, OrderedActionGroup):
                raise ValueError(
                    "OrderedActionSpec.groups entries must be OrderedActionGroup, "
                    f"got {type(group).__name__}"
                )
            identity = (group.articulation, group.name)
            if identity in identities:
                raise ValueError(
                    "OrderedActionSpec: duplicate group identity "
                    f"(articulation, name)={identity!r}"
                )
            identities.add(identity)
            for joint in group.joints:
                key = (group.articulation, joint)
                previous = joint_owner.get(key)
                if previous is not None:
                    raise ValueError(
                        f"OrderedActionSpec: articulation {group.articulation!r} has "
                        f"overlapping joints between groups {previous!r} and "
                        f"{group.name!r} (joint {joint!r})"
                    )
                joint_owner[key] = group.name
        object.__setattr__(self, "groups", groups)

    @property
    def dim(self) -> int:
        """Return the total flattened action dimension."""
        return sum(len(group.joints) for group in self.groups)

    def dim_table(self) -> tuple[ActionDim, ...]:
        """Return the declaration-ordered action dimensions."""
        rows: list[ActionDim] = []
        index = 0
        for group in self.groups:
            for joint in group.joints:
                rows.append(
                    ActionDim(
                        index=index,
                        articulation=group.articulation,
                        group=group.name,
                        joint=joint,
                        target=group.target,
                    )
                )
                index += 1
        return tuple(rows)

    def __str__(self) -> str:
        """Render the compact ``idx articulation group joint target`` table."""
        rows: list[tuple[str, ...]] = [("idx", "articulation", "group", "joint", "target")]
        rows.extend(
            (str(row.index), row.articulation, row.group, row.joint, row.target)
            for row in self.dim_table()
        )
        widths = [max(len(row[column]) for row in rows) for column in range(5)]
        return "\n".join(
            "  ".join(cell.ljust(widths[column]) for column, cell in enumerate(row)).rstrip()
            for row in rows
        )

    def __repr__(self) -> str:
        """Return a concise one-line summary."""
        groups = ", ".join(
            f"{group.articulation}/{group.name}:{len(group.joints)}" for group in self.groups
        )
        return f"OrderedActionSpec(dim={self.dim}, groups=[{groups}])"


# --------------------------------------------------------------------------- #
# the contract type
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NamedAction:
    """One self-describing action group value.

    Identity is ``(articulation, group)``: duplicate joint names across
    articulations are legal and are disambiguated by the articulation.
    The order of a ``NamedAction`` sequence is not an action layout.

    Attributes:
        articulation: owning articulation name (non-empty).
        group: owning action-group name within that articulation (non-empty).
        values: ``(N, dim)`` float tensor — targets in the units of *kind*
            (rad for position, rad/s for velocity, N·m for torque/effort,
            A for current), one column per entry in *joint_names*.
        joint_names: joint names, ``len(joint_names) == values.shape[1]``.
            The tuple IS the contract: consumers resolve by name, never by
            position.  Names are scoped to ``(articulation, group)``.
        kind: the explicit target kind
            (``position | velocity | torque | effort | current``).  The
            engine validates it against the configured control group's
            ``target`` every tick.

    Construction validates the identity, shape/name/kind agreement
    (fail-fast) and rejects duplicate names (a duplicate would make the
    routing ambiguous).
    """

    articulation: str
    group: str
    values: torch.Tensor
    joint_names: tuple[str, ...]
    kind: str

    def __post_init__(self):
        _require_text(self.articulation, "NamedAction.articulation")
        _require_text(self.group, "NamedAction.group")
        if self.kind not in LEGAL_ACTION_KINDS:
            raise ValueError(
                f"NamedAction.kind must be one of {list(LEGAL_ACTION_KINDS)}, " f"got {self.kind!r}"
            )
        if not isinstance(self.values, torch.Tensor):
            raise ValueError(
                f"NamedAction.values must be a torch tensor, got " f"{type(self.values).__name__}"
            )
        if self.values.ndim != 2:
            raise ValueError(
                f"NamedAction.values must be (N, dim), got " f"{tuple(self.values.shape)}"
            )
        if isinstance(self.joint_names, (str, bytes)) or not isinstance(self.joint_names, Sequence):
            raise ValueError(
                "NamedAction.joint_names must be a sequence of joint names, "
                f"got {type(self.joint_names).__name__}"
            )
        names = tuple(self.joint_names)
        if not names:
            raise ValueError("NamedAction.joint_names must be non-empty")
        if len(names) != self.values.shape[1]:
            raise ValueError(
                f"NamedAction: {self.values.shape[1]} value columns vs "
                f"{len(names)} joint_names (must match 1:1)"
            )
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"NamedAction: duplicate joint names {dupes}")
        object.__setattr__(self, "joint_names", names)

    @property
    def num_envs(self) -> int:
        """Batch dimension N of the values tensor."""
        return int(self.values.shape[0])

    @property
    def dim(self) -> int:
        """Per-env action dimension (== len(joint_names))."""
        return int(self.values.shape[1])

    @classmethod
    def zeros(
        cls,
        num_envs: int,
        articulation: str,
        group: str,
        joint_names: Sequence[str],
        kind: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> NamedAction:
        """All-zero action over *joint_names* for ``(articulation, group)``."""
        if isinstance(joint_names, (str, bytes)) or not isinstance(joint_names, Sequence):
            raise ValueError(
                "NamedAction.zeros joint_names must be a sequence of joint names, "
                f"got {type(joint_names).__name__}"
            )
        names = tuple(joint_names)
        return cls(
            articulation=articulation,
            group=group,
            values=torch.zeros(int(num_envs), len(names), dtype=dtype, device=device),
            joint_names=names,
            kind=kind,
        )


# --------------------------------------------------------------------------- #
# the resolver protocol (backend-traits: the typed seam both backends
# implement; NamedAction consumers dispatch here, never on engine names)
# --------------------------------------------------------------------------- #
@runtime_checkable
class JointResolver(Protocol):
    """Name-addressed action entry -> backend addressing primitive.

    isaaclab: :class:`IsaaclabJointResolver` (gather indices into one
    preserve_order action term's flat input); mujoco:
    :class:`MujocoJointResolver` (pol/hand/prop ctrl-address routing).
    Both are cached per name-tuple and fail fast on unknown names.
    """

    def resolve(
        self,
        joint_names: Sequence[str],
    ) -> ResolvedJointTargets | torch.Tensor:
        """Resolve entry names to the backend addressing primitive."""
        ...


# --------------------------------------------------------------------------- #
# isaaclab side: entry names -> positions in a term's flat input slice
# --------------------------------------------------------------------------- #
class IsaaclabJointResolver:
    """Maps entry joint names to indices into one action term's input.

    The term must be configured with ``preserve_order=True`` and an explicit
    joint-name list, so its flat input dim ``i`` corresponds to
    ``term_joint_names[i]`` (plan §1).  The gather
    index is cached per name-tuple (first resolve, then reuse).

    Unknown names fail fast listing the accepted names.
    """

    def __init__(self, term_joint_names: Sequence[str]) -> None:
        self.term_names: list[str] = list(term_joint_names)
        self._index: dict[str, int] = {n: i for i, n in enumerate(self.term_names)}
        self._cache: dict[tuple[str, ...], torch.Tensor] = {}

    def gather(self, joint_names: Sequence[str]) -> torch.Tensor:
        """``(len(joint_names),)`` int64 gather index.

        For each entry column its position in the term's flat input
        (``term_input[:, gather] = entry.values``); unknown names raise
        ValueError.
        """
        key = tuple(joint_names)
        idx = self._cache.get(key)
        if idx is None:
            missing = [n for n in key if n not in self._index]
            if missing:
                raise ValueError(
                    f"unknown joint name(s) {missing} for this action term; "
                    f"accepted names: {self.term_names}"
                )
            idx = torch.tensor([self._index[n] for n in key], dtype=torch.int64)
            self._cache[key] = idx
        return idx

    # JointResolver protocol name: the isaaclab addressing primitive
    # is the gather index; ``gather`` stays the historical caller face.
    resolve = gather


# --------------------------------------------------------------------------- #
# mujoco side: entry names -> actuator addresses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ResolvedJointTargets:
    """Per-entry routing table into the MJ target vectors (numpy).

    ``pol_idx`` / ``hand_idx`` / ``prop_adr`` have one slot per entry
    column; a column that addresses a policy joint carries its index in
    *pol_idx* (and ``-1`` elsewhere), a finger joint carries its
    hand-vector index in *hand_idx*, and a PROP actuated joint
    carries its actuator ctrl address in *prop_adr*.  The three are
    mutually exclusive; the env scatters values accordingly (policy joints
    MUST be fully covered, finger/prop slots fall back to their hold
    value when no entry addresses them).
    """

    pol_idx: np.ndarray  # int64, -1 where not a policy joint
    hand_idx: np.ndarray  # int64, -1 where not a hand joint
    prop_adr: np.ndarray  # int64, ctrl adr where a prop joint


class MujocoJointResolver:
    """Name -> address resolution over an MJ model (cached per name-tuple).

    Built from the env's two name tables — the policy joint order (the 29
    SONIC body joints) and the hand joint order (the non-policy finger DOFs,
    i.e. the env's ``_hand_names``) — plus the PROP actuator tables
    (``{prop_name: {jname: ctrl_adr}}`` for wired articulation
    props; prop-scoped names must not collide with the robot tables --
    namespace fail-fast).  Names outside all tables (or without an
    actuator) fail fast listing the accepted names.
    """

    def __init__(
        self,
        model: MjModel,
        policy_names: Sequence[str],
        hand_joint_names: Sequence[str],
        prop_actuators: Mapping[str, Mapping[str, int]] | None = None,
        model_ir: ModelIR | None = None,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self._pol_names: list[str] = list(policy_names)
        self._hand_names: list[str] = list(hand_joint_names)
        self._pol: dict[str, int] = {n: i for i, n in enumerate(self._pol_names)}
        self._hand: dict[str, int] = {n: i for i, n in enumerate(self._hand_names)}
        self._prop: dict[str, dict[str, int]] = {
            str(k): dict(v) for k, v in (prop_actuators or {}).items()
        }
        conflict = sorted(
            {n for t in self._prop.values() for n in t} & (set(self._pol) | set(self._hand))
        )
        if conflict:
            raise ValueError(
                f"mujoco action resolver: prop joint names {conflict} "
                f"collide with robot joint names (namespace conflict)"
            )
        # name -> (qpos_adr, dof_adr) from the compiled ModelIR; the
        # address table is derived here once and is not a hot-path lookup.
        # The MjModel fallback is the legacy path for stand-alone
        # callers (tests) that do not assemble an IR: this module stays
        # torch-only and may not import unigym.model (layer contract).
        self.addresses: dict[str, tuple[int, int]]
        if model_ir is None:
            self.addresses = {
                model.joint(idx).name: (int(model.jnt_qposadr[idx]), int(model.jnt_dofadr[idx]))
                for idx in range(model.njnt)
            }
            actuated = {model.joint(int(model.actuator_trnid[a, 0])).name for a in range(model.nu)}
        else:
            self.addresses = {
                row.name: (int(row.qpos_adr), int(row.dof_adr))
                for row in model_ir.joints
                if row.qpos_adr is not None and row.dof_adr is not None
            }
            actuated = {
                act.target
                for act in model_ir.actuators
                if act.trntype in ("joint", "jointinparent")
            }
        # every resolvable joint must have an actuator (fail-fast here, not
        # at PD-write time)
        missing_act = sorted(n for n in self._pol_names + self._hand_names if n not in actuated)
        if missing_act:
            raise ValueError(f"mujoco action resolver: no actuator for joints {missing_act}")
        self._cache: dict[tuple[str | None, tuple[str, ...]], ResolvedJointTargets] = {}

    def resolve(
        self,
        joint_names: Sequence[str],
        prop: str | None = None,
    ) -> ResolvedJointTargets:
        """Resolve entry names to pol/hand/prop addresses (cached).

        Args:
            joint_names: Entry column names (policy / hand / prop-scoped).
            prop: Prop name routing the entry into that prop's actuator
                table (None = the robot tables).

        Returns:
            ResolvedJointTargets with one slot per column.

        Raises:
            ValueError: Unknown name / unknown prop / name outside the
                prop's actuated set.
        """
        key = (prop, tuple(joint_names))
        out = self._cache.get(key)
        if out is None:
            import numpy as np

            pol_idx: list[int] = []
            hand_idx: list[int] = []
            prop_adr: list[int] = []
            if prop is not None:
                table = self._prop.get(prop)
                if table is None:
                    raise ValueError(
                        f"no prop action entry {prop!r}; known prop "
                        f"entries: {sorted(self._prop)}"
                    )
                for n in key[1]:
                    if n not in table:
                        raise ValueError(
                            f"joint name {n!r} is not an actuated joint "
                            f"of prop {prop!r}; actuated joints: "
                            f"{sorted(table)}"
                        )
                    pol_idx.append(-1)
                    hand_idx.append(-1)
                    prop_adr.append(table[n])
            else:
                for n in key[1]:
                    if n in self._pol:
                        pol_idx.append(self._pol[n])
                        hand_idx.append(-1)
                        prop_adr.append(-1)
                    elif n in self._hand:
                        pol_idx.append(-1)
                        hand_idx.append(self._hand[n])
                        prop_adr.append(-1)
                    else:
                        raise ValueError(
                            f"unknown joint name {n!r} in the MJ model action "
                            f"tables; policy joints: {self._pol_names}; hand "
                            f"joints: {self._hand_names}; prop entries: "
                            f"{sorted(self._prop)}"
                        )
            out = ResolvedJointTargets(
                pol_idx=np.asarray(pol_idx, dtype=np.int64),
                hand_idx=np.asarray(hand_idx, dtype=np.int64),
                prop_adr=np.asarray(prop_adr, dtype=np.int64),
            )
            self._cache[key] = out
        return out
