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

Stage A2 connects those templates to the self-describing runtime values:
:meth:`OrderedActionSpec.unflatten` and :meth:`OrderedActionSpec.flatten`
share the ``dim_table`` group-then-joint traversal and round-trip
bit-identically (``flatten`` also accepts any NamedAction tuple order and any
per-entry joint-name permutation).

Stage A3a adds the independent :class:`PolicyBinding` declaration plus the
mapping parsers (:meth:`OrderedActionGroup.from_mapping`,
:meth:`OrderedActionSpec.from_mapping`, :meth:`PolicyBinding.from_mapping`)
for the action-authoring surface.  Run/build wiring is deliberately not part
of this stage.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

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
    "PolicyBinding",
    "ResolvedJointTargets",
]


def _require_text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value


def _require_mapping(entry: object, what: str) -> Mapping[str, object]:
    if not isinstance(entry, Mapping):
        raise TypeError(f"{what} expects a mapping, got {type(entry).__name__}")
    return entry


def _reject_unknown_keys(
    entry: Mapping[str, object],
    allowed: tuple[str, ...],
    what: str,
) -> None:
    unknown = sorted(set(entry) - set(allowed))
    if unknown:
        raise ValueError(f"{what} does not accept key(s) {unknown}; allowed keys: {list(allowed)}")


def _require_keys(
    entry: Mapping[str, object],
    required: tuple[str, ...],
    what: str,
) -> None:
    missing = [key for key in required if key not in entry]
    if missing:
        raise ValueError(
            f"{what} is missing required key(s) {missing}; required keys: {list(required)}"
        )


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

    @classmethod
    def from_mapping(cls, entry: Mapping[str, object]) -> OrderedActionGroup:
        """Build one group from an explicit declaration mapping.

        Accepted keys are exactly ``name``, ``articulation``, ``joints`` and
        ``target``; all are required.  ``joints`` must already be an explicit
        non-empty list/tuple of joint names -- selector keys such as
        ``joint_set`` are resolved outside this parser (Stage A3b) and are
        rejected here as unknown keys.  Construction runs the regular
        :class:`OrderedActionGroup` validation.

        Args:
            entry: Mapping with the four declaration keys.

        Returns:
            The validated declaration-ordered group.

        Raises:
            TypeError: *entry* is not a mapping.
            ValueError: Unknown or missing keys, or constructor validation
                fails (empty/duplicate/non-text joints, illegal target).
        """
        data = _require_mapping(entry, "OrderedActionGroup.from_mapping")
        allowed = ("name", "articulation", "joints", "target")
        _reject_unknown_keys(data, allowed, "OrderedActionGroup.from_mapping")
        _require_keys(data, allowed, "OrderedActionGroup.from_mapping")
        raw_joints = data["joints"]
        if isinstance(raw_joints, (str, bytes)) or not isinstance(raw_joints, Sequence):
            raise ValueError(
                "OrderedActionGroup.from_mapping joints must be an explicit "
                "list/tuple of joint names (selectors such as joint_set are "
                f"resolved outside this parser), got {type(raw_joints).__name__}"
            )
        joints = tuple(
            _require_text(joint, "OrderedActionGroup.from_mapping joint name")
            for joint in raw_joints
        )
        return cls(
            name=_require_text(data["name"], "OrderedActionGroup.from_mapping name"),
            articulation=_require_text(
                data["articulation"], "OrderedActionGroup.from_mapping articulation"
            ),
            joints=joints,
            target=_require_text(data["target"], "OrderedActionGroup.from_mapping target"),
        )


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

    @classmethod
    def from_mapping(cls, entries: Sequence[object]) -> OrderedActionSpec:
        """Build a spec from an ordered sequence of group declarations.

        Each entry may already be an :class:`OrderedActionGroup` or a mapping
        accepted by :meth:`OrderedActionGroup.from_mapping`.  Declared order
        is preserved.  Strings, bytes and mappings are rejected as the outer
        container (a spec is a sequence, not a single declaration), and
        construction runs the regular :class:`OrderedActionSpec` validation.

        Args:
            entries: Sequence of group declarations (mappings or groups).

        Returns:
            The validated declaration-ordered spec.

        Raises:
            TypeError: *entries* is a string/bytes/mapping/non-sequence, or an
                entry is neither an OrderedActionGroup nor a mapping.
            ValueError: *entries* is empty or constructor validation fails
                (duplicate identity, overlapping joints, bad group).
        """
        if isinstance(entries, (str, bytes, Mapping)) or not isinstance(entries, Sequence):
            raise TypeError(
                "OrderedActionSpec.from_mapping expects a sequence of "
                "OrderedActionGroup or mapping entries, "
                f"got {type(entries).__name__}"
            )
        groups: list[OrderedActionGroup] = []
        for index, entry in enumerate(entries):
            if isinstance(entry, OrderedActionGroup):
                groups.append(entry)
            elif isinstance(entry, Mapping):
                groups.append(OrderedActionGroup.from_mapping(entry))
            else:
                raise TypeError(
                    "OrderedActionSpec.from_mapping entry must be an "
                    f"OrderedActionGroup or mapping; entry {index} is "
                    f"{type(entry).__name__}"
                )
        return cls(groups=tuple(groups))

    @property
    def dim(self) -> int:
        """Return the total flattened action dimension."""
        return sum(len(group.joints) for group in self.groups)

    def _group_spans(self) -> tuple[tuple[OrderedActionGroup, int, int], ...]:
        """Return ``(group, start, end)`` column spans in declaration order.

        This is the single traversal shared by :meth:`dim_table`,
        :meth:`unflatten` and :meth:`flatten`, so the flat layout can never
        drift between the table view and the conversion methods.
        """
        spans: list[tuple[OrderedActionGroup, int, int]] = []
        start = 0
        for group in self.groups:
            end = start + len(group.joints)
            spans.append((group, start, end))
            start = end
        return tuple(spans)

    def dim_table(self) -> tuple[ActionDim, ...]:
        """Return the declaration-ordered action dimensions."""
        rows: list[ActionDim] = []
        for group, start, _ in self._group_spans():
            for offset, joint in enumerate(group.joints):
                rows.append(
                    ActionDim(
                        index=start + offset,
                        articulation=group.articulation,
                        group=group.name,
                        joint=joint,
                        target=group.target,
                    )
                )
        return tuple(rows)

    def unflatten(self, flat: torch.Tensor) -> tuple[NamedAction, ...]:
        """Split a flat ``(N, dim)`` action tensor into NamedAction values.

        Columns are sliced in declaration order (group order, then each
        group's joint order) and wrapped as one self-describing
        :class:`NamedAction` per group.  Slices are views: dtype and device
        are preserved and no value is copied.

        Args:
            flat: ``(N, dim)`` action tensor where ``dim == self.dim``.

        Returns:
            One ``NamedAction`` per spec group, in declaration order.

        Raises:
            TypeError: *flat* is not a torch tensor.
            ValueError: *flat* is not rank 2 or its column count differs
                from ``self.dim``.
        """
        if not isinstance(flat, torch.Tensor):
            raise TypeError(
                "OrderedActionSpec.unflatten expects a torch tensor, " f"got {type(flat).__name__}"
            )
        if flat.ndim != 2:
            raise ValueError(
                "OrderedActionSpec.unflatten expects a (N, dim) tensor, "
                f"got shape {tuple(flat.shape)}"
            )
        if flat.shape[1] != self.dim:
            raise ValueError(
                f"OrderedActionSpec.unflatten expects {self.dim} action columns, "
                f"got {flat.shape[1]} (shape {tuple(flat.shape)})"
            )
        return tuple(
            NamedAction(
                articulation=group.articulation,
                group=group.name,
                values=flat[:, start:end],
                joint_names=group.joints,
                kind=group.target,
            )
            for group, start, end in self._group_spans()
        )

    def flatten(self, actions: Sequence[NamedAction]) -> torch.Tensor:
        """Pack self-describing NamedAction values into flat ``(N, dim)`` form.

        The sequence order is not semantic: entries are matched by
        ``(articulation, group)`` and must cover exactly the spec identities.
        Each entry's ``joint_names`` must be a permutation of that template
        group's joints (columns are reordered back to template order), and all
        entries must agree on ``num_envs``, dtype and device.

        Args:
            actions: NamedAction entries in any order.

        Returns:
            ``(N, dim)`` tensor in spec declaration order.  It is the bitwise
            inverse of :meth:`unflatten`.

        Raises:
            TypeError: *actions* is not a sequence or an entry is not a
                NamedAction.
            ValueError: identities are duplicated, missing or extra; a
                joint-name set, kind, ``num_envs``, dtype or device does not
                match the template.
        """
        if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
            raise TypeError(
                "OrderedActionSpec.flatten expects a sequence of NamedAction, "
                f"got {type(actions).__name__}"
            )
        spans = self._group_spans()
        templates = {(group.articulation, group.name): group for group, _, _ in spans}
        accepted = sorted(templates)
        entries: dict[tuple[str, str], NamedAction] = {}
        for entry in actions:
            if not isinstance(entry, NamedAction):
                raise TypeError(
                    "OrderedActionSpec.flatten expects a sequence of NamedAction, "
                    f"got {type(entry).__name__} entry"
                )
            identity = (entry.articulation, entry.group)
            if identity in entries:
                raise ValueError(
                    f"OrderedActionSpec.flatten: group {identity!r} is addressed "
                    "by more than one NamedAction"
                )
            entries[identity] = entry
        unknown = sorted(set(entries) - set(templates))
        if unknown:
            raise ValueError(
                f"OrderedActionSpec.flatten: NamedAction groups {unknown} are not "
                f"declared by this spec; accepted groups: {accepted}"
            )
        missing = [identity for identity in templates if identity not in entries]
        if missing:
            raise ValueError(
                "OrderedActionSpec.flatten: NamedAction entries leave spec groups "
                f"unaddressed: {missing}; accepted groups: {accepted}"
            )
        first_group, _, _ = spans[0]
        reference_identity = (first_group.articulation, first_group.name)
        reference = entries[reference_identity]
        for group, _, _ in spans:
            identity = (group.articulation, group.name)
            entry = entries[identity]
            if entry.kind != group.target:
                raise ValueError(
                    f"OrderedActionSpec.flatten: group {identity!r} has kind "
                    f"{entry.kind!r} but the template expects target {group.target!r}"
                )
            if len(entry.joint_names) != len(group.joints) or set(entry.joint_names) != set(
                group.joints
            ):
                raise ValueError(
                    f"OrderedActionSpec.flatten: group {identity!r} joint_names "
                    f"{list(entry.joint_names)} are not a permutation of template "
                    f"joints {list(group.joints)}"
                )
            if entry.num_envs != reference.num_envs:
                raise ValueError(
                    f"OrderedActionSpec.flatten: group {identity!r} has "
                    f"{entry.num_envs} env rows but {reference_identity!r} has "
                    f"{reference.num_envs}; all entries must share num_envs"
                )
            if entry.values.dtype != reference.values.dtype:
                raise ValueError(
                    f"OrderedActionSpec.flatten: group {identity!r} values dtype "
                    f"{entry.values.dtype} differs from {reference_identity!r} "
                    f"dtype {reference.values.dtype}; mixed dtypes are not allowed"
                )
            if entry.values.device != reference.values.device:
                raise ValueError(
                    f"OrderedActionSpec.flatten: group {identity!r} values device "
                    f"{entry.values.device} differs from {reference_identity!r} "
                    f"device {reference.values.device}; mixed devices are not allowed"
                )
        chunks = []
        for group, _, _ in spans:
            entry = entries[(group.articulation, group.name)]
            if entry.joint_names == group.joints:
                block = entry.values
            else:
                positions = {name: index for index, name in enumerate(entry.joint_names)}
                gather = torch.tensor(
                    [positions[name] for name in group.joints],
                    dtype=torch.int64,
                    device=entry.values.device,
                )
                block = entry.values.index_select(1, gather)
            chunks.append(block)
        return torch.cat(chunks, dim=1)

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
# independent policy binding (Stage A3a: authoring surface only, no run/build
# wiring yet)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolicyBinding:
    """Which adapter and action groups a policy declares it produces.

    This is the independent declaration half of the action-authoring surface:
    it names the adapter plus the addressed ``(articulation, group)``
    identities (and optional timing/agent metadata) without pointing at any
    run or config object yet.

    Attributes:
        adapter: adapter name that produces the actions (non-empty).
        groups: addressed ``(articulation, group)`` identities, stored as a
            tuple of tuples; order is preserved.  The same group name may
            appear on different articulations, but each identity is unique.
        control_dt: optional control period in seconds (finite and > 0).
        agent: optional agent name (non-empty when present).

    Construction validates the declaration and normalizes *groups* to a tuple
    of tuples (fail-fast).
    """

    adapter: str
    groups: tuple[tuple[str, str], ...]  # (articulation, group)
    control_dt: float | None = None
    agent: str | None = None

    def __post_init__(self):
        _require_text(self.adapter, "PolicyBinding.adapter")
        if isinstance(self.groups, (str, bytes)) or not isinstance(self.groups, Sequence):
            raise ValueError(
                "PolicyBinding.groups must be a non-empty sequence of "
                f"(articulation, group) pairs, got {type(self.groups).__name__}"
            )
        raw_groups = tuple(self.groups)
        if not raw_groups:
            raise ValueError(
                "PolicyBinding.groups must contain at least one " "(articulation, group) pair"
            )
        groups: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for pair in raw_groups:
            if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence):
                raise ValueError(
                    "PolicyBinding.groups entries must be 2-item "
                    "(articulation, group) sequences, "
                    f"got {type(pair).__name__}"
                )
            if len(pair) != 2:
                raise ValueError(
                    "PolicyBinding.groups entry must have exactly 2 items "
                    f"(articulation, group), got {tuple(pair)!r}"
                )
            identity = (
                _require_text(pair[0], "PolicyBinding group articulation"),
                _require_text(pair[1], "PolicyBinding group name"),
            )
            if identity in seen:
                raise ValueError(
                    "PolicyBinding: duplicate group identity " f"(articulation, group)={identity!r}"
                )
            seen.add(identity)
            groups.append(identity)
        object.__setattr__(self, "groups", tuple(groups))
        if self.control_dt is not None:
            if isinstance(self.control_dt, bool) or not isinstance(self.control_dt, (int, float)):
                raise ValueError(
                    "PolicyBinding.control_dt must be a positive finite number "
                    f"or None, got {self.control_dt!r}"
                )
            value = float(self.control_dt)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "PolicyBinding.control_dt must be a positive finite number "
                    f"or None, got {self.control_dt!r}"
                )
            object.__setattr__(self, "control_dt", value)
        if self.agent is not None:
            _require_text(self.agent, "PolicyBinding.agent")

    @classmethod
    def from_mapping(cls, entry: Mapping[str, object]) -> PolicyBinding:
        """Build a binding from an explicit declaration mapping.

        Accepted keys are exactly ``adapter``, ``groups``, ``control_dt`` and
        ``agent``; ``adapter`` and ``groups`` are required.  ``groups`` must
        be a non-empty list of 2-item lists/tuples ``[articulation, group]``.
        Construction runs the regular :class:`PolicyBinding` validation
        (normalization included).

        Args:
            entry: Mapping with the binding declaration.

        Returns:
            The validated policy binding.

        Raises:
            TypeError: *entry* is not a mapping.
            ValueError: Unknown or missing keys, or constructor validation
                fails (empty identities, duplicates, bad dt/agent).
        """
        data = _require_mapping(entry, "PolicyBinding.from_mapping")
        allowed = ("adapter", "groups", "control_dt", "agent")
        _reject_unknown_keys(data, allowed, "PolicyBinding.from_mapping")
        _require_keys(data, ("adapter", "groups"), "PolicyBinding.from_mapping")
        groups = data["groups"]
        if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
            raise ValueError(
                "PolicyBinding.from_mapping groups must be a non-empty list "
                "of [articulation, group] pairs, "
                f"got {type(groups).__name__}"
            )
        return cls(
            adapter=cast("str", data["adapter"]),
            groups=cast("tuple[tuple[str, str], ...]", groups),
            control_dt=cast("float | None", data.get("control_dt")),
            agent=cast("str | None", data.get("agent")),
        )


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
