"""Joint-addressed action routing over a physical control table.

The routing unit is one ``(articulation, joint)`` pair.  The physical control
table is an opaque sequence (or mapping) of ``ControlRuntime``-like rows; this
module never imports a backend or an action template and never treats a
control row label as an action identity.  A :class:`JointActionIndex` is built
once from the controls and then reused for every tick: routing maps each
column of a :class:`~unigym.runtime.actions.NamedAction` to the unique slot
that physically owns that joint.

The helper is intentionally free of simulator and tensor-framework imports.
Only action metadata (articulation, joint names, kind and batch size) is read;
the values tensor is passed through untouched inside :class:`RoutedColumn`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeGuard

if TYPE_CHECKING:
    from unigym.runtime.actions import NamedAction

__all__ = ["JointActionIndex", "JointSlot", "RoutedColumn"]


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    """Whether *value* is a non-string sequence."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _require_text(value: object, what: str) -> str:
    """Validate one non-empty text field."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value


def _owner_label(owner: object) -> str:
    """Return a diagnostic label for one opaque control row."""
    name = getattr(owner, "name", None)
    if isinstance(name, str) and name:
        return name
    return f"<{type(owner).__name__} at {id(owner):#x}>"


@dataclass(frozen=True)
class JointSlot:
    """One physical action column owned by a control row.

    Attributes:
        articulation: owning articulation name.
        joint: joint name addressed by this column.
        kind: target kind expected by the owning control row.
        owner: the opaque control row that owns the joint (never interpreted
            as an action identity).
        column: zero-based column inside ``owner.joint_names``.
        required: whether the owning control row requires full coverage.
        prop: whether the owning control row is an articulated-prop servo.
    """

    articulation: str
    joint: str
    kind: str
    owner: object
    column: int
    required: bool
    prop: bool

    def __post_init__(self) -> None:
        _require_text(self.articulation, "JointSlot.articulation")
        _require_text(self.joint, "JointSlot.joint")
        _require_text(self.kind, "JointSlot.kind")
        if isinstance(self.column, bool) or not isinstance(self.column, int) or self.column < 0:
            raise ValueError(
                f"JointSlot.column must be a non-negative integer, got {self.column!r}"
            )
        if not isinstance(self.required, bool):
            raise ValueError(f"JointSlot.required must be a bool, got {self.required!r}")
        if not isinstance(self.prop, bool):
            raise ValueError(f"JointSlot.prop must be a bool, got {self.prop!r}")


@dataclass(frozen=True)
class RoutedColumn:
    """One routed action column: entry tensor slice to one physical slot.

    Attributes:
        entry: the self-describing action value that carries the column.
        source_index: zero-based column inside ``entry.values``.
        slot: the physical joint slot the column addresses.
    """

    entry: NamedAction
    source_index: int
    slot: JointSlot

    def __post_init__(self) -> None:
        if isinstance(self.source_index, bool) or not isinstance(self.source_index, int):
            raise ValueError(
                "RoutedColumn.source_index must be a non-negative integer, "
                f"got {self.source_index!r}"
            )
        if self.source_index < 0:
            raise ValueError(
                "RoutedColumn.source_index must be a non-negative integer, "
                f"got {self.source_index!r}"
            )
        if not isinstance(self.slot, JointSlot):
            raise ValueError(
                f"RoutedColumn.slot must be a JointSlot, got {type(self.slot).__name__}"
            )


class JointActionIndex:
    """A ``(articulation, joint)`` index over a physical control table.

    Build with :meth:`from_controls` once per environment, then call
    :meth:`route` every tick.  The index keys never contain an action
    template label: a joint is owned by exactly one control row, and the row
    travels opaquely inside the :class:`JointSlot`.
    """

    def __init__(
        self,
        *,
        label: str,
        slots: dict[tuple[str, str], JointSlot],
        control_slots: tuple[tuple[JointSlot, ...], ...],
    ) -> None:
        self._label = label
        self._slots = slots
        self._control_slots = control_slots
        self._joints: dict[str, tuple[str, ...]] = {}
        self._prop_joints: dict[str, tuple[str, ...]] = {}
        for slot in slots.values():
            self._joints.setdefault(slot.articulation, ())
            self._joints[slot.articulation] += (slot.joint,)
            if slot.prop:
                self._prop_joints.setdefault(slot.articulation, ())
                self._prop_joints[slot.articulation] += (slot.joint,)
        self._joints = {
            articulation: tuple(sorted(joints)) for articulation, joints in self._joints.items()
        }
        self._prop_joints = {
            articulation: tuple(sorted(joints))
            for articulation, joints in self._prop_joints.items()
        }

    @classmethod
    def from_controls(cls, controls: object, *, label: str) -> JointActionIndex:
        """Build the index from a physical control table.

        Controls are consumed in physical declaration order.  A mapping is
        read through its values; a non-string sequence is iterated directly.
        Each control row must expose ``articulation``, ``joint_names`` and
        ``target``; ``required`` and ``prop`` default to ``True`` / ``False``.
        The row itself is stored opaquely as ``owner``.

        Args:
            controls: Physical control table (sequence or mapping of
                ``ControlRuntime``-like rows).
            label: Backend or caller label used as error context.

        Returns:
            The built joint action index.

        Raises:
            TypeError: *controls* is not a sequence or mapping.
            ValueError: A row is malformed, or two rows own the same
                ``(articulation, joint)`` pair.
        """
        label = _require_text(label, "JointActionIndex label")
        if isinstance(controls, Mapping):
            rows = tuple(controls.values())
        elif _is_sequence(controls):
            rows = tuple(controls)
        else:
            raise TypeError(
                "JointActionIndex.from_controls expects a sequence or mapping of "
                f"control rows, got {type(controls).__name__}"
            )

        slots: dict[tuple[str, str], JointSlot] = {}
        control_slots: list[tuple[JointSlot, ...]] = []
        for control in rows:
            articulation = _require_text(
                getattr(control, "articulation", None),
                f"{label} control articulation",
            )
            raw_joints = getattr(control, "joint_names", ())
            if not _is_sequence(raw_joints) or not raw_joints:
                raise ValueError(
                    f"{label}: control row {_owner_label(control)!r} of articulation "
                    f"{articulation!r} must declare a non-empty joint_names sequence"
                )
            kind = _require_text(
                getattr(control, "target", None),
                f"{label} control target for {_owner_label(control)!r}",
            )
            required = bool(getattr(control, "required", True))
            prop = bool(getattr(control, "prop", False))
            row_slots: list[JointSlot] = []
            for column, raw_joint in enumerate(raw_joints):
                joint = _require_text(raw_joint, f"{label} control joint name")
                key = (articulation, joint)
                previous = slots.get(key)
                if previous is not None:
                    raise ValueError(
                        f"{label}: duplicate joint slot for articulation "
                        f"{articulation!r} joint {joint!r}; owned by both control "
                        f"rows {_owner_label(previous.owner)!r} and "
                        f"{_owner_label(control)!r}"
                    )
                slot = JointSlot(
                    articulation=articulation,
                    joint=joint,
                    kind=kind,
                    owner=control,
                    column=column,
                    required=required,
                    prop=prop,
                )
                slots[key] = slot
                row_slots.append(slot)
            control_slots.append(tuple(row_slots))

        return cls(label=label, slots=slots, control_slots=tuple(control_slots))

    def route(self, actions: object, num_envs: int) -> tuple[RoutedColumn, ...]:
        """Route every action column to its owning physical joint slot.

        Entries may appear in any order and one entry may span several control
        rows of one articulation as long as every column's kind matches the
        owning control's target.  Every addressed joint is consumed exactly
        once per tick and every required control joint must be covered.

        Args:
            actions: Non-string sequence of
                :class:`~unigym.runtime.actions.NamedAction` entries.
            num_envs: Batch size the backend expects.

        Returns:
            One :class:`RoutedColumn` per action column, in entry then column
            order.

        Raises:
            TypeError: *actions* is not a sequence or an entry is not a
                ``NamedAction``.
            ValueError: Batch size, joint name, kind, duplicate coverage or
                required coverage validation fails.
        """
        from unigym.runtime.actions import NamedAction

        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 0:
            raise ValueError(f"{self._label}: num_envs must be a non-negative integer")
        if not _is_sequence(actions):
            raise TypeError(
                f"{self._label}: route expects a sequence of NamedAction, "
                f"got {type(actions).__name__}"
            )

        addressed: set[tuple[str, str]] = set()
        routed: list[RoutedColumn] = []
        for entry in actions:
            if not isinstance(entry, NamedAction):
                raise TypeError(
                    f"{self._label}: route expects a sequence of NamedAction, "
                    f"got {type(entry).__name__} entry"
                )
            if entry.num_envs != num_envs:
                raise ValueError(
                    f"{self._label}: action entry for articulation "
                    f"{entry.articulation!r} has {entry.num_envs} env rows, but "
                    f"the backend expects {num_envs}"
                )
            for source_index, joint in enumerate(entry.joint_names):
                key = (entry.articulation, joint)
                slot = self._slots.get(key)
                if slot is None:
                    raise ValueError(self._unknown_joint_message(entry.articulation, joint))
                if entry.kind != slot.kind:
                    raise ValueError(
                        f"{self._label}: articulation {entry.articulation!r} joint "
                        f"{joint!r} has action kind {entry.kind!r} but the control "
                        f"expects target kind {slot.kind!r}"
                    )
                if key in addressed:
                    raise ValueError(
                        f"{self._label}: articulation {entry.articulation!r} joint "
                        f"{joint!r} is addressed more than once in one tick"
                    )
                addressed.add(key)
                routed.append(RoutedColumn(entry=entry, source_index=source_index, slot=slot))

        missing = self.missing_required(routed)
        if missing:
            details = "; ".join(
                f"articulation {articulation!r}: joints {list(joints)}"
                for articulation, joints in missing.items()
            )
            raise ValueError(f"{self._label}: required joints are unaddressed: {details}")
        return tuple(routed)

    def missing_required(self, routed: Sequence[RoutedColumn]) -> dict[str, tuple[str, ...]]:
        """Return required-but-unaddressed joints grouped by articulation.

        Args:
            routed: Columns already routed for this tick.

        Returns:
            Mapping ``articulation -> (missing joint names,)``; empty when
            every required control joint is covered.

        Raises:
            TypeError: *routed* is not a sequence or contains a non-routed
                entry.
        """
        if not _is_sequence(routed):
            raise TypeError(
                "JointActionIndex.missing_required expects a sequence of "
                f"RoutedColumn, got {type(routed).__name__}"
            )
        covered: set[tuple[str, str]] = set()
        for column in routed:
            if not isinstance(column, RoutedColumn):
                raise TypeError(
                    "JointActionIndex.missing_required expects a sequence of "
                    f"RoutedColumn, got {type(column).__name__} entry"
                )
            covered.add((column.slot.articulation, column.slot.joint))

        missing: dict[str, list[str]] = {}
        for row in self._control_slots:
            for slot in row:
                if not slot.required:
                    continue
                if (slot.articulation, slot.joint) in covered:
                    continue
                bucket = missing.setdefault(slot.articulation, [])
                if slot.joint not in bucket:
                    bucket.append(slot.joint)
        return {articulation: tuple(joints) for articulation, joints in missing.items()}

    def _unknown_joint_message(self, articulation: str, joint: str) -> str:
        """Build the unknown-joint teaching error for one address."""
        accepted = self._joints.get(articulation)
        if accepted is None:
            accepted_articulations = sorted(self._joints)
            return (
                f"{self._label}: unknown action joint {joint!r} for unknown "
                f"articulation {articulation!r}; accepted articulations: "
                f"{accepted_articulations}"
            )
        message = (
            f"{self._label}: unknown action joint {joint!r} for articulation "
            f"{articulation!r}; accepted joints: {list(accepted)}"
        )
        prop_joints = self._prop_joints.get(articulation)
        if prop_joints:
            message += f"; prop joints: {list(prop_joints)}"
        return message
