"""Backend-neutral request/response types for runtime query.

The query face is name-addressed and never exposes raw engine ids.  Every
response type validates its own tensors at construction and stores detached
CPU copies, so a caller cannot mutate live simulator state through a result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import torch

from .errors import QueryShapeError

__all__ = [
    "AABB",
    "ActuatorFacts",
    "ActuatorRuntime",
    "ActuatorSlotKind",
    "ComRef",
    "ContactJacobian",
    "ContactManifold",
    "ContactPoint",
    "DistanceResult",
    "DofLayout",
    "EntityRef",
    "EnvSelection",
    "Jacobian",
    "PointRef",
    "Pose",
]

EntityKind = Literal["body", "site", "geom"]

ActuatorSlotKind = Literal["mujoco_actuator", "isaaclab_joint", "isaaclab_group"]

_ACTUATOR_SLOT_KINDS = frozenset({"mujoco_actuator", "isaaclab_joint", "isaaclab_group"})

EnvSelection = int | Sequence[int] | torch.Tensor | None


def _env_tensor(env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
    """Validate an output env-id vector as a detached 1-D int64 CPU tensor."""
    if isinstance(env_ids, torch.Tensor):
        ids = env_ids.detach().to("cpu")
    else:
        ids = torch.as_tensor(list(env_ids), dtype=torch.int64)
    if ids.ndim != 1 or ids.dtype != torch.int64:
        raise QueryShapeError(
            "env_ids",
            "a 1-D int64 tensor",
            f"shape={tuple(ids.shape)} dtype={ids.dtype}",
        )
    if ids.numel() and int(ids.min()) < 0:
        raise QueryShapeError("env_ids", "non-negative env ids", str(ids.tolist()))
    return ids.clone()


def _matrix(value: torch.Tensor, field: str, rows: int, cols: int) -> torch.Tensor:
    """Validate a ``(rows, cols)`` floating tensor and return a detached copy."""
    if not isinstance(value, torch.Tensor):
        raise QueryShapeError(
            field,
            f"a torch.Tensor of shape ({rows}, {cols})",
            type(value).__name__,
        )
    if not value.is_floating_point():
        raise QueryShapeError(field, "a floating tensor", str(value.dtype))
    if tuple(value.shape) != (rows, cols):
        raise QueryShapeError(field, f"shape ({rows}, {cols})", str(tuple(value.shape)))
    return value.detach().to("cpu").clone()


def _vector(value: torch.Tensor, field: str, length: int) -> torch.Tensor:
    """Validate a ``(length,)`` floating tensor and return a detached copy."""
    if not isinstance(value, torch.Tensor):
        raise QueryShapeError(field, f"a torch.Tensor of shape ({length},)", type(value).__name__)
    if not value.is_floating_point():
        raise QueryShapeError(field, "a floating tensor", str(value.dtype))
    if tuple(value.shape) != (length,):
        raise QueryShapeError(field, f"shape ({length},)", str(tuple(value.shape)))
    return value.detach().to("cpu").clone()


def _tensor3(value: torch.Tensor, field: str, rows: int, cols: int) -> torch.Tensor:
    """Validate a ``(rows, 3, cols)`` floating tensor and return a detached copy."""
    if not isinstance(value, torch.Tensor):
        raise QueryShapeError(
            field, f"a torch.Tensor of shape ({rows}, 3, {cols})", type(value).__name__
        )
    if not value.is_floating_point():
        raise QueryShapeError(field, "a floating tensor", str(value.dtype))
    if tuple(value.shape) != (rows, 3, cols):
        raise QueryShapeError(field, f"shape ({rows}, 3, {cols})", str(tuple(value.shape)))
    return value.detach().to("cpu").clone()


def _finite_vector(value: torch.Tensor, field: str, length: int) -> torch.Tensor:
    """Validate a finite ``(length,)`` tensor and return a detached copy."""
    vector = _vector(value, field, length)
    if not bool(torch.isfinite(vector).all()):
        raise QueryShapeError(field, "finite values", str(vector.tolist()))
    return vector


def _finite_matrix(value: torch.Tensor, field: str, rows: int, cols: int) -> torch.Tensor:
    """Validate a finite ``(rows, cols)`` tensor and return a detached copy."""
    matrix = _matrix(value, field, rows, cols)
    if not bool(torch.isfinite(matrix).all()):
        raise QueryShapeError(field, "finite values", str(matrix.tolist()))
    return matrix


def _finite_columns(value: torch.Tensor, field: str, cols: int) -> torch.Tensor:
    """Validate a finite float64 ``(dim, cols)`` tensor and return a detached copy.

    ``dim`` is the contact's own effective constraint-row count, so only the
    column count is fixed by the articulation's DOF layout.
    """
    if not isinstance(value, torch.Tensor):
        raise QueryShapeError(field, f"a torch.Tensor of shape (dim, {cols})", type(value).__name__)
    if value.dtype != torch.float64:
        raise QueryShapeError(field, "a float64 tensor", str(value.dtype))
    if value.ndim != 2 or value.shape[1] != cols:
        raise QueryShapeError(field, f"shape (dim, {cols})", str(tuple(value.shape)))
    if value.shape[0] < 1:
        raise QueryShapeError(field, "at least one constraint row", str(tuple(value.shape)))
    matrix = value.detach().to("cpu").clone()
    if not bool(torch.isfinite(matrix).all()):
        raise QueryShapeError(field, "finite values", str(matrix.tolist()))
    return matrix


def _finite_scalar(value: float, field: str) -> float:
    """Validate a finite scalar and return it as a plain float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QueryShapeError(field, "a finite float", type(value).__name__)
    scalar = float(value)
    if not isfinite(scalar):
        raise QueryShapeError(field, "finite values", repr(scalar))
    return scalar


def _non_negative_int(value: int, field: str) -> int:
    """Validate a non-negative integer identity field."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise QueryShapeError(field, "a non-negative int", type(value).__name__)
    if value < 0:
        raise QueryShapeError(field, "a non-negative int", str(value))
    return value


def _name(value: str | None, field: str, *, optional: bool = False) -> str | None:
    """Validate a non-empty entity name (or ``None`` when optional)."""
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be {'None or ' if optional else ''}a non-empty string")
    return value


def _name_tuple(value: Sequence[str], field: str) -> tuple[str, ...]:
    """Validate a sequence of non-empty names and return it as a tuple."""
    if isinstance(value, (str, bytes)):
        raise QueryShapeError(field, "a sequence of names", type(value).__name__)
    names: list[str] = []
    for item in value:
        name = _name(item, field)
        if not isinstance(name, str):
            raise ValueError(f"{field} must contain non-empty strings")
        names.append(name)
    return tuple(names)


def _float_tuple(value: Sequence[float] | None, field: str) -> tuple[float, ...] | None:
    """Validate a non-empty finite float tuple, preserving ``None``."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise QueryShapeError(field, "a non-empty sequence of finite floats", type(value).__name__)
    values = tuple(_finite_scalar(item, field) for item in value)
    if not values:
        raise QueryShapeError(field, "a non-empty sequence of finite floats", "an empty sequence")
    return values


def _float_pair(value: Sequence[float] | None, field: str) -> tuple[float, float] | None:
    """Validate a finite ``(low, high)`` pair, preserving ``None``."""
    values = _float_tuple(value, field)
    if values is None:
        return None
    if len(values) != 2:
        raise QueryShapeError(field, "a (low, high) pair", str(len(values)))
    return (values[0], values[1])


def _optional_vector(value: torch.Tensor | None, field: str, length: int) -> torch.Tensor | None:
    """Validate an optional ``(length,)`` floating tensor and return a copy."""
    if value is None:
        return None
    return _vector(value, field, length)


@dataclass(frozen=True)
class EntityRef:
    """A name-addressed body, site or geom."""

    kind: EntityKind
    name: str
    articulation: str | None = None

    def __post_init__(self) -> None:
        """Reject an unknown kind and empty names."""
        if self.kind not in ("body", "site", "geom"):
            raise ValueError(f"EntityRef.kind must be 'body'|'site'|'geom', got {self.kind!r}")
        if not self.name:
            raise ValueError("EntityRef.name must be non-empty")
        if self.articulation is not None and not self.articulation:
            raise ValueError("EntityRef.articulation must be None or non-empty")


@dataclass(frozen=True)
class ComRef:
    """A center-of-mass reference inside one articulation."""

    articulation: str
    body: str | None = None
    subtree: bool = False

    def __post_init__(self) -> None:
        """Reject an empty articulation or body name."""
        if not self.articulation:
            raise ValueError("ComRef.articulation must be non-empty")
        if self.body is not None and not self.body:
            raise ValueError("ComRef.body must be None or non-empty")


@dataclass(frozen=True)
class PointRef:
    """A world point attached to a body for Jacobian purposes."""

    entity: EntityRef
    point_w: torch.Tensor

    def __post_init__(self) -> None:
        """Require a body entity and a finite ``(3,)`` world point."""
        if self.entity.kind != "body":
            raise ValueError("PointRef.entity must be a body reference")
        point = _vector(self.point_w, "point_w", 3)
        if not bool(torch.isfinite(point).all()):
            raise QueryShapeError("point_w", "finite values", str(point.tolist()))
        object.__setattr__(self, "point_w", point)


@dataclass(frozen=True)
class Pose:
    """Per-env world poses (position + wxyz quaternion)."""

    env_ids: torch.Tensor
    pos_w: torch.Tensor
    quat_wxyz: torch.Tensor

    def __post_init__(self) -> None:
        """Validate and copy the env ids, positions and quaternions."""
        ids = _env_tensor(self.env_ids)
        rows = ids.numel()
        object.__setattr__(self, "env_ids", ids)
        object.__setattr__(self, "pos_w", _matrix(self.pos_w, "pos_w", rows, 3))
        object.__setattr__(self, "quat_wxyz", _matrix(self.quat_wxyz, "quat_wxyz", rows, 4))


@dataclass(frozen=True)
class DistanceResult:
    """Per-env signed geometry distance and closest points."""

    env_ids: torch.Tensor
    signed_distance_m: torch.Tensor
    closest_a_w: torch.Tensor
    closest_b_w: torch.Tensor
    normal_w: torch.Tensor
    capped: torch.Tensor
    mesh_exact: bool = False

    def __post_init__(self) -> None:
        """Validate every row-count-linked tensor."""
        ids = _env_tensor(self.env_ids)
        rows = ids.numel()
        distance = _vector(self.signed_distance_m, "signed_distance_m", rows)
        capped = self.capped
        if not isinstance(capped, torch.Tensor) or capped.dtype != torch.bool:
            raise QueryShapeError("capped", "a bool tensor", type(capped).__name__)
        if tuple(capped.shape) != (rows,):
            raise QueryShapeError("capped", f"shape ({rows},)", str(tuple(capped.shape)))
        object.__setattr__(self, "env_ids", ids)
        object.__setattr__(self, "signed_distance_m", distance)
        object.__setattr__(self, "closest_a_w", _matrix(self.closest_a_w, "closest_a_w", rows, 3))
        object.__setattr__(self, "closest_b_w", _matrix(self.closest_b_w, "closest_b_w", rows, 3))
        object.__setattr__(self, "normal_w", _matrix(self.normal_w, "normal_w", rows, 3))
        object.__setattr__(self, "capped", capped.detach().to("cpu").clone())


@dataclass(frozen=True)
class AABB:
    """Per-env world-space axis-aligned bounding box."""

    env_ids: torch.Tensor
    min_w: torch.Tensor
    max_w: torch.Tensor

    def __post_init__(self) -> None:
        """Validate bounds and require ``min_w <= max_w`` on every row."""
        ids = _env_tensor(self.env_ids)
        rows = ids.numel()
        min_w = _matrix(self.min_w, "min_w", rows, 3)
        max_w = _matrix(self.max_w, "max_w", rows, 3)
        if bool((min_w > max_w).any()):
            raise QueryShapeError("min_w/max_w", "min_w <= max_w", "a reversed bound")
        object.__setattr__(self, "env_ids", ids)
        object.__setattr__(self, "min_w", min_w)
        object.__setattr__(self, "max_w", max_w)


@dataclass(frozen=True)
class DofLayout:
    """The per-column DOF layout of one articulation's Jacobian."""

    articulation: str
    base_dofs: tuple[str, ...]
    base_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    nv: int
    order: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate tuple normalization and the column-count invariant."""
        if not self.articulation:
            raise ValueError("DofLayout.articulation must be non-empty")
        base_dofs = tuple(str(name) for name in self.base_dofs)
        base_names = tuple(str(name) for name in self.base_names)
        joint_names = tuple(str(name) for name in self.joint_names)
        order = tuple(str(name) for name in self.order)
        if self.nv < 0:
            raise ValueError(f"DofLayout.nv must be >= 0, got {self.nv}")
        if len(order) != self.nv:
            raise QueryShapeError("order", f"length {self.nv}", str(len(order)))
        if len(base_dofs) not in (0, 6):
            raise ValueError(f"DofLayout.base_dofs must have 0 or 6 entries, got {len(base_dofs)}")
        if len(base_names) not in (0, 1):
            raise ValueError(
                f"DofLayout.base_names must have 0 or 1 entries, got {len(base_names)}"
            )
        if order[: len(base_dofs)] != base_dofs:
            raise ValueError("DofLayout.order must start with base_dofs")
        if any(not name for name in (*base_dofs, *base_names, *joint_names, *order)):
            raise ValueError("DofLayout names must be non-empty")
        object.__setattr__(self, "base_dofs", base_dofs)
        object.__setattr__(self, "base_names", base_names)
        object.__setattr__(self, "joint_names", joint_names)
        object.__setattr__(self, "order", order)


@dataclass(frozen=True)
class Jacobian:
    """Per-env translational and rotational Jacobians in a named layout."""

    env_ids: torch.Tensor
    transl: torch.Tensor
    rot: torch.Tensor
    layout: DofLayout
    frame: str
    reference: str

    def __post_init__(self) -> None:
        """Validate the ``(E, 3, nv)`` jacobian tensors against the layout."""
        ids = _env_tensor(self.env_ids)
        rows = ids.numel()
        columns = self.layout.nv
        transl = _tensor3(self.transl, "transl", rows, columns)
        rot = _tensor3(self.rot, "rot", rows, columns)
        object.__setattr__(self, "env_ids", ids)
        object.__setattr__(self, "transl", transl)
        object.__setattr__(self, "rot", rot)


@dataclass(frozen=True)
class ContactPoint:
    """One active contact point in world coordinates.

    The field set is backend-neutral and never exposes a raw engine handle.
    ``body1``/``body2`` and ``geom1``/``geom2`` keep the backend's contact
    order; ``geom`` names are ``None`` when the compiled geom has no name.

    Force convention (MuJoCo ``mj_contactForce``): ``normal_force_n`` and
    ``friction_force_w`` act on the **geom2 body**.  ``normal_w`` points from
    geom1 to geom2, and ``frame_w`` rows are the contact axes
    (normal, tangent1, tangent2) in world coordinates.  ``frame_w`` is a
    basis for those three axes, not a per-constraint-row axis label: a
    pyramidal contact's rows are friction-pyramid face normals such as
    ``n +/- t1`` and ``n +/- t2`` (no pure normal row for ``condim=3``), and
    ``condim`` 4/6 torsion/rolling rows are not described by ``frame_w`` at
    all.  ``separation_m`` is the signed surface separation (positive =
    separated).  ``contact_id`` is an index into the backend's current contact
    list for that env; it is not a stable cross-step identifier.
    """

    env_id: int
    group: str
    body1: str
    body2: str
    geom1: str | None
    geom2: str | None
    pos_w: torch.Tensor
    normal_w: torch.Tensor | None
    frame_w: torch.Tensor | None
    separation_m: float
    normal_force_n: float
    friction_force_w: torch.Tensor | None
    contact_id: int

    def __post_init__(self) -> None:
        """Validate identities, shapes and finiteness, then store CPU copies."""
        env_id = _non_negative_int(self.env_id, "env_id")
        contact_id = _non_negative_int(self.contact_id, "contact_id")
        group = _name(self.group, "group")
        body1 = _name(self.body1, "body1")
        body2 = _name(self.body2, "body2")
        geom1 = _name(self.geom1, "geom1", optional=True)
        geom2 = _name(self.geom2, "geom2", optional=True)
        if not isinstance(group, str) or not isinstance(body1, str) or not isinstance(body2, str):
            raise ValueError("contact point names must be strings")
        pos = _finite_vector(self.pos_w, "pos_w", 3)
        normal = None if self.normal_w is None else _finite_vector(self.normal_w, "normal_w", 3)
        frame = None if self.frame_w is None else _finite_matrix(self.frame_w, "frame_w", 3, 3)
        friction = (
            None
            if self.friction_force_w is None
            else _finite_vector(self.friction_force_w, "friction_force_w", 3)
        )
        object.__setattr__(self, "env_id", env_id)
        object.__setattr__(self, "contact_id", contact_id)
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "body1", body1)
        object.__setattr__(self, "body2", body2)
        object.__setattr__(self, "geom1", geom1)
        object.__setattr__(self, "geom2", geom2)
        object.__setattr__(self, "pos_w", pos)
        object.__setattr__(self, "normal_w", normal)
        object.__setattr__(self, "frame_w", frame)
        object.__setattr__(self, "separation_m", _finite_scalar(self.separation_m, "separation_m"))
        object.__setattr__(
            self, "normal_force_n", _finite_scalar(self.normal_force_n, "normal_force_n")
        )
        object.__setattr__(self, "friction_force_w", friction)


@dataclass(frozen=True)
class ContactManifold:
    """All active contact points of one declared contact group.

    Points are in backend contact order, env-major; ``env_ids`` is the
    selected env vector.  An empty selection or a group with no active
    contacts yields an empty ``points`` tuple, never an error.
    """

    env_ids: torch.Tensor
    group: str
    points: tuple[ContactPoint, ...]

    def __post_init__(self) -> None:
        """Validate the env selection and every point's group/env ownership."""
        ids = _env_tensor(self.env_ids)
        group = _name(self.group, "group")
        if not isinstance(group, str):
            raise ValueError("contact manifold group must be a string")
        if not isinstance(self.points, tuple):
            raise TypeError(f"points must be a tuple, got {type(self.points).__name__}")
        selected = {int(value) for value in ids.tolist()}
        for point in self.points:
            if not isinstance(point, ContactPoint):
                raise TypeError(f"points must hold ContactPoint, got {type(point).__name__}")
            if point.group != group:
                raise ValueError(
                    f"contact point group {point.group!r} does not match manifold group {group!r}"
                )
            if point.env_id not in selected:
                raise ValueError(
                    f"contact point env_id {point.env_id} is not in env_ids {sorted(selected)}"
                )
        object.__setattr__(self, "env_ids", ids)
        object.__setattr__(self, "group", group)


@dataclass(frozen=True)
class ContactJacobian:
    """One active contact's constraint-row Jacobian block.

    ``rows`` is the ``(rows, nv)`` contact constraint Jacobian for
    ``ContactPoint.contact_id``, column-scoped to one articulation's DOF
    layout.  ``rows`` may have more entries than the contact dimensionality:
    it is the effective friction-cone row count, not ``ContactPoint``'s
    contact dimensionality.

    ``frame`` is ``"contact"``.  Rows are MuJoCo constraint rows, not a
    per-row contact-axis list:

    - elliptic contacts with ``condim >= 3`` have rows 0..2 equal to the
      contact frame axes (normal, tangent1, tangent2); rows 3+ are
      torsion/rolling constraints;
    - pyramidal rows are friction-pyramid face normals such as ``n + t1``,
      ``n - t1``, ``n + t2`` and ``n - t2``.  There is no pure normal row for
      ``condim=3``; ``ContactPoint.frame_w`` is a basis for expressing a row
      as a linear combination, not an axis label per row;
    - ``condim`` 4/6 torsion and rolling rows are not described by
      ``frame_w``.

    The tensor is a detached CPU float64 copy.
    """

    env_id: int
    frame: str
    rows: torch.Tensor
    layout: DofLayout

    def __post_init__(self) -> None:
        """Validate the frame, layout and ``(rows, nv)`` row block."""
        env_id = _non_negative_int(self.env_id, "env_id")
        if self.frame != "contact":
            raise QueryShapeError("frame", "'contact'", repr(self.frame))
        if not isinstance(self.layout, DofLayout):
            raise QueryShapeError("layout", "a DofLayout", type(self.layout).__name__)
        object.__setattr__(self, "env_id", env_id)
        object.__setattr__(self, "frame", self.frame)
        object.__setattr__(self, "rows", _finite_columns(self.rows, "rows", self.layout.nv))


@dataclass(frozen=True)
class ActuatorFacts:
    """Static facts of one actuator, resolved from the published ``ModelIR``.

    ``slot`` is the backend's compiled actuator slot; ``slot_kind`` names the
    slot namespace so it is never mistaken for a neutral array index.  For the
    MuJoCo family ``slot`` is the compiled ``MjModel`` actuator id and
    ``slot_kind`` is ``"mujoco_actuator"``.  ``target`` is the neutral target
    name (``None`` when the target has no neutral name); ``ctrlrange``,
    ``gear``, ``gainprm`` and ``biasprm`` are ``None`` when the compiled row
    does not declare them.  ``joint_names`` names the neutral joints the
    actuator drives: one joint for a MuJoCo joint actuator, the declared group
    members for an IsaacLab actuator group, and empty when the backend publishes
    no such list.
    """

    name: str
    target: str | None
    kind: str
    slot: int | None
    slot_kind: ActuatorSlotKind
    ctrlrange: tuple[float, float] | None = None
    gear: tuple[float, ...] | None = None
    gainprm: tuple[float, ...] | None = None
    biasprm: tuple[float, ...] | None = None
    joint_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate identities and declared parameter tuples."""
        name = _name(self.name, "name")
        target = _name(self.target, "target", optional=True)
        kind = _name(self.kind, "kind")
        if not isinstance(name, str) or not isinstance(kind, str):
            raise ValueError("ActuatorFacts.name/kind must be strings")
        if self.slot_kind not in _ACTUATOR_SLOT_KINDS:
            raise ValueError(
                f"ActuatorFacts.slot_kind must be one of {sorted(_ACTUATOR_SLOT_KINDS)}, "
                f"got {self.slot_kind!r}"
            )
        slot = None if self.slot is None else _non_negative_int(self.slot, "slot")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "slot", slot)
        object.__setattr__(self, "ctrlrange", _float_pair(self.ctrlrange, "ctrlrange"))
        object.__setattr__(self, "gear", _float_tuple(self.gear, "gear"))
        object.__setattr__(self, "gainprm", _float_tuple(self.gainprm, "gainprm"))
        object.__setattr__(self, "biasprm", _float_tuple(self.biasprm, "biasprm"))
        object.__setattr__(self, "joint_names", _name_tuple(self.joint_names, "joint_names"))


@dataclass(frozen=True)
class ActuatorRuntime:
    """One actuator's per-env live runtime channel values.

    ``env_ids`` selects the rows and preserves the requested order.  Each
    field is a ``(E,)`` float tensor, or ``None`` when the backend cannot read
    that channel.  Every tensor is a detached CPU copy, so mutating the result
    cannot change live simulator state.
    """

    env_ids: torch.Tensor
    ctrl: torch.Tensor | None = None
    force: torch.Tensor | None = None
    length: torch.Tensor | None = None
    velocity: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate row counts and store detached CPU copies."""
        ids = _env_tensor(self.env_ids)
        rows = ids.numel()
        object.__setattr__(self, "env_ids", ids)
        object.__setattr__(self, "ctrl", _optional_vector(self.ctrl, "ctrl", rows))
        object.__setattr__(self, "force", _optional_vector(self.force, "force", rows))
        object.__setattr__(self, "length", _optional_vector(self.length, "length", rows))
        object.__setattr__(self, "velocity", _optional_vector(self.velocity, "velocity", rows))
