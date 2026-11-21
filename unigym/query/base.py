"""Backend-neutral request/response types for runtime query.

The query face is name-addressed and never exposes raw engine ids.  Every
response type validates its own tensors at construction and stores detached
CPU copies, so a caller cannot mutate live simulator state through a result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from .errors import QueryShapeError

__all__ = [
    "AABB",
    "ComRef",
    "DistanceResult",
    "DofLayout",
    "EntityRef",
    "EnvSelection",
    "Jacobian",
    "PointRef",
    "Pose",
]

EntityKind = Literal["body", "site", "geom"]

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
