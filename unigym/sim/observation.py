"""The typed observation record of the experimental simulation facade.

One :class:`Observation` is a read-only view of the state produced by one tick.
``record`` is the producing record mapping (canonical artifact keys without
the bookkeeping keys) and ``obs`` is the producing
:class:`~unigym.env.obs.ObsSnapshot` (typed environment registry).  The typed
fields are projections of ``record``; every typed tensor is the backend tensor
itself.  A missing required state key is an
:class:`~unigym.sim.errors.ObservationError`; there is no zero-fill.

Every projected tensor has exactly ``num_envs`` rows: the row count is the
declared env count, never inferred from one field, so a state with a wrong row
count fails as an :class:`~unigym.sim.errors.ObservationError` before any
field is returned.  The two id masks are distinct: ``reset_env_ids`` names the
rows the producing operation reset, ``reset_due_env_ids`` names the
backend-owned due rows at the producing tick.

Typed camera and command/prev_action values come from the observation
snapshot; only tensors are accepted for those projected slots.

``torch`` is not imported by this module at import time: it is imported lazily
inside :func:`build_observation` and the private projection helpers, so
``import unigym.sim`` stays torch-free.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from .errors import ObservationError

if TYPE_CHECKING:
    import torch

    from unigym.env.obs import ObsSnapshot

    from .handles import ArticulationHandle, CameraHandle

__all__ = ["ArticulationObservation", "CameraFrame", "Observation", "build_observation"]

_STATE_BOOKKEEPING_KEYS = frozenset({"reset_env_ids", "should_reset"})


@dataclass(frozen=True, eq=False)
class Observation:
    """One read-only view of the state produced by tick :attr:`t`.

    Attributes:
        t: the policy tick that produced the cached state (``0`` at spawn).
        num_envs: the declared env count; every projected tensor has this
            many rows.
        articulations: handle-keyed typed articulation projections.
        cameras: handle-keyed published camera frames (only cameras whose
            ``cam/<name>/rgb`` or ``cam/<name>/depth`` key is in the cache).
        contacts: ``contact/`` record tensors carried by the live state.
        reset_env_ids: the env rows this operation reset (empty after a step).
        reset_due_env_ids: the backend-owned due rows at the producing tick
            (empty after a reset).
        terminated: the producing tick's termination mask, or ``None`` when
            the tick was a spawn/reset read.
        record: the producing record mapping without ``reset_env_ids`` /
            ``should_reset``.
        obs: the producing typed :class:`~unigym.env.obs.ObsSnapshot`.
    """

    t: int
    num_envs: int
    articulations: Mapping[ArticulationHandle, ArticulationObservation]
    cameras: Mapping[CameraHandle, CameraFrame]
    contacts: Mapping[str, torch.Tensor]
    reset_env_ids: torch.Tensor
    reset_due_env_ids: torch.Tensor
    terminated: torch.Tensor | None
    record: Mapping[str, torch.Tensor | bool]
    obs: ObsSnapshot

    def __post_init__(self) -> None:
        object.__setattr__(self, "articulations", MappingProxyType(dict(self.articulations)))
        object.__setattr__(self, "cameras", MappingProxyType(dict(self.cameras)))
        object.__setattr__(self, "contacts", MappingProxyType(dict(self.contacts)))
        object.__setattr__(self, "record", MappingProxyType(dict(self.record)))


@dataclass(frozen=True, eq=False)
class ArticulationObservation:
    """The typed per-articulation projection of one tick.

    Attributes:
        handle: the declaration handle this record belongs to.
        joint_names: ``profile.joint_order``, the order of every ``(N, J)``
            joint column.
        joint_pos: absolute joint positions ``(N, J)``.
        joint_pos_rel: ``joint_pos - q0`` ``(N, J)``; ``q0`` comes from
            ``profile.default_joint_pos`` in ``joint_names`` order.
        joint_vel: joint velocities ``(N, J)``.
        applied_torque: applied joint torques ``(N, J)``.
        gravity_b: gravity direction in the base frame ``(N, 3)``.
        base_pos_w: base position in world ``(N, 3)``.
        base_quat_w: base orientation ``(N, 4)``, ``w-x-y-z``.
        base_lin_vel_b: base linear velocity in the base frame ``(N, 3)``.
        base_ang_vel_b: base angular velocity in the base frame ``(N, 3)``.
        prev_action: the optional cached previous action ``(N, ...)``.
        command: the optional cached command ``(N, ...)``.
    """

    handle: ArticulationHandle
    joint_names: tuple[str, ...]
    joint_pos: torch.Tensor
    joint_pos_rel: torch.Tensor
    joint_vel: torch.Tensor
    applied_torque: torch.Tensor
    gravity_b: torch.Tensor
    base_pos_w: torch.Tensor
    base_quat_w: torch.Tensor
    base_lin_vel_b: torch.Tensor
    base_ang_vel_b: torch.Tensor
    prev_action: torch.Tensor | None
    command: torch.Tensor | None


@dataclass(frozen=True, eq=False)
class CameraFrame:
    """One camera's published frame for tick :attr:`t` (never rendered here).

    Attributes:
        handle: the declared camera handle.
        t: the policy tick that published the frame.
        rgb: the cached RGB tensor, or ``None`` when only depth was published.
        depth: the cached depth tensor, or ``None`` when only RGB was published.
    """

    handle: CameraHandle
    t: int
    rgb: torch.Tensor | None
    depth: torch.Tensor | None


def _required_tensor(state: Mapping[str, object], key: str) -> torch.Tensor:
    """Read one required state key as a tensor or fail with the key named."""
    import torch

    value = state.get(key)
    if not isinstance(value, torch.Tensor):
        raise ObservationError(
            f"state key {key!r} must be a torch.Tensor, got {type(value).__name__}"
        )
    return value


def _matrix(state: Mapping[str, object], key: str, columns: int, rows: int | None) -> torch.Tensor:
    """Read one required ``(N, columns)`` state matrix."""
    value = _required_tensor(state, key)
    if value.ndim != 2 or value.shape[1] != columns:
        raise ObservationError(
            f"state key {key!r} must have shape (N, {columns}), got {tuple(value.shape)}"
        )
    if rows is not None and value.shape[0] != rows:
        raise ObservationError(f"state key {key!r} has {value.shape[0]} env rows, expected {rows}")
    return value


def _vector(state: Mapping[str, object], key: str, width: int, rows: int) -> torch.Tensor:
    """Read one required ``(N, width)`` state vector."""
    value = _required_tensor(state, key)
    if value.ndim != 2 or value.shape[1] != width:
        raise ObservationError(
            f"state key {key!r} must have shape (N, {width}), got {tuple(value.shape)}"
        )
    if value.shape[0] != rows:
        raise ObservationError(f"state key {key!r} has {value.shape[0]} env rows, expected {rows}")
    return value


def _optional_cached(snapshot: ObsSnapshot, key: str, num_envs: int) -> torch.Tensor | None:
    """Read one optional projected ``(N, ...)`` snapshot tensor (absent -> None)."""
    import torch

    value = snapshot.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ObservationError(
            f"observation key {key!r} must be a torch.Tensor, " f"got {type(value).__name__}"
        )
    if value.ndim == 0 or value.shape[0] != num_envs:
        raise ObservationError(
            f"observation key {key!r} must have {num_envs} env rows, "
            f"got shape {tuple(value.shape)}"
        )
    return value


def _default_positions(
    handle: ArticulationHandle, joint_names: tuple[str, ...], like: torch.Tensor
) -> torch.Tensor:
    """Resolve ``profile.default_joint_pos`` in ``joint_names`` order."""
    import torch

    defaults = handle.profile.default_joint_pos
    values: list[float] = []
    for joint in joint_names:
        if joint not in defaults:
            raise ObservationError(
                f"articulation {handle.name!r} profile has no default_joint_pos for "
                f"joint {joint!r}; declared defaults: {sorted(defaults)}"
            )
        raw = defaults[joint]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ObservationError(
                f"articulation {handle.name!r} default_joint_pos[{joint!r}] must be a "
                f"finite number, got {raw!r}"
            )
        number = float(raw)
        if not math.isfinite(number):
            raise ObservationError(
                f"articulation {handle.name!r} default_joint_pos[{joint!r}] must be "
                f"finite, got {raw!r}"
            )
        values.append(number)
    return torch.tensor(values, dtype=like.dtype, device=like.device)


def _articulation_record(
    handle: ArticulationHandle,
    state: Mapping[str, object],
    snapshot: ObsSnapshot,
    num_envs: int,
) -> ArticulationObservation:
    """Project one articulation's namespaced state into its typed record."""
    joint_names = tuple(handle.profile.joint_order)
    prefix = f"articulation/{handle.name}"
    rows = num_envs
    joint_pos = _matrix(state, f"{prefix}/joint_pos", len(joint_names), rows)
    joint_vel = _matrix(state, f"{prefix}/joint_vel", len(joint_names), rows)
    applied_torque = _matrix(state, f"{prefix}/applied_torque", len(joint_names), rows)
    gravity_b = _vector(state, f"{prefix}/gravity_b", 3, rows)
    base_pos_w = _vector(state, f"{prefix}/base_pos_w", 3, rows)
    base_quat_w = _vector(state, f"{prefix}/base_quat_w", 4, rows)
    base_lin_vel_b = _vector(state, f"{prefix}/base_lin_vel_b", 3, rows)
    base_ang_vel_b = _vector(state, f"{prefix}/base_ang_vel_b", 3, rows)
    q0 = _default_positions(handle, joint_names, joint_pos)
    return ArticulationObservation(
        handle=handle,
        joint_names=joint_names,
        joint_pos=joint_pos,
        joint_pos_rel=joint_pos - q0,
        joint_vel=joint_vel,
        applied_torque=applied_torque,
        gravity_b=gravity_b,
        base_pos_w=base_pos_w,
        base_quat_w=base_quat_w,
        base_lin_vel_b=base_lin_vel_b,
        base_ang_vel_b=base_ang_vel_b,
        prev_action=snapshot.prev_action.value,
        command=snapshot.command.value,
    )


def _mask(value: object, key: str, num_envs: int) -> torch.Tensor:
    """Validate one reset-id mask: 1-D ``int64``, unique, ids in ``[0, N)``."""
    import torch

    if not isinstance(value, torch.Tensor):
        raise ObservationError(
            f"observation {key} must be a torch.Tensor, got {type(value).__name__}"
        )
    if value.dtype != torch.int64 or value.ndim != 1:
        raise ObservationError(
            f"observation {key} must be a 1-D int64 tensor, got dtype "
            f"{value.dtype} and shape {tuple(value.shape)}"
        )
    if value.numel() > 0 and bool(((value < 0) | (value >= num_envs)).any()):
        raise ObservationError(
            f"observation {key} {value} are outside the legal range [0, {num_envs})"
        )
    if value.numel() > 1:
        ordered = value.sort().values
        if bool((ordered[1:] == ordered[:-1]).any()):
            raise ObservationError(f"observation {key} {value} contain duplicate ids")
    return value


def _terminated(value: object, num_envs: int) -> torch.Tensor | None:
    """Validate the termination mask: ``None`` or a ``(N,)`` bool tensor."""
    import torch

    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ObservationError(
            f"observation terminated must be a torch.Tensor or None, " f"got {type(value).__name__}"
        )
    if value.dtype != torch.bool or value.ndim != 1 or value.shape[0] != num_envs:
        raise ObservationError(
            f"observation terminated must be a ({num_envs},) bool tensor, got "
            f"dtype {value.dtype} and shape {tuple(value.shape)}"
        )
    return value


def build_observation(
    *,
    t: int,
    num_envs: int,
    state: Mapping[str, object],
    snapshot: ObsSnapshot,
    handles: Sequence[ArticulationHandle],
    cameras: Sequence[CameraHandle],
    reset_env_ids: torch.Tensor,
    reset_due_env_ids: torch.Tensor,
    terminated: torch.Tensor | None,
) -> Observation:
    """Project one state mapping and its observation cache into a record.

    Args:
        t: the policy tick that produced ``state``.
        num_envs: the declared env count; every projected tensor must have
            exactly this many rows.
        state: the live state mapping (the step ``extra``).
        snapshot: the producing typed ``ObsSnapshot`` (camera/command slots).
        handles: the declared articulation handles, in declaration order.
        cameras: the declared camera handles, in declaration order.
        reset_env_ids: the env rows this operation reset.
        reset_due_env_ids: the backend-owned due rows at the producing tick.
        terminated: the producing tick's termination mask, or ``None``.

    Raises:
        ObservationError: ``num_envs``, a mask or ``terminated`` violates its
            contract, a required state key is missing, has the wrong type or
            row count, a joint has no valid default position, or ``raw``
            carries a value that is neither a tensor nor a bool.
    """
    import torch

    if isinstance(t, bool) or not isinstance(t, int) or t < 0:
        raise ObservationError(f"observation t must be an int >= 0, got {t!r}")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ObservationError(f"observation num_envs must be an int >= 1, got {num_envs!r}")
    _mask(reset_env_ids, "reset_env_ids", num_envs)
    _mask(reset_due_env_ids, "reset_due_env_ids", num_envs)
    _terminated(terminated, num_envs)

    raw: dict[str, torch.Tensor | bool] = {}
    for key, value in state.items():
        if key in _STATE_BOOKKEEPING_KEYS:
            continue
        if not isinstance(key, str):
            raise ObservationError(
                f"observation state keys must be strings, got {type(key).__name__}"
            )
        if not isinstance(value, (torch.Tensor, bool)):
            raise ObservationError(
                f"observation state key {key!r} must be a torch.Tensor or bool in "
                f"raw, got {type(value).__name__}"
            )
        raw[key] = value

    contacts: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not key.startswith("contact/"):
            continue
        if not isinstance(value, torch.Tensor):
            raise ObservationError(
                f"observation contact key {key!r} must be a torch.Tensor, "
                f"got {type(value).__name__}"
            )
        contacts[key] = value

    records = {
        handle: _articulation_record(handle, state, snapshot, num_envs) for handle in handles
    }
    frames: dict[CameraHandle, CameraFrame] = {}
    for camera in cameras:
        rgb = _optional_cached(snapshot, f"cam/{camera.name}/rgb", num_envs)
        depth = _optional_cached(snapshot, f"cam/{camera.name}/depth", num_envs)
        if rgb is None and depth is None:
            continue
        frames[camera] = CameraFrame(handle=camera, t=t, rgb=rgb, depth=depth)

    return Observation(
        t=t,
        num_envs=num_envs,
        articulations=records,
        cameras=frames,
        contacts=contacts,
        reset_env_ids=reset_env_ids,
        reset_due_env_ids=reset_due_env_ids,
        terminated=terminated,
        record=raw,
        obs=snapshot,
    )
