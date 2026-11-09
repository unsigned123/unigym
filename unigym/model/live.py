"""Live zero-copy proxies over the static model IR.

``LiveModel`` binds one :class:`~unigym.model.ir.ModelIR` to a live env
without copying any static row: every proxy keeps a reference to its
static ``Joint`` / ``Body`` row and resolves the compiled address or id
on demand from the env's native model (cached, never duplicated as IR).

Dynamic reads return read-only numpy views of the backend's own state
(``qpos``/``qvel``/``xpos``/``xquat``); a single-env read is a view, all
envs may be stacked.  After ``env.close()`` every dynamic read raises
:class:`~unigym.model.errors.StaleModelError`; static IR reads keep
working because they never touch the native world.

Only the MuJoCo family is bound today (``MujocoEnv`` and its ``mjwarp``
mirror, which shares the ``MjModel``/``MjData`` contract).  Any other
backend fails fast with ``NotImplementedError``.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from ._serialize import JsonValue, row_dict
from .ir import Body, Joint, ModelIR
from .table import Table

if TYPE_CHECKING:
    from unigym.core.quat_types import QuatWXYZ

__all__ = ["LiveBody", "LiveJoint", "LiveModel", "LiveTable"]

_LiveT = TypeVar("_LiveT")

_QPOS_SIZE = {"hinge": 1, "slide": 1, "ball": 4, "free": 7}
_QVEL_SIZE = {"hinge": 1, "slide": 1, "ball": 3, "free": 6}

_Floats = NDArray[np.float64]


class _MjData(Protocol):
    """The ``MjData`` mirror fields the live proxies read."""

    qpos: _Floats
    qvel: _Floats
    xpos: _Floats
    xquat: _Floats


@runtime_checkable
class _LiveBinder(Protocol):
    """The backend half of a live binding (implemented by the engine layer)."""

    def select(
        self, env: int | None,
    ) -> tuple[object, Sequence[_MjData], Sequence[int]]:
        """Return ``(native_env, datas, indices)`` for an env selector."""
        ...

    def joint_addresses(
        self, native: object, info: Joint,
    ) -> tuple[int, int]:
        """Return ``(qpos_adr, dof_adr)`` for one joint."""
        ...

    def body_id(self, native: object, info: Body) -> int:
        """Return the compiled body id for one body."""
        ...


def _readonly(array: _Floats) -> _Floats:
    """Return *array* with its write flag cleared (a view stays a view)."""
    array.flags.writeable = False
    return array


class LiveTable(Table[_LiveT], Generic[_LiveT]):
    """A ``Table`` whose slicing/filtering keeps the live row type."""

    __slots__ = ()

    @overload
    def __getitem__(self, key: int) -> _LiveT:
        ...

    @overload
    def __getitem__(self, key: str) -> _LiveT:
        ...

    @overload
    def __getitem__(self, key: slice) -> LiveTable[_LiveT]:
        ...

    def __getitem__(
        self, key: int | str | slice,
    ) -> _LiveT | LiveTable[_LiveT]:
        """Slice into a ``LiveTable``; int/name lookup stays Table-like."""
        if isinstance(key, slice):
            return LiveTable(self.values()[key])
        return super().__getitem__(key)

    def matching(self, regex: str | re.Pattern[str]) -> LiveTable[_LiveT]:
        """Return a ``LiveTable`` of rows matching *regex*."""
        return LiveTable(super().matching(regex))

    def filter(
        self, pred: Callable[[_LiveT], bool] | None = None,
        **fields: JsonValue,
    ) -> LiveTable[_LiveT]:
        """Return a ``LiveTable`` satisfying *pred* and equality fields."""
        return LiveTable(super().filter(pred, **fields))


class LiveJoint:
    """One joint proxy: static fields from the IR, state from the env."""

    __slots__ = ("_model", "_info")

    def __init__(self, model: LiveModel, info: Joint) -> None:
        """Bind the static joint *info* to *model* (no data is copied)."""
        self._model = model
        self._info = info

    @property
    def info(self) -> Joint:
        """The static :class:`~unigym.model.ir.Joint` row."""
        return self._info

    @property
    def name(self) -> str:
        """The joint's compiled global name."""
        return self._info.name

    def __getattr__(self, name: str) -> Any:
        """Delegate every static field (``type``, ``range``, ...) to the IR.

        The IR row is the source of truth and carries dozens of stable
        dataclass fields; reflecting them here is the intended dynamic
        surface, so the return type is genuinely open.
        """
        return getattr(self._info, name)

    def qpos(self, env: int | None = None) -> _Floats:
        """Return this joint's ``qpos`` as a read-only numpy array.

        Args:
            env: ``None`` reads every env (``(N, k)``); an int reads one
                env (``(k,)``, a zero-copy view of the backend state).

        Returns:
            The position coordinates: 1 for hinge/slide, 4 for ball and
            7 for free joints.
        """
        single = env is not None
        native, datas, indices = self._model._select(env)
        adr, _ = self._model._joint_addresses(native, self._info)
        size = _qpos_size(self._info)
        if single:
            return _readonly(datas[indices[0]].qpos[adr:adr + size])
        rows = [datas[e].qpos[adr:adr + size] for e in indices]
        return _readonly(np.stack(rows))

    def qvel(self, env: int | None = None) -> _Floats:
        """Return this joint's ``qvel`` as a read-only numpy array.

        Args:
            env: ``None`` reads every env (``(N, k)``); an int reads one
                env (``(k,)``, a zero-copy view of the backend state).

        Returns:
            The velocity coordinates: 1 for hinge/slide, 3 for ball and
            6 for free joints.
        """
        single = env is not None
        native, datas, indices = self._model._select(env)
        _, dof_adr = self._model._joint_addresses(native, self._info)
        size = _qvel_size(self._info)
        if single:
            return _readonly(datas[indices[0]].qvel[dof_adr:dof_adr + size])
        rows = [datas[e].qvel[dof_adr:dof_adr + size] for e in indices]
        return _readonly(np.stack(rows))

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the static joint row as a plain dictionary."""
        return row_dict(self._info)

    def __repr__(self) -> str:
        """Return a compact proxy description."""
        return f"LiveJoint(name={self.name!r}, info={self._info!r})"


class LiveBody:
    """One body proxy: static fields from the IR, world pose from the env."""

    __slots__ = ("_model", "_info")

    def __init__(self, model: LiveModel, info: Body) -> None:
        """Bind the static body *info* to *model* (no data is copied)."""
        self._model = model
        self._info = info

    @property
    def info(self) -> Body:
        """The static :class:`~unigym.model.ir.Body` row."""
        return self._info

    @property
    def name(self) -> str:
        """The body's compiled global name."""
        return self._info.name

    def __getattr__(self, name: str) -> Any:
        """Delegate every static field (``parent``, ``mass``, ...) to the IR.

        As with :class:`LiveJoint`, the dataclass row is the source of
        truth and this dynamic delegation is the intended surface.
        """
        return getattr(self._info, name)

    def world_pose(
        self, env: int | None = None,
    ) -> (
        tuple[_Floats, QuatWXYZ]
        | tuple[_Floats, list[QuatWXYZ]]
    ):
        """Return ``(pos, quat)`` in the world frame, read-only.

        Args:
            env: ``None`` reads every env (``(N, 3)`` plus a list of
                ``QuatWXYZ``); an int reads one env (``(3,)`` position
                view plus a ``QuatWXYZ`` wrapping the ``xquat`` view).

        Returns:
            A ``(position, quaternion)`` pair; the quaternion is a
            :class:`~unigym.core.quat_types.QuatWXYZ` (Q0 order).
        """
        from unigym.core.quat_types import QuatWXYZ

        single = env is not None
        native, datas, indices = self._model._select(env)
        body_id = self._model._body_id(native, self._info)
        if single:
            data = datas[indices[0]]
            return (_readonly(data.xpos[body_id]),
                    QuatWXYZ(_readonly(data.xquat[body_id])))
        positions = np.stack([datas[e].xpos[body_id] for e in indices])
        quats = [QuatWXYZ(_readonly(datas[e].xquat[body_id]))
                 for e in indices]
        return _readonly(positions), quats

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the static body row as a plain dictionary."""
        return row_dict(self._info)

    def __repr__(self) -> str:
        """Return a compact proxy description."""
        return f"LiveBody(name={self.name!r}, info={self._info!r})"


def _qpos_size(info: Joint) -> int:
    """Return the qpos width for one joint row or fail loudly."""
    size = _QPOS_SIZE.get(info.type)
    if size is None:
        raise ValueError(
            f"live model: joint {info.name!r} has unsupported type "
            f"{info.type!r} (expected hinge/slide/ball/free)")
    return size


def _qvel_size(info: Joint) -> int:
    """Return the qvel width for one joint row or fail loudly."""
    size = _QVEL_SIZE.get(info.type)
    if size is None:
        raise ValueError(
            f"live model: joint {info.name!r} has unsupported type "
            f"{info.type!r} (expected hinge/slide/ball/free)")
    return size


def _backend_label(ir: ModelIR, env: object) -> str:
    """Return the backend name used in errors (IR first, env fallback)."""
    return ir.backend or type(env).__name__


def _resolve_live_binder(
    ir: ModelIR, env: object, label: str,
) -> _LiveBinder:
    """Resolve the backend's ``live_provider`` face (registry, lazy).

    The neutral layer names no engine: the binder class is imported only
    when a ``BackendInfo.live_provider`` factory is actually invoked.
    """
    provider: Callable[[], type] | None = None
    try:
        from unigym.backends.registry import backend_info, build_name

        provider = backend_info(build_name(label)).live_provider
    except (ValueError, RuntimeError):
        provider = None
    if provider is None:
        raise NotImplementedError(
            f"live proxies: backend {label!r} has no live model binding "
            f"(hint: live proxies: mujoco/mjwarp today)")
    binder = provider()(ir, env)
    if not isinstance(binder, _LiveBinder):
        raise NotImplementedError(
            f"live proxies: backend {label!r} returned an invalid binder")
    return binder


class LiveModel:
    """The static IR plus live reads, bound to one backend env."""

    __slots__ = ("_ir", "_backend", "_binder", "_joints", "_bodies")

    def __init__(self, ir: ModelIR, env: object) -> None:
        """Bind *ir* to *env* without copying any static row.

        Args:
            ir: The assembled :class:`~unigym.model.ir.ModelIR`.
            env: The live backend env (weakly referenced by the binder).

        Raises:
            NotImplementedError: The backend has no live binding yet.
        """
        self._ir = ir
        self._backend = _backend_label(ir, env)
        self._binder = _resolve_live_binder(ir, env, self._backend)
        self._joints = LiveTable[LiveJoint](
            [LiveJoint(self, row) for row in ir.joints])
        self._bodies = LiveTable[LiveBody](
            [LiveBody(self, row) for row in ir.bodies])

    @property
    def ir(self) -> ModelIR:
        """The bound static model IR (never a copy)."""
        return self._ir

    @property
    def backend(self) -> str:
        """The backend label used by this binding."""
        return self._backend

    @property
    def joints(self) -> LiveTable[LiveJoint]:
        """The live joint table (same lookup surface as the static Table)."""
        return self._joints

    @property
    def bodies(self) -> LiveTable[LiveBody]:
        """The live body table (same lookup surface as the static Table)."""
        return self._bodies

    def _select(
        self, env: int | None,
    ) -> tuple[object, Sequence[_MjData], Sequence[int]]:
        """Return ``(native_env, datas, indices)`` for an env selector."""
        return self._binder.select(env)

    def _joint_addresses(
        self, native: object, info: Joint,
    ) -> tuple[int, int]:
        """Return and cache ``(qpos_adr, dof_adr)`` for one joint."""
        return self._binder.joint_addresses(native, info)

    def _body_id(self, native: object, info: Body) -> int:
        """Return and cache the compiled body id for one body."""
        return self._binder.body_id(native, info)

    def __repr__(self) -> str:
        """Return a compact binding summary."""
        return (f"LiveModel(backend={self._backend!r}, "
                f"joints={len(self._joints)}, bodies={len(self._bodies)})")
