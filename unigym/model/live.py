"""Live zero-copy proxies over the static model IR (design v098 N3).

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

import numpy as np

from .table import Table

__all__ = ["LiveBody", "LiveJoint", "LiveModel", "LiveTable"]

_QPOS_SIZE = {"hinge": 1, "slide": 1, "ball": 4, "free": 7}
_QVEL_SIZE = {"hinge": 1, "slide": 1, "ball": 3, "free": 6}


def _readonly(array):
    """Return *array* with its write flag cleared (a view stays a view)."""
    array.flags.writeable = False
    return array


def _row_dict(info):
    """Return one static dataclass row as a plain dictionary."""
    from dataclasses import fields

    return {field.name: getattr(info, field.name) for field in fields(info)}


class LiveTable(Table):
    """A ``Table`` whose slicing/filtering keeps the live row type."""

    __slots__ = ()

    def __getitem__(self, key):
        """Slice into a ``LiveTable``; int/name lookup stays Table-like."""
        if isinstance(key, slice):
            return LiveTable(self.values()[key])
        return super().__getitem__(key)

    def matching(self, regex):
        """Return a ``LiveTable`` of rows matching *regex*."""
        return LiveTable(super().matching(regex))

    def filter(self, pred=None, **fields):
        """Return a ``LiveTable`` satisfying *pred* and equality fields."""
        return LiveTable(super().filter(pred, **fields))


class LiveJoint:
    """One joint proxy: static fields from the IR, state from the env."""

    __slots__ = ("_model", "_info")

    def __init__(self, model, info):
        """Bind the static joint *info* to *model* (no data is copied)."""
        self._model = model
        self._info = info

    @property
    def info(self):
        """The static :class:`~unigym.model.ir.Joint` row."""
        return self._info

    @property
    def name(self):
        """The joint's compiled global name."""
        return self._info.name

    def __getattr__(self, name):
        """Delegate every static field (``type``, ``range``, ...) to the IR."""
        return getattr(self._info, name)

    def qpos(self, env=None):
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

    def qvel(self, env=None):
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

    def to_dict(self):
        """Return the static joint row as a plain dictionary."""
        return _row_dict(self._info)

    def __repr__(self):
        """Return a compact proxy description."""
        return f"LiveJoint(name={self.name!r}, info={self._info!r})"


class LiveBody:
    """One body proxy: static fields from the IR, world pose from the env."""

    __slots__ = ("_model", "_info")

    def __init__(self, model, info):
        """Bind the static body *info* to *model* (no data is copied)."""
        self._model = model
        self._info = info

    @property
    def info(self):
        """The static :class:`~unigym.model.ir.Body` row."""
        return self._info

    @property
    def name(self):
        """The body's compiled global name."""
        return self._info.name

    def __getattr__(self, name):
        """Delegate every static field (``parent``, ``mass``, ...) to the IR."""
        return getattr(self._info, name)

    def world_pose(self, env=None):
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

    def to_dict(self):
        """Return the static body row as a plain dictionary."""
        return _row_dict(self._info)

    def __repr__(self):
        """Return a compact proxy description."""
        return f"LiveBody(name={self.name!r}, info={self._info!r})"


def _qpos_size(info):
    """Return the qpos width for one joint row or fail loudly."""
    size = _QPOS_SIZE.get(info.type)
    if size is None:
        raise ValueError(
            f"live model: joint {info.name!r} has unsupported type "
            f"{info.type!r} (expected hinge/slide/ball/free)")
    return size


def _qvel_size(info):
    """Return the qvel width for one joint row or fail loudly."""
    size = _QVEL_SIZE.get(info.type)
    if size is None:
        raise ValueError(
            f"live model: joint {info.name!r} has unsupported type "
            f"{info.type!r} (expected hinge/slide/ball/free)")
    return size


def _backend_label(ir, env):
    """Return the backend name used in errors (IR first, env fallback)."""
    return getattr(ir, "backend", None) or type(env).__name__


def _resolve_live_binder(ir, env, label):
    """Resolve the backend's ``live_provider`` face (registry, lazy).

    The neutral layer names no engine: the binder class is imported only
    when a ``BackendInfo.live_provider`` factory is actually invoked.
    """
    provider = None
    try:
        from unigym.backends.registry import backend_info, build_name

        provider = backend_info(build_name(label)).live_provider
    except (ValueError, RuntimeError):
        provider = None
    if provider is None:
        raise NotImplementedError(
            f"live proxies: backend {label!r} has no live model binding "
            f"(hint: live proxies: mujoco/mjwarp today)")
    return provider()(ir, env)


class LiveModel:
    """The static IR plus live reads, bound to one backend env."""

    __slots__ = ("_ir", "_backend", "_binder", "_joints", "_bodies")

    def __init__(self, ir, env):
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
        self._joints = LiveTable(
            [LiveJoint(self, row) for row in ir.joints])
        self._bodies = LiveTable(
            [LiveBody(self, row) for row in ir.bodies])

    @property
    def ir(self):
        """The bound static model IR (never a copy)."""
        return self._ir

    @property
    def backend(self):
        """The backend label used by this binding."""
        return self._backend

    @property
    def joints(self):
        """The live joint table (same lookup surface as the static Table)."""
        return self._joints

    @property
    def bodies(self):
        """The live body table (same lookup surface as the static Table)."""
        return self._bodies

    def _select(self, env):
        """Return ``(native_env, datas, indices)`` for an env selector."""
        return self._binder.select(env)

    def _joint_addresses(self, native, info):
        """Return and cache ``(qpos_adr, dof_adr)`` for one joint."""
        return self._binder.joint_addresses(native, info)

    def _body_id(self, native, info):
        """Return and cache the compiled body id for one body."""
        return self._binder.body_id(native, info)

    def __repr__(self):
        """Return a compact binding summary."""
        return (f"LiveModel(backend={self._backend!r}, "
                f"joints={len(self._joints)}, bodies={len(self._bodies)})")
