"""MuJoCo-family live model binder.

The engine half of :mod:`unigym.model.live`: resolves compiled joint/body
addresses from the backend's ``MjModel`` and hands out the env's ``MjData``
mirrors (``mj_data`` for N=1, ``mj_datas`` for multi-env).  ``mjwarp`` shares
this binder (its CPU mirrors expose the same ``MjModel``/``MjData``
contract); the registry's ``live_provider`` face keeps the neutral layer
engine-free at import time.
"""
from __future__ import annotations

import weakref

from unigym.model.errors import StaleModelError


class MujocoLiveBackend:
    """Zero-copy live binding over one MuJoCo-family env."""

    def __init__(self, ir, env):
        """Bind *ir* to *env* (weakly referenced) without copying IR rows."""
        self.ir = ir
        self._env_ref = weakref.ref(env)
        self.backend = getattr(ir, "backend", None) or type(env).__name__
        self._cache = {}

    def live_env(self):
        """Resolve the live env or raise :class:`StaleModelError`."""
        env = self._env_ref()
        if env is None:
            raise StaleModelError(
                f"live model proxy for backend {self.backend!r} is stale: "
                f"the env object was released; static ModelIR reads remain "
                f"valid")
        if (getattr(env, "_closed", False)
                or getattr(env, "mj_model", None) is None
                or getattr(env, "mj_data", None) is None):
            raise StaleModelError(
                f"live model proxy for backend {self.backend!r} was read "
                f"after env.close(); static ModelIR reads remain valid")
        return env

    def select(self, env):
        """Return ``(native_env, datas, indices)`` for an env selector."""
        native = self.live_env()
        datas = getattr(native, "mj_datas", None)
        if datas is None:
            data = getattr(native, "mj_data", None)
            datas = [data] if data is not None else []
        if env is None:
            return native, datas, range(len(datas))
        index = int(env)
        count = len(datas)
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise IndexError(
                f"live model: env index {env!r} out of range for "
                f"{count} envs")
        return native, datas, (index,)

    def joint_addresses(self, native, info):
        """Return and cache ``(qpos_adr, dof_adr)`` for one joint."""
        key = ("joint", info.name)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        model = native.mj_model
        import mujoco

        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, info.name)
        if joint_id >= 0:
            value = (int(model.jnt_qposadr[joint_id]),
                     int(model.jnt_dofadr[joint_id]))
        elif info.qpos_adr is not None and info.dof_adr is not None:
            value = (int(info.qpos_adr), int(info.dof_adr))
        else:
            raise ValueError(
                f"live model: joint {info.name!r} is not in the compiled "
                f"model")
        self._cache[key] = value
        return value

    def body_id(self, native, info):
        """Return and cache the compiled body id for one body."""
        key = ("body", info.name)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        model = native.mj_model
        import mujoco

        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, info.name)
        if body_id < 0 and info.body_id is not None:
            body_id = int(info.body_id)
        if body_id < 0:
            raise ValueError(
                f"live model: body {info.name!r} is not in the compiled "
                f"model")
        self._cache[key] = body_id
        return body_id
