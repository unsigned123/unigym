"""MuJoCo-Warp camera rigs (v095 C3).

Two render paths behind one factory:

* **N == 1** -- the historical CPU-mirror path, verbatim.  ``MjwarpRig``
  delegates to the exact same ``mujoco.Renderer`` rig classes the mujoco
  backend uses (``backends/mujoco/cameras.build_mj_rig`` via the
  neutral registry hub), so the single-env
  mjwarp camera/physics trajectory stays bit-identical (no golden churn).
* **N > 1** -- the native warp batch renderer (``mjwarp/render.py``): one
  render context over the batched ``Data`` (``nworld=N``), per-world camera
  poses written into the model's ``cam_pos``/``cam_quat`` batch columns and
  refreshed with ``mjw.camlight`` (``mjw.kinematics`` does not refresh
  camera frames in mujoco-warp 3.17).

Camera mounts (the same MjSpec injection the mujoco backend uses):

* ``body`` -- the ``<camera>`` is welded to the mount body; its model
  ``cam_pos``/``cam_quat`` are BODY-relative, so ``camlight`` derives a
  per-world world pose from each world's own robot FK.  We only write the
  model column when a live mount offset is set (default zero writes the
  injected value back unchanged).
* ``world`` -- worldbody camera; per-world poses come from the C2 state table
  (``shared`` broadcasts one row, ``per_instance`` keeps one per env).
* ``chase`` -- worldbody placeholder re-posed every step tail from each
  world's mirror root (robot-yaw-rotated offsets + look-at aim).

Depth is METRIC planar z from ``rc.depth_data`` (never ``get_depth``); see
``render.py``.  Obs/record key space is untouched: ``read()`` always returns
``(N, ·)`` blocks.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from unigym.cameras.base import CameraSpec, depth_key, rgb_key, rig_type
from unigym.cameras.state import make_state_table
from unigym.cameras.rigs import DEFAULT_CHASE_OFFSET, lookat_quat, \
    yaw_pitch_roll_quat

from ..shared.mjfamily.cameras.helpers import (MJ_CAMERA_PREFIX,
                                              _quat_mul_np,
                                              _rig_offset_quat,
                                              inject_spec_camera,
                                              mj_camera_quat)


def _cpu_rig_factory():
    """Load the N=1 CPU-mirror rig builder without importing the backend.

    The concrete ``mujoco.Renderer`` rig implementation stays in
    ``backends/mujoco/cameras.py`` (v100 G6); the neutral registry is the
    sanctioned lazy hub, so mjwarp reuses the exact same rig classes while
    importing no other backend module.
    """
    from unigym.backends.registry import backend_info

    return backend_info("mujoco").camera_backend()

__all__ = ["MjwarpRig", "build_mjwarp_rig"]


def _ypr_quat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Per-env yaw/pitch/roll quaternion (wxyz), or identity."""
    return yaw_pitch_roll_quat(float(yaw), float(pitch), float(roll))


class _WarpRigBase:
    """Shared warp-rig plumbing: pose writes, render-on-read, live state."""

    def __init__(self, spec: CameraSpec):
        self.spec = spec
        self.name = spec.name
        self._mj_name = MJ_CAMERA_PREFIX + spec.name
        self._env = None
        self._cam_id = None
        self._renderer = None
        self._wm = None
        self._wd = None
        rig = spec.rig or {}
        self.target_env = int(rig.get("target_env", 0))
        self.render_spec = None
        self.render_source = None
        self.state_table = make_state_table(self.name,
                                            spec.state or "shared")
        self._mount_pos = np.asarray(
            [float(x) for x in rig.get("pos", (0.0, 0.0, 0.0))],
            dtype=np.float64)
        self._mount_quat = np.asarray(_rig_offset_quat(spec),
                                      dtype=np.float64)
        # live per-instance properties (mount-frame offsets)
        self._pose_off = None   # (N, 3)
        self._yaw = None        # (N,)
        self._pitch = None
        self._roll = None

    # -- lifecycle -------------------------------------------------------- #
    def setup_scene(self, mjspec) -> None:
        """Inject the ``<camera>`` element (pre-compile, shared code)."""
        inject_spec_camera(self.spec, mjspec)

    def bind_env(self, env) -> None:
        import mujoco
        from .render import WarpRenderer

        self._env = env
        cid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA,
                                self._mj_name)
        if cid < 0:
            raise RuntimeError(
                f"mjwarp camera '{self._mj_name}' missing from the compiled "
                f"model (injection lost?)")
        self._cam_id = int(cid)
        self._wm, self._wd = env._wm, env._wdata
        self._renderer = WarpRenderer(
            env.mj_model, self._wm, self._wd, int(env.num_envs),
            [self._mj_name], int(self.spec.width), int(self.spec.height),
            with_depth="depth" in self.spec.types)
        self._init_state_rows()

    def close(self) -> None:
        """Drop the render context + context-local warp arrays."""
        self._renderer = None

    # -- addressing ------------------------------------------------------- #
    def _shared(self) -> bool:
        return self.spec.state == "shared"

    def _validate_ids(self, env_ids) -> None:
        if self._env is None or env_ids is None:
            return
        n = int(self._env.num_envs)
        ids = sorted({int(i) for i in env_ids})
        for i in ids:
            if not 0 <= i < n:
                raise ValueError(
                    f"camera '{self.name}': env_id {i} out of range for "
                    f"{n} env(s)")
        if self._shared() and ids != [self.target_env]:
            raise ValueError(
                f"camera '{self.name}' is state=shared: it has a single "
                f"meaningful instance (env {self.target_env} = target_env); "
                f"env_ids must be omitted or [{self.target_env}]")

    # -- live properties -------------------------------------------------- #
    def _ensure_live(self) -> None:
        if self._pose_off is None:
            n = int(self._env.num_envs)
            self._pose_off = np.zeros((n, 3), dtype=np.float64)
            self._yaw = np.zeros(n, dtype=np.float64)
            self._pitch = np.zeros(n, dtype=np.float64)
            self._roll = np.zeros(n, dtype=np.float64)

    def apply_properties(self, pose_offset=None, yaw_deg=None,
                         pitch_deg=None, roll_deg=None, target_env=None,
                         env_ids: list[int] | None = None) -> None:
        """Fold the live whitelist into the per-instance pose rows."""
        self._validate_ids(env_ids)
        self._ensure_live()
        sel = np.arange(int(self._env.num_envs)) if env_ids is None \
            else np.asarray([int(i) for i in env_ids], dtype=np.int64)
        if pose_offset is not None:
            off = np.asarray([float(x) for x in pose_offset], dtype=np.float64)
            if off.shape != (3,):
                raise ValueError(f"camera '{self.name}': pose_offset must be "
                                 f"[dx, dy, dz], got {list(off)}")
            self._pose_off[sel] = off
        if yaw_deg is not None:
            self._yaw[sel] = math.radians(float(yaw_deg))
        if pitch_deg is not None:
            self._pitch[sel] = math.radians(float(pitch_deg))
        if roll_deg is not None:
            self._roll[sel] = math.radians(float(roll_deg))
        if target_env is not None:
            self.target_env = int(target_env)
        self.sync_pose(env_ids=env_ids)

    def _live_pose(self, env_idx: int, base_pos, base_quat):
        """Base mount pose ∘ this instance's live offset (isaac quat)."""
        p = np.asarray(base_pos, dtype=np.float64)
        q = np.asarray(base_quat, dtype=np.float64)
        if self._pose_off is not None:
            p = p + self._pose_off[env_idx]
            y, pt, r = (self._yaw[env_idx], self._pitch[env_idx],
                        self._roll[env_idx])
            if y or pt or r:
                q = _quat_mul_np(_ypr_quat(y, pt, r), q)
        return p, q

    # -- posing ----------------------------------------------------------- #
    def _init_state_rows(self) -> None:
        """Seed the state table from the declared pose (world rigs)."""

    def apply_declared_poses(self, env_poses) -> None:
        """Apply declared per-env poses at bind/reset (world rig overrides)."""
        del env_poses

    def sync_pose(self, env_ids=None) -> None:
        """Write this camera's model column and refresh ``cam_xpos``."""
        if self._env is None:
            return
        self._validate_ids(env_ids)
        pos, quat = self._model_poses()
        self._write_column(pos, quat)

    def on_reset(self, env_ids) -> None:
        # shared cameras have one meaningful instance; the reset subset may
        # name any worlds, so route them to their single target_env row.
        self.sync_pose(None if self._shared() else env_ids)

    def _model_poses(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def _write_column(self, pos, quat_isaac) -> None:
        """Write ``(N,3)/(N,4)`` model-relative poses + camlight."""
        import mujoco_warp as mjw
        import warp as wp

        pos = np.ascontiguousarray(pos, dtype=np.float32)
        quat = np.asarray([mj_camera_quat(q) for q in quat_isaac],
                          dtype=np.float32)
        dev = self._wm.cam_pos.device
        wp.copy(self._wm.cam_pos[:, self._cam_id],
                wp.array(pos, dtype=wp.vec3, device=dev))
        wp.copy(self._wm.cam_quat[:, self._cam_id],
                wp.array(quat, dtype=wp.quat, device=dev))
        # kinematics() does not refresh camera frames in mujoco-warp 3.17.
        mjw.camlight(self._wm, self._wd)

    # -- reading ---------------------------------------------------------- #
    def read(self, env_ids=None) -> dict:
        """Render once, return this rig's ``(N, ·)`` streams."""
        self._validate_ids(env_ids)
        self._renderer.render()
        out = {}
        for key in self.spec.keys():
            if key == rgb_key(self.name):
                out[key] = torch.from_numpy(self._renderer.rgb(0))
            elif key == depth_key(self.name):
                out[key] = torch.from_numpy(
                    self._renderer.depth(0, float(self.spec.far_clip)))
            else:  # pragma: no cover - CameraSpec.keys() is closed
                raise KeyError(key)
        return out

    # -- world pose faces ------------------------------------------------- #
    def get_world_pose(self):
        raise NotImplementedError(
            f"camera '{self.name}': world-pose read-back is supported on "
            f"world rigs only")


class WarpBodyRig(_WarpRigBase):
    """Body-welded camera: model-relative pose, per-world FK derives it."""

    def _model_poses(self):
        n = int(self._env.num_envs)
        pos, quat = [], []
        for e in range(n):
            p, q = self._live_pose(e, self._mount_pos, self._mount_quat)
            pos.append(p)
            quat.append(q)
        return pos, quat


class WarpWorldRig(_WarpRigBase):
    """World-anchored tripod with the C2 ``(camera, env)`` pose table."""

    def __init__(self, spec: CameraSpec):
        super().__init__(spec)
        rig = spec.rig or {}
        pos = [float(x) for x in rig.get("pos", (0.0, 0.0, 0.0))]
        if rig.get("rot") is not None:
            rot = np.asarray([float(x) for x in rig["rot"]],
                             dtype=np.float64)
        else:
            rot = np.asarray(lookat_quat(pos, rig.get("lookat",
                                                      (1.0, 0.0, 0.25))),
                            dtype=np.float64)
        self._base_pos = np.asarray(pos, dtype=np.float64)
        self._base_rot = rot
        self._declared: dict[int, tuple] | None = None

    def _init_state_rows(self) -> None:
        for i in range(int(self._env.num_envs)):
            self.state_table.set(self.name, i, self._base_pos.copy(),
                                 self._base_rot.copy())

    def apply_declared_poses(self, env_poses) -> None:
        n = int(self._env.num_envs)
        if not env_poses:
            self._declared = None
        else:
            rows = [env_poses[0]] * n if len(env_poses) == 1 \
                else list(env_poses)
            self._declared = {
                i: (np.asarray(rows[i]["pos"], dtype=np.float64),
                    np.asarray(rows[i]["quat"], dtype=np.float64))
                for i in range(n)}
        self.sync_pose(None if self._shared() else list(range(n)))

    def _base_for(self, env_idx: int) -> tuple:
        if self._declared is not None:
            return self._declared[env_idx]
        return self._base_pos, self._base_rot

    def _model_poses(self):
        n = int(self._env.num_envs)
        if self._shared():
            bp, bq = self._base_for(self.target_env)
            p, q = self._live_pose(self.target_env, bp, bq)
            self.state_table.set(self.name, None, p, q)
            return [p] * n, [q] * n
        pos, quat = [], []
        for e in range(n):
            bp, bq = self._base_for(e)
            p, q = self._live_pose(e, bp, bq)
            self.state_table.set(self.name, e, p, q)
            pos.append(p)
            quat.append(q)
        return pos, quat

    def get_world_pose(self):
        row = self.state_table.get(self.name, self.target_env)
        if row is None:
            self.sync_pose()
            row = self.state_table.get(self.name, self.target_env)
        return row.pos, row.quat

    def set_world_pose(self, pos, quat_wxyz, env_idx: int | None = None):
        idx = self.target_env if env_idx is None else int(env_idx)
        if not 0 <= idx < int(self._env.num_envs):
            raise ValueError(f"camera '{self.name}': env_idx {idx} out of "
                             f"range")
        self.state_table.set(self.name, None if self._shared() else idx,
                             np.asarray(pos, dtype=np.float64),
                             np.asarray(quat_wxyz, dtype=np.float64))
        self.sync_pose()

    def begin_world_pose(self, pos, quat_wxyz):
        saved = {k: (v.pos.copy(), v.quat.copy())
                 for k, v in self.state_table.rows(self.name).items()}
        self.set_world_pose(pos, quat_wxyz)
        return saved

    def end_world_pose(self, saved) -> None:
        for env, (p, q) in saved.items():
            self.state_table.set(self.name, env, p.copy(), q.copy())
        self.sync_pose()


class WarpChaseRig(_WarpRigBase):
    """Per-world chase: robot-yaw offsets around each mirror's root."""

    def __init__(self, spec: CameraSpec):
        super().__init__(spec)
        rig = spec.rig or {}
        self._pos_off = np.asarray(
            [float(x) for x in rig.get("pos", DEFAULT_CHASE_OFFSET["pos"])],
            dtype=np.float64)
        self._look_off = np.asarray(
            [float(x) for x in rig.get("lookat",
                                       DEFAULT_CHASE_OFFSET["lookat"])],
            dtype=np.float64)

    def _root(self, env_idx: int) -> tuple:
        env = self._env
        if not getattr(env, "_has_freejoint", True):
            return (np.asarray(env._fixed_base_pos_w, dtype=np.float64),
                    np.asarray(env._fixed_base_quat_w, dtype=np.float64))
        d = env.mj_datas[env_idx]
        return d.qpos[0:3].copy(), d.qpos[3:7].copy()

    def _chase_pose(self, env_idx: int) -> tuple:
        root, root_q = self._root(env_idx)
        w, x, y, z = root_q
        yaw = math.atan2(2.0 * (w * z + x * y),
                         1.0 - 2.0 * (y * y + z * z))
        c, s = math.cos(yaw), math.sin(yaw)

        def rot_xy(off):
            off3 = np.asarray(off, dtype=np.float64)
            return np.array([off3[0] * c - off3[1] * s,
                             off3[0] * s + off3[1] * c, off3[2]])

        live = np.zeros(3) if self._pose_off is None else self._pose_off[env_idx]
        pos = root + rot_xy(self._pos_off + live)
        lookat = root + rot_xy(self._look_off)
        return pos, np.asarray(lookat_quat(pos, lookat), dtype=np.float64)

    def _model_poses(self):
        n = int(self._env.num_envs)
        if self._shared():
            p, q = self._chase_pose(self.target_env)
            return [p] * n, [q] * n
        pos, quat = [], []
        for e in range(n):
            p, q = self._chase_pose(e)
            pos.append(p)
            quat.append(q)
        return pos, quat

    def get_world_pose(self):
        p, q = self._chase_pose(self.target_env)
        return p, q


_WARP_RIGS = {"body": WarpBodyRig, "world": WarpWorldRig,
              "chase": WarpChaseRig}


class MjwarpRig:
    """N-selecting facade: N=1 -> CPU-mirror rig, N>1 -> warp batch rig."""

    def __init__(self, spec: CameraSpec, render_groups=None):
        self.spec = spec
        self.name = spec.name
        self._cpu = _cpu_rig_factory()(spec, render_groups=render_groups)
        self._impl = None

    def setup_scene(self, scene_cfg) -> None:
        """Inject once through the proven mujoco rig (pre-compile)."""
        self._cpu.setup_scene(scene_cfg)

    def bind_env(self, env) -> None:
        """Pick the render path from ``env.num_envs`` and bind it."""
        if int(env.num_envs) == 1:
            self._impl = self._cpu
        else:
            rtype = rig_type(self.spec)
            cls = _WARP_RIGS.get(rtype)
            if cls is None:
                raise ValueError(
                    f"camera {self.name!r}: unknown rig type {rtype!r} "
                    f"(body|world|chase)")
            self._impl = cls(self.spec)
        self._impl.bind_env(env)

    def __getattr__(self, item):
        impl = self.__dict__.get("_impl")
        if impl is None:
            raise AttributeError(item)
        return getattr(impl, item)


def build_mjwarp_rig(spec: CameraSpec, render_groups=None) -> MjwarpRig:
    """Factory for the mjwarp backend's camera rigs (v095 C3)."""
    return MjwarpRig(spec, render_groups=render_groups)
