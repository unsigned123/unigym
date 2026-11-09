"""IsaacLab camera rigs: the three mountings (body / world / chase).

This is the backend-side home of the rig classes that used to live in the
neutral ``unigym/cameras/rigs.py``.  The module is **torch-only at
import time**; isaaclab imports happen inside methods
(``setup_scene``/``sync_pose``/``read``), mirroring the lazy-import
convention of the package.

* :class:`BodyRig` — mounted on the robot base (default) or, with
  ``attach: <body_name>``, on ANY link: the mount pose is that body's
  FK ∘ the mount offset/orientation.  Base mounts re-posed to
  ``root_pose ∘ offset`` at every step TAIL (the next step's render then
  captures this pose) and after every reset — the generalized, exact
  port of the classic ``_sync_depth_camera_pose`` (the standard Camera
  render product is pinned to the prim's world transform, so an explicit
  per-step ``set_world_poses`` is what makes the frame follow the
  robot).  Link mounts own their pose through the kinematic tree.
* :class:`WorldRig` — world-anchored tripod at the env origin (the cfg
  offset poses it at spawn); never re-posed by the physics loop.  NOT
  frozen hardware though: the live-property whitelist re-aims the head
  (yaw/pitch), moves the base (pose_offset) and a transient world pose
  can overwrite one frame — "world" names the ANCHORING, not a lack of
  actuators.
* :class:`ChaseRig` — world-frame follow: held at a robot-YAW-rotated
  offset behind/above the root, aimed at the root, re-posed relative to
  ``target_env``'s robot base every step (the port of the old recorder
  single-env follow; ``recorder/camera.py`` was removed — the
  recorder binds declared rigs only, no self-built camera path).
"""

from __future__ import annotations

import math

import torch

from unigym.cameras.base import CameraSpec, CameraStateTable, rig_type
from unigym.cameras.rigs import (
    DEFAULT_CHASE_OFFSET,
    _lookat_quat_batch,
    _quat_mul_batch,
    _yaw_from_quat,
    _yaw_pitch_roll_quat_batch,
    lookat_quat,
)


def build_isaaclab_rig(spec: CameraSpec, render_groups=None):
    """IsaacLab rig factory: rig instance from the spec's ``rig.type``.

    Resolved through :attr:`BackendInfo.camera_backend` for
    ``isaaclab2``/``isaaclab3``; ``render_groups`` is accepted for the
    registry face signature (mujoco-only in v1).
    """
    del render_groups
    rtype = rig_type(spec)
    if rtype == "body":
        return BodyRig(spec)
    if rtype == "world":
        return WorldRig(spec)
    if rtype == "chase":
        return ChaseRig(spec)
    if rtype == "fixed":
        raise ValueError(
            f"camera {spec.name!r}: rig type 'fixed' was renamed 'world' "
            f"(camera unification) -- update the config; there is no alias"
        )
    raise ValueError(f"camera {spec.name!r}: unknown rig type {rtype!r} " "(body|world|chase)")


class _RigBase:
    """Shared rig plumbing (scene setup / env binding / frame reading).

    Live camera properties (whitelist, docs/contracts/camera_contract.md):
    ``pose_offset`` (mount-frame 3-vector), ``yaw_deg``/``pitch_deg``/
    ``roll_deg`` (mount-frame orientation offset, positive pitch = down,
    roll about the view axis) and ``target_env`` are stored per instance
    and consumed by ``sync_pose``;
    ``apply_properties`` is the single entry (keys already whitelisted by
    the manager).  Zero-valued state keeps the reference math bit-identical.
    """

    def __init__(self, spec: CameraSpec):
        self.spec = spec
        self.name = spec.name
        self._env = None
        self._sensor = None
        self.target_env = int((spec.rig or {}).get("target_env", 0))
        self._target_env_declared = "target_env" in (spec.rig or {})
        self._pose_dirty = False
        self._pose_off = None  # (N, 3) torch, mount frame
        self._yaw = None  # (N,) torch, rad
        self._pitch = None  # (N,) torch, rad
        self._roll = None  # (N,) torch, rad (view-axis roll)
        # the (camera, env) pose state.  ``world``/``chase`` WRITE
        # resolved poses here and the sensor write is driven from these
        # rows; ``body`` stays derived (FK ∘ offset, never stored).
        self.state_table = CameraStateTable()
        self.state_table.declare(self.name, spec.state or "shared")

    # -- lifecycle -------------------------------------------------------- #
    def setup_scene(self, scene_cfg) -> None:
        """Add the sensor cfg to the scene cfg (before env construction)."""
        # lazy: the sensor-cfg builder lives beside this module
        from .cfg import build_sensor_cfg

        setattr(scene_cfg, f"cam_{self.spec.name}", build_sensor_cfg(self.spec))

    def bind_env(self, env) -> None:
        """Grab the cloned sensor handle after env construction.

        NOTE (measured): InteractiveScene keys sensor entities by the
        SCENE-CFG FIELD NAME (the ``cam_<name>`` attribute set in
        ``setup_scene``), not by the prim path's last component.
        """
        self._env = env
        self._sensor = env.scene[f"cam_{self.spec.name}"]
        self._init_state_rows()

    def read(self, env_ids=None) -> dict:
        """Current sensor frames; ``{key: tensor | None}`` (see backend).

        ``env_ids`` scopes which instances are REFRESHED from the annotator;
        the returned block is always ``(N, ·)`` -- the obs/record key space
        never changes (design §4).  The isaaclab annotator is a single
        batched render product, so the refresh is inherently all-instance;
        the argument is accepted for protocol symmetry (mujoco N=1 too).
        """
        from .cfg import read_frames

        del env_ids
        return read_frames(self._sensor, self.spec, self._env.num_envs)

    # -- (camera, env) state -------------------------------- #
    def _state_shared(self) -> bool:
        return self.spec.state == "shared"

    def _all_envs(self) -> list[int]:
        return list(range(int(self._env.num_envs)))

    def _check_env_ids(self, env_ids) -> list[int]:
        n = int(self._env.num_envs)
        ids = self._all_envs() if env_ids is None else sorted({int(i) for i in env_ids})
        for i in ids:
            if not 0 <= i < n:
                raise ValueError(
                    f"camera '{self.name}': env_id {i} out of range for " f"{n} env(s)"
                )
        return ids

    def _served_envs(self, env_ids=None) -> list[int]:
        """Instances addressed by one sync (shared -> target_env only)."""
        if self._state_shared():
            if env_ids is None:
                te = self.target_env
                if not 0 <= te < int(self._env.num_envs):
                    raise ValueError(
                        f"camera '{self.name}': target_env {te} out of "
                        f"range for {int(self._env.num_envs)} env(s)"
                    )
                return [te]
            ids = self._check_env_ids(env_ids)
            if ids != [self.target_env]:
                raise ValueError(
                    f"camera '{self.name}' is state=shared: it has a single "
                    f"meaningful instance (env {self.target_env} = "
                    f"target_env); env_ids must be omitted or "
                    f"[{self.target_env}]"
                )
            return ids
        return self._check_env_ids(env_ids)

    def _init_state_rows(self) -> None:
        """Seed the state table from the declared pose (rigs override)."""

    def apply_declared_poses(self, env_poses) -> None:
        """Apply declared per-env poses at bind/reset (rigs override)."""
        del env_poses

    # -- live properties  ------------------------------------------ #
    def _live_state(self, device):
        """(pose_off, yaw, pitch) on *device*, lazily allocated (N from env)."""
        if self._pose_off is None:
            n = int(self._env.num_envs)
            self._pose_off = torch.zeros(n, 3)
            self._yaw = torch.zeros(n)
            self._pitch = torch.zeros(n)
            self._roll = torch.zeros(n)
        return (
            self._pose_off.to(device),
            self._yaw.to(device),
            self._pitch.to(device),
            self._roll.to(device),
        )

    def _has_orientation_offset(self) -> bool:
        return self._yaw is not None and (
            bool((self._yaw != 0).any())
            or bool((self._pitch != 0).any())
            or bool((self._roll != 0).any())
        )

    def _has_pose_offset(self) -> bool:
        return self._pose_off is not None and bool((self._pose_off != 0).any())

    def _sync_selection(self) -> tuple[slice, list[int] | None]:
        """(row slice, env_ids for set_world_poses) of the pose sync.

        shared -> only the target_env clone (the single meaningful
        instance); per_env -> every clone (body) / target_env (chase, which
        stays single-target — docs/contracts/camera_contract.md §5).
        """
        if self.spec.instance == "shared":
            te = self.target_env
            return slice(te, te + 1), [te]
        return slice(None), None

    def apply_properties(
        self,
        pose_offset=None,
        yaw_deg=None,
        pitch_deg=None,
        roll_deg=None,
        target_env=None,
        env_ids: list[int] | None = None,
    ) -> None:
        """Update the live pose properties (manager pre-validates keys).

        ``env_ids=None`` applies to every instance (per_env semantics);
        a shared camera is addressed without env_id (the manager routes
        that to the target_env instance).  Ends with an immediate re-pose.
        """
        n = int(self._env.num_envs)
        if self._pose_off is None:
            self._pose_off = torch.zeros(n, 3)
            self._yaw = torch.zeros(n)
            self._pitch = torch.zeros(n)
            self._roll = torch.zeros(n)
        sel: any = slice(None) if env_ids is None else torch.tensor(env_ids, dtype=torch.long)
        if pose_offset is not None:
            self._pose_off[sel] = torch.tensor([float(x) for x in pose_offset], dtype=torch.float32)
        if yaw_deg is not None:
            self._yaw[sel] = math.radians(float(yaw_deg))
        if pitch_deg is not None:
            self._pitch[sel] = math.radians(float(pitch_deg))
        if roll_deg is not None:
            self._roll[sel] = math.radians(float(roll_deg))
        if target_env is not None:
            self.target_env = int(target_env)
            self._target_env_declared = True
        self._pose_dirty = True
        self.sync_pose(env_ids=env_ids)

    # -- pose hooks --------------------------------------------------------- #
    def sync_pose(self, env_ids=None) -> None:  # pragma: no cover - overridden
        """Re-pose the camera (subclasses own the math)."""
        raise NotImplementedError

    def on_reset(self, env_ids) -> None:
        """After-reset hook: re-pose to the spawn state (default = sync)."""
        self.sync_pose(env_ids=env_ids)

    # -- transient world pose (ux-7; world rigs only) ----------------------- #
    def begin_world_pose(self, pos, quat_wxyz):
        """Apply a world-frame pose override; return a saved-state token.

        The token feeds :meth:`end_world_pose`.  World rigs only -- every
        other rig type fails fast here (they are re-posed from the robot
        each step, so an override would be misleading).
        """
        raise NotImplementedError(
            f"camera '{self.name}': transient world pose is supported on "
            f"world rigs only (this rig: "
            f"{str((self.spec.rig or {}).get('type', '')).strip().lower()!r}"
            f")"
        )

    def end_world_pose(self, saved) -> None:
        """Restore the pose saved by :meth:`begin_world_pose`."""
        raise NotImplementedError(
            f"camera '{self.name}': transient world pose is supported on " f"world rigs only"
        )

    def set_world_pose(self, pos, quat_wxyz):
        """Persistent absolute world pose (freecam face; world rigs only).

        Body/chase rigs are re-posed from the robot every step and reject
        the override.
        """
        raise NotImplementedError(
            f"camera '{self.name}': world-pose set is supported on "
            f"world rigs only (this rig: "
            f"{str((self.spec.rig or {}).get('type', '')).strip().lower()!r}"
            f")"
        )

    def get_world_pose(self):
        """Read-back: ``(pos(3,), quat_wxyz(4,))`` world convention."""
        raise NotImplementedError(
            f"camera '{self.name}': world-pose read-back is supported on " f"world rigs only"
        )


class BodyRig(_RigBase):
    """Robot mount; per-step pose follows the mount body's FK ∘ offset.

    Default (no ``attach``): the base mount — ``rig.pos`` is base-frame;
    re-posed to ``root ∘ offset`` at every step tail (step-tail
    semantics, ``_sync_depth_camera_pose`` port — see cameras/base.py).
    ``attach: <body_name>`` welds the camera to ANY link: the pose is
    that body's FK ∘ the mount offset/orientation (isaaclab: the sensor
    prim is a robot-link child — see ``cameras.cfg.build_sensor_cfg`` —
    so the kinematic tree owns the pose and per-step re-posing is
    DISABLED; mujoco: the MJCF camera is injected on that body, see
    ``backends/mujoco/cameras.inject_spec_camera``).

    Orientation: explicit ``rig.rot`` (w,x,y,z) or ``pitch_down_deg``
    (rotation about +Y, the pitch quaternion
    ``[cos(p/2), 0, sin(p/2), 0]`` — positive pitch tilts the forward
    axis DOWN).
    """

    def __init__(self, spec: CameraSpec):
        super().__init__(spec)
        rig = spec.rig or {}
        self.attach = str(rig.get("attach", "") or "").strip()
        self._prim_mounted = bool(self.attach)
        self._off_pos = torch.tensor(
            [float(x) for x in rig.get("pos", (0.0, 0.0, 0.0))], dtype=torch.float32
        )
        if rig.get("rot") is not None:
            self._off_rot = torch.tensor([float(x) for x in rig["rot"]], dtype=torch.float32)
        else:
            pitch = math.radians(float(rig.get("pitch_down_deg", 0.0)))
            self._off_rot = torch.tensor(
                [math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0], dtype=torch.float32
            )

    def sync_pose(self, env_ids=None) -> None:
        """Re-pose the render camera to the robot (step-tail semantics).

        The ``_sync_depth_camera_pose`` port — see cameras/base.py.
        shared state: only the ``target_env`` clone follows its robot;
        per_instance (default for body): every addressed clone follows its
        own env's robot.  ``env_ids`` scopes WHICH instances are refreshed;
        the pose is DERIVED (env robot FK ∘ mount offset) and never stored
        (design §4).  Link-mounted prims (``attach``) skip this entirely
        (kinematic parent poses the prim; see class docstring).
        """
        if self._sensor is None or self._prim_mounted:
            return
        from isaaclab.utils import math as math_utils  # lazy

        robot = self._env.scene["robot"]
        n = self._env.num_envs
        device = robot.data.root_pos_w.device
        ids = self._served_envs(env_ids)
        sel = torch.tensor(ids, dtype=torch.long, device=device)
        if len(ids) == n and ids == self._all_envs():
            sel = slice(None)
            pose_env_ids = None
        else:
            pose_env_ids = ids
        root_quat = robot.data.root_quat_w[sel]
        off_pos = self._off_pos.to(device).expand(root_quat.shape[0], 3)
        if self._has_pose_offset() or self._has_orientation_offset():
            pose_off, yaw, pitch, roll = self._live_state(device)
        if self._has_pose_offset():
            off_pos = off_pos + pose_off[sel]
        off_rot = self._off_rot.to(device).expand(root_quat.shape[0], 4)
        pos = robot.data.root_pos_w[sel, :3] + math_utils.quat_apply(root_quat, off_pos)
        rot = math_utils.quat_mul(root_quat, off_rot)
        if self._has_orientation_offset():
            rot = math_utils.quat_mul(
                _yaw_pitch_roll_quat_batch(yaw[sel], pitch[sel], roll[sel]), rot
            )
        self._sensor.set_world_poses(pos, rot, env_ids=pose_env_ids, convention="world")


class WorldRig(_RigBase):
    """World-anchored tripod; posed once by the cfg offset at spawn.

    The NAME names the anchoring (world frame, never follows the body),
    not a lack of actuators: the live-property whitelist re-aims the
    head (``yaw_deg``/``pitch_deg``), moves the base (``pose_offset``)
    and a transient world pose can overwrite a single frame
    (``begin_world_pose``).  Live pose properties: a one-shot re-pose
    from ``base ∘ offset`` the next time ``sync_pose`` runs after an
    ``apply_properties`` (prim poses survive resets, so ``on_reset``
    stays a no-op).
    """

    def __init__(self, spec: CameraSpec):
        super().__init__(spec)
        rig = spec.rig or {}
        pos = [float(x) for x in rig.get("pos", (0.0, 0.0, 0.0))]
        self._base_pos = torch.tensor(pos, dtype=torch.float32)
        if rig.get("rot") is not None:
            self._base_rot = torch.tensor([float(x) for x in rig["rot"]], dtype=torch.float32)
        else:
            lookat = rig.get("lookat", (1.0, 0.0, 0.25))
            self._base_rot = torch.tensor(lookat_quat(pos, lookat), dtype=torch.float32)
        # per-env declared pose override (CameraSpec.env_poses); None = the
        # single base pose for every env
        self._declared_override: dict[int, tuple] | None = None

    # -- state table ------------------------------------------------------- #
    def _init_state_rows(self) -> None:
        """Seed the pose table from the single declared base pose."""
        for i in range(int(self._env.num_envs)):
            self.state_table.set(self.name, i, self._base_pos.clone(), self._base_rot.clone())

    def apply_declared_poses(self, env_poses) -> None:
        """Write the declared pose rows (bind + reset re-application)."""
        n = int(self._env.num_envs)
        if not env_poses:
            self._declared_override = None
            for i in range(n):
                self.state_table.set(self.name, i, self._base_pos.clone(), self._base_rot.clone())
            return
        rows = [env_poses[0]] * n if len(env_poses) == 1 else list(env_poses)
        self._declared_override = {
            i: (
                torch.tensor(rows[i]["pos"], dtype=torch.float32),
                torch.tensor(rows[i]["quat"], dtype=torch.float32),
            )
            for i in range(n)
        }
        self._pose_dirty = True
        self.sync_pose(env_ids=list(range(n)))

    def _base_for(self, env_idx: int) -> tuple:
        if self._declared_override is not None:
            return self._declared_override[env_idx]
        return (self._base_pos, self._base_rot)

    def _served_envs(self, env_ids=None) -> list[int]:
        # world rigs BROADCAST: a shared state writes one row and pushes it
        # to every served instance (the historical world write-all), while
        # per_instance keeps one independent row per env.
        return self._check_env_ids(env_ids)

    def sync_pose(self, env_ids=None) -> None:
        """Write the resolved world pose rows and push them to the sensor.

        A no-op until the first ``apply_properties``/declared pose marks
        the rig dirty (the cfg spawn pose already posed the prim).
        """
        if self._sensor is None:
            return
        if not self._pose_dirty and env_ids is None:
            return
        device = getattr(self._env, "device", "cpu")
        n = int(self._env.num_envs)
        ids = self._served_envs(env_ids)
        pose_off, yaw, pitch, roll = self._live_state(device)
        shared = self._state_shared()
        compute_ids = [ids[0]] if shared else ids
        src0 = ids[0]
        if shared and 0 <= self.target_env < n:
            src0 = self.target_env
        for i in compute_ids:
            bp, br = self._base_for(i)
            src = src0 if shared else i
            p = bp.to(device) + pose_off[src]
            r = br.to(device)
            if self._has_orientation_offset():
                r = _quat_mul_batch(
                    _yaw_pitch_roll_quat_batch(
                        yaw[src : src + 1], pitch[src : src + 1], roll[src : src + 1]
                    ),
                    r.unsqueeze(0),
                )[0]
            self.state_table.set(self.name, None if shared else i, p, r)
        self._push_state(ids, n)
        self._pose_dirty = False

    def _push_state(self, ids, num_envs) -> None:
        from .cfg import push_state_table_poses

        push_state_table_poses(self._sensor, self.state_table, self.name, ids, num_envs=num_envs)

    def on_reset(self, env_ids) -> None:
        """Re-apply the declared env_poses after a reset (if declared)."""
        # declared env_poses must be re-applied after the reset chain;
        # without them the prims keep their (static) world pose.
        if self._declared_override is not None:
            self._pose_dirty = True
            self.sync_pose(env_ids=env_ids)

    # -- transient world pose (ux-7) ---------------------------------------- #
    def begin_world_pose(self, pos, quat_wxyz):
        """Write the forensic pose onto the target clone immediately.

        The saved token is opaque (None): restore recomputes the rig's
        canonical pose from its own base+offset state (no sensor readback,
        no convention round-trip) -- see :meth:`end_world_pose`.
        """
        if self._sensor is None:
            raise RuntimeError(f"camera '{self.name}': sensor not bound (bind_env)")
        device = getattr(self._env, "device", "cpu")
        n = int(self._env.num_envs)
        te = self.target_env
        if not 0 <= te < n:
            raise ValueError(
                f"camera '{self.name}': target_env {te} out " f"of range for {n} env(s)"
            )
        self.state_table.set(
            self.name,
            None if self._state_shared() else te,
            torch.tensor(pos, dtype=torch.float32, device=device),
            torch.tensor(quat_wxyz, dtype=torch.float32, device=device),
        )
        self._push_state([te], n)
        return None

    def end_world_pose(self, saved) -> None:
        """Restore = recompute from the rig's own base+offset state.

        Rewrites through the exact sync_pose write path -- bit-identical
        to the pre-override pose within one command dispatch.
        """
        del saved
        self._pose_dirty = True
        self.sync_pose()

    def get_world_pose(self):
        """Read-back of the state-table row (sensor fallback)."""
        if self._sensor is None:
            raise RuntimeError(f"camera '{self.name}': sensor not bound (bind_env)")
        idx = 0 if self._state_shared() else self.target_env
        row = self.state_table.get(self.name, idx)
        if row is not None:
            pos, quat = row.pos, row.quat
            return (
                torch.as_tensor(pos).detach().cpu().numpy().copy(),
                torch.as_tensor(quat).detach().cpu().numpy().copy(),
            )
        idx = self._sync_selection()[1] or [0]
        data = self._sensor.data
        pos = data.pos_w[idx[0]]
        quat = data.quat_w[idx[0]]
        return (pos.detach().cpu().numpy().copy(), quat.detach().cpu().numpy().copy())

    def set_world_pose(self, pos, quat_wxyz, env_idx: int | None = None):
        """Persistent absolute world pose (freecam face).

        The exact :meth:`begin_world_pose` write minus the save/restore
        protocol -- the pose sticks until the next pose write (a later
        offset-keyed ``apply_properties`` recomputes from the mount,
        last write wins).  ``env_idx`` None = target_env.
        """
        if self._sensor is None:
            raise RuntimeError(f"camera '{self.name}': sensor not bound (bind_env)")
        device = getattr(self._env, "device", "cpu")
        n = int(self._env.num_envs)
        te = self.target_env if env_idx is None else int(env_idx)
        if not 0 <= te < n:
            raise ValueError(
                f"camera '{self.name}': target_env {te} out " f"of range for {n} env(s)"
            )
        self.state_table.set(
            self.name,
            None if self._state_shared() else te,
            torch.tensor(pos, dtype=torch.float32, device=device),
            torch.tensor(quat_wxyz, dtype=torch.float32, device=device),
        )
        self._push_state([te], n)


class ChaseRig(_RigBase):
    """World-frame chase mount; per-step follow of ONE env's robot root.

    ``target_env`` (yaml ``rig.target_env``, default 0) names the env whose
    robot base the chase is re-posed against at every step tail + reset:
    held at ``rig.pos`` (robot-YAW-rotated, root-relative), aimed at
    ``rig.lookat`` (same rotation).

    Multi-env semantics (docs/contracts/camera_contract.md §5): every sensor clone
    lives in its own env namespace (``{ENV_REGEX_NS}``), but the chase
    re-poses ONLY clone ``target_env`` — the exact port of the classic
    recorder follow mode (``set_world_poses(env_ids=[record_env])``).
    Clones of other envs keep the cfg spawn pose (static tripods); set the
    recorder's ``record_env`` equal to the rig's ``target_env`` to film the
    chased robot.
    """

    def __init__(self, spec: CameraSpec):
        super().__init__(spec)
        rig = spec.rig or {}
        pos = rig.get("pos", DEFAULT_CHASE_OFFSET["pos"])
        lookat = rig.get("lookat", DEFAULT_CHASE_OFFSET["lookat"])
        self._pos_off = torch.tensor([float(x) for x in pos], dtype=torch.float32)
        self._look_off = torch.tensor([float(x) for x in lookat], dtype=torch.float32)

    def _sync_selection(self) -> tuple[slice, list[int] | None]:
        # legacy single-target selector (get_world_pose fallback / the
        # contract §5 shared rule).  C2 per_instance addressing lives in
        # ``sync_pose`` via ``_served_envs`` below.
        te = self.target_env
        return slice(te, te + 1), [te]

    def _served_envs(self, env_ids=None) -> list[int]:
        if self._state_shared():
            return super()._served_envs(env_ids)
        return self._check_env_ids(env_ids)

    def _chase_target(self, env_idx: int) -> int:
        # per_instance: the instance's own env unless target_env was
        # declared explicitly (then all instances chase that named env)
        if self._target_env_declared:
            return self.target_env
        return env_idx

    def sync_pose(self, env_ids=None) -> None:
        """Re-pose every served clone against its chased target root."""
        if self._sensor is None:
            return
        robot = self._env.scene["robot"]
        n = int(self._env.num_envs)
        ids = self._served_envs(env_ids)
        if self._state_shared() and not 0 <= self.target_env < n:
            raise ValueError(
                f"chase camera '{self.name}': target_env {self.target_env} "
                f"out of range for {n} env(s)"
            )
        device = robot.data.root_pos_w.device
        for env_idx in ids:
            te = self.target_env if self._state_shared() else self._chase_target(env_idx)
            if not 0 <= te < n:
                raise ValueError(
                    f"chase camera '{self.name}': target_env {te} out of " f"range for {n} env(s)"
                )
            # chase offsets are computed against TARGET env's root only
            root_pos = robot.data.root_pos_w[te : te + 1, :3]
            yaw = _yaw_from_quat(robot.data.root_quat_w[te : te + 1])
            c, s = torch.cos(yaw), torch.sin(yaw)
            pos_off = self._pos_off.to(device)
            if self._has_pose_offset():
                pos_off = pos_off + self._live_state(device)[0][te]
            look_off = self._look_off.to(device)

            def _rot_xy(off):
                return torch.stack(
                    [off[0] * c - off[1] * s, off[0] * s + off[1] * c, off[2].expand_as(c)], dim=-1
                )

            pos = root_pos + _rot_xy(pos_off)
            lookat = root_pos + _rot_xy(look_off)
            quat = _lookat_quat_batch(pos, lookat)
            if self._has_orientation_offset():
                quat = _quat_mul_batch(
                    _yaw_pitch_roll_quat_batch(
                        self._yaw.to(device)[te : te + 1],
                        self._pitch.to(device)[te : te + 1],
                        self._roll.to(device)[te : te + 1],
                    ),
                    quat,
                )
            self.state_table.set(
                self.name, None if self._state_shared() else env_idx, pos[0], quat[0]
            )
        from .cfg import push_state_table_poses

        push_state_table_poses(
            self._sensor, self.state_table, self.name, ids, num_envs=n, force_env_ids=True
        )
