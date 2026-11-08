"""Anchored articulation prop handler (usd_articulation + latch).

The articulated-prop kind done through the existing named surfaces:

* **asset**  -- a URDF or MJCF fragment whose joints are defined inside the
  asset. The root body is FIXED to the world at the sampled placement
  (anchored semantics; a fragment carrying a freejoint fails fast at the
  backend build/bind time). The former free-floating door handler
  (props/door.py) is subsumed by this one -- door.urdf is reused as-is.
  The ``type: usd_articulation`` kind: a native USD articulation asset
  (embedded revolute joints) spawned via ``UsdFileCfg`` with
  ``fix_root_link`` anchoring -- the HumanoidArena open_door door route
  (docs/designs/v031-ha-open-door.md). isaaclab-only (the mujoco backend
  has no USD articulation loader and fails fast at spec-build time).
* **action surface** -- every joint declared ``mode: actuated``
  automatically becomes a NamedAction entry named after the prop
  (``{prop_name: NamedAction(values, joint_names)}``, alongside ``g1`` /
  ``hand``); no extra action-section wiring. ``mode: passive`` joints get
  damping injection only and never join the action surface (a
  passive joint MAY still declare ``stiffness`` / ``effort_limit`` so
  USD-authored drive semantics are reproducible without policy driving --
  e.g. the HA door handle: stiffness 5.0 behind an effort cap of 0.001).
* **state surface** -- the anchor root pose flows into the
  ``object/<name>/{pos,quat}`` record keys like any prop, plus
  ``object/<name>/joints/<jname>`` (N, 1) per declared joint, consumed
  directly by FormulaTask terms (``final`` / ``range_check_fraction``).
* **latch** (optional ``spec.latch``) -- runtime drive switching
  on ONE declared joint, gated by ANOTHER joint's angle crossing a
  threshold: locked drive on reset, and once the sense joint crosses the
  threshold the lock joint's drive is restored to its declared (default)
  values. Mirrors the HumanoidArena door latch (handle turned past -20deg
  releases the leaf's 1e5-stiffness lock; move_open_door_*_env_cfg.py:
  401-422,540-656,1481-1485). Angles are ABSOLUTE joint positions in
  degrees -- our on_reset zeros the declared joints, so absolute equals
  relative-to-reset as upstream computes it.

fail-fast (declaration level here, asset level at backend bind time):
unknown mode / ``actuated`` without ``stiffness`` / ``passive`` without
``damping`` / an asset joint left undeclared / a declared joint missing
from the asset / a latch naming a joint outside the declared table.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from unigym.props.articulation import validate_joints, validate_latch
from unigym.props.base import (
    PropHandler,
    PropPlacement,
    PropSpec,
    _quat_mul,
    _roll_quat,
    _yaw_quat,
)
from ..spawn import _build_urdf_spawn

class PropArticulation(PropHandler):
    """Anchored articulated prop (root pose + joints state surface)."""

    def __init__(self, spec: PropSpec, num_envs: int, device: str,
                 base_dir: str | None = None):
        self.spec = spec
        self.num_envs = num_envs
        self.device = device
        self.joints = validate_joints(spec)
        """Declared joint table (yaml order preserved)."""
        self.actuated_joints = [n for n, p in self.joints.items()
                                if p["mode"] == "actuated"]
        self.latch = validate_latch(spec, self.joints)
        """Normalized latch table, or None ."""
        self.latch_enabled = self.latch is not None
        self._latch_unlocked: set[int] = set()
        self.placement = PropPlacement(spec.placement)
        self._roll_quat = _roll_quat(spec.init_roll)
        # populated by bind_env()
        self._asset = None
        self._jidx: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # scene configuration
    # ------------------------------------------------------------------ #
    def setup_scene(self, *, classes=None) -> tuple[str, Any]:
        """Return the isaaclab scene cfg for this prop."""
        c = classes
        spawn = _build_urdf_spawn(self.spec, c)
        roll_q = self._roll_quat
        cfg = c["ArticulationCfg"](
            prim_path=f"{{ENV_REGEX_NS}}/Prop_{self.spec.name}",
            spawn=spawn,
            init_state=c["ArticulationCfg"].InitialStateCfg(
                pos=(0.0, 0.0, self.placement.height_offset),
                rot=(float(roll_q[0]), float(roll_q[1]),
                     float(roll_q[2]), float(roll_q[3])),
                joint_pos={".*": 0.0},
            ),
            actuators=_build_actuators(self.joints, c),
            collision_group=0,
            debug_vis=False,
        )
        return self.spec.name, cfg

    # ------------------------------------------------------------------ #
    # runtime
    # ------------------------------------------------------------------ #
    def bind_env(self, env) -> None:
        """Cache the spawned asset handle after scene init."""
        self._asset = env.scene[f"prop_{self.spec.name}"]
        asset_joints = list(self._asset.joint_names)
        missing = [n for n in self.joints if n not in asset_joints]
        if missing:
            raise ValueError(
                f"articulation prop {self.spec.name!r}: declared joints "
                f"{missing} not in the asset joints {asset_joints}")
        undeclared = [n for n in asset_joints if n not in self.joints]
        if undeclared:
            raise ValueError(
                f"articulation prop {self.spec.name!r}: asset joints "
                f"{undeclared} are not declared in spec.joints "
                f"(every asset joint must be declared)")
        self._jidx = {n: asset_joints.index(n) for n in self.joints}

    def on_reset(self, env_ids, env) -> None:
        """Re-sample placement + reset prop state for the env subset."""
        if self._asset is None:
            return
        env_ids = torch.as_tensor(env_ids, device=env.device)
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        origins = env.scene.env_origins[env_ids]
        placements = self.placement.sample(n, env.device)
        world_pos = origins + placements[:, :3]
        yaw_q = _yaw_quat(placements[:, 3])
        roll_q = self._roll_quat.to(env.device).type_as(yaw_q)
        pose = torch.zeros(n, 7, device=env.device, dtype=world_pos.dtype)
        pose[:, 0:3] = world_pos
        pose[:, 3:7] = _quat_mul(yaw_q, roll_q.expand(n, 4))
        # anchored: re-anchor the fixed root at the sampled placement
        self._asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        vel = torch.zeros(n, 6, device=env.device, dtype=world_pos.dtype)
        self._asset.write_root_velocity_to_sim(vel, env_ids=env_ids)
        # declared joint reset state (T-B1 ``init``, default 0.0) for the
        # subset -- position + velocity (the velocity is always zeroed)
        inits = torch.tensor(
            [self.joints[n]["init"] for n in self.joints],
            device=env.device, dtype=world_pos.dtype).expand(n, -1)
        self._asset.write_joint_state_to_sim(
            inits, torch.zeros(n, len(self.joints), device=env.device,
                               dtype=world_pos.dtype),
            joint_ids=[self._jidx[n] for n in self.joints],
            env_ids=env_ids)
        # latch: re-lock (upstream semantics -- reset always re-locks and the
        # unlocked set starts empty; move_open_door reset_open_door_latch)
        for eid in env_ids.tolist():
            self._latch_unlocked.discard(int(eid))
        if self.latch_enabled:
            self._write_drive(
                env, env_ids,
                self.latch["locked"]["stiffness"],
                self.latch["locked"]["damping"],
                self.latch["locked"]["friction"],
                self.latch["lock_joint"])

    # ------------------------------------------------------------------ #
    # latch : per-control-step poll + drive switching
    # ------------------------------------------------------------------ #
    def on_step(self, env) -> None:
        """Poll the sense joint; unlock the lock joint past the threshold.

        Called once per control step by the PropManager (wired from
        the env backend's extra build -- upstream polls at the same 0.02 s
        control cadence).  Reads ABSOLUTE joint positions in degrees: our
        on_reset zeros the declared joints, matching the upstream
        reference-subtracted read (env_cfg:976-1005).  Batch semantics: one
        write per unlocking step for the subset that crosses the threshold;
        latching is permanent for the episode (upstream never re-locks).
        """
        if not self.latch_enabled or self._asset is None:
            return
        pending = [i for i in range(self.num_envs)
                   if i not in self._latch_unlocked]
        if not pending:
            return
        deg = self._sense_deg(env)
        if deg is None:
            return
        thr = self.latch["unlock_threshold_deg"]
        if thr >= 0.0:
            crossed = [i for i in pending if float(deg[i]) >= thr]
        else:
            crossed = [i for i in pending if float(deg[i]) <= thr]
        if not crossed:
            return
        ids = torch.tensor(crossed, dtype=torch.int64, device=env.device)
        self._write_drive(
            env, ids,
            self.latch["unlocked"]["stiffness"],
            self.latch["unlocked"]["damping"],
            self.latch["unlocked"]["friction"],
            self.latch["lock_joint"])
        self._latch_unlocked.update(crossed)

    def _sense_deg(self, env) -> torch.Tensor | None:
        """Sense-joint angle in degrees, (num_envs,) on env.device."""
        sense = self.latch["sense_joint"]
        jp = self._asset.data.joint_pos
        return jp[:, self._jidx[sense]] * (180.0 / math.pi)

    def _write_drive(self, env, env_ids, stiffness: float, damping: float,
                     friction: float, joint_name: str) -> None:
        """Write drive gains (+ joint friction) for one joint, one env subset.

        Uses the same runtime surfaces the upstream latch uses
        (write_joint_stiffness_to_sim / write_joint_damping_to_sim /
        write_joint_friction_coefficient_to_sim).  Zero-valued entries are
        still written (restore-to-passive semantics), friction only when
        non-zero or when explicitly restoring (keeps the API surface small
        on backends lacking the friction writer -- but fails fast when a
        non-zero friction cannot be written).
        """
        n = env_ids.shape[0] if torch.is_tensor(env_ids) else len(env_ids)
        dev = env.device
        jid = [self._jidx[joint_name]]
        vals = torch.full((n, 1), float(stiffness), device=dev,
                          dtype=torch.float32)
        self._asset.write_joint_stiffness_to_sim(vals, joint_ids=jid,
                                                 env_ids=env_ids)
        vals = torch.full((n, 1), float(damping), device=dev,
                          dtype=torch.float32)
        self._asset.write_joint_damping_to_sim(vals, joint_ids=jid,
                                               env_ids=env_ids)
        if friction != 0.0 or self.latch_enabled:
            writer = getattr(self._asset,
                             "write_joint_friction_coefficient_to_sim", None)
            if writer is None:
                if friction != 0.0:
                    raise RuntimeError(
                        f"articulation prop {self.spec.name!r}: latch "
                        f"friction {friction} requested but the backend "
                        f"articulation lacks "
                        f"write_joint_friction_coefficient_to_sim")
                return
            vals = torch.full((n, 1), float(friction), device=dev,
                              dtype=torch.float32)
            writer(vals, joint_ids=jid, env_ids=env_ids)

    def latch_unlocked_mask(self, env) -> torch.Tensor:
        """``(N,) bool`` -- latch state per env (True = unlocked).

        False everywhere when no latch is declared (the upstream reward's
        ``latch_enabled=False -> latch_unlocked=True`` equivalence is the
        FormulaTask's job via its ``required: false`` face).
        """
        mask = torch.zeros(self.num_envs, dtype=torch.bool,
                           device=env.device)
        for i in self._latch_unlocked:
            mask[i] = True
        return mask

    def get_world_pose(self, env):
        """``(pos (N,3), quat (N,4) wxyz)`` world-frame ANCHOR pose."""
        if self._asset is None or not hasattr(self._asset, "data"):
            return None
        return (self._asset.data.root_pos_w.clone(),
                self._asset.data.root_quat_w.clone())

    def get_joint_positions(self, env) -> dict[str, torch.Tensor]:
        """``{jname: (N,1) f32}`` joint positions (record/state surface)."""
        if self._asset is None or not hasattr(self._asset, "data"):
            return {}
        jp = self._asset.data.joint_pos
        return {n: jp[:, i:i + 1].clone() for n, i in self._jidx.items()}

    @property
    def state_requests(self) -> list:
        """No obs face: the articulation state is the record-key family."""
        return []

    def get_state(self, env) -> dict[str, torch.Tensor]:
        """Empty obs contribution (the state surface is the record keys)."""
        return {}


def _build_actuators(joints: dict[str, dict], c: dict) -> dict:
    """Implicit actuator cfgs from the declared joint table.

    ``passive`` -> declared stiffness (default 0) + declared damping
    (damping injection; may carry a USD-authored stiffness);
    ``actuated`` -> declared stiffness/damping (the JointPositionAction
    term writes the position target).  A declared ``effort_limit`` (either
    mode) feeds ``effort_limit_sim`` -- the HA door handle's capped
    passive drive is expressed that way.
    """
    ImplicitActuatorCfg = c["ImplicitActuatorCfg"]
    out: dict[str, Any] = {}
    for name, p in joints.items():
        kw: dict[str, Any] = {
            "joint_names_expr": [name],
            "stiffness": p["stiffness"],
            "damping": p["damping"],
        }
        if p["effort_limit"] is not None:
            kw["effort_limit_sim"] = p["effort_limit"]
        out[name] = ImplicitActuatorCfg(**kw)
    return out


class UsdArticulationProp(PropArticulation):
    """``type: usd_articulation`` : native USD articulation.

    Same declared surfaces as :class:`PropArticulation` (joints/latch/
    state/action faces are inherited unchanged); only the spawn builder
    differs -- the USD is loaded AS AUTHORED via ``UsdFileCfg`` with
    ``fix_root_link`` anchoring, so the asset's embedded revolute joints,
    limits and joint friction come with the file (the HumanoidArena
    door001 route).  ``spec.spawn`` knobs (all optional, defaults mirror
    the upstream DOOR_SPAWN_CFG, base_scene_open_door.py:23-47):
    ``disable_gravity`` (default True), ``max_depenetration_velocity``
    (default 5.0), ``contact_offset`` (default 0.01).
    """

    def setup_scene(self, *, classes=None) -> tuple[str, Any]:
        """Return the isaaclab scene cfg for this prop."""
        c = classes
        spawn = self._build_spawn(c)
        roll_q = self._roll_quat
        cfg = c["ArticulationCfg"](
            prim_path=f"{{ENV_REGEX_NS}}/Prop_{self.spec.name}",
            spawn=spawn,
            init_state=c["ArticulationCfg"].InitialStateCfg(
                pos=(0.0, 0.0, self.placement.height_offset),
                rot=(float(roll_q[0]), float(roll_q[1]),
                     float(roll_q[2]), float(roll_q[3])),
                joint_pos={".*": 0.0},
            ),
            actuators=_build_actuators(self.joints, c),
            collision_group=0,
            debug_vis=False,
        )
        return self.spec.name, cfg

    def _build_spawn(self, c: dict):
        """UsdFileCfg with HA-door anchoring defaults (contract §9)."""
        UsdFileCfg = c["UsdFileCfg"]
        s = dict(self.spec.spawn)
        RigidBodyPropertiesCfg = c["RigidBodyPropertiesCfg"]
        ArticulationRootPropertiesCfg = c["ArticulationRootPropertiesCfg"]
        CollisionPropertiesCfg = c["CollisionPropertiesCfg"]
        kwargs = {}
        if s.get("visible") is not None:
            # render knob (round-3 ha_open_door: the managed articulation
            # copy runs invisible while a static visual copy renders the
            # asset -- the articulation spawner breaks the leaf materials)
            kwargs["visible"] = bool(s["visible"])
        return UsdFileCfg(
            usd_path=self.spec.usd_path,
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=bool(s.get("disable_gravity", True)),
                kinematic_enabled=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=float(
                    s.get("max_depenetration_velocity", 5.0)),
            ),
            articulation_props=ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=float(s.get("contact_offset", 0.01)),
                rest_offset=0.0,
            ),
            **kwargs,
        )


class FreeArticulationProp(PropArticulation):
    """``type: free_articulation`` : a FREE-FLOATING articulated prop.

    The GAP-A kind (T-B3, the robocasa Laptop family): the asset's ROOT
    body KEEPS its freejoint (a pushable/knockable table object -- the
    root pose is live physics state) plus internal passive joints (the
    laptop lid hinge).  Everything the anchored kind declares about the
    INTERNAL joints carries over verbatim (``validate_joints`` -- the
    ``init`` reset face included); the freejoint itself is NOT a
    declarable joint -- the reset pose comes from ``placement`` (the
    freejoint qpos0 face, dual-backend: the mujoco on_reset writes the
    7-dof qpos + zeroes the 6-dof velocity, exactly the rigid-prop
    reset), and the runtime ``object/<name>/{pos,quat}`` record keys are
    the live root pose.

    Two DECLARED narrowings (v1):
    * passive-only -- an ``actuated`` internal joint fails fast at the
      mujoco build (the laptop family is damping-only; the servo/action
      face of a free root is unexercised scope);
    * mujoco-only -- :meth:`setup_scene` (the isaaclab face) refuses
      with an error+hint (the hide_geom_groups precedent): the isaaclab
      articulation route has no free-root keep-face to honor (its URDF
      converter anchors or welds; an honest refusal beats a silently
      re-anchored laptop).
    """

    def setup_scene(self, *, classes=None) -> tuple[str, Any]:
        """Refuse the isaaclab face (mujoco-only kind)."""
        raise ValueError(
            f"free_articulation prop {self.spec.name!r}: mujoco-only kind "
            f"(the isaaclab articulation spawner keeps no free root: "
            f"fix_root_link=True anchors the laptop, False has no "
            f"converted-URDF route) -- run this pack on backend: mujoco")
