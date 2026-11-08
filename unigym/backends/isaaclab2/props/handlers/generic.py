"""Generic prop handler: arbitrary URDF rigid object / USD asset.

Tracks a URDF-based rigid prop by its **root** link in the robot body
frame, mirroring the ball handler but loading a URDF (or a native USD
asset) instead of a sphere primitive. Useful for boxes, pallets, or
other interactive objects the policy should react to.

  - ``prop_pos_b`` (3): prop root position in the robot body frame.
  - ``prop_vel_b`` (3): prop root velocity relative to the robot, body frame.

a ``usd`` prop with ``spawn.static: true`` becomes a STAGE
decoration (``AssetBaseCfg`` -- loaded as authored, USD-internal physics
APIs preserved, e.g. the HumanoidArena digital-twin room): no RigidObject,
no state keys, no object/* record keys, reset is a no-op.  The mujoco
backend keeps the §8 representation tiers (static scenery included).

Articulated props moved to :class:`props.articulation.PropArticulation`
; this handler serves ``rigid_object`` / ``usd`` only.
"""
from __future__ import annotations

from typing import Any

import torch

from unigym.props.base import (
    PropHandler,
    PropPlacement,
    PropSpec,
    _quat_mul,
    _roll_quat,
    _yaw_quat,
    quat_rotate_inverse,
)
from ..spawn import _build_urdf_spawn, _build_usd_spawn
from unigym.adapters.base import RawObsRequest

_STATE_PREFIX = "prop"   # default state-key prefix; name-specific keys override


def _static_init_state(placement: PropPlacement, roll_quat,
                       ) -> tuple[tuple[float, float, float],
                                  tuple[float, float, float, float]]:
    """Init (pos, rot) for a STATIC stage-decoration prop from its placement.

    Static prims have no reset surface, so the placement sample is applied
    once at scene build (round-3 ha_open_door root cause: this used to
    ignore placement entirely and spawn stage props at the env origin).
    """
    p0 = placement.sample(1, device="cpu")[0]
    yaw_q = _yaw_quat(p0[3:4])
    total_q = _quat_mul(yaw_q, roll_quat.expand(1, 4))[0]
    return ((float(p0[0]), float(p0[1]),
             float(p0[2]) + placement.height_offset),
            (float(total_q[0]), float(total_q[1]),
             float(total_q[2]), float(total_q[3])))


class GenericProp(PropHandler):
    """Generic URDF prop handler (prop_pos_b / prop_vel_b in body frame)."""

    def __init__(self, spec: PropSpec, num_envs: int, device: str,
                 base_dir: str | None = None):
        self.spec = spec
        self.num_envs = num_envs
        self.device = device
        self.static = bool(self.spec.type == "usd"
                           and self.spec.spawn.get("static", False))
        """Stage-decoration mode : no dynamic state surface."""
        self.placement = PropPlacement(spec.placement)
        self._roll_quat = _roll_quat(spec.init_roll)
        self._asset = None
        # state-key names: spec.state_keys overrides the default prefix so two
        # generic props in one scene don't collide (e.g. ["box_pos_b", ...]).
        # Falls back to prop_pos_b / prop_vel_b when the spec leaves it empty.
        if spec.state_keys:
            self._pos_key = spec.state_keys[0]
            self._vel_key = spec.state_keys[1] if len(spec.state_keys) > 1 else (
                spec.state_keys[0].replace("_pos_", "_vel_"))
        else:
            self._pos_key = f"{_STATE_PREFIX}_pos_b"
            self._vel_key = f"{_STATE_PREFIX}_vel_b"
        # v057 additive decision-package feed (design v057 §2): the
        # orientation key rides the pos key's stem; backends that emit
        # it (mujoco get_state) serve it, the rest leave obs_cache
        # without it -- the adapter degrades honestly to quat None
        self._quat_key = self._pos_key.replace("_pos_", "_quat_")

    # ------------------------------------------------------------------ #
    # scene configuration
    # ------------------------------------------------------------------ #
    def setup_scene(self, *, classes=None) -> tuple[str, Any]:
        """Return the isaaclab scene cfg for this prop."""
        c = classes
        if self.spec.type == "usd":
            spawn = _build_usd_spawn(self.spec, c)
            roll_q = self._roll_quat
            if self.static:
                # stage-only decoration (AssetBaseCfg) -- the USD
                # loads as authored with any physics APIs it carries
                # internally; IsaacLab keeps such entries out of the asset
                # registry, hence no bind/state/reset surface at all.
                # Placement IS applied at build time (see
                # _static_init_state); no per-episode re-sampling exists
                # for stage prims.
                pos, rot = _static_init_state(self.placement, self._roll_quat)
                cfg = c["AssetBaseCfg"](
                    prim_path=f"{{ENV_REGEX_NS}}/Prop_{self.spec.name}",
                    spawn=spawn,
                    init_state=c["AssetBaseCfg"].InitialStateCfg(
                        pos=pos, rot=rot),
                )
                return self.spec.name, cfg
            # native USD spawn -- RigidObject rooted at the USD's
            # default prim; materials/textures come with the file (the
            # asset must be sim-ready: scripts/usd_prep_prop.py applies the
            # root RigidBodyAPI that SIMPLE-authored USDs lack).
            cfg = c["RigidObjectCfg"](
                prim_path=f"{{ENV_REGEX_NS}}/Prop_{self.spec.name}",
                spawn=spawn,
                init_state=c["RigidObjectCfg"].InitialStateCfg(
                    pos=(0.0, 0.0, self.placement.height_offset),
                    rot=(float(roll_q[0]), float(roll_q[1]),
                         float(roll_q[2]), float(roll_q[3])),
                ),
                collision_group=0,
                debug_vis=False,
            )
            return self.spec.name, cfg
        spawn = _build_urdf_spawn(self.spec, c)
        roll_q = self._roll_quat
        init_state_pos = (0.0, 0.0, self.placement.height_offset)
        cfg = c["RigidObjectCfg"](
            prim_path=f"{{ENV_REGEX_NS}}/Prop_{self.spec.name}",
            spawn=spawn,
            init_state=c["RigidObjectCfg"].InitialStateCfg(
                pos=init_state_pos,
                rot=(float(roll_q[0]), float(roll_q[1]),
                     float(roll_q[2]), float(roll_q[3])),
            ),
            collision_group=0,
            debug_vis=False,
        )
        return self.spec.name, cfg

    # ------------------------------------------------------------------ #
    # runtime
    # ------------------------------------------------------------------ #
    def bind_env(self, env) -> None:
        """Cache the spawned asset handle (static stage props stay unbound)."""
        if self.static:
            # AssetBaseCfg entries stay out of the IsaacLab asset registry
            self._asset = None
            return
        self._asset = env.scene[f"prop_{self.spec.name}"]

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
        yaw = placements[:, 3]
        yaw_q = _yaw_quat(yaw)
        roll_q = self._roll_quat.to(env.device).type_as(yaw_q)
        total_q = _quat_mul(yaw_q, roll_q.expand(n, 4))
        pose = torch.zeros(n, 7, device=env.device, dtype=world_pos.dtype)
        pose[:, 0:3] = world_pos
        pose[:, 3:7] = total_q
        self._asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        vel = torch.zeros(n, 6, device=env.device, dtype=world_pos.dtype)
        self._asset.write_root_velocity_to_sim(vel, env_ids=env_ids)

    def get_state(self, env) -> dict[str, torch.Tensor]:
        """Body-frame state tensors for this prop."""
        if self._asset is None or not hasattr(self._asset, "data"):
            return {}
        robot = env.robot
        rd = robot.data
        bd = self._asset.data
        base_quat = rd.root_quat_w
        rel = bd.root_pos_w - rd.root_pos_w
        pos_b = quat_rotate_inverse(base_quat, rel)
        rel_v = bd.root_lin_vel_w - rd.root_lin_vel_w
        vel_b = quat_rotate_inverse(base_quat, rel_v)
        return {self._pos_key: pos_b, self._vel_key: vel_b}

    @property
    def state_requests(self) -> list[RawObsRequest]:
        """Declare the obs keys this prop exposes (none for static props)."""
        if self.static:
            # stage decoration: no obs face at all
            return []
        return [
            RawObsRequest(self._pos_key, 3, "prop"),
            RawObsRequest(self._vel_key, 3, "prop"),
            # v057 §2: the catalog face of the additive quat feed (the
            # mujoco backend emits it; backends without the emission
            # never fill obs_cache -- "requested keys only", the
            # adapter's .get() degrades to None)
            RawObsRequest(self._quat_key, 4, "prop"),
        ]

    def get_world_pose(self, env):
        """``(pos (N,3), quat (N,4) wxyz)`` world-frame root pose (clones)."""
        if self._asset is None or not hasattr(self._asset, "data"):
            return None
        return (self._asset.data.root_pos_w.clone(),
                self._asset.data.root_quat_w.clone())
