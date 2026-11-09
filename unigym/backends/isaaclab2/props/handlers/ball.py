"""Ball prop handler (sphere rigid object).

Spawns a physics sphere via Isaac Lab's :class:`SphereCfg` (no URDF needed) and
exposes the ball state in the **robot body frame** so a dribbling/interacting
policy can consume it directly:

  - ``ball_pos_b`` (3): ball centre position in the robot body frame =
    ``quat_rotate_inverse(base_quat, ball_world - base_world)``.
  - ``ball_vel_b`` (3): ball velocity relative to the robot, in the body frame
    = ``quat_rotate_inverse(base_quat, ball_lin_vel_w - base_lin_vel_w)``.

The spawn config carries ``radius``, ``mass``, ``color`` (rgb) and optional
``friction`` / ``restitution`` for the physics material. Reset re-samples
``(x, y, z, yaw)`` from :class:`PropPlacement` (z typically the ball radius so
it rests on the ground) and parks the velocity to zero.
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
from unigym.adapters.base import RawObsRequest


def _build_sphere_spawn(spec: PropSpec, c: dict):
    """Build the SphereCfg spawn from spec.spawn (radius/mass/color/material).

    Applies the RigidBodyAPI (``rigid_props``) and collision (``collision_props``
    enabled) so the spawned sphere resolves as a :class:`RigidObject` and
    physically interacts with the robot; Isaac Lab only applies these schemas
    when the corresponding cfg field is non-None (defaults are None).
    """
    SphereCfg = c["SphereCfg"]
    MassPropertiesCfg = c["MassPropertiesCfg"]
    RigidBodyPropertiesCfg = c["RigidBodyPropertiesCfg"]
    CollisionPropertiesCfg = c["CollisionPropertiesCfg"]
    PreviewSurfaceCfg = c["PreviewSurfaceCfg"]
    RigidBodyMaterialCfg = c["RigidBodyMaterialCfg"]
    s = dict(spec.spawn)
    radius = spec.radius if spec.radius is not None else s.get("radius")
    if radius is None:
        raise ValueError(
            f"sphere prop {spec.name!r}: radius is required " f"(declare radius or spawn.radius)"
        )
    radius = float(radius)
    from unigym.props.primitive import resolve_mass

    mass = resolve_mass(spec, default=0.1)
    color = s.get("color")
    if color is None:
        color = list(s.get("rgba", (1.0, 0.8, 0.0)))[:3]
    color = tuple(float(v) for v in color)
    kw = dict(
        radius=radius,
        mass_props=MassPropertiesCfg(mass=mass),
        # apply RigidBodyAPI (motion: dynamic, gravity on) + enable collision
        rigid_props=RigidBodyPropertiesCfg(),
        collision_props=CollisionPropertiesCfg(collision_enabled=True),
        visual_material=PreviewSurfaceCfg(diffuse_color=color),
    )
    friction = s.get("friction")
    restitution = s.get("restitution")
    if friction is not None or restitution is not None:
        mat_kw: dict = {}
        if friction is not None:
            mat_kw["static_friction"] = float(friction)
            mat_kw["dynamic_friction"] = float(friction)
        if restitution is not None:
            mat_kw["restitution"] = float(restitution)
        kw["physics_material"] = RigidBodyMaterialCfg(**mat_kw)
    return SphereCfg(**kw)


class BallProp(PropHandler):
    """Sphere rigid-object handler (ball_pos_b / ball_vel_b in body frame)."""

    def __init__(self, spec: PropSpec, num_envs: int, device: str, base_dir: str | None = None):
        from unigym.props.primitive import resolve_mass

        radius = spec.radius if spec.radius is not None else spec.spawn.get("radius")
        if radius is None:
            raise ValueError(
                f"sphere prop {spec.name!r}: radius is required "
                f"(declare radius or spawn.radius)"
            )
        resolve_mass(spec, default=0.1)
        self.spec = spec
        self.num_envs = num_envs
        self.device = device
        self.placement = PropPlacement(spec.placement)
        self._roll_quat = _roll_quat(spec.init_roll)
        self._asset = None

    # ------------------------------------------------------------------ #
    # scene configuration
    # ------------------------------------------------------------------ #
    def setup_scene(self, *, classes=None) -> tuple[str, Any]:
        """Return the isaaclab scene cfg for this prop."""
        c = classes
        spawn = _build_sphere_spawn(self.spec, c)
        roll_q = self._roll_quat
        init_state_pos = (0.0, 0.0, self.placement.height_offset)
        cfg = c["RigidObjectCfg"](
            prim_path=f"{{ENV_REGEX_NS}}/Prop_{self.spec.name}",
            spawn=spawn,
            init_state=c["RigidObjectCfg"].InitialStateCfg(
                pos=init_state_pos,
                rot=(float(roll_q[0]), float(roll_q[1]), float(roll_q[2]), float(roll_q[3])),
            ),
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
        # park the ball's velocity (no carry-over from the previous episode)
        vel = torch.zeros(n, 6, device=env.device, dtype=world_pos.dtype)
        self._asset.write_root_velocity_to_sim(vel, env_ids=env_ids)

    def get_state(self, env) -> dict[str, torch.Tensor]:
        """Body-frame state tensors for this prop."""
        if self._asset is None or not hasattr(self._asset, "data"):
            return {}
        robot = env.robot
        rd = robot.data
        bd = self._asset.data
        base_quat = rd.root_quat_w  # (N, 4) wxyz
        # ball position relative to base, in body frame
        rel = bd.root_pos_w - rd.root_pos_w  # (N, 3) world
        ball_pos_b = quat_rotate_inverse(base_quat, rel)
        # ball velocity relative to base, in body frame
        rel_v = bd.root_lin_vel_w - rd.root_lin_vel_w  # (N, 3) world
        ball_vel_b = quat_rotate_inverse(base_quat, rel_v)
        return {
            "ball_pos_b": ball_pos_b,
            "ball_vel_b": ball_vel_b,
        }

    @property
    def state_requests(self) -> list[RawObsRequest]:
        """Declare the obs keys this prop exposes."""
        return [
            RawObsRequest("ball_pos_b", 3, "prop"),
            RawObsRequest("ball_vel_b", 3, "prop"),
        ]

    def get_world_pose(self, env):
        """``(pos (N,3), quat (N,4) wxyz)`` world-frame root pose (clones)."""
        if self._asset is None or not hasattr(self._asset, "data"):
            return None
        return (self._asset.data.root_pos_w.clone(), self._asset.data.root_quat_w.clone())
