"""Dynamic primitive props: cylinder / capsule (S2c).

The content/props path owns dynamic free bodies (a freejoint plus
mass/inertia).  Sphere and box already have handlers with that behaviour
(:class:`~unigym.props.ball.BallProp` and
:class:`~unigym.props.container.BoxProp`); this module adds the missing
cylinder and capsule kinds with the SAME declaration vocabulary:

    objects:
      types:
        - name: peg
          type: cylinder
          radius: 0.03
          half_height: 0.05          # local z, matching the scene primitives
          spawn: {mass: 0.2, friction: 1.0, rgba: [0.8, 0.6, 0.2, 1.0]}
          placement: {mode: fixed, x_range: [0.4, 0.4],
                      y_range: [0.0, 0.0], z_range: [0.3, 0.3]}

Dimensions are the neutral scene-primitive ones: ``radius`` and
``half_height`` (``cylinder``'s half-height; for ``capsule`` the
half-length of its cylindrical section, so its resting centre height is
``radius + half_height``).  ``spawn.mass`` and ``spawn.density`` are
mutually exclusive; mass wins as an explicit declaration, density is
multiplied by the analytic volume, and an absent pair falls back to
``DEFAULT_MASS``.  The backend always receives the resolved MASS: mujoco
derives inertia from the geom (engine-computed), isaaclab applies
``MassPropertiesCfg`` (engine-computed from the spawner shape).

The box and sphere kinds keep their existing handlers (``CuboidCfg`` on
isaaclab for the single-box kind, generated URDF for container/table, and
``SphereCfg`` respectively) so their long-standing configs and state keys
are unchanged; they share :func:`resolve_mass` for the density vocabulary.
"""

from __future__ import annotations

from typing import Any

import torch

from unigym.props.base import (
    _quat_mul,
    _roll_quat,
    _yaw_quat,
    PropHandler,
    PropPlacement,
    PropSpec,
    quat_rotate_inverse,
)
from unigym.props.primitive import (
    material_colors,
    resolve_mass,
    validate_primitive_spec,
)
from unigym.adapters.base import RawObsRequest


def _build_primitive_spawn(spec: PropSpec, c: dict):
    """Native isaaclab spawner cfg for a cylinder / capsule prop.

    IsaacLab's ``CylinderCfg``/``CapsuleCfg`` ``height`` is the AXIAL
    length: for a cylinder the full height, for a capsule the cylindrical
    section (USD adds the two hemispherical caps on top).  The neutral
    ``half_height`` is that same cylindrical half-length, so BOTH kinds
    map to ``2 * half_height`` (S2b container verification caught the
    earlier capsule ``2 * (half_height + radius)`` double-count).  Inertia
    stays engine-computed from the shape plus the resolved mass.
    """
    CylinderCfg = c["CylinderCfg"]
    CapsuleCfg = c["CapsuleCfg"]
    MassPropertiesCfg = c["MassPropertiesCfg"]
    RigidBodyPropertiesCfg = c["RigidBodyPropertiesCfg"]
    CollisionPropertiesCfg = c["CollisionPropertiesCfg"]
    PreviewSurfaceCfg = c["PreviewSurfaceCfg"]
    RigidBodyMaterialCfg = c["RigidBodyMaterialCfg"]
    mass = resolve_mass(spec)
    color = material_colors(dict(spec.spawn))
    kwargs: dict[str, Any] = dict(
        mass_props=MassPropertiesCfg(mass=mass),
        rigid_props=RigidBodyPropertiesCfg(),
        collision_props=CollisionPropertiesCfg(collision_enabled=True),
        visual_material=PreviewSurfaceCfg(diffuse_color=color),
    )
    friction = spec.spawn.get("friction")
    restitution = spec.spawn.get("restitution")
    if friction is not None or restitution is not None:
        mat: dict = {}
        if friction is not None:
            mat["static_friction"] = float(friction)
            mat["dynamic_friction"] = float(friction)
        if restitution is not None:
            mat["restitution"] = float(restitution)
        kwargs["physics_material"] = RigidBodyMaterialCfg(**mat)
    if spec.type == "cylinder":
        return CylinderCfg(
            radius=float(spec.radius), height=2.0 * float(spec.half_height), **kwargs
        )
    return CapsuleCfg(radius=float(spec.radius), height=2.0 * float(spec.half_height), **kwargs)


class PrimitiveProp(PropHandler):
    """Dynamic cylinder / capsule prop (freejoint rigid body).

    State keys default to ``<name>_pos_b`` / ``<name>_vel_b`` in the robot
    body frame (the box-family convention); ``state_keys`` overrides them.
    """

    def __init__(self, spec: PropSpec, num_envs: int, device: str, base_dir: str | None = None):
        validate_primitive_spec(spec)
        self.spec = spec
        self.num_envs = num_envs
        self.device = device
        self.placement = PropPlacement(spec.placement)
        self._roll_quat = _roll_quat(spec.init_roll)
        self._asset = None
        keys = list(spec.state_keys)
        if keys:
            self._pos_key = keys[0]
            self._vel_key = keys[1] if len(keys) > 1 else keys[0].replace("_pos_", "_vel_")
        else:
            self._pos_key = f"{spec.name}_pos_b"
            self._vel_key = f"{spec.name}_vel_b"

    # ------------------------------------------------------------------ #
    # scene configuration
    # ------------------------------------------------------------------ #
    def setup_scene(self, *, classes=None) -> tuple[str, Any]:
        """Return the isaaclab ``RigidObjectCfg`` for this primitive."""
        c = classes
        spawn = _build_primitive_spawn(self.spec, c)
        roll_q = self._roll_quat
        cfg = c["RigidObjectCfg"](
            prim_path=f"{{ENV_REGEX_NS}}/Prop_{self.spec.name}",
            spawn=spawn,
            init_state=c["RigidObjectCfg"].InitialStateCfg(
                pos=(0.0, 0.0, self.placement.height_offset),
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
        """Cache the spawned rigid-object handle after scene init."""
        self._asset = env.scene[f"prop_{self.spec.name}"]

    def on_reset(self, env_ids, env) -> None:
        """Re-sample this prop's placement and zero its root velocity."""
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
        """Body-frame ``<name>_pos_b`` / ``<name>_vel_b`` state tensors."""
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
        """Declare the two body-frame obs keys this prop exposes."""
        return [
            RawObsRequest(self._pos_key, 3, "prop"),
            RawObsRequest(self._vel_key, 3, "prop"),
        ]

    def get_world_pose(self, env):
        """``(pos (N,3), quat (N,4) wxyz)`` world-frame root pose (clones)."""
        if self._asset is None or not hasattr(self._asset, "data"):
            return None
        return (self._asset.data.root_pos_w.clone(), self._asset.data.root_quat_w.clone())
