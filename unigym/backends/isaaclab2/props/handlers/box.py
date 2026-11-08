"""Box-family props: container / table / box (L1).

Three registered kinds (props yaml ``type:``), one handler:

  - ``table``     : single slab box (desktop), origin at the outer bottom
                    face center;
  - ``box``       : single solid cube, origin at the bottom face center --
                    the classic pick object (any hand-contact criterion is
                    a declared contact group, contacts/base.py);
  - ``container`` : open-top box = bottom slab + 4 side walls; the yaml
                    declares the INNER cavity as
                    ``aabb: {center: [x,y,z], half_extents: [hx,hy,hz]}``
                    (relative to the prop origin) -- the FormulaTask
                    ``point_in_box_fraction`` parameter source.

Geometry is generated, not authored: a PropSpec carries ``size`` (outer
envelope ``[sx, sy, sz]``) and (for container) the declared ``aabb``;
``box_layout()`` turns them into axis-aligned box primitives
``[(center(3), half(3))]`` which feed either a minimal URDF (single link,
multiple visual+collision boxes, generated into the repo-local
content-addressed cache for isaaclab's ``UrdfFileCfg``) or the MjSpec body
builder (props mujoco backend) -- "equivalent combination", one source of
geometry.  The isaaclab spawner uses a native ``CuboidCfg`` for the
single-box ``box`` kind (no generated URDF); the multi-box
``container``/``table`` kinds keep the generated URDF.  Box assets are
deliberately plain (no meshes/textures); color is optional and cosmetic.

The handler subclasses :class:`GenericProp` (root-pose tracking, free-joint
dynamic rigid body) and adds: aabb validation, aabb metadata exposure
(``aabb_center``/``aabb_half``), and world-frame ``get_world_pose`` for the
``object/*`` record-key layer (props/object_state).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from unigym.props.base import (
    PropSpec, _quat_mul, _yaw_quat, quat_rotate, quat_rotate_inverse,
)
from unigym.props import container as container_mod
from unigym.props.container import (
    _DEFAULT_COLOR, _DEFAULT_MASS, _declared_aabb, _positive_size,
    validate_box_spec,
)
from unigym.props.primitive import resolve_mass
from .generic import GenericProp


def _build_cuboid_spawn(spec: PropSpec, c: dict):
    """Native isaaclab ``CuboidCfg`` spawn for the single-box ``box`` kind.

    ``spawn.friction`` / ``spawn.restitution`` (when declared) become the
    IsaacLab physics material (``RigidBodyMaterialCfg``), matching the
    sphere / cylinder / capsule spawners.  ``friction`` sets both the
    static and dynamic coefficient; an undeclared pair leaves IsaacLab's
    defaults untouched.
    """
    s = dict(spec.spawn)
    size = _positive_size(spec)
    mass = resolve_mass(spec, default=_DEFAULT_MASS["box"])
    color = tuple(float(v) for v in list(s.get(
        "color", _DEFAULT_COLOR["box"]))[:3])
    kwargs: dict[str, Any] = dict(
        size=tuple(float(v) for v in size),
        mass_props=c["MassPropertiesCfg"](mass=mass),
        rigid_props=c["RigidBodyPropertiesCfg"](),
        collision_props=c["CollisionPropertiesCfg"](collision_enabled=True),
        visual_material=c["PreviewSurfaceCfg"](diffuse_color=color),
    )
    friction = s.get("friction")
    restitution = s.get("restitution")
    if friction is not None or restitution is not None:
        material: dict[str, float] = {}
        if friction is not None:
            material["static_friction"] = float(friction)
            material["dynamic_friction"] = float(friction)
        if restitution is not None:
            material["restitution"] = float(restitution)
        kwargs["physics_material"] = c["RigidBodyMaterialCfg"](**material)
    return c["CuboidCfg"](**kwargs)


def _box_origin_offset(spec: PropSpec, quat: torch.Tensor) -> torch.Tensor:
    """World offset from the logical bottom-face origin to the box centre.

    ``CuboidCfg`` roots the body at its geometric centre while the public
    prop origin (placement.z, aabb, state keys) is the outer bottom-face
    centre.  The offset is the local ``[0, 0, sz/2]`` rotated by the body
    orientation, so the two conventions stay interchangeable under yaw/roll.
    """
    offset = torch.zeros((quat.shape[0], 3), dtype=quat.dtype,
                         device=quat.device)
    offset[:, 2] = float(_positive_size(spec)[2]) / 2.0
    return quat_rotate(quat, offset)


# --------------------------------------------------------------------------- #
# handler
# --------------------------------------------------------------------------- #
class BoxProp(GenericProp):
    """container / table / box handler (aabb metadata + free rigid body).

    ``container``/``table`` keep the generated multi-box URDF; the
    single-box ``box`` kind uses isaaclab's native ``CuboidCfg``.  Both keep
    the public bottom-face origin: the CuboidCfg root (geometric centre) is
    offset by ``sz/2`` at spawn/reset and the offset is removed from the
    world-pose / state / aabb faces.

    State keys follow GenericProp's name-specific convention:
    ``<name>_pos_b`` / ``<name>_vel_b`` in the robot body frame.
    """

    def __init__(self, spec: PropSpec, num_envs: int, device: str,
                 base_dir: str | None = None):
        validate_box_spec(spec)
        if spec.type == "box":
            spec.urdf_path = None
        else:
            spec.urdf_path = container_mod.generate_box_urdf(spec)
        if not spec.state_keys:
            spec.state_keys = [f"{spec.name}_pos_b", f"{spec.name}_vel_b"]
        super().__init__(spec, num_envs=num_envs, device=device,
                         base_dir=base_dir)

    # ---- isaaclab scene build ------------------------------------------ #
    def setup_scene(self, *, classes=None) -> tuple[str, Any]:
        """Return the ``RigidObjectCfg`` (CuboidCfg for box, URDF otherwise)."""
        if self.spec.type != "box":
            return super().setup_scene(classes=classes)
        c = classes
        spawn = _build_cuboid_spawn(self.spec, c)
        roll_q = self._roll_quat
        offset = _box_origin_offset(self.spec, roll_q.reshape(1, 4))[0]
        init_pos = (float(offset[0]), float(offset[1]),
                    float(self.placement.height_offset + offset[2]))
        cfg = c["RigidObjectCfg"](
            prim_path=f"{{ENV_REGEX_NS}}/Prop_{self.spec.name}",
            spawn=spawn,
            init_state=c["RigidObjectCfg"].InitialStateCfg(
                pos=init_pos,
                rot=(float(roll_q[0]), float(roll_q[1]),
                     float(roll_q[2]), float(roll_q[3])),
            ),
            collision_group=0,
            debug_vis=False,
        )
        return self.spec.name, cfg

    # ---- runtime: keep the logical bottom-face origin ------------------ #
    def _logical_pos(self, root_pos: torch.Tensor,
                     quat: torch.Tensor) -> torch.Tensor:
        if self.spec.type != "box":
            return root_pos
        return root_pos - _box_origin_offset(self.spec, quat)

    def on_reset(self, env_ids, env) -> None:
        """Re-sample placement + reset prop state for the env subset."""
        if self.spec.type != "box":
            return super().on_reset(env_ids, env)
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
        total_q = _quat_mul(yaw_q, roll_q.expand(n, 4))
        world_pos = world_pos + _box_origin_offset(self.spec, total_q)
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
        root_pos = self._logical_pos(bd.root_pos_w, bd.root_quat_w)
        rel = root_pos - rd.root_pos_w
        pos_b = quat_rotate_inverse(rd.root_quat_w, rel)
        rel_v = bd.root_lin_vel_w - rd.root_lin_vel_w
        vel_b = quat_rotate_inverse(rd.root_quat_w, rel_v)
        return {self._pos_key: pos_b, self._vel_key: vel_b}

    # ---- aabb metadata (declared; relative to the prop origin) --------- #
    @property
    def aabb_center(self) -> np.ndarray:
        """Declared aabb centre relative to the prop origin."""
        center, _ = _declared_aabb(self.spec)
        return center

    @property
    def aabb_half(self) -> np.ndarray:
        """Declared aabb half extents."""
        _, half = _declared_aabb(self.spec)
        return half

    def get_aabb_world(self, env) -> tuple[torch.Tensor, np.ndarray]:
        """``(center (N,3) world tensor, half_extents (3,))`` for scoring.

        The half extents are orientation-independent (axis-aligned boxes stay
        axis-aligned under the prop's yaw placement only when yaw == 0; for
        yawed props use the conservative bounding variant -- the benchmark
        configs place containers yaw-free).
        """
        pose = self.get_world_pose(env)
        if pose is None:
            raise RuntimeError(
                f"prop {self.spec.name!r}: get_aabb_world before bind_env")
        pos, quat = pose
        center_w = pos + quat_rotate(quat, torch.as_tensor(
            self.aabb_center, dtype=pos.dtype, device=pos.device
        ).unsqueeze(0).expand_as(pos))
        return center_w, self.aabb_half

    # ---- world pose for the object/* record layer ---------------------- #
    def get_world_pose(self, env) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(pos (N,3), quat (N,4) wxyz)`` world logical (bottom-face) pose."""
        if self._asset is None or not hasattr(self._asset, "data"):
            return None
        quat = self._asset.data.root_quat_w.clone()
        pos = self._logical_pos(self._asset.data.root_pos_w, quat).clone()
        return pos, quat
