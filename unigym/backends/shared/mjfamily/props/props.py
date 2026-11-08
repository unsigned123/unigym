"""mujoco props backend: MjSpec body builder + runtime state reader (L2).

Rigid-prop half; its mujoco-facing siblings are ``props_usd.py`` (the usd
representation tiers), ``props_articulation.py`` (anchored + free
articulations) and ``asset_render.py`` (generated IR payloads, v072
S2.0) -- the props-package modules allowed to touch the mujoco API, all
lazily imported by the mujoco env backend when a ``props:`` section is
wired.

Rigid bodies are appended to the MjSpec BEFORE compilation (the same
pattern as the camera rigs): one dynamic body per prop named
``Prop_<name>`` with a freejoint, geoms from the shared box layout
(props/container.box_layout -- the same geometry the isaaclab URDF
carries, "equivalent combination").  Runtime:

* ``bind_env``  -- resolve body ids + freejoint qpos/dof addresses
  (fail-fast on a missing body);
* ``on_reset``  -- re-sample ``(x, y, z, yaw)`` from the spec's
  PropPlacement (single env at the world origin), write the freejoint
  qpos and zero the joint velocity (no carry-over across episodes);
* ``get_world_poses`` -- ``{name: (pos (3,), quat (4,) wxyz)}`` from
  ``data.xpos``/``data.xquat`` for the object/* record layer;
* ``get_state`` -- body-frame ``<name>_pos_b``/``<name>_vel_b`` relative
  to the robot base (pelvis), mirroring the isaaclab handlers' obs
  semantics for adapters that request prop keys.

N is always 1 (MujocoEnv is single-env); record tensors are batched
to (1, ...) so the object layer is backend-agnostic.
"""
from __future__ import annotations

import numpy as np
import torch

from unigym.adapters.base import RawObsRequest
from unigym.props.articulation import validate_joints, validate_latch
from unigym.props.base import PropHandler, PropPlacement, PropSpec
from unigym.props import container as container_mod
from unigym.props.container import _declared_aabb, validate_box_spec
from unigym.props.primitive import (
    resolve_mass, validate_primitive_spec,
)


def _spawn_geom_kwargs(spec, material: str | None = None) -> dict:
    """Shared spawn-controlled geom kwargs (friction / contact / colour).

    Mirrors the SIMPLE-alignment pass-through used by the box-family path:
    ``spawn.friction`` (scalar or 3-vector), ``spawn.condim`` /
    ``priority`` / ``solref`` / ``solimp``, ``spawn.rgba`` and an optional
    material name.  A silent field keeps the engine default.
    """
    kwargs: dict = {}
    friction = spec.spawn.get("friction")
    if friction is not None:
        if isinstance(friction, (list, tuple)):
            kwargs["friction"] = [float(v) for v in friction]
        else:
            f = float(friction)
            kwargs["friction"] = [f, f, f]   # slide, spin, roll
    for key in ("condim", "priority", "solref", "solimp"):
        if spec.spawn.get(key) is not None:
            kwargs[key] = spec.spawn[key]
    rgba = spec.spawn.get("rgba")
    if rgba is not None:
        kwargs["rgba"] = [float(v) for v in rgba]
    if material is not None:
        kwargs["material"] = material
    return kwargs


# --------------------------------------------------------------------------- #
# MuJoCo prop handlers (the mujoco half of the neutral prop declaration)
#
# The mujoco backend consumes the handler's declaration surface (spec,
# placement, validated joint table) and drives the runtime through
# :class:`MujocoPropBackend`; the isaaclab scene hooks live in the isaaclab
# backend.  These classes deliberately implement only the neutral declaration
# face plus the obs-key catalog the manager exposes.
# --------------------------------------------------------------------------- #
class MujocoPropHandler(PropHandler):
    """Declaration-only mujoco prop handler (no scene-cfg face)."""

    def __init__(self, spec: PropSpec, num_envs: int, device: str,
                 base_dir: str | None = None):
        self.spec = spec
        self.num_envs = num_envs
        self.device = device
        self.placement = PropPlacement(spec.placement)
        self._pos_key = f"{spec.name}_pos_b"
        self._vel_key = f"{spec.name}_vel_b"

    @property
    def state_requests(self) -> list[RawObsRequest]:
        """Declare the body-frame obs keys this prop exposes."""
        return [
            RawObsRequest(self._pos_key, 3, "prop"),
            RawObsRequest(self._vel_key, 3, "prop"),
        ]


class MujocoBallProp(MujocoPropHandler):
    """Sphere prop declaration (mujoco)."""

    def __init__(self, spec: PropSpec, num_envs: int, device: str,
                 base_dir: str | None = None):
        radius = spec.radius if spec.radius is not None \
            else spec.spawn.get("radius")
        if radius is None:
            raise ValueError(
                f"sphere prop {spec.name!r}: radius is required "
                f"(declare radius or spawn.radius)")
        resolve_mass(spec, default=0.1)
        super().__init__(spec, num_envs, device, base_dir)
        self._pos_key = "ball_pos_b"
        self._vel_key = "ball_vel_b"


class MujocoGenericProp(MujocoPropHandler):
    """URDF/USD rigid-object prop declaration (mujoco)."""

    def __init__(self, spec: PropSpec, num_envs: int, device: str,
                 base_dir: str | None = None):
        super().__init__(spec, num_envs, device, base_dir)
        self.static = bool(spec.type == "usd"
                           and spec.spawn.get("static", False))
        if spec.state_keys:
            self._pos_key = spec.state_keys[0]
            self._vel_key = spec.state_keys[1] if len(spec.state_keys) > 1 \
                else spec.state_keys[0].replace("_pos_", "_vel_")
        self._quat_key = self._pos_key.replace("_pos_", "_quat_")

    @property
    def state_requests(self) -> list[RawObsRequest]:
        """Declare the body-frame obs keys (none for static stage props)."""
        if self.static:
            return []
        return [
            RawObsRequest(self._pos_key, 3, "prop"),
            RawObsRequest(self._vel_key, 3, "prop"),
            RawObsRequest(self._quat_key, 4, "prop"),
        ]


class MujocoPrimitiveProp(MujocoPropHandler):
    """Cylinder / capsule prop declaration (mujoco)."""

    def __init__(self, spec: PropSpec, num_envs: int, device: str,
                 base_dir: str | None = None):
        validate_primitive_spec(spec)
        super().__init__(spec, num_envs, device, base_dir)
        keys = list(spec.state_keys)
        if keys:
            self._pos_key = keys[0]
            self._vel_key = (keys[1] if len(keys) > 1
                             else keys[0].replace("_pos_", "_vel_"))


class MujocoBoxProp(MujocoGenericProp):
    """Container / table / box prop declaration (mujoco)."""

    def __init__(self, spec: PropSpec, num_envs: int, device: str,
                 base_dir: str | None = None):
        validate_box_spec(spec)
        if spec.type == "box":
            spec.urdf_path = None
        else:
            spec.urdf_path = container_mod.generate_box_urdf(spec)
        if not spec.state_keys:
            spec.state_keys = [f"{spec.name}_pos_b", f"{spec.name}_vel_b"]
        super().__init__(spec, num_envs, device, base_dir)

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


class MujocoArticulationProp(MujocoPropHandler):
    """Anchored articulation prop declaration (mujoco)."""

    def __init__(self, spec: PropSpec, num_envs: int, device: str,
                 base_dir: str | None = None):
        self.joints = validate_joints(spec)
        self.actuated_joints = [n for n, p in self.joints.items()
                                if p["mode"] == "actuated"]
        self.latch = validate_latch(spec, self.joints)
        self.latch_enabled = self.latch is not None
        self._latch_unlocked: set[int] = set()
        super().__init__(spec, num_envs, device, base_dir)
        self._asset = None
        self._jidx: dict[str, int] = {}

    @property
    def state_requests(self) -> list:
        """No obs face: the articulation state is the record-key family."""
        return []


class MujocoUsdArticulationProp(MujocoArticulationProp):
    """Native USD articulation declaration (isaaclab-only runtime kind)."""


class MujocoFreeArticulationProp(MujocoArticulationProp):
    """Free-floating articulation prop declaration (mujoco)."""


_MUJOCO_REGISTRY: dict[str, type[PropHandler]] = {
    "sphere": MujocoBallProp,
    "rigid_object": MujocoGenericProp,
    "usd": MujocoGenericProp,
    "articulation": MujocoArticulationProp,
    "usd_articulation": MujocoUsdArticulationProp,
    "free_articulation": MujocoFreeArticulationProp,
    "container": MujocoBoxProp,
    "table": MujocoBoxProp,
    "box": MujocoBoxProp,
    "cylinder": MujocoPrimitiveProp,
    "capsule": MujocoPrimitiveProp,
}


class MujocoPropBackendFace:
    """Prop backend face for ``mujoco``/``mjwarp`` (spec-first form)."""

    name = "mujoco"
    #: this face injects prop bodies into the MjSpec pre-compile
    injects_mjspec = True
    handler_registry = _MUJOCO_REGISTRY

    def handler_for(self, spec: PropSpec) -> type[PropHandler]:
        """Return the handler class declared for ``spec.type`` (fail fast)."""
        try:
            return _MUJOCO_REGISTRY[spec.type]
        except KeyError:
            known = ", ".join(sorted(_MUJOCO_REGISTRY))
            raise ValueError(
                f"prop {spec.name!r}: unknown type {spec.type!r} "
                f"(known: {known}); a plane is static scene geometry and "
                f"cannot be a dynamic prop body") from None

    def compat(self, name: str):
        """The mujoco face has no public legacy class names."""
        raise AttributeError(name)

    def build_handler(self, spec: PropSpec, *, num_envs: int, device: str,
                      base_dir: str | None = None) -> PropHandler:
        """Instantiate the declared handler for one spec."""
        return self.handler_for(spec)(
            spec, num_envs=num_envs, device=device, base_dir=base_dir)

    def scene_fields(self, manager, *, classes=None) -> dict:
        """No-op: the mujoco form injects into the MjSpec pre-compile."""
        del manager, classes
        return {}

    def inject_spec(self, manager, mjspec) -> None:
        """Append the declared prop bodies to the MjSpec before compile."""
        add_prop_bodies(manager, mjspec)


def build_mujoco_prop_backend() -> MujocoPropBackendFace:
    """Registry factory: the mujoco prop backend face."""
    return MujocoPropBackendFace()


def add_prop_bodies(prop_manager, mjspec) -> None:
    """Append one free body per prop to the MjSpec (pre-compile)."""
    import mujoco

    for handler in prop_manager._handlers:
        spec = handler.spec
        asset = prop_manager.asset_definitions.get(spec.name)
        if asset is not None:
            # generated payload (v072 S2.0): the neutral IR renders here;
            # runtime is the REUSED articulation view (the compiled Prop_
            # subtree is exactly what it addresses).  Two attach forms:
            # anchored (S2.0 fixtures) and free-root (v072 §13 step 3,
            # the jointless robocasa data objects -- the renderer
            # injects the root freejoint).
            if spec.type == "free_articulation":
                from ..scene.asset_render import add_ir_free_body
                add_ir_free_body(mjspec, handler, asset)
            else:
                from ..scene.asset_render import add_ir_body
                add_ir_body(mjspec, handler, asset)
            continue
        if spec.type == "usd_articulation":
            # native USD articulation is an isaaclab-only kind
            # (no mujoco USD loader); fail-fast instead of silently
            # substituting geometry (the §8 usd tiers cover rigid bodies
            # only)
            raise ValueError(
                f"usd_articulation prop {spec.name!r}: isaaclab-only kind "
                f"(no mujoco USD articulation loader); use type: "
                f"articulation with a URDF fragment on the mujoco backend")
        if spec.type == "articulation":
            # anchored articulated props (jointed fragment; build +
            # runtime live in mujoco_articulation.py -- same package, same
            # mujoco-only convention, split by the file-size discipline)
            from .props_articulation import add_articulation_body
            add_articulation_body(mjspec, handler)
            continue
        if spec.type == "free_articulation":
            # free-floating articulated props (T-B3 GAP-A: the root
            # keeps its freejoint; build + runtime in props_articulation.py)
            from .props_articulation import add_free_articulation_body
            add_free_articulation_body(mjspec, handler)
            continue
        name = f"Prop_{spec.name}"
        if spec.type == "usd" or (
                spec.type == "rigid_object"
                and (spec.mujoco or {}).get("representation")):
            # declared mjcf representation (usd kind, or the T2.4
            # URDF-less rigid_object declaration): attachable fragment /
            # envelope tiers live in props_usd.add_usd_body
            from .props_usd import add_usd_body
            add_usd_body(mjspec, spec, mjspec.worldbody)
            continue
        body = mjspec.worldbody.add_body(name=name)
        body.add_joint(name=f"{name}_free", type=mujoco.mjtJoint.mjJNT_FREE)
        from unigym.props.primitive import resolve_mass
        if spec.type == "sphere":
            # sphere props reuse the spawn parameters (radius/mass) with the
            # S2c top-level radius / density vocabulary as the declared form
            radius = (spec.radius if spec.radius is not None
                      else spec.spawn.get("radius", 0.05))
            mass = resolve_mass(spec, default=0.1)
            body.add_geom(name=f"{name}_geom",
                          type=mujoco.mjtGeom.mjGEOM_SPHERE,
                          size=[float(radius), 0.0, 0.0], mass=mass)
            continue
        if spec.type in ("cylinder", "capsule"):
            # dynamic primitive free body (S2c): the neutral half_height is
            # mujoco's axial size slot, inertia stays engine-computed
            geom = (mujoco.mjtGeom.mjGEOM_CYLINDER
                    if spec.type == "cylinder"
                    else mujoco.mjtGeom.mjGEOM_CAPSULE)
            from .props_usd import add_material
            kwargs = _spawn_geom_kwargs(
                spec, add_material(mjspec, spec))
            body.add_geom(name=f"{name}_geom", type=geom,
                          size=[float(spec.radius),
                                float(spec.half_height), 0.0],
                          mass=resolve_mass(spec), **kwargs)
            continue
        from unigym.props.container import box_layout
        boxes = box_layout(spec)
        mass = resolve_mass(spec, default=1.0)
        from .props_usd import add_material
        mat_name = add_material(mjspec, spec)
        per = mass / len(boxes)
        for i, (center, half) in enumerate(boxes):
            # SIMPLE-alignment pass-through (test/sim2repro-probe): desktop
            # contact params live in spawn, e.g. {condim: 6, priority: 10}
            # (SIMPLE engines/mujoco.py _build_primitive).
            kwargs = _spawn_geom_kwargs(spec, mat_name)
            body.add_geom(name=f"{name}_geom{i}",
                          type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=[float(v) for v in center],
                          size=[float(v) for v in half],
                          mass=per, **kwargs)


class MujocoPropBackend:
    """Runtime side of the wired props (mujoco, N=1)."""

    def __init__(self, prop_manager):
        self.prop_manager = prop_manager
        self._ids: dict[str, int] = {}
        self._qadr: dict[str, int] = {}
        self._vadr: dict[str, int] = {}
        self._articulations = {}   # name -> MujocoPropArticulation
        self._env = None

    # ------------------------------------------------------------------ #
    def bind_env(self, env) -> None:
        """Resolve body ids + freejoint addresses; fail-fast on misses."""
        from ..model.ir_access import model_ir_of

        ir = model_ir_of(env)
        self._env = env
        for handler in self.prop_manager._handlers:
            spec = handler.spec
            if spec.type == "articulation":
                from .props_articulation import MujocoPropArticulation
                art = MujocoPropArticulation(handler)
                art.bind_env(env)
                self._articulations[spec.name] = art
                continue
            if spec.type == "free_articulation":
                from .props_articulation import MujocoPropFreeArticulation
                art = MujocoPropFreeArticulation(handler)
                art.bind_env(env)
                self._articulations[spec.name] = art
                continue
            name = f"Prop_{spec.name}"
            body_row = ir.bodies[name] if name in ir.bodies else None
            bid = -1 if body_row is None or body_row.body_id is None \
                else int(body_row.body_id)
            if bid < 0:
                raise ValueError(f"prop body {name!r} not in the MJ model")
            joints = [j for j in ir.joints if j.body == name]
            if len(joints) != 1 or joints[0].type != "free":
                raise ValueError(
                    f"prop body {name!r} must carry exactly one freejoint")
            joint = joints[0]
            if joint.qpos_adr is None or joint.dof_adr is None:
                raise ValueError(
                    f"prop body {name!r} freejoint has no compiled address")
            self._ids[handler.spec.name] = bid
            self._qadr[handler.spec.name] = int(joint.qpos_adr)
            self._vadr[handler.spec.name] = int(joint.dof_adr)

    def on_reset(self, env_ids, env) -> None:
        """Re-place props for the reset subset (given id order)."""
        import mujoco

        if self._env is None or env_ids is None:
            return
        ids = torch.as_tensor(env_ids).reshape(-1)
        if ids.numel() == 0:
            return
        m = env.mj_model
        datas = getattr(env, "mj_datas", None) or [env.mj_data]
        for e in ids.tolist():
            d = datas[e]
            for handler in self.prop_manager._handlers:
                spec = handler.spec
                art = self._articulations.get(spec.name)
                if art is not None:
                    art.on_reset(m, d)     # anchored: re-anchor + zero joints
                    continue
                x, y, z, yaw = handler.placement.sample(1, "cpu")[0].tolist()
                adr = self._qadr[spec.name]
                d.qpos[adr:adr + 3] = [x, y, z]
                d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0,
                                           np.sin(yaw / 2)]
                vadr = self._vadr[spec.name]
                d.qvel[vadr:vadr + 6] = 0.0
            mujoco.mj_forward(m, d)

    # ------------------------------------------------------------------ #
    def get_world_poses(self, env=None) -> dict[str, tuple]:
        """``{name: (pos (N,3) f32, quat (N,4) wxyz f32)}`` (torch, cpu)."""
        env = self._env if env is None else env
        datas = getattr(env, "mj_datas", None) or [env.mj_data]
        out = {}
        for name, bid in self._ids.items():
            poss, quats = [], []
            for d in datas:
                poss.append(torch.from_numpy(
                    np.asarray(d.xpos[bid], dtype=np.float32)))
                quats.append(torch.from_numpy(
                    np.asarray(d.xquat[bid], dtype=np.float32)))
            out[name] = (torch.stack(poss, dim=0),
                         torch.stack(quats, dim=0))
        for name, art in self._articulations.items():
            poses = [art.get_world_pose(e) for e in range(len(datas))]
            out[name] = (torch.cat([p[0] for p in poses], dim=0),
                         torch.cat([p[1] for p in poses], dim=0))
        return out

    def get_joint_states(self, env=None) -> dict:
        """``{name: {jname: (N,1) f32}}`` articulation joint positions."""
        env = self._env if env is None else env
        datas = getattr(env, "mj_datas", None) or [env.mj_data]
        out = {}
        for name, art in self._articulations.items():
            per = [art.get_joint_states(e) for e in range(len(datas))]
            out[name] = {j: torch.cat([p[j] for p in per], dim=0)
                         for j in per[0]}
        return out

    def actuated_addresses(self) -> dict:
        """``{name: {jname: ctrl_adr}}`` for the resolver's prop tables."""
        return {name: art.actuated_addresses()
                for name, art in self._articulations.items()}

    # ---- director accessors (unigym/director) -------------------- #
    def free_prop_names(self) -> list[str]:
        """Rigid (freejoint) prop names, spec order."""
        return list(self._ids)

    def free_addresses(self, name: str) -> tuple[int, int, int]:
        """``(body_id, freejoint_qpos_adr, freejoint_dof_adr)``."""
        return (self._ids[name], self._qadr[name], self._vadr[name])

    def body_id(self, name: str) -> int | None:
        """Body id of a prop (rigid OR articulation root), else None."""
        if name in self._ids:
            return self._ids[name]
        art = self._articulations.get(name)
        return art.root_bid if art is not None else None

    def articulations(self) -> dict:
        """``{name: MujocoPropArticulation}`` (director joint writes)."""
        return dict(self._articulations)

    def get_state(self, env) -> dict[str, torch.Tensor]:
        """Body-frame obs keys relative to the robot base (pelvis).

        ``<name>_pos_b``: R_b^T (prop_pos - base_pos);
        ``<name>_vel_b``: R_b^T (prop_vel - base_vel) -- the mujoco mirror
        of the isaaclab handlers' obs semantics.
        ``<name>_quat_b`` (v057 additive): the prop orientation relative
        to the base, ``q_b^-1 (x) q_prop`` (wxyz) -- the decision
        package's obj_pose quat source (design v057 §2; requested keys
        only, never a default).
        """
        datas = getattr(env, "mj_datas", None) or [env.mj_data]
        base = int(getattr(env, "_base_body_id"))
        out: dict[str, torch.Tensor] = {}
        for name, bid in self._ids.items():
            poss, vels, quats = [], [], []
            for d in datas:
                bp = d.xpos[base]
                bq = d.xquat[base]
                bv = d.qvel[0:3]
                rot_t = _quat_to_rotmat_t(bq)
                bq_conj = np.array([bq[0], -bq[1], -bq[2], -bq[3]])
                rel = d.xpos[bid] - bp
                jadr = self._vadr[name]
                vel = d.qvel[jadr:jadr + 3] - bv
                poss.append(torch.tensor(
                    rot_t @ rel, dtype=torch.float32))
                vels.append(torch.tensor(
                    rot_t @ vel, dtype=torch.float32))
                q = np.asarray(d.xquat[bid], dtype=np.float64)
                quats.append(torch.tensor(
                    _quat_mul_np(bq_conj, q), dtype=torch.float32))
            out[f"{name}_pos_b"] = torch.stack(poss, dim=0)
            out[f"{name}_vel_b"] = torch.stack(vels, dim=0)
            out[f"{name}_quat_b"] = torch.stack(quats, dim=0)
        return out

    def close(self) -> None:
        """Release prop handles and the articulation backends."""
        self._ids.clear()
        self._qadr.clear()
        self._vadr.clear()
        for art in self._articulations.values():
            art.close()
        self._articulations.clear()
        self._env = None


def _quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a (x) b, wxyz, plain numpy.

    Same convention
    as core.quat.mul; kept local so this module stays torch-only-facing.
    """
    w1, x1, y1, z1 = (float(v) for v in a)
    w2, x2, y2, z2 = (float(v) for v in b)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_to_rotmat_t(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotation-matrix TRANSPOSE (world <- body -> body <- world)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    rot = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return rot.T
