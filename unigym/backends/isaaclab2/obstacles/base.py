"""URDF obstacle manager (Route D).

Loads URDF obstacles directly via Isaac Lab's :class:`UrdfFileCfg` (which runs
:class:`UrdfConverter` at spawn time with on-disk caching — no manual
URDF→USD pre-conversion).  No curriculum.

``placement.mode`` in ``{random, fixed}`` keeps the classic per-env behaviour:
each reset samples one placement (x, y, yaw) from the strategy, the type
assigned to an env sits at that pose, all other types are parked far below
ground.  ``placement.mode == "track"`` reproduces a fixed serpentine course
(``frame_pos`` style): *every* obstacle type owns one ``track_pos``
slot relative to the env origin (robot travels +X) and all of them are placed
active per env — nothing is parked.

The manager is track-agnostic: all track special-casing (slot validation,
init-state baking, reset no-op) lives in :class:`PlacementStrategy` behind the
generic interface :meth:`PlacementStrategy.validate`,
:meth:`PlacementStrategy.initial_pose_for` and
:meth:`PlacementStrategy.needs_reset_placement`.  ``ObstacleManager`` never
reads ``mode == "track"`` itself.

``num_joints == 0`` => :class:`RigidObjectCfg`, ``num_joints > 0`` =>
:class:`ArticulationCfg` with implicit PD actuators built from
``joint_params``.  Anchoring: use ``heavy_dynamic=True`` (verified working
recipe, see ``obstacles/border.py``): the body stays a **normal dynamic rigid
body** — so it really collides with the robot — but is made practically
immovable with a 2000 kg mass plus strong linear/angular damping.  This is
required because *kinematic* URDF bodies (``kinematic=True``) generate NO
contact in GridCloner-cloned multi-env scenes (verified empirically: walls and
the whole obstacle course were walked straight through), so ``kinematic`` is
kept only as a discouraged fallback.  ``fix_base=True`` (root fixed joint) is
only reliable in non-cloned / articulated setups.

The manager is isaaclab-free at import time (cfg classes resolved lazily via
``_resolve``); unit tests inject fake classes through ``classes=``.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .border import border_geometry, build_border_scene_fields
from .placement import PlacementStrategy

# ``heavy_dynamic`` anchor recipe (obstacles/border.py, verified): a *normal*
# dynamic rigid body — so it really collides with the robot — made practically
# immovable with a 2000 kg mass + strong linear/angular damping and velocity
# clamps.  Per-episode ``RigidObject`` reset restores the baked spawn pose.
DEFAULT_HEAVY_MASS = 2000.0       # kg  (dog ~13 kg; ratio ~150: stable)
DEFAULT_HEAVY_LINEAR_DAMPING = 1000.0
DEFAULT_HEAVY_ANGULAR_DAMPING = 1000.0
DEFAULT_HEAVY_MAX_LINEAR_VELOCITY = 2.0
DEFAULT_HEAVY_MAX_ANGULAR_VELOCITY = 2.0


@dataclass
class ObstacleSpec:
    """Configuration for a single obstacle type."""

    name: str
    urdf_path: str
    proportion: float = 1.0
    num_joints: int = 0
    joint_params: dict = field(default_factory=dict)
    init_roll: float = -1.5708  # -pi/2: most URDFs lie flat after this roll
    fix_base: bool = False      # anchor base_link to the world via a root
                                # fixed joint (works for non-cloned scenes /
                                # articulated obstacles; unreliable in
                                # GridCloner-cloned scenes — prefer
                                # ``kinematic`` there)
    kinematic: bool = False     # spawn as an immovable *kinematic* rigid body
                                # (immune to forces, never flies, collides
                                # solidly with the robot).  The robust way to
                                # anchor URDF obstacle shells in multi-env
                                # cloned scenes (fix_base's root joint is not
                                # cloned with a valid body rel).  Requires
                                # num_joints == 0.
                                #
                                # NOTE (2026-08): DEPRECATED.  Empirically a
                                # kinematic URDF body generates NO contact in
                                # GridCloner-cloned scenes (walls and this
                                # whole course were walked straight through,
                                # see obstacles/border.py header).  The working
                                # anchor is ``heavy_dynamic`` — a normal
                                # dynamic rigid body with 2000 kg + strong
                                # damping.
    heavy_dynamic: bool = False  # spawn as a normal *dynamic* rigid body with
                                # an artificially heavy mass (border.py recipe:
                                # 2000 kg) + strong damping, so it collides
                                # really with the robot yet is practically
                                # immovable.  Per-episode ``RigidObject`` reset
                                # restores the baked spawn pose.  Requires
                                # num_joints == 0 and exclusive with
                                # ``kinematic``.
    mass: float | None = None   # heavy_dynamic body mass override (kg);
                                # None -> 2000.0 (border.py recipe)
    linear_damping: float | None = None     # heavy_dynamic linear damping;
                                            # None -> 1000.0 (border.py)
    angular_damping: float | None = None    # heavy_dynamic angular damping;
                                            # None -> 1000.0 (border.py)
    height: float | None = None     # spawn z of base_link origin; None uses
                                    # placement.height_offset. Set to the STL
                                    # min-z magnitude so the mesh rests on ground.
    merge_fixed_joints: bool = False    # merge fixed-jointed links into one
                                        # rigid body (static multi-mesh URDFs)


def _resolve(classes):
    """Return the isaaclab cfg-class table (lazy import when classes is None)."""
    if classes is not None:
        return classes
    import isaaclab.sim as sim_utils  # noqa: F401  (ensures spawners loaded)
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg
    from isaaclab.sim.converters import UrdfConverterCfg
    from isaaclab.sim.schemas.schemas_cfg import (
        ArticulationRootPropertiesCfg,
        CollisionPropertiesCfg,
        MassPropertiesCfg,
        RigidBodyPropertiesCfg,
    )
    from isaaclab.sim.spawners import UrdfFileCfg

    return {
        "UrdfFileCfg": UrdfFileCfg,
        "RigidObjectCfg": RigidObjectCfg,
        "ArticulationCfg": ArticulationCfg,
        "ImplicitActuatorCfg": ImplicitActuatorCfg,
        "UrdfConverterCfg": UrdfConverterCfg,
        "ArticulationRootPropertiesCfg": ArticulationRootPropertiesCfg,
        "MassPropertiesCfg": MassPropertiesCfg,
        "RigidBodyPropertiesCfg": RigidBodyPropertiesCfg,
        "CollisionPropertiesCfg": CollisionPropertiesCfg,
    }


def _resolve_urdf_path(path: str, base_dir: str | None) -> str:
    """Resolve a (possibly relative) urdf path to an absolute path.

    Tries ``base_dir`` first; if the file isn't there, falls back to the
    parent of ``base_dir`` so a path like ``obstacles/box.urdf`` resolves
    whether the config lives next to ``obstacles/`` or one level below it
    (the common ``config/`` layout).  Third candidate (``obstacles/``-
    prefixed declarations only): the vendored assets that ship with THIS
    package (``backends/isaaclab2/obstacles/``) — they moved with the M2.5
    backend layout, the resolution contract does not.
    """
    if os.path.isabs(path) or "://" in path:
        return path
    if base_dir is None:
        base_dir = os.getcwd()
    vendored_parent = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))  # .../backends/isaaclab2
    candidates = [
        os.path.join(base_dir, path),
        os.path.join(os.path.dirname(base_dir), path),
    ]
    if path.startswith("obstacles/"):
        candidates.append(os.path.join(vendored_parent, path))
    for cand in candidates:
        if os.path.exists(cand):
            return os.path.abspath(cand)
    # fall back to the first candidate (best error message downstream)
    return os.path.abspath(candidates[0])


# --------------------------------------------------------------------------- #
# quaternion helpers (wxyz, world-frame composition: q = q_yaw ⊗ q_roll)
# --------------------------------------------------------------------------- #
def _roll_quat(roll: float) -> torch.Tensor:
    return torch.tensor(
        [math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0], dtype=torch.float32)


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    half = yaw * 0.5
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device, dtype=yaw.dtype)
    q[:, 0] = torch.cos(half)
    q[:, 3] = torch.sin(half)
    return q


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a ⊗ b (both (N,4) wxyz)."""
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    out = torch.empty_like(a)
    out[:, 0] = aw * bw - ax * bx - ay * by - az * bz
    out[:, 1] = aw * bx + ax * bw + ay * bz - az * by
    out[:, 2] = aw * by - ax * bz + ay * bw + az * bx
    out[:, 3] = aw * bz + ax * by - ay * bx + az * bw
    return out


class ObstacleManager:
    """URDF obstacle scene manager (direct URDF load, no curriculum)."""

    def __init__(self, config: dict, num_envs: int, env_spacing: float,
                 device: str = "cpu", base_dir: str | None = None):
        self.num_envs = int(num_envs)
        self.env_spacing = float(env_spacing)
        self.device = device
        self.base_dir = base_dir
        self.specs: list[ObstacleSpec] = [
            ObstacleSpec(**s) for s in config.get("types", [])]
        self.placement = PlacementStrategy(config.get("placement", {}))
        # resolve absolute urdf paths up-front (cheap; aids error messages)
        for s in self.specs:
            s.urdf_path = _resolve_urdf_path(s.urdf_path, base_dir)
        for s in self.specs:
            if s.kinematic and s.num_joints != 0:
                raise ValueError(
                    f"obstacle '{s.name}': kinematic=True requires "
                    f"num_joints == 0 (single rigid body)")
            if s.heavy_dynamic and s.num_joints != 0:
                raise ValueError(
                    f"obstacle '{s.name}': heavy_dynamic=True requires "
                    f"num_joints == 0 (single rigid body)")
            if s.kinematic and s.heavy_dynamic:
                raise ValueError(
                    f"obstacle '{s.name}': 'kinematic' and 'heavy_dynamic' "
                    f"are mutually exclusive; use heavy_dynamic=True (the "
                    f"working anchor) instead")
        # track-specific constraints live in the placement strategy
        self.placement.validate(self.specs)
        # static corridor walls (see obstacles/border.py): URDF kinematic
        # bodies do not generate contact in this cloned setup, so the border
        # uses dynamic heavy rigid cuboids; course obstacles use the same
        # recipe via ``heavy_dynamic=True``.
        self.border_cfg = config.get("borders") or None
        self._border_geom = border_geometry(self.border_cfg)
        # populated by bind_env()
        self._assets: list[Any] = []
        self._roll_quat: list[torch.Tensor] = []
        self.env_type: torch.Tensor = self._assign_types()

    # ------------------------------------------------------------------ #
    # env -> obstacle-type assignment (deterministic, by proportion)
    # ------------------------------------------------------------------ #
    def _assign_types(self) -> torch.Tensor:
        n = self.num_envs
        if not self.specs:
            return torch.zeros(n, dtype=torch.long)
        props = np.array([s.proportion for s in self.specs], dtype=np.float64)
        props = props / props.sum()
        cum = np.cumsum(props)
        types = np.zeros(n, dtype=np.int64)
        for i in range(n):
            f = (i + 0.5) / n
            types[i] = int(np.searchsorted(cum, f, side="right"))
        types = np.clip(types, 0, len(self.specs) - 1)
        return torch.tensor(types, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # scene configuration
    # ------------------------------------------------------------------ #
    def setup_scene(self, *, classes=None) -> dict:
        """Return ``{SceneCfg field: cfg}`` to merge into the scene config.

        URDF obstacle fields plus the static corridor border walls
        (``obstacles.borders``, represented as static collision cuboids).
        """
        c = _resolve(classes)
        fields: dict[str, Any] = {}
        for si, spec in enumerate(self.specs):
            spawn = self._build_spawn(spec, c)
            roll_q = _roll_quat(spec.init_roll)
            z0 = (spec.height if spec.height is not None
                  else self.placement.height_offset)
            baked = self.placement.initial_pose_for(si)
            if baked is not None:
                # fixed course: bake the type's track slot into the spawn so
                # every clone is anchored there at construction (and
                # ``scene.reset`` restores it) — no reset-time teleports.
                sx, sy, syaw = baked
                syaw_q = _yaw_quat(torch.tensor([syaw]))
                init_rot = _quat_mul(syaw_q, roll_q.unsqueeze(0))[0]
                init_pos = (sx, sy, z0)
            else:
                init_rot = roll_q
                init_pos = (0.0, 0.0, z0)
            if spec.num_joints > 0:
                cfg = c["ArticulationCfg"](
                    prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{spec.name}",
                    spawn=spawn,
                    init_state=c["ArticulationCfg"].InitialStateCfg(
                        pos=init_pos,
                        rot=(float(init_rot[0]), float(init_rot[1]),
                             float(init_rot[2]), float(init_rot[3])),
                        joint_pos={".*": 0.0},
                    ),
                    actuators=self._build_actuators(spec, c),
                    collision_group=0,
                    debug_vis=False,
                )
            else:
                cfg = c["RigidObjectCfg"](
                    prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{spec.name}",
                    spawn=spawn,
                    init_state=c["RigidObjectCfg"].InitialStateCfg(
                        pos=init_pos,
                        rot=(float(init_rot[0]), float(init_rot[1]),
                             float(init_rot[2]), float(init_rot[3])),
                    ),
                    collision_group=0,
                    debug_vis=False,
                )
            fields[f"obstacle_{spec.name}"] = cfg
        fields.update(build_border_scene_fields(self.border_cfg, classes=classes))
        return fields

    def _build_spawn(self, spec: ObstacleSpec, c: dict):
        UrdfFileCfg = c["UrdfFileCfg"]
        kwargs = dict(
            asset_path=spec.urdf_path,
            merge_fixed_joints=spec.merge_fixed_joints,
            fix_base=spec.fix_base,    # anchor base_link (articulated obstacles)
            collision_from_visuals=True,
            make_instanceable=True,
        )
        if spec.num_joints > 0:
            # give the USD joint drive valid (zero) defaults; the Articulation
            # implicit actuators override with the real gains.
            JointDriveCfg = c["UrdfConverterCfg"].JointDriveCfg
            PDGainsCfg = JointDriveCfg.PDGainsCfg
            kwargs["joint_drive"] = JointDriveCfg(
                gains=PDGainsCfg(stiffness=0.0, damping=0.0))
        else:
            kwargs["joint_drive"] = None
        if spec.kinematic:
            # immovable kinematic body: no root joint at all (its world-fixed
            # joint would corrupt cloned scenes), immune to forces -> never
            # flies, still collides with the robot.
            # NOTE (2026-08): DEPRECATED — kinematic URDF bodies generate NO
            # contact in GridCloner-cloned scenes (see border.py header).
            kwargs["fix_base"] = False
            from .kinematic import spawn_kinematic_urdf
            spawn = UrdfFileCfg(**kwargs)
            spawn.func = spawn_kinematic_urdf
            return spawn
        if spec.heavy_dynamic:
            # border.py recipe (verified): keep the body a *normal dynamic*
            # rigid body so it really collides, but give it a 2000 kg mass and
            # strong damping + velocity clamps -> practically immovable; the
            # baked spawn pose is restored every episode by RigidObject reset.
            # The spawner-config props are authored on the source prim before
            # the @clone decorator duplicates it into every env, exactly like
            # the working ``border_*`` CuboidCfgs.
            kwargs["fix_base"] = False   # no root joint (clone-hostile)
            MassPropertiesCfg = c["MassPropertiesCfg"]
            RigidBodyPropertiesCfg = c["RigidBodyPropertiesCfg"]
            CollisionPropertiesCfg = c["CollisionPropertiesCfg"]
            kwargs["mass_props"] = MassPropertiesCfg(
                mass=spec.mass if spec.mass is not None else DEFAULT_HEAVY_MASS)
            kwargs["rigid_props"] = RigidBodyPropertiesCfg(
                linear_damping=(spec.linear_damping
                                if spec.linear_damping is not None
                                else DEFAULT_HEAVY_LINEAR_DAMPING),
                angular_damping=(spec.angular_damping
                                 if spec.angular_damping is not None
                                 else DEFAULT_HEAVY_ANGULAR_DAMPING),
                max_linear_velocity=DEFAULT_HEAVY_MAX_LINEAR_VELOCITY,
                max_angular_velocity=DEFAULT_HEAVY_MAX_ANGULAR_VELOCITY,
            )
            kwargs["collision_props"] = CollisionPropertiesCfg(
                collision_enabled=True)
            return UrdfFileCfg(**kwargs)
        if spec.num_joints == 0 and spec.fix_base:
            # fix_base makes the URDF importer insert a fixed root joint
            # (articulation root). RigidObject rejects *enabled* articulation
            # roots, so disable articulation for single-body obstacles: the
            # body stays anchored via the root fixed joint (RigidObjectCfg
            # path keeps working; ArticulationCfg path is unaffected).
            ArticRootCfg = c["ArticulationRootPropertiesCfg"]
            kwargs["articulation_props"] = ArticRootCfg(
                articulation_enabled=False)
        return UrdfFileCfg(**kwargs)

    def _build_actuators(self, spec: ObstacleSpec, c: dict) -> dict:
        ImplicitActuatorCfg = c["ImplicitActuatorCfg"]
        actuators: dict[str, Any] = {}
        if spec.joint_params:
            for jname, gains in spec.joint_params.items():
                actuators[jname] = ImplicitActuatorCfg(
                    joint_names_expr=[jname],
                    stiffness=float(gains.get("stiffness", 0.0)),
                    damping=float(gains.get("damping", 0.0)),
                )
        else:
            actuators["all"] = ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=0.0, damping=0.0)
        return actuators

    # ------------------------------------------------------------------ #
    # runtime
    # ------------------------------------------------------------------ #
    def bind_env(self, env) -> None:
        """Cache asset handles + per-spec roll quaternions (after scene init)."""
        self._assets = [env.scene[f"obstacle_{s.name}"] for s in self.specs]
        self._roll_quat = [_roll_quat(s.init_roll) for s in self.specs]

    def on_reset(self, env_ids, env) -> None:
        """Re-sample obstacle placements for the given env subset.

        Called by ``IsaaclabEnv.reset_envs`` *after* ``_reset_idx`` (robot already
        teleported to spawn) and *before* ``write_data_to_sim`` so the new
        obstacle poses are committed together with the robot reset.

        ``placement.needs_reset_placement()`` is ``False`` for ``track`` mode:
        the course is baked into the spawn ``init_state`` (each type anchored
        at its own slot), so ``scene.reset`` inside ``_reset_idx`` already
        restores every obstacle to its slot.  No reset-time teleports ->
        nothing to snap back.  Any other mode places the env-assigned type at
        a fresh random pose and parks the other types below ground.
        """
        if not self.specs or not self._assets:
            return
        if not self.placement.needs_reset_placement():
            return
        env_ids = torch.as_tensor(env_ids, device=env.device)
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        origins = env.scene.env_origins[env_ids]  # (n, 3)
        placements = self.placement.sample(n, env.device)  # (n, 4)
        world_pos = origins + placements[:, :3]
        yaw = placements[:, 3]
        yaw_q = _yaw_quat(yaw)
        env_type = self.env_type.to(env.device)[env_ids]
        for si, (asset, roll_q) in enumerate(zip(self._assets, self._roll_quat)):
            roll_q = roll_q.to(env.device).type_as(yaw_q)
            total_q = _quat_mul(yaw_q, roll_q.expand(n, 4))
            mask = env_type == si
            spec_z = self.specs[si].height
            pose = torch.zeros(n, 7, device=env.device, dtype=world_pos.dtype)
            # park unassigned instances far below ground
            pose[:, 0] = origins[:, 0]
            pose[:, 1] = origins[:, 1]
            pose[:, 2] = -100.0
            pose[:, 3] = 1.0
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                pose[idx, 0:3] = world_pos[idx]
                if spec_z is not None:
                    pose[idx, 2] = origins[idx, 2] + spec_z
                pose[idx, 3:7] = total_q[idx]
            asset.write_root_pose_to_sim(pose, env_ids=env_ids)

    def get_obstacle_positions(self, env) -> dict:
        """Return ``{spec_name: root_pos_w (N,3)}`` for inspection (optional)."""
        out: dict[str, torch.Tensor] = {}
        for spec, asset in zip(self.specs, self._assets):
            if asset is not None and hasattr(asset, "data"):
                out[spec.name] = asset.data.root_pos_w.clone()
        return out

    def get_border_positions(self, env) -> dict:
        """World poses (N,3) of the static corridor walls, if any.

        Derived from the border geometry plus ``env_origins`` (the walls are
        static scene assets; there is no asset handle to query).
        """
        if not self._border_geom:
            return {}
        origins = env.scene.env_origins
        out = {}
        for name, x, y, z in self._border_geom:
            out[name] = origins[:, :3] + torch.tensor(
                [x, y, z], device=origins.device, dtype=origins.dtype)
        return out

    def monitor_envs(self, env) -> dict:
        """Per-env obstacle sanity metrics for progress logging (n_env,).

        ``disp_max`` = max lateral distance of any obstacle clone from its env
        origin (anchored course obstacles sit within a few metres; a flying or
        snap-back obstacle blows this up).  ``z_max`` = max root z over all
        clones.  Empty dict when no assets are bound (or no obstacles).
        """
        if not self._assets:
            return {}
        origins = env.scene.env_origins
        disp_list, z_list = [], []
        for asset in self._assets:
            try:
                p = asset.data.root_pos_w
            except AttributeError:
                continue
            disp_list.append(torch.norm(p[:, :2] - origins[:, :2], dim=-1))
            z_list.append(p[:, 2])
        if not disp_list:
            return {}
        return {
            "disp_max": torch.stack(disp_list).max(dim=0).values,
            "z_max": torch.stack(z_list).max(dim=0).values,
        }
