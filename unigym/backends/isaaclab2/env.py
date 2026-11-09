"""IsaaclabEnv: ManagerBasedEnv wrapper for policy rollout data collection.

No reward/termination (pure kinematic/dynamic logging). P2 episode handling is
**per-env**: each env carries its own episode clock and (possibly randomized)
duration; ``step()`` returns ``extra["reset_env_ids"]`` listing envs whose
episode just ended. The caller records the final step first (data invariant:
record-before-reset) and then calls :meth:`reset_envs with those ids, which
replays the parent-class reset chain for the subset:
``_reset_idx`` (scene teleport + mode="reset" DR events + manager resets)
-> ``write_data_to_sim`` -> ``sim.forward`` -> bookkeeping + obs refresh.

Domain randomization rides on the native EventManager: yaml ``events`` terms
are converted to EventTermCfg (see ``dr_events.py``). Because
``ManagerBasedEnv.load_managers`` applies "startup" events only for the exact
base class, IsaaclabEnv applies them itself right after construction, followed
by one initial application of mode="reset" events so per-episode DR (mass,
com, ...) is active from episode 0.

All joint/foot quantities exposed to the adapter and written to disk are in
the adapter's declared (policy) order; the env gathers them from Isaac Lab
buffers by name, so ordering is correct by construction.
"""
from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from unigym.cameras.manager import validate_image_frame
from unigym.core import quat

from unigym.env.episode import parse_episode_length, sample_durations
from .init_pose import build_init_pose_cfg, sample_init_pose
from .terrains.native import NativeTerrainPreset
from unigym.env.naming import contact_sensor_prim_path
from unigym.env.obs_catalog import (
    build_obs_catalog,
    canonicalize_obs_requests,
    split_articulation_key,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from unigym.cameras import CameraManager
    from unigym.contacts import ContactManager


from unigym.env.backends.base import BackendCapabilities

BACKEND_CAPABILITIES = BackendCapabilities.full()

__all__ = ["IsaaclabEnv", "BACKEND_CAPABILITIES"]


def _base_z_obs(env):
    return env.scene["robot"].data.root_pos_w[:, 2:3]


def _resolve_terrain_cfg(terrain_cfg, terrain, num_envs, env_spacing):
    """Return the scene terrain fields: native IR wins over the yaml preset."""
    if terrain is not None:
        return NativeTerrainPreset(terrain).build(num_envs, env_spacing)
    if terrain_cfg is None:
        raise ValueError(
            "IsaaclabEnv requires a terrain preset: pass terrain_cfg "
            "(yaml terrain) or terrain=<Terrain>")
    return terrain_cfg


def _add_static_shape_fields(scene, static_shapes) -> None:
    """Merge static ``World.add`` primitives as native static colliders.

    Static shapes are ``AssetBaseCfg`` fields with collision-only shape
    spawners (no RigidBodyAPI), cloned per env at their declared pose.
    A field-name collision fails fast instead of shadowing a scene entry.
    """
    if not static_shapes:
        return
    from .static_shapes import build_static_scene_fields

    for key, value in build_static_scene_fields(static_shapes).items():
        if hasattr(scene, key):
            raise ValueError(
                f"static primitive scene field {key!r} collides with an "
                f"existing scene entry")
        setattr(scene, key, value)


def _az_el_to_quat(az_deg: float, el_deg: float) -> tuple:
    """Azimuth/elevation (degrees) -> quaternion (w, x, y, z) for a DistantLight.

    A USDLux DistantLight emits along its local -Z, i.e. light travels from
    +Z toward -Z. We rotate the prim so local +Z points toward the sun
    position defined by azimuth (CCW from +X about world Z) and elevation
    (above the XY plane). The result is (w, x, y, z) as expected by
    ``AssetBaseCfg.InitialStateCfg.rot``.

    (DELEGATED to unigym.core.quat.az_el, M1 convergence)
    """
    return quat.az_el(az_deg, el_deg)


def _configure_robot_scene(cfg, robot_profile, adapter, controls,
                           articulations, recipes, prop_manager, *, events_cfg,
                           recorder, camera_manager,
                           contact_manager, director_cfg, init_pose_cfg):
    """Fill the scene/action cfg for the single or partitioned layout.

    Returns ``(partitioned, fields)``.  Kept out of ``IsaaclabEnv.__init__``
    so the single-articulation tail and the partitioned branch each stay
    inside the complexity budget.
    """
    ars = tuple(articulations or ())
    partitioned = len(ars) > 1
    if partitioned:
        from . import articulation_runtime as _articulation_runtime

        _articulation_runtime.validate_options(
            adapter, events_cfg=events_cfg, recorder=recorder,
            prop_manager=prop_manager,
            camera_manager=camera_manager, contact_manager=contact_manager,
            director_cfg=director_cfg, init_pose_cfg=init_pose_cfg)
        fields = _articulation_runtime.configure_scene(
            cfg, controls, ars, dict(recipes or {}))
        return partitioned, fields
    cfg.scene.robot = robot_profile.articulation_cfg
    # The generic `.*_foot` regex only matches foot-style body names;
    # derive an explicit alternation from the profile's declared feet
    # (body_parts["feet"]) when available, else keep the default
    # (profiles without body_parts — behavior unchanged for all
    # existing robot profiles).
    cfg.scene.contact_forces.prim_path = contact_sensor_prim_path(
        (robot_profile.body_parts or {}).get("feet"))
    # P3 (G1): the Nucleus g1.usd carries 14 finger DOFs beyond the 29
    # policy joints, so the action term must target exactly the profile's
    # declared joints instead of all asset DOFs.  When joint_order covers
    # the full asset set, the resolved joint ids (and thus the behavior) are
    # identical to the [".*"] default.
    if robot_profile.joint_order:
        cfg.actions.g1.joint_names = list(robot_profile.joint_order)
    # actuated prop joints join the action surface AUTOMATICALLY
    # (declaration IS capability): one preserve_order JointPositionAction term per
    # articulation prop, field name == prop name == the NamedAction
    # entry key adapters supply (alongside "g1"/"hand").
    if prop_manager is not None:
        for h in prop_manager._handlers:
            actuated = list(getattr(h, "actuated_joints", []) or [])
            if not actuated:
                continue
            if hasattr(cfg.actions, h.spec.name):
                raise ValueError(
                    f"prop action entry {h.spec.name!r} collides with "
                    f"an existing action term name")
            setattr(cfg.actions, h.spec.name, JointPositionActionCfg(
                asset_name=f"prop_{h.spec.name}",
                joint_names=actuated,
                scale=1.0,
                use_default_offset=False,
                preserve_order=True))
    # profile-declared extra NamedAction faces (the G1 "hand"
    # dex3 finger face on with-hand assets) -- one preserve_order term
    # per declared entry, field name == the NamedAction entry key
    # (declaration-becomes-capability, same rule as the prop terms
    # above).  ``None`` (every profile built without actuate_hands)
    # leaves the action surface untouched.
    for term_name, term_joints in (
            robot_profile.extra_action_terms or {}).items():
        if hasattr(cfg.actions, term_name):
            raise ValueError(
                f"extra action entry {term_name!r} collides with an "
                f"existing action term name")
        setattr(cfg.actions, term_name, JointPositionActionCfg(
            asset_name="robot",
            joint_names=list(term_joints),
            scale=1.0,
            use_default_offset=False,
            preserve_order=True))
    return partitioned, None


@configclass
class _IsaaclabSceneCfg(InteractiveSceneCfg):
    num_envs = 1  # overwritten at runtime
    env_spacing = 4.0
    ground: AssetBaseCfg | None = None  # flat terrain preset
    terrain: TerrainImporterCfg | None = None  # mesh terrain preset (steps, ...)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.95, 0.95, 0.95)),
    )
    dome_light: AssetBaseCfg | None = None  # optional sky-fill light (Task 1: brighter video)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        history_length=1,
    )
    robot: ArticulationCfg | None = None  # filled from RobotRecipe
    # camera sensors (obs + recorder camera_ref) join the scene ONLY via
    # cameras/ CameraManager.setup_scene under cam_<name> fields (the
    # recorder's self-built tiled_camera path was removed).


@configclass
class _IsaaclabActionsCfg:
    # named action faces.  The field name IS the entry key adapters
    # use in their postprocess_action dict ("g1" = the G1 body face).
    # preserve_order=True keeps the term input dim i == joint_names[i], which
    # is what makes the NamedAction name addressing trivially correct
    # (IsaaclabJointResolver contract); explicit name lists are set from the
    # profile at construction.  With the full profile joint_order the
    # resolved joint ids (and thus the behavior) are identical to the reference
    # [".*"] term.
    g1: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class _PolicyObsGroup(ObservationGroupCfg):
    concatenate_terms = True
    base_z = ObservationTermCfg(func=_base_z_obs)


@configclass
class _IsaaclabObsCfg:
    policy = _PolicyObsGroup()


@configclass
class _IsaaclabEnvCfg(ManagerBasedEnvCfg):
    decimation = 4
    episode_length_s = 10000.0
    scene: _IsaaclabSceneCfg = _IsaaclabSceneCfg()
    actions: _IsaaclabActionsCfg = _IsaaclabActionsCfg()
    observations: _IsaaclabObsCfg = _IsaaclabObsCfg()

    def __post_init__(self):
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.81)


class IsaaclabEnv(ManagerBasedEnv):
    """Data-collection environment (non-RL ManagerBasedEnv).

    Implements the EnvBackend contract (isaaclab backend); see
    ``unigym/env/backends/base.py`` and docs/contracts/backend_contract.md.
    """

    def __init__(self, robot_profile, adapter, terrain_cfg, num_envs: int,
                 episode_length_s: float, env_spacing: float = 4.0,
                 decimation: int = 4, sim_dt: float = 0.005,
                 contact_force_threshold: float = 1.0,
                 events_cfg=None, recorder=None,
                 prop_manager=None,
                 lighting: dict | None = None,
                 init_pose: dict | None = None,
                 camera_manager: CameraManager | None = None,
                 contact_manager: ContactManager | None = None,
                 director_cfg: dict | None = None,
                 device: str = "cuda:0", terrain=None,
                 static_shapes=None,
                 articulations=None, controls=None, recipes=None,
                 backend_options=None):
        """Build the collect env: cfg stack, scene fields and manager seams.

        Args:
            robot_profile: RobotRecipe (articulation cfg + joint order +
                feet/body parts); source of the action term's joint list.
            adapter: Adapter consuming this env (read for its
                ``wants_reset_render`` opt-in only).
            num_envs: parallel env count (>1 requires the multi_env
                capability; gated in build_env_backend).
            episode_length_s: sampled episode duration (s; scalar or
                ``[lo, hi]`` pair parsed by ``parse_episode_length``).
            env_spacing: scene grid spacing between cloned envs (m).
            decimation: sim steps per policy step (policy dt = decimation *
                ``sim_dt``).
            sim_dt: physics step size (s; written to ``cfg.sim.dt``).
            contact_force_threshold: foot-contact binary threshold (N; feeds
                the ``foot_contact`` extra key).
            events_cfg: optional yaml ``randomization.events`` dict
                (mode/interval DR events; backend capability ``events``).
            terrain_cfg: ``{SceneCfg field: cfg}`` dict from a TerrainPreset
                (e.g. ``{"ground": AssetBaseCfg}`` for flat or
                ``{"terrain": TerrainImporterCfg}`` for mesh terrain).
            terrain: optional native :class:`~unigym.scenes.terrain.Terrain`
                IR; when given it overrides ``terrain_cfg`` with
                :class:`NativeTerrainPreset` (one generator tile per env).
            static_shapes: optional static ``World.add`` primitive
                declarations (``World.static_declarations()``); mapped to
                native static colliders (``AssetBaseCfg`` + collision-only
                shape spawners) per env.
            recorder: retained kwarg for backend capability gating
                only (``requested_features`` reads ``recorder.enabled``).
                The recorder adds NO scene sensor — it binds a
                CameraManager rig (``camera_ref``); video frames are read
                from the manager's cache by the CollectSink.
            prop_manager: optional :class:`unigym.props.PropManager`;
                its ``setup_scene`` adds interactive prop (door/ball/generic)
                fields to the scene, ``on_reset`` re-places them each episode,
                and ``get_state`` injects per-step prop state into obs_cache
                for adapters that request it (P15).
            lighting: optional dict from yaml ``env.lighting``; configures a
                DomeLight (sky fill) and/or overrides the DistantLight to
                brighten the camera image (Task 1).
            init_pose: optional dict from yaml ``init_pose``; when present,
                a fraction ``prob`` of resetting envs spawn at a randomized
                orientation (roll/pitch/yaw ranges in degrees) instead of the
                default upright pose.  Used to collect recovery/fall data
                where the robot starts already lying down.
            camera_manager: optional
                :class:`unigym.cameras.CameraManager` (built by
                collect.py from yaml ``cameras`` + the adapter's image obs
                requests, docs/contracts/camera_contract.md).  Its sensors join the
                scene before construction; ``bind_env`` grabs the handles;
                ``sync_pose`` runs at each step tail; ``on_reset`` after each
                reset chain; ``tick_and_read`` fills the image obs keys of
                the cache (shape-validated against the manager's
                ImageObsRequests).
            contact_manager: optional
                :class:`unigym.contacts.ContactManager` (built by collect.py
                from yaml ``contacts``, docs/contracts/contacts_contract.md).  Its
                ContactSensorCfgs join the scene before construction
                (``setup_scene``); ``bind_env`` resolves bodies + grabs the
                sensor handles; ``read()`` is merged into ``extra`` as
                ``contact/<name>`` keys every step (record channel L2) and
                into obs_cache for adapter-requested keys.
            director_cfg: optional yaml ``director`` dict; mounts the virtual
                gripper scene fields (director_contract §4.3).
            device: torch device string for the sim (default ``cuda:0``).
            articulations: the run's ordered neutral articulation specs.
            controls: neutral ``ControlRuntime`` entries built from
                ``RunSpec.controls`` + the robot recipes.
            recipes: ``{articulation name: RobotRecipe}`` supplying each
                articulation's ArticulationCfg.
            backend_options: backend-private options; isaaclab declares none
                (``env.backend_options.isaaclab`` fails fast before build).
        """
        self._dur_lo, self._dur_hi = parse_episode_length(episode_length_s)
        self._contact_force_threshold = float(contact_force_threshold)
        self._init_pose = build_init_pose_cfg(init_pose)
        self._articulation_specs = tuple(articulations or ())
        self._recipes = dict(recipes or {})
        if backend_options:
            raise ValueError(
                "isaaclab backend declares no backend_options (unknown keys "
                f"{sorted(backend_options)}); supported options: []")
        self._planned_controls = tuple(controls or ())

        cfg = _IsaaclabEnvCfg()
        cfg.decimation = decimation
        # internal resets are driven by our own counter
        cfg.episode_length_s = max(self._dur_hi, 1e6)
        cfg.sim.device = device
        cfg.sim.dt = sim_dt
        cfg.sim.render_interval = decimation
        cfg.scene.num_envs = num_envs
        cfg.scene.env_spacing = env_spacing
        terrain_cfg = _resolve_terrain_cfg(
            terrain_cfg, terrain, num_envs, env_spacing)
        for key, value in terrain_cfg.items():  # merge terrain preset fields
            setattr(cfg.scene, key, value)
        # v102b: more than one policy binding switches the scene/action cfg
        # to the multi-articulation layout (independent articulation clones +
        # one action term per binding).  One binding (or none) keeps the
        # historical single-robot configuration byte-identically.
        (self._partitioned, self._articulation_fields) = \
            _configure_robot_scene(
                cfg, robot_profile, adapter, controls, articulations,
                recipes, prop_manager, events_cfg=events_cfg,
                recorder=recorder,
                camera_manager=camera_manager,
                contact_manager=contact_manager,
                director_cfg=director_cfg, init_pose_cfg=self._init_pose)
        # note: `recorder` intentionally unused here (no scene contribution
        # kept in the signature for capability gating kwargs).
        if prop_manager is not None:  # merge interactive prop fields (P15)
            for key, value in prop_manager.setup_scene().items():
                setattr(cfg.scene, key, value)
        # static World primitives (S2b): the same declarations the mujoco
        # inline scene renders become native static colliders here.
        _add_static_shape_fields(cfg.scene, static_shapes)
        # director: the virtual gripper joins the scene as a fixed-
        # base KINEMATIC rigid body (--direct only; director_contract §4.3)
        self.director_cfg = director_cfg
        if director_cfg is not None:
            from unigym.backends.isaaclab2.director.gripper import (
                isaaclab_scene_fields)
            from unigym.director.gripper import normalize_gripper_cfg
            gcfg = normalize_gripper_cfg(director_cfg.get("gripper"))
            if gcfg is not None:
                for key, value in isaaclab_scene_fields(gcfg).items():
                    setattr(cfg.scene, key, value)
        if lighting is not None:  # Task 1: brighter scene for video capture
            self._apply_lighting(cfg.scene, lighting)
        # feat/camera-obs: camera-manager sensors join the scene before
        # cloning (same pattern as props above).  This
        # INCLUDES the implicit depth sugar-shim camera when the adapter
        # requests the depth alias obs keys (yaml ``camera_depth`` params
        # were folded into the manager spec by collect.py).
        self.camera_manager = camera_manager
        # r4 ha_open_door: opt-in reset-time render (adapter-declared t0
        # frame contract -- upstream provider fills the frame BEFORE the
        # first control tick; applied in reset_envs)
        self._render_on_reset = bool(
            getattr(adapter, "wants_reset_render", False))
        if self.camera_manager is not None:
            self.camera_manager.setup_scene(cfg.scene)
        # feat/contact-manager: contact-group sensors join the scene before
        # cloning (same pattern as props/cameras above)
        self.contact_manager = contact_manager
        if self.contact_manager is not None:
            for key, value in self.contact_manager.setup_scene().items():
                setattr(cfg.scene, key, value)
        if events_cfg is not None:
            cfg.events = events_cfg
        super().__init__(cfg=cfg)

        if self._partitioned:
            self._bind_partitioned(adapter, num_envs)
            return

        self.robot = self.scene["robot"]
        self.contact = self.scene["contact_forces"]
        # named-action routing tables.  Entry keys must equal the
        # action-term names (field names of _IsaaclabActionsCfg); each term is
        # configured with preserve_order=True + explicit name lists, so term
        # input dim i == cfg.joint_names[i] and IsaaclabJointResolver maps
        # adapter entry columns onto the term's flat input slice.
        from unigym.runtime.actions import IsaaclabJointResolver
        self.action_entry_names = list(self.action_manager.active_terms)
        self._term_resolvers = {}
        for name in self.action_entry_names:
            term = self.action_manager.get_term(name)
            joint_names = getattr(getattr(term, "cfg", None), "joint_names",
                                  None)
            if not joint_names:
                raise ValueError(
                    f"action term {name!r} has no explicit joint_names; the "
                    f"NamedAction contract requires preserve_order=True "
                    f"terms with explicit name lists")
            self._term_resolvers[name] = IsaaclabJointResolver(joint_names)
        if self.camera_manager is not None:
            self.camera_manager.bind_env(self)
        if self.contact_manager is not None:
            self.contact_manager.bind_env(self)

        # ---- DR events: base class only auto-applies startup for the exact
        # ManagerBasedEnv class, so a subclass must apply it itself. One initial
        # pass of mode="reset" events makes per-episode DR active from step 0.
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        if "reset" in self.event_manager.available_modes:
            all_ids = torch.arange(num_envs, device=self.device)
            self.event_manager.apply(
                mode="reset", env_ids=all_ids,
                global_env_step_count=self._sim_step_counter // self.cfg.decimation)
            self.scene.write_data_to_sim()
            self.sim.forward()

        # ---- props (P15): bind asset handles + initial placement + close joints
        self.prop_manager = prop_manager
        if self.prop_manager is not None:
            self.prop_manager.bind_env(self)
            all_ids = torch.arange(num_envs, device=self.device)
            self.prop_manager.on_reset(all_ids, self)
            self.scene.write_data_to_sim()
            self.sim.forward()

        # ---- articulation/control runtimes (v104.2) ----
        from . import articulation_runtime as _articulation_runtime

        _articulation_runtime.bind_articulations(self, robot_profile)
        # single articulation keeps the historical flat compatibility attrs
        art = next(iter(self.articulations.values()))
        pol_names = list(art.addresses.get("policy_joints") or
                         (adapter.joint_order or robot_profile.joint_order))
        body_control = next(
            (c for c in self._control_order
             if c.articulation == art.name and c.action_face == art.name),
            self._control_order[0] if self._control_order else None)
        if body_control is not None:
            pol_names = list(body_control.joint_names)
        il_names = list(art.joint_names)
        missing = sorted(set(pol_names) - set(il_names))
        extra = sorted(set(il_names) - set(pol_names))
        if missing:
            raise ValueError(
                f"joint name mismatch between adapter/profile and asset:\n"
                f"  asset:   {il_names}\n  policy: {pol_names}\n"
                f"  missing in asset: {missing}")
        if extra:
            print(f"[isaaclab2.env] note: asset has {len(extra)} non-policy "
                  f"DOFs (actuated, excluded from obs/action): {extra}",
                  flush=True)
        self._perm_il2pol = torch.tensor(
            [il_names.index(n) for n in pol_names], device=self.device)
        self._perm_pol2il = torch.argsort(self._perm_il2pol)
        self._q0_art = art.q0_art
        self._joint_gathers: dict = {}

        foot_names = list(adapter.foot_order or robot_profile.foot_bodies)
        body_names = list(self.contact.body_names)
        if set(body_names) != set(foot_names):
            raise ValueError(
                f"foot name mismatch between adapter/profile and contact sensor:\n"
                f"  sensor:  {body_names}\n  policy: {foot_names}")
        self._foot_perm = torch.tensor(
            [body_names.index(f) for f in foot_names], device=self.device)
        self._foot_body_idx = torch.tensor(
            [self.robot.data.body_names.index(n) for n in foot_names],
            device=self.device, dtype=torch.long)

        # ---- obs cache: fill only what the adapter requested ----
        self._obs_requests = adapter.get_obs_requests()
        if self.contact_manager is not None:
            self.contact_manager.register_obs_requests(
                {r.key for r in self._obs_requests})
        self._obs_requests = canonicalize_obs_requests(self._obs_requests)
        # rebuild the catalogs with the wired managers (camera/contact/props)
        entries = {}
        requestable = set()
        self.obs_catalogs = {}
        for a in self.articulations.values():
            cat = build_obs_catalog(
                a.recipe,
                camera_manager=self.camera_manager,
                prop_manager=self.prop_manager,
                contact_manager=self.contact_manager,
                zero_fill_props=True,
                articulation_joint_names=list(a.joint_names),
                articulation=a.name)
            a.catalog = cat
            self.obs_catalogs[a.name] = cat
            entries.update(cat.entries)
            requestable |= cat.requestable
        from unigym.env.obs_catalog import ObsCatalog
        self.obs_catalog = ObsCatalog(entries, requestable)
        for req in self._obs_requests:
            self.obs_catalog.validate_request(req)
            if split_articulation_key(req.key)[1] in (
                    "joint_pos_rel", "joint_vel", "applied_torque"):
                self._joint_gather(req.joint_names, req)
        self.obs_cache: dict = {}

        # ---- episode bookkeeping (per-env) ----
        self._episode_id = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._time_in_episode = torch.zeros(num_envs, device=self.device)
        self._episode_dur = sample_durations(
            self._dur_lo, self._dur_hi, num_envs, self.device)

        adapter.bind_env(self)
        ids = torch.arange(num_envs, device=self.device)
        adapter.on_reset(ids)
        self.reset_all()
        self._rebuild_model_set()

    def _bind_partitioned(self, adapter, num_envs: int) -> None:
        """The partitioned-articulation tail (one env, external/manual driver)."""
        from . import articulation_runtime as _articulation_runtime

        self.prop_manager = None
        self._obs_requests = adapter.get_obs_requests()
        if self._obs_requests:  # pragma: no cover - rejected pre-super
            raise NotImplementedError(
                "isaaclab2 backend: adapter obs requests are not supported "
                "with multiple articulations")
        _articulation_runtime.bind_articulations(self)
        self.obs_cache: dict = {}

        self._episode_id = torch.zeros(num_envs, dtype=torch.int64,
                                       device=self.device)
        self._time_in_episode = torch.zeros(num_envs, device=self.device)
        self._episode_dur = sample_durations(
            self._dur_lo, self._dur_hi, num_envs, self.device)

        adapter.bind_env(self)
        ids = torch.arange(num_envs, device=self.device)
        adapter.on_reset(ids)
        self.reset_all()
        self._rebuild_model_set()

    def _rebuild_model_set(self, meta=None) -> None:
        """Rebuild ``model_irs`` / ``model_set`` from the live articulations.

        Runs once at construction and again after the run builder annotates
        the IRs with ``meta["articulations"]``.  ``merged`` is the single
        binding's IR; with several bindings it is ``None`` (no honest merge).
        """
        from dataclasses import replace as dataclass_replace

        from unigym.model import ModelSet

        from .model import IsaacLabModelProvider

        provider = IsaacLabModelProvider()
        irs = {}
        for name, asset in self.robots.items():
            ir = provider.describe(asset)
            if meta:
                ir = dataclass_replace(ir, meta={**ir.meta, **meta})
            irs[name] = ir
        merged = None
        if not self._partitioned and len(irs) == 1:
            merged = next(iter(irs.values()))
        self.model_irs = irs
        self.model_set = ModelSet(irs, merged=merged)

    # ------------------------------------------------------------------ #
    # lighting (Task 1: brighter video)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_lighting(scene, lighting: dict) -> None:
        """Configure scene lights from a yaml ``lighting`` section.

        Adds a DomeLight (uniform sky-fill, the main brightness booster) and
        optionally overrides the DistantLight intensity/direction. All keys
        are optional; an absent section leaves the default single distant
        light unchanged.

        Supported keys (under yaml ``env.lighting``):
            dome_intensity: float      # 0/absent = no dome light
            dome_color: [r,g,b]        # default white
            dome_texture: str|null     # optional HDR sky texture path
            distant_enabled: bool     # default true
            distant_intensity: float  # default 3000
            distant_color: [r,g,b]    # default (0.95,0.95,0.95)
            distant_angle: [az, el]   # degrees; azimuth/elevation of the sun
        """
        dome_intensity = float(lighting.get("dome_intensity", 0.0))
        if dome_intensity > 0.0:
            color = lighting.get("dome_color", (1.0, 1.0, 1.0))
            color = tuple(float(c) for c in color)
            kw = dict(intensity=dome_intensity, color=color)
            tex = lighting.get("dome_texture")
            if tex:
                kw["texture_file"] = tex
            scene.dome_light = AssetBaseCfg(
                prim_path="/World/dome_light",
                spawn=sim_utils.DomeLightCfg(**kw),
            )
        if lighting.get("distant_enabled", True):
            di_raw = lighting.get("distant_intensity")
            di = float(di_raw) if di_raw is not None else 3000.0
            dcolor = tuple(float(c) for c in lighting.get(
                "distant_color", (0.95, 0.95, 0.95)))
            spawn = sim_utils.DistantLightCfg(intensity=di, color=dcolor)
            init = AssetBaseCfg.InitialStateCfg()
            angle = lighting.get("distant_angle")
            if angle is not None:
                init.rot = _az_el_to_quat(float(angle[0]), float(angle[1]))
            scene.light = AssetBaseCfg(
                prim_path="/World/light", spawn=spawn, init_state=init)
        else:
            scene.light = None

    # ------------------------------------------------------------------ #
    # ordering helpers
    # ------------------------------------------------------------------ #
    @property
    def perm_il2pol(self) -> torch.Tensor:
        """Gather index: ``data_il[:, perm_il2pol]`` -> policy joint order."""
        return self._perm_il2pol

    @property
    def perm_pol2il(self) -> torch.Tensor:
        """Gather index: ``data_pol[:, perm_pol2il]`` -> Isaac Lab joint order."""
        return self._perm_pol2il

    def to_policy_order(self, x_il: torch.Tensor) -> torch.Tensor:
        """Gather ``(N, ...)`` IL-order joint tensors into policy order."""
        return x_il[:, self._perm_il2pol]

    def to_il_order(self, x_pol: torch.Tensor) -> torch.Tensor:
        """Gather ``(N, ...)`` policy-order joint tensors into IL order."""
        return x_pol[:, self._perm_pol2il]

    @property
    def policy_joint_pos_limits(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Soft joint position limits in policy order: ``(lower, upper)``.

        Shape ``(N, num_joints)`` each, gathered from the articulation's
        ``joint_pos_limits`` (IL order, (N, dof, 2)).  Returns ``None``
        when the underlying data is unavailable (older Isaac Lab or before
        sim init).  Feeds ``TerminationState.joint_pos_limits`` (P10).
        """
        lim = getattr(self.robot.data, "joint_pos_limits", None)
        if lim is None:
            return None
        lim = torch.as_tensor(lim)
        return (lim[:, self._perm_il2pol, 0], lim[:, self._perm_il2pol, 1])

    @property
    def joint_pos_limits_by_articulation(self):
        """``{articulation: (lower, upper)}`` for joint-limit terminations."""
        return getattr(self, "_joint_limits_by_articulation", None)

    # ------------------------------------------------------------------ #
    # obs cache
    # ------------------------------------------------------------------ #
    def _articulation_for_key(self, key):
        name, _ = split_articulation_key(key)
        if name is None:
            if len(self.articulations) != 1:
                raise ValueError(
                    f"isaaclab2 backend: obs key {key!r} does not name an "
                    f"articulation; this run has {sorted(self.articulations)}")
            return next(iter(self.articulations.values()))
        try:
            return self.articulations[name]
        except KeyError:
            raise ValueError(
                f"isaaclab2 backend: obs key {key!r} names unknown "
                f"articulation {name!r}; declared: "
                f"{sorted(self.articulations)}") from None

    def _joint_gather(self, names, req=None) -> torch.Tensor:
        art = (self._articulation_for_key(req.key) if req is not None
               else next(iter(self.articulations.values())))
        key = (art.name, tuple(names))
        g = self._joint_gathers.get(key)
        if g is None:
            index = {n: i for i, n in enumerate(art.joint_names)}
            missing = sorted(set(names) - set(index))
            if missing:
                raise ValueError(
                    f"isaaclab2 backend: obs joint names {missing} are not "
                    f"asset joints of articulation {art.name!r}")
            g = torch.tensor([index[n] for n in names], dtype=torch.int64,
                             device=self.device)
            self._joint_gathers[key] = g
        return g

    def _joint_obs(self, req) -> torch.Tensor:
        """Serve a joint-vector request by name gather over the articulation."""
        art = self._articulation_for_key(req.key)
        base = split_articulation_key(req.key)[1]
        g = self._joint_gather(req.joint_names, req)
        d = art.asset.data
        if base == "joint_pos_rel":
            return d.joint_pos[:, g] - art.q0_art[g]
        if base == "applied_torque":
            return d.applied_torque[:, g]
        return d.joint_vel[:, g]

    def _refresh_obs_cache(self, raw_action: torch.Tensor | None = None,
                           cmd: torch.Tensor | None = None) -> None:
        if self._partitioned:
            from . import articulation_runtime as _articulation_runtime

            _articulation_runtime.refresh_obs_cache(self, raw_action, cmd)
        # P15: prop state (when present) -- injected every step so adapters that
        # request prop keys (door/ball) read live values. No props -> the fetch
        # loop below zero-fills any prop key the adapter still requests.
        if self.prop_manager is not None:
            prop_state = self.prop_manager.get_state(self)
            for k, v in prop_state.items():
                self.obs_cache[k] = v
        d = self.robot.data
        # feat/camera-obs: image keys come from the CameraManager (interval
        # sampling + placeholder fallback live there); entries are already
        # shape/dtype-validated inside tick_and_read, and re-checked below
        # against each request's shape branch.
        if self.camera_manager is not None:
            for k, v in self.camera_manager.tick_and_read().items():
                self.obs_cache[k] = v
        # feat/contact-manager: adapter-requested contact/<name> keys are
        # injected every step (same serve pattern as prop keys above); NOT
        # requested -> never enters the obs cache, recording unaffected.
        if (self.contact_manager is not None
                and self.contact_manager.obs_keys()):
            for k, v in self.contact_manager.read().items():
                if k in self.contact_manager.obs_keys():
                    self.obs_cache[k] = v
        fetch = {
            "gravity_b": lambda: d.projected_gravity_b,
            # joint_pos_rel / joint_vel are served per-request by
            # _joint_obs (name gather over the full articulation)
            "command": lambda: cmd,
            "prev_action": lambda: raw_action,
            # additive obs sources (policy foot order)
            "base_ang_vel_b": lambda: d.root_ang_vel_b,
            "foot_contact_force": lambda: torch.norm(
                self.contact.data.net_forces_w[:, self._foot_perm], dim=-1),
            "base_yaw": lambda: math_utils.euler_xyz_from_quat(d.root_quat_w)[2]
                .unsqueeze(-1),
            # additive obs source (P11: recovery-switching termination state)
            "base_pos_w": lambda: d.root_pos_w,
            # additive obs source (P14: SymmLoco projected_forward_vec needs
            # the full base orientation; pure-raw quat in (w,x,y,z))
            "base_quat_w": lambda: d.root_quat_w,
        }
        for req in self._obs_requests:
            # camera-served keys (ImageObsRequest or depth alias): validate
            # by FULL SHAPE + dtype (the manager contract), not last-dim
            if (self.camera_manager is not None
                    and req.key in self.camera_manager.obs_keys()):
                val = self.obs_cache.get(req.key)
                if val is not None:
                    validate_image_frame(
                        self.camera_manager.frame_request(req.key), val,
                        self.num_envs)
                continue
            base = split_articulation_key(req.key)[1]
            if base in ("joint_pos_rel", "joint_vel",
                        "applied_torque"):
                val = self._joint_obs(req)
            elif base not in fetch:
                # prop-provided key (or zero-fill when no props configured):
                # validate shape if it was injected by the PropManager this step.
                v = self.obs_cache.get(req.key)
                if v is None:
                    v = torch.zeros(self.num_envs, req.dim, device=self.device)
                    self.obs_cache[req.key] = v
                elif v.shape[-1] != req.dim:
                    raise ValueError(
                        f"obs '{req.key}' dim {v.shape[-1]} != requested {req.dim}")
                continue
            else:
                val = fetch[base]()
            if val is None:
                continue  # passthrough value not provided this call; keep stale entry
            if val.shape[-1] != req.dim:
                raise ValueError(
                    f"obs '{req.key}' dim {val.shape[-1]} != requested {req.dim}")
            self.obs_cache[req.key] = val

    # ------------------------------------------------------------------ #
    # stepping
    # ------------------------------------------------------------------ #
    def step(self, actions: dict,
             raw_action: torch.Tensor | None = None,
             cmd: torch.Tensor | None = None) -> tuple[dict, dict]:
        """Advance one policy step.

        Args:
            actions: ``dict[str, NamedAction]`` — named action faces; each
                key must be an action-term name (``self.action_entry_names``)
                and every term must receive exactly one entry.  Entry joint
                names are resolved onto the term's flat input slice
                (preserve_order terms; resolver cached per name-tuple).
            raw_action: adapter raw action (policy order) -> obs_cache["prev_action"].
            cmd: current command tensor -> obs_cache["command"].

        Returns:
            (obs_cache copy, extra_info). ``extra["reset_env_ids"]`` lists the
            envs whose episode clock reached its duration (caller records this
            step first, then calls :meth:`reset_envs with the same ids).
            ``extra["should_reset"]`` (bool) part of the stable extra face.
        """
        # also applies mode="interval" DR events (pushes)
        super().step(self._route_named_actions(actions))
        self._time_in_episode += self.step_dt
        self._refresh_obs_cache(raw_action, cmd)
        extra = self._build_extra()
        due = self._time_in_episode >= self._episode_dur
        extra["reset_env_ids"] = torch.nonzero(due).squeeze(-1)
        extra["should_reset"] = bool(due.any())
        if self.camera_manager is not None:
            # step-tail posing: the render inside the NEXT step's sim.step
            # captures this pose (the classic re-pose semantics,
            # now rig-driven; see cameras/rigs.py)
            self.camera_manager.sync_pose()
        return dict(self.obs_cache), extra

    def _route_named_actions(self, actions: dict) -> torch.Tensor:
        """Resolve + gather the named action faces onto the flat term input.

        Builds the ``(N, total_action_dim)`` tensor ManagerBasedEnv.step
        expects, with each term's slice filled in the term's own input order
        (preserve_order terms: input dim i == cfg.joint_names[i]).  Entries
        that do not match any term, missing entries and partial term
        coverage all fail fast.
        """
        from . import articulation_runtime as _articulation_runtime

        return _articulation_runtime.route_actions(self, actions)
        if not isinstance(actions, dict):
            raise TypeError(
                f"IsaaclabEnv.step expects dict[str, NamedAction], got "
                f"{type(actions).__name__}")
        unknown = sorted(set(actions) - set(self.action_entry_names))
        if unknown:
            raise ValueError(
                f"action entries {unknown} match no action term; known "
                f"terms: {self.action_entry_names}")
        missing = [k for k in self.action_entry_names if k not in actions]
        if missing:
            raise ValueError(
                f"action entries missing for terms {missing}; every action "
                f"term must receive exactly one NamedAction entry "
                f"(terms: {self.action_entry_names})")
        n = self.num_envs
        chunks = []
        for key in self.action_entry_names:
            na = actions[key]
            term = self.action_manager.get_term(key)
            dim = int(term.action_dim)
            if na.num_envs != n:
                raise ValueError(
                    f"action entry {key!r} has {na.num_envs} env rows, "
                    f"expected {n}")
            if na.dim != dim:
                raise ValueError(
                    f"action entry {key!r} addresses {na.dim} joints but "
                    f"term {key!r} covers {dim}")
            gather = self._term_resolvers[key].gather(
                na.joint_names).to(self.device)
            buf = torch.zeros(n, dim, dtype=torch.float32, device=self.device)
            buf[:, gather] = na.values.to(device=self.device,
                                          dtype=torch.float32)
            chunks.append(buf)
        return torch.cat(chunks, dim=-1)

    def _build_extra(self) -> dict:
        """Per-step recorded quantities, read post-step like the verify script."""
        from . import articulation_runtime as _articulation_runtime

        extra = _articulation_runtime.build_extra(self)
        contact_values = self.contact_manager.read() \
            if self.contact_manager is not None else None
        if contact_values is not None:
            extra.update(contact_values)
        if self.prop_manager is not None:
            from unigym.props.object_state import build_object_extra
            self.prop_manager.on_step(self)
            extra.update(build_object_extra(
                self.prop_manager, self.prop_manager.get_world_poses(self),
                joint_states=self.prop_manager.get_articulation_joint_states(
                    self)))
        return extra

    # ------------------------------------------------------------------ #
    # episode reset (per-env subset; also covers the batch case)
    # ------------------------------------------------------------------ #
    def reset_envs(self, env_ids) -> torch.Tensor:
        """Reset the given env subset through the full parent-class reset chain.

        Order mirrors ``ManagerBasedEnv.reset``: ``_reset_idx`` (scene teleport
        to default state + mode="reset" DR events for these envs + manager
        resets incl. per-env interval-event timers) -> write_data_to_sim ->
        sim.forward; then episode bookkeeping and obs-cache refresh.

        .. note::
            When mode="reset" DR events are active, ``_reset_idx`` applies
            ``scene.reset`` (teleport) *before* ``event_manager.apply`` (DR).
            PhysX view writes in DR (``set_coms``, ``set_masses``, etc.)
            corrupt the root state, so the teleport is silently lost — the
            robot stays at its end-of-episode position instead of spawn.
            We re-issue the teleport *after* DR via low-level Articulation
            write methods, which bypass the corrupted ``root_state_w``
            buffer and write directly to PhysX.  See IsaacLab issue #7311.
        """
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device)
        if env_ids.numel() == 0:
            return env_ids
        self._reset_idx(env_ids)
        if self.prop_manager is not None:  # P15: re-place props + close joints
            self.prop_manager.on_reset(env_ids, self)
        if self._partitioned:
            from . import articulation_runtime as _articulation_runtime

            _articulation_runtime.reset_articulations(self, env_ids)
        # Re-teleport after DR: set_coms/set_masses in mode="reset" events
        # corrupt the root pose written by scene.reset in _reset_idx.
        # Writing via write_root_pose_to_sim (explicit args, bypasses the
        # root_state_w buffer) restores the teleport; COM/mass offsets
        # survive because they are separate PhysX properties.
        # When ``init_pose`` is configured we additionally override the root
        # pose of a (prob) fraction of envs with a randomized orientation so
        # the episode starts lying/fallen (recovery data collection).
        if (not self._partitioned and (self._init_pose is not None
                or "reset" in self.event_manager.available_modes)):
            robot = self.scene["robot"]
            dr = robot.data.default_root_state[env_ids].clone()
            dj = robot.data.default_joint_pos[env_ids]
            if self._init_pose is not None:
                n = env_ids.numel()
                ip = self._init_pose
                pick = torch.rand(n, device=self.device) < ip["prob"]
                if pick.any():
                    n_pick = int(pick.sum().item())
                    pos, quat = sample_init_pose(
                        n_pick, ip["roll_range"], ip["pitch_range"],
                        ip["yaw_range"], ip["height"], self.device)
                    drm = dr[pick].clone()
                    drm[:, :3] = pos
                    drm[:, 3:7] = quat
                    drm[:, 7:13] = 0.0  # at rest before first contact
                    dr[pick] = drm
            robot.write_root_pose_to_sim(
                root_pose=dr[:, :7], env_ids=env_ids)
            robot.write_root_velocity_to_sim(
                root_velocity=dr[:, 7:13], env_ids=env_ids)
            robot.write_joint_state_to_sim(
                position=dj, velocity=torch.zeros_like(dj), env_ids=env_ids)
            robot.data.root_state_w[env_ids] = dr
        self.scene.write_data_to_sim()
        self.sim.forward()
        self._episode_id[env_ids] += 1
        self._time_in_episode[env_ids] = 0.0
        self._episode_dur[env_ids] = sample_durations(
            self._dur_lo, self._dur_hi, env_ids.numel(), self.device)
        if self.camera_manager is not None:
            # re-pose body/chase rigs to the spawn state (the reference chain
            # called _sync_depth_camera_pose at this exact point)
            self.camera_manager.on_reset(env_ids)
            if self._render_on_reset:
                # r4 first-fetch phase fix: render once at reset so the obs
                # cache carries a REAL spawn-state frame into act(t0).  The
                # upstream contract has the t0 frame ready before the first
                # control tick (official tick0 consumes a live frame -- its
                # pos_hold == chunk0[9:38] bit-exact, round-3 dump) while our
                # annotator holds no frame until the first in-step render,
                # which pushed the first chunk fetch to t1 and made tick0
                # hold the spawn reference (round-3 residual item 2).
                # sim.render() alone does not refresh annotators -- the
                # scene.update(0) pairing is the verified combo.
                self.sim.render()
                self.scene.update(dt=0)
        self._refresh_obs_cache()  # cache now reflects spawn state
        return env_ids

    def reset_all(self) -> torch.Tensor:
        """Batch-synchronous episode reset (the P1 shorthand)."""
        return self.reset_envs(torch.arange(self.num_envs, device=self.device))

    # ------------------------------------------------------------------ #
    # snapshot / restore (Round B B1 rewind): DECLARED ABSENT
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict:
        """Not implemented on isaaclab v1 (rewind is mujoco-only, Round B)."""
        raise NotImplementedError(
            "isaaclab2 backend does not implement snapshot()/restore() "
            "(rewind superpower, design v052-round-b §1.1); run the "
            "rewind face against env.backend: mujoco")

    def restore(self, state) -> int:
        """Not implemented on isaaclab v1 (rewind is mujoco-only, Round B)."""
        raise NotImplementedError(
            "isaaclab2 backend does not implement snapshot()/restore() "
            "(rewind superpower, design v052-round-b §1.1); run the "
            "rewind face against env.backend: mujoco")
