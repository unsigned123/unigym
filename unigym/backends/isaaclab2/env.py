"""IsaaclabEnv: ManagerBasedEnv wrapper for policy rollout data collection.

No reward/termination (pure kinematic/dynamic logging). Episode handling is
**per-env**: core owns no clock, so ``step()`` reports no due row; an episode
ends only through an explicit runner terminate or a sink terminal. The caller
records the final step first (data invariant: record-before-reset) and then
calls :meth:`reset_envs with the ended ids, which replays the parent-class
reset chain for the subset:
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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from unigym._typing import JsonValue, is_json_object
from unigym.backends.isaaclab2.robots import articulation_cfg
from unigym.backends.shared.isaaclab_bodies import resolve_site_body_index
from unigym.core import quat

from unigym.env.native import NativeHandle
from .init_pose import build_init_pose_cfg, sample_init_pose
from .terrains.native import NativeTerrainPreset
from unigym.env.obs import ObservationConfig, ObservationRegistry
from unigym.backends.obs_schema import build_schema

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.assets import Articulation

    from unigym.backends.isaaclab2.asset_scene.engine import (
        AssetSceneEngine,
        ComposedAssetScene,
    )
    from unigym.cameras import CameraManager
    from unigym.contacts import ContactManager
    from unigym.query.facade import Query
    from unigym.runtime.actions import NamedAction


from unigym.env.backends.base import (
    BackendCapabilities,
    EnvStep,
    normalize_state_env_ids,
    normalize_state_quat,
    validate_state_tensor,
)
from unigym.query.errors import QueryStaleError

BACKEND_CAPABILITIES = BackendCapabilities.full()

__all__ = ["IsaaclabEnv", "BACKEND_CAPABILITIES"]


@dataclass(frozen=True)
class _IsaacObsSite:
    """One IsaacLab site handle: robot asset, body index and local pose."""

    name: str
    asset: Articulation
    body_index: int
    local_pos: tuple[float, float, float]
    local_quat: tuple[float, float, float, float]


def _base_z_obs(env):
    return env.scene["robot"].data.root_pos_w[:, 2:3]


def _resolve_terrain_cfg(terrain_cfg, terrain, num_envs, env_spacing):
    """Return the scene terrain fields: native IR wins over the yaml preset."""
    if terrain is not None:
        return NativeTerrainPreset(terrain).build(num_envs, env_spacing)
    if terrain_cfg is None:
        raise ValueError(
            "IsaaclabEnv requires a terrain preset: pass terrain_cfg "
            "(yaml terrain) or terrain=<Terrain>"
        )
    return terrain_cfg


def _scene_provider_values(params: Mapping[str, object] | None) -> dict[str, JsonValue]:
    """Narrow the routed scene-provider params to the JSON tree providers take.

    The run layer routes ``env.params.scene_provider_params`` straight from
    the config, so every value is a JSON tree by construction; the copy keeps
    the provider dispatch from observing the caller's mapping.

    Args:
        params: The routed provider parameter mapping, or ``None``.

    Returns:
        A plain JSON mapping (empty when *params* is ``None``).

    Raises:
        ValueError: *params* is not a string-keyed mapping.
    """
    value: object = {} if params is None else dict(params)
    if not is_json_object(value):
        raise ValueError(
            f"env.params.scene_provider_params must be a JSON mapping, "
            f"got {type(value).__name__}"
        )
    return value


def _resolve_asset_scene_engine() -> AssetSceneEngine:
    """Resolve ``isaaclab2`` through the neutral engine registry.

    The registry returns the protocol view; this env can only attach the
    backend's own engine, so the concrete check runs here and the caller
    receives the typed product (no cast).

    Raises:
        ValueError: The neutral registry does not return an
            :class:`AssetSceneEngine` for ``isaaclab2``.
    """
    from unigym.scenes.engine import scene_engine

    from .asset_scene.engine import AssetSceneEngine

    engine = scene_engine("isaaclab2")
    if not isinstance(engine, AssetSceneEngine):
        raise ValueError(
            f"the neutral scene-engine registry returned {type(engine).__name__} "
            f"for 'isaaclab2'; this env can only attach the backend's own "
            f"AssetSceneEngine"
        )
    return engine


def _compose_asset_scene(
    cfg,
    scene_provider: str | None,
    scene_provider_params: Mapping[str, object] | None,
    scene_provider_preset: str | None,
    asset_plan,
    static_shapes,
    artifact_dir: str | Path | None,
    terrain=None,
) -> tuple[ComposedAssetScene | None, dict[str, str]]:
    """Compose + attach the routed scene provider before the scene is cloned.

    Returns the composed product and the written artifact paths, or
    ``(None, {})`` when the run declares no scene provider (the historical
    build path stays untouched).

    The neutral scene is built first.  An ``asset_plan`` is required only when
    that scene declares file-backed asset instances; a primitive-only scene
    composes with ``None``.  The engine owns that guard.  When the run also
    carries native ``terrain`` IR, that run-level terrain is the single ground
    owner: the inline/provider scene's duplicate terrain declaration is dropped
    before composition (the asset-scene engine still refuses a scene terrain
    when the run declares none).

    Raises:
        ValueError: *static_shapes* and a scene provider are both declared,
            the provider/scene validation fails, the composed scene has asset
            instances but no asset plan, or the neutral registry does not
            return this backend's engine.
    """
    if scene_provider is None:
        return None, {}
    if static_shapes:
        raise ValueError(
            "isaaclab backend: env.params.scene_provider and static_shapes "
            f"cannot both be declared ({len(static_shapes)} static "
            "declaration(s) plus a composed scene); declare one or the other"
        )
    from unigym.scenes.pipeline import validate_asset_scene
    from unigym.scenes.providers import build_scene_from_provider

    scene = build_scene_from_provider(
        str(scene_provider),
        _scene_provider_values(scene_provider_params),
        preset=(str(scene_provider_preset) if scene_provider_preset is not None else None),
    )
    if terrain is not None and scene.terrain is not None:
        scene = replace(scene, terrain=None)
    validate_asset_scene(scene)
    engine = _resolve_asset_scene_engine()
    composed = engine.compose(scene, asset_plan)
    engine.attach(cfg.scene, composed, frame=f"Scene_{scene.name}_attach")
    artifact_paths: dict[str, str] = {}
    if artifact_dir is not None:
        from .asset_scene.artifacts import write_asset_scene_artifacts

        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        artifact_paths = write_asset_scene_artifacts(artifact_dir, scene, composed, engine)
    return composed, artifact_paths


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
                f"static primitive scene field {key!r} collides with an " f"existing scene entry"
            )
        setattr(scene, key, value)


def _az_el_to_quat(az_deg: float, el_deg: float) -> tuple:
    """Azimuth/elevation (degrees) -> quaternion (w, x, y, z) for a DistantLight.

    A USDLux DistantLight emits along its local -Z, i.e. light travels from
    +Z toward -Z. We rotate the prim so local +Z points toward the sun
    position defined by azimuth (CCW from +X about world Z) and elevation
    (above the XY plane). The result is (w, x, y, z) as expected by
    ``AssetBaseCfg.InitialStateCfg.rot``.

    (DELEGATED to unigym.core.quat.az_el)
    """
    return quat.az_el(az_deg, el_deg)


def _configure_robot_scene(
    cfg,
    robot_profile,
    adapter,
    controls,
    articulations,
    profiles,
    robot_realizations,
    prop_manager,
    *,
    events_cfg,
    recorder,
    camera_manager,
    contact_manager,
    director_cfg,
    init_pose_cfg,
):
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
            adapter,
            events_cfg=events_cfg,
            recorder=recorder,
            prop_manager=prop_manager,
            camera_manager=camera_manager,
            contact_manager=contact_manager,
            director_cfg=director_cfg,
            init_pose_cfg=init_pose_cfg,
        )
        fields = _articulation_runtime.configure_scene(
            cfg,
            controls,
            ars,
            dict(profiles or {}),
            dict(robot_realizations or {}),
        )
        return partitioned, fields
    if ars:
        cfg.scene.robot = articulation_cfg(robot_realizations, ars[0].name)
    else:
        raise ValueError(
            "isaaclab2 backend: a run must declare at least one articulation "
            "with an ArticulationCfg realization"
        )
    # The body action term name is the body control's action-group name (kept
    # as a legacy backend term identity).  Resolve the body group from its
    # explicit role instead of inferring it from the articulation name.
    single_name = ars[0].name if ars else None
    body_controls = [
        control
        for control in tuple(controls or ())
        if not control.prop
        and control.role == "body"
        and (single_name is None or control.articulation == single_name)
    ]
    if len(body_controls) != 1:
        accepted = [
            (control.name, control.articulation, control.role) for control in tuple(controls or ())
        ]
        raise ValueError(
            "isaaclab2 backend: the single-articulation path needs exactly "
            "one non-prop body control (role == 'body'), got "
            f"{len(body_controls)}; accepted controls (name, articulation, "
            f"role): {accepted}"
        )
    body_control = body_controls[0]
    term_name = str(body_control.name)
    if not term_name.isidentifier():
        raise ValueError(
            f"isaaclab2 backend: control {term_name!r} of articulation "
            f"{body_control.articulation!r} is not a valid action-term name"
        )
    # Keep the joint list source unchanged: the profile's declared body order
    # (falling back to the asset default when the profile declares none).  The
    # G1 Nucleus asset carries 14 finger DOFs beyond the 29 policy joints, so
    # the term targets exactly the profile's declared joints instead of all
    # asset DOFs.
    joint_names = list(robot_profile.joint_order) if robot_profile.joint_order else [".*"]
    setattr(
        cfg.actions,
        term_name,
        JointPositionActionCfg(
            asset_name="robot",
            joint_names=joint_names,
            scale=1.0,
            use_default_offset=False,
            preserve_order=True,
        ),
    )
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
                    f"an existing action term name"
                )
            setattr(
                cfg.actions,
                h.spec.name,
                JointPositionActionCfg(
                    asset_name=f"prop_{h.spec.name}",
                    joint_names=actuated,
                    scale=1.0,
                    use_default_offset=False,
                    preserve_order=True,
                ),
            )
    # profile-declared extra action sets (the G1 "hand"
    # dex3 finger set on with-hand assets) -- one preserve_order term
    # per declared entry, field name == the NamedAction entry key
    # (declaration-becomes-capability, same rule as the prop terms
    # above).  ``None`` (a profile without extra action sets)
    # leaves the action surface untouched.
    for term_name, term_joints in (robot_profile.extra_action_sets or {}).items():
        if hasattr(cfg.actions, term_name):
            raise ValueError(
                f"extra action entry {term_name!r} collides with an " f"existing action term name"
            )
        setattr(
            cfg.actions,
            term_name,
            JointPositionActionCfg(
                asset_name="robot",
                joint_names=list(term_joints),
                scale=1.0,
                use_default_offset=False,
                preserve_order=True,
            ),
        )
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
    robot: ArticulationCfg | None = None  # filled from the backend realizer
    # camera sensors (obs + recorder camera_ref) join the scene ONLY via
    # cameras/ CameraManager.setup_scene under cam_<name> fields (the
    # recorder's self-built tiled_camera path was removed).


@configclass
class _IsaaclabActionsCfg:
    """Action groups created dynamically from the run's controls.

    The single-articulation path creates the body term under the body
    control's action-group name; the partitioned path creates one term per
    control group in :mod:`articulation_runtime`.  Extra action sets (hands,
    articulated props) are added the same way.  ``preserve_order=True`` keeps
    the term input dim ``i == joint_names[i]``, which is what makes the
    NamedAction name addressing trivially correct (IsaaclabJointResolver
    contract).
    """


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

    def __init__(
        self,
        robot_profile,
        adapter,
        terrain_cfg,
        num_envs: int,
        env_spacing: float = 4.0,
        decimation: int = 4,
        sim_dt: float = 0.005,
        events_cfg=None,
        recorder=None,
        prop_manager=None,
        lighting: dict | None = None,
        init_pose: dict | None = None,
        camera_manager: CameraManager | None = None,
        contact_manager: ContactManager | None = None,
        director_cfg: dict | None = None,
        device: str = "cuda:0",
        terrain=None,
        static_shapes=None,
        articulations=None,
        controls=None,
        profiles=None,
        robot_realizations=None,
        backend_options=None,
        asset_plan=None,
        scene_provider: str | None = None,
        scene_provider_params: Mapping[str, object] | None = None,
        scene_provider_preset: str | None = None,
        artifact_dir: str | Path | None = None,
        observations: ObservationConfig | None = None,
    ):
        """Build the collect env: cfg stack, scene fields and manager seams.

        Args:
            robot_profile: the neutral RobotProfile (joint order + body
                facts); source of the action term's joint list.
            adapter: Adapter consuming this env (read for its
                ``wants_reset_render`` opt-in only).
            num_envs: parallel env count (>1 requires the multi_env
                capability; gated in build_env_backend).
            env_spacing: scene grid spacing between cloned envs (m).
            decimation: sim steps per policy step (policy dt = decimation *
                ``sim_dt``).
            sim_dt: physics step size (s; written to ``cfg.sim.dt``).
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
                and ``get_world_poses`` feeds the typed object observation views
                for adapters that request it.
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
                camera frames).
            contact_manager: optional
                :class:`unigym.contacts.ContactManager` (built by collect.py
                from yaml ``contacts``, docs/contracts/contacts_contract.md).  Its
                ContactSensorCfgs join the scene before construction
                (``setup_scene``); ``bind_env`` resolves bodies + grabs the
                sensor handles; ``read()`` is merged into ``extra`` as
                ``contact/<name>`` keys every step (record channel L2) and
                into the typed observation registry for selected groups.
            director_cfg: optional yaml ``director`` dict; mounts the virtual
                gripper scene fields (director_contract §4.3).
            device: torch device string for the sim (default ``cuda:0``).
            articulations: the run's ordered neutral articulation specs.
            controls: neutral ``ControlRuntime`` entries derived from
                ``RunSpec.action_spec`` + ``RunSpec.policies`` + the robot
                profiles.
            profiles: ``{articulation name: RobotProfile}`` neutral robot
                data for every articulation.
            robot_realizations: ``{articulation name: ArticulationCfg}``
                built by this backend's robot realizer.
            backend_options: backend-private options; isaaclab declares none
                (``env.backend_options.isaaclab`` fails fast before build).
            asset_plan: the composition boundary that resolves every
                file-backed asset reference of the routed scene; required
                only when that scene declares asset instances (a
                primitive-only scene composes without one).
            scene_provider: the registered scene provider whose neutral
                :class:`~unigym.scenes.scene_ir.Scene` is composed into
                IsaacLab asset fields (``env.params.scene_provider``).
                ``None`` keeps the historical build path byte-identical.
            scene_provider_params: the provider parameter mapping routed by
                ``apply_scene_preset``; ``None`` means no params.
            scene_provider_preset: the preset catalog id when the dispatch
                came from a preset, else ``None``.
            artifact_dir: optional run artifact directory; with a composed
                scene the env writes ``scene.json`` / ``resolution.json`` /
                ``payload_index.json`` / ``compose.txt`` / ``describe.json``
                there.  ``None`` disables the write.
            observations: the run's ``ObservationConfig`` (config-declared
                observation sources).
        """
        # teardown state is initialised before the parent constructor so a
        # failure there still leaves close()/the guards well-defined
        self._closed = False
        self._closing = False
        self._stepping = False
        self._model_revision = 0
        self._query_handle: Query | None = None
        self._asset_scene: ComposedAssetScene | None = None
        self.artifact_paths: dict[str, str] = {}
        self._init_pose = build_init_pose_cfg(init_pose)
        self.observations = observations
        self._articulation_specs = tuple(articulations or ())
        self._profiles = dict(profiles or {})
        if backend_options:
            raise ValueError(
                "isaaclab backend declares no backend_options (unknown keys "
                f"{sorted(backend_options)}); supported options: []"
            )
        self._planned_controls = tuple(controls or ())

        cfg = _IsaaclabEnvCfg()
        cfg.decimation = decimation
        # ManagerBasedEnv (non-RL) has no episode-horizon schedule; this
        # wrapper never derives a due mask from the engine.
        cfg.sim.device = device
        cfg.sim.dt = sim_dt
        cfg.sim.render_interval = decimation
        cfg.scene.num_envs = num_envs
        cfg.scene.env_spacing = env_spacing
        terrain_cfg = _resolve_terrain_cfg(terrain_cfg, terrain, num_envs, env_spacing)
        for key, value in terrain_cfg.items():  # merge terrain preset fields
            setattr(cfg.scene, key, value)
        # scene-provider composition: one neutral scene -> IsaacLab asset
        # fields, attached to the real InteractiveSceneCfg BEFORE the robot/
        # prop/camera/contact fields are merged, so the engine's collision
        # check covers the base cfg fields and the reserved names only --
        # not the robot/prop/camera/contact fields added after this point.
        # The fields clone natively.  Composing and attaching before
        # ``super().__init__`` is the whole contract: the engine never
        # touches sim state.  No provider -> the historical path stays
        # untouched.
        self._asset_scene, self.artifact_paths = _compose_asset_scene(
            cfg,
            scene_provider,
            scene_provider_params,
            scene_provider_preset,
            asset_plan,
            static_shapes,
            artifact_dir,
            terrain=terrain,
        )
        # v102b: more than one policy binding switches the scene/action cfg
        # to the multi-articulation layout (independent articulation clones +
        # one action term per binding).  One binding (or none) keeps the
        # historical single-robot configuration byte-identically.
        (self._partitioned, self._articulation_fields) = _configure_robot_scene(
            cfg,
            robot_profile,
            adapter,
            controls,
            articulations,
            profiles,
            robot_realizations,
            prop_manager,
            events_cfg=events_cfg,
            recorder=recorder,
            camera_manager=camera_manager,
            contact_manager=contact_manager,
            director_cfg=director_cfg,
            init_pose_cfg=self._init_pose,
        )
        # note: `recorder` intentionally unused here (no scene contribution
        # kept in the signature for capability gating kwargs).
        if prop_manager is not None:  # merge interactive prop fields
            for key, value in prop_manager.setup_scene().items():
                setattr(cfg.scene, key, value)
        # static World primitives (S2b): the same declarations the mujoco
        # inline scene renders become native static colliders here.  A
        # composed asset scene already attached its primitives through the
        # engine, so the two mappings never run against the same scene.
        if self._asset_scene is None:
            _add_static_shape_fields(cfg.scene, static_shapes)
        # director: the virtual gripper joins the scene as a fixed-
        # base KINEMATIC rigid body (--direct only; director_contract §4.3)
        self.director_cfg = director_cfg
        if director_cfg is not None:
            from unigym.backends.isaaclab2.director.gripper import isaaclab_scene_fields
            from unigym.director.gripper import normalize_gripper_cfg

            gcfg = normalize_gripper_cfg(director_cfg.get("gripper"))
            if gcfg is not None:
                for key, value in isaaclab_scene_fields(gcfg).items():
                    setattr(cfg.scene, key, value)
        if lighting is not None:  # Task 1: brighter scene for video capture
            self._apply_lighting(cfg.scene, lighting)
        # feat/camera-obs: camera-manager sensors join the scene before
        # cloning (same pattern as props above).  Depth is served only by
        # an explicitly declared ``depth`` camera; there is no implicit
        # shim camera.
        self.camera_manager = camera_manager
        # r4 ha_open_door: opt-in reset-time render (adapter-declared t0
        # frame contract -- upstream provider fills the frame BEFORE the
        # first control tick; applied in reset_envs)
        self._render_on_reset = bool(getattr(adapter, "wants_reset_render", False))
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
            joint_names = getattr(getattr(term, "cfg", None), "joint_names", None)
            if not joint_names:
                raise ValueError(
                    f"action term {name!r} has no explicit joint_names; the "
                    f"NamedAction contract requires preserve_order=True "
                    f"terms with explicit name lists"
                )
            self._term_resolvers[name] = IsaaclabJointResolver(joint_names)
        if self.camera_manager is not None:
            self.camera_manager.bind_env(self)
        if self.contact_manager is not None:
            self.contact_manager.bind_env(self)
        self.observations = (
            observations if observations is not None else ObservationConfig.from_mapping({})
        )
        self._obs_sites = self._resolve_obs_sites(self.observations.site)

        # ---- DR events: base class only auto-applies startup for the exact
        # ManagerBasedEnv class, so a subclass must apply it itself. One initial
        # pass of mode="reset" events makes per-episode DR active from step 0.
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        if "reset" in self.event_manager.available_modes:
            all_ids = torch.arange(num_envs, device=self.device)
            self.event_manager.apply(
                mode="reset",
                env_ids=all_ids,
                global_env_step_count=self._sim_step_counter // self.cfg.decimation,
            )
            self.scene.write_data_to_sim()
            self.sim.forward()

        # ---- props: bind asset handles + initial placement + close joints
        self.prop_manager = prop_manager
        if self.prop_manager is not None:
            self.prop_manager.bind_env(self)
            all_ids = torch.arange(num_envs, device=self.device)
            self.prop_manager.on_reset(all_ids, self)
            self.scene.write_data_to_sim()
            self.sim.forward()

        # ---- articulation/control runtimes ----
        from . import articulation_runtime as _articulation_runtime

        _articulation_runtime.bind_articulations(self)
        # single articulation keeps the historical flat compatibility attrs
        art = next(iter(self.articulations.values()))
        pol_names = list(
            art.addresses.get("policy_joints") or (adapter.joint_order or robot_profile.joint_order)
        )
        body_control = next(
            (c for c in self._control_order if c.articulation == art.name and c.role == "body"),
            self._control_order[0] if self._control_order else None,
        )
        if body_control is not None:
            pol_names = list(body_control.joint_names)
        il_names = list(art.joint_names)
        missing = sorted(set(pol_names) - set(il_names))
        extra = sorted(set(il_names) - set(pol_names))
        if missing:
            raise ValueError(
                f"joint name mismatch between adapter/profile and asset:\n"
                f"  asset:   {il_names}\n  policy: {pol_names}\n"
                f"  missing in asset: {missing}"
            )
        if extra:
            print(
                f"[isaaclab2.env] note: asset has {len(extra)} non-policy "
                f"DOFs (actuated, excluded from obs/action): {extra}",
                flush=True,
            )
        self._perm_il2pol = torch.tensor([il_names.index(n) for n in pol_names], device=self.device)
        self._perm_pol2il = torch.argsort(self._perm_il2pol)
        self._q0_art = art.q0_art

        # ---- typed observation registry (config-declared sources) ----
        self.obs = ObservationRegistry(
            self.observations,
            build_schema(
                self.observations,
                env=self,
                camera_manager=self.camera_manager,
                contact_manager=self.contact_manager,
                prop_manager=self.prop_manager,
                sites=tuple(handle.name for handle in self._obs_sites),
            ),
            num_envs=self.num_envs,
            device=self.device,
        )
        self._t = 0

        # ---- episode bookkeeping (per-env) ----
        # Core owns no clock: these are honest per-env counters, not a due
        # source.  ``step()`` reports no backend-owned due.
        self._episode_id = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._time_in_episode = torch.zeros(num_envs, device=self.device)

        adapter.bind_env(self)
        ids = torch.arange(num_envs, device=self.device)
        adapter.on_reset(ids)
        self.reset_all()
        self._rebuild_model_set()

    def _bind_partitioned(self, adapter, num_envs: int) -> None:
        """The partitioned-articulation tail (one env, external/manual driver)."""
        from . import articulation_runtime as _articulation_runtime

        self.prop_manager = None
        _articulation_runtime.bind_articulations(self)
        if self.contact_manager is not None:
            self.contact_manager.bind_env(self)
        self.observations = getattr(self, "observations", None) or ObservationConfig.from_mapping(
            {}
        )
        self._obs_sites = self._resolve_obs_sites(self.observations.site)
        self.obs = ObservationRegistry(
            self.observations,
            build_schema(
                self.observations,
                env=self,
                camera_manager=self.camera_manager,
                contact_manager=self.contact_manager,
                prop_manager=None,
                sites=tuple(handle.name for handle in self._obs_sites),
            ),
            num_envs=self.num_envs,
            device=self.device,
        )
        self._t = 0

        self._episode_id = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._time_in_episode = torch.zeros(num_envs, device=self.device)

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

    def model_edit(self):
        """Open one global body model-edit transaction between steps.

        Geom fields are declared absent (no published geom IR); the provider
        raises a capability error for them.

        Raises:
            RuntimeError: The environment is closed or currently stepping.
        """
        if self._closed or self._closing:
            raise RuntimeError("isaaclab2 backend: environment is closed")
        if self._stepping:
            raise RuntimeError("isaaclab2 backend: model edits are only allowed between steps")
        from unigym.backends.isaaclab2.model_edit import IsaacLab2ModelEditProvider
        from unigym.model.edit import ModelEditor

        provider = IsaacLab2ModelEditProvider()

        def commit(bodies, geoms):
            return provider.apply(self, bodies, geoms)

        return ModelEditor(
            backend="isaaclab2",
            fields=provider.fields,
            model_set=self.model_set,
            revision=self._model_revision,
            commit=commit,
            is_open=lambda: not (self._closed or self._closing or self._stepping),
            current_revision=lambda: self._model_revision,
        )

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
            dcolor = tuple(float(c) for c in lighting.get("distant_color", (0.95, 0.95, 0.95)))
            spawn = sim_utils.DistantLightCfg(intensity=di, color=dcolor)
            init = AssetBaseCfg.InitialStateCfg()
            angle = lighting.get("distant_angle")
            if angle is not None:
                init.rot = _az_el_to_quat(float(angle[0]), float(angle[1]))
            scene.light = AssetBaseCfg(prim_path="/World/light", spawn=spawn, init_state=init)
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
        """First articulation's soft limits in policy order: ``(lower, upper)``.

        Shape ``(N, num_joints)`` each.  This is a compatibility view over the
        first declared ``ArticulationRuntime.joint_pos_limits`` (record/body-
        control policy order); ``None`` when that articulation declares no
        limits.
        """
        first = next(iter(self.articulations.values()), None)
        return None if first is None else first.joint_pos_limits

    @property
    def joint_pos_limits_by_articulation(
        self,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor] | None]:
        """Compatibility view over every ``ArticulationRuntime.joint_pos_limits``."""
        return {name: art.joint_pos_limits for name, art in self.articulations.items()}

    # ------------------------------------------------------------------ #
    # obs cache
    # ------------------------------------------------------------------ #
    def _resolve_obs_sites(self, declared) -> list:
        """Resolve robot-profile sites to ``body + local pose`` handles.

        IsaacLab has no native site readback face and no generated prop-site
        payload; only asset-level robot-profile declarations are supported.
        There is no USD prim/FrameTransformer fallback.
        """
        if not declared:
            return []
        from unigym.backends.obs_schema import build_site_inventory
        from unigym.robots.base import normalize_canonical_body_id

        inventory = build_site_inventory(
            getattr(self, "_profiles", None) or {},
            getattr(getattr(self, "prop_manager", None), "asset_definitions", None) or {},
        )
        handles: list[_IsaacObsSite] = []
        for name in tuple(declared or ()):
            entry = inventory.get(name)
            if entry is None:
                raise ValueError(
                    f"isaaclab2 backend: observations.site {name!r} is not an "
                    f"asset-level site; IsaacLab sites require a robot-profile "
                    f"declaration (no USD prim fallback); declared asset sites: "
                    f"{sorted(inventory)}"
                )
            if entry.kind == "prop":
                raise ValueError(
                    f"isaaclab2 backend: observations.site {name!r} is a generated "
                    f"prop site; IsaacLab sites currently require a robot profile "
                    f"(no USD prim fallback)"
                )
            profile = self._profiles.get(entry.owner)
            if profile is None:
                raise ValueError(
                    f"isaaclab2 backend: observations.site {name!r} is owned by "
                    f"articulation {entry.owner!r} but no profile was supplied"
                )
            canonical_body = normalize_canonical_body_id(
                entry.spec.body, f"observations.site {name!r} body"
            )
            binding = profile.body_bindings.get(canonical_body)
            if binding is None or "usd" not in binding.names:
                raise ValueError(
                    f"isaaclab2 backend: observations.site {name!r} body "
                    f"{entry.spec.body!r} has no usd binding in profile {entry.owner!r}"
                )
            assets = getattr(self, "robots", None) or {}
            asset = assets.get(entry.owner)
            if asset is None:
                if len(self._profiles) != 1:
                    raise ValueError(
                        f"isaaclab2 backend: observations.site {name!r} cannot "
                        f"resolve articulation {entry.owner!r} to a live scene asset"
                    )
                asset = self.robot
            data = getattr(asset, "data", None)
            body_index = resolve_site_body_index(
                data,
                usd_name=binding.names["usd"],
                site_name=name,
                owner=entry.owner,
                backend="isaaclab2 backend",
            )
            handles.append(
                _IsaacObsSite(
                    name=name,
                    asset=asset,
                    body_index=body_index,
                    local_pos=entry.spec.pos,
                    local_quat=entry.spec.quat_wxyz,
                )
            )
        return handles

    def _refresh_obs(self, raw_action=None, cmd=None) -> None:
        """Refresh the typed observation registry exactly once per step."""
        self.obs.t = int(self._t)
        schema = self.obs.schema
        if self.camera_manager is not None:
            for key, value in self.camera_manager.tick_and_read().items():
                if key in schema:
                    self.obs.set_camera(key, value)
        if self.contact_manager is not None:
            contacts = self.contact_manager.read()
            for key, value in contacts.items():
                if key in schema:
                    self.obs.set_contact(key[len("contact/") :], value)
            # episode facts: force/active-body facts ride the record mapping
            # (the one raw read above already produced them).
            self._last_contact_facts = {
                key: value
                for key, value in contacts.items()
                if key.endswith("/force") or key.endswith("/active_bodies")
            }
            for name in self.observations.contact:
                if self.contact_manager.has_detail(name):
                    self.obs.set_contact_detail_provider(
                        name,
                        lambda name=name: self.contact_manager.read_detail(name),
                    )
        if self.prop_manager is not None:
            self.prop_manager.on_step(self)
            poses = self.prop_manager.get_world_poses(self)
            for name in self.observations.object:
                if name in poses:
                    pos, quat_w = poses[name]
                    self.obs.set_object(name, "pos", pos)
                    self.obs.set_object(name, "quat", quat_w)
            joints = self.prop_manager.get_articulation_joint_states(self)
            for name in self.observations.object:
                for joint_name, value in (joints.get(name) or {}).items():
                    if f"object/{name}/joints/{joint_name}" in schema:
                        self.obs.set_object(name, f"joints/{joint_name}", value)
        self._refresh_articulations()
        self._refresh_sites()
        if self.observations.command:
            self.obs.set_command(cmd)
        if self.observations.prev_action:
            self.obs.set_prev_action(raw_action)

    def _refresh_articulations(self) -> None:
        for name in self.observations.robot:
            art = self.articulations[name]
            d = art.asset.data
            perm = art.perm_il2pol
            full_q = d.joint_pos
            q0 = art.q0_art
            self.obs.set_robot(name, "joint_pos", full_q[:, perm])
            self.obs.set_robot(name, "joint_pos_rel", full_q - q0)
            self.obs.set_robot(name, "joint_vel", d.joint_vel[:, perm])
            self.obs.set_robot(name, "joint_vel_full", d.joint_vel)
            self.obs.set_robot(name, "applied_torque", d.applied_torque[:, perm])
            self.obs.set_robot(name, "gravity_b", d.projected_gravity_b)
            self.obs.set_robot(name, "base_pos_w", d.root_pos_w)
            self.obs.set_robot(name, "base_quat_w", d.root_quat_w)
            self.obs.set_robot(name, "base_lin_vel_w", d.root_lin_vel_w)
            self.obs.set_robot(name, "base_ang_vel_w", d.root_ang_vel_w)
            self.obs.set_robot(name, "base_lin_vel_b", d.root_lin_vel_b)
            self.obs.set_robot(name, "base_ang_vel_b", d.root_ang_vel_b)
            self.obs.set_robot(
                name,
                "base_yaw",
                math_utils.euler_xyz_from_quat(d.root_quat_w)[2].unsqueeze(-1),
            )

    def _refresh_sites(self) -> None:
        """Compose selected robot-profile site poses from the live body pose."""
        for handle in self._obs_sites:
            data = handle.asset.data
            body_pos = torch.as_tensor(data.body_pos_w)[:, handle.body_index]
            body_quat = torch.as_tensor(data.body_quat_w)[:, handle.body_index]
            local_pos = torch.as_tensor(
                handle.local_pos, dtype=body_pos.dtype, device=body_pos.device
            )
            local_quat = torch.as_tensor(
                handle.local_quat, dtype=body_quat.dtype, device=body_quat.device
            )
            pos_w = torch.as_tensor(body_pos + quat.rotate(body_quat, local_pos))
            quat_w = torch.as_tensor(quat.normalize(quat.mul(body_quat, local_quat)))
            self.obs.set_site(
                handle.name,
                pos_w.to(dtype=torch.float32),
                quat_w.to(dtype=torch.float32),
            )

    def record(self) -> dict[str, object]:
        """The current record mapping (canonical keys + bookkeeping)."""
        record: dict[str, object] = dict(self.obs.record())
        record["time_in_episode"] = self._time_in_episode.clone()
        record["episode_id"] = self._episode_id.clone()
        record.update(getattr(self, "_last_contact_facts", None) or {})
        return record

    # ------------------------------------------------------------------ #
    # stepping
    # ------------------------------------------------------------------ #
    def step(
        self,
        actions: Sequence[NamedAction],
        raw_action: torch.Tensor | None = None,
        cmd: torch.Tensor | None = None,
    ) -> EnvStep:
        """Advance one policy step behind the model-edit step guard."""
        if self._stepping:  # pragma: no cover - one owner per step
            raise RuntimeError("isaaclab2 backend: step() is not re-entrant")
        self._stepping = True
        try:
            return self._step_impl(actions, raw_action, cmd)
        finally:
            self._stepping = False

    def _step_impl(
        self,
        actions: Sequence[NamedAction],
        raw_action: torch.Tensor | None = None,
        cmd: torch.Tensor | None = None,
    ) -> EnvStep:
        """Advance one policy step.

        Args:
            actions: ``Sequence[NamedAction]`` — joint-addressed action
                batches routed by ``(articulation, joint)``; every required
                control joint must be addressed exactly once.  Entry joint
                names are resolved onto the term's flat input slice
                (preserve_order terms; resolver cached per name-tuple).
            raw_action: adapter raw action (policy order) -> the prev_action slot.
            cmd: current command tensor -> the command slot.

        Returns:
            an ``EnvStep``. ``step.reset_env_ids`` is always empty: core owns
            no episode clock, so an episode ends only through an explicit
            terminate or a sink terminal. ``extra["should_reset"]`` (bool) is
            part of the stable extra face.
        """
        # also applies mode="interval" DR events (pushes)
        self._raise_if_closed()
        super().step(self._route_named_actions(actions))
        self._time_in_episode += self.step_dt
        self._refresh_obs(raw_action, cmd)
        reset_ids = torch.empty(0, dtype=torch.int64, device=self.device)
        should_reset = False
        if self.camera_manager is not None:
            # step-tail posing: the render inside the NEXT step's sim.step
            # captures this pose (the classic re-pose semantics,
            # now rig-driven; see cameras/rigs.py)
            self.camera_manager.sync_pose()
        step = EnvStep(
            t=int(self._t),
            obs=self.obs.snapshot(),
            record=self.record(),
            reset_env_ids=reset_ids,
            should_reset=should_reset,
        )
        self._t += 1
        return step

    def _route_named_actions(self, actions: Sequence[NamedAction]) -> torch.Tensor:
        """Resolve + gather joint-addressed actions onto the flat term input.

        Builds the ``(N, total_action_dim)`` tensor ManagerBasedEnv.step
        expects, with each column routed by ``(articulation, joint)`` and each
        term's slice filled in the term's own input order (preserve_order
        terms: input dim i == cfg.joint_names[i]).  Unknown joints, wrong
        kinds, duplicate coverage and missing required joints all fail fast.
        """
        from . import articulation_runtime as _articulation_runtime

        return _articulation_runtime.route_actions(self, actions)

    # ------------------------------------------------------------------ #
    # optional state-write face (docs/state_write_contract.md)
    # ------------------------------------------------------------------ #
    def _state_root_asset(self, name: str):
        """Resolve *name* to a writable root asset; fail honestly otherwise."""
        art = self.articulations.get(name)
        if art is not None:
            asset = art.asset
            if bool(getattr(asset, "is_fixed_base", False)):
                raise ValueError(
                    f"isaaclab2 backend: articulation {name!r} has a fixed root; "
                    f"set_root_state needs a free root (use set_joint_state "
                    f"for its joints)"
                )
            return asset
        if self.prop_manager is not None:
            handler = self.prop_manager.handler(name)
            if handler is not None:
                asset = handler.asset
                if asset is None:
                    raise ValueError(
                        f"isaaclab2 backend: prop {name!r} has no writable "
                        f"asset (static stage decoration or unbound); "
                        f"set_root_state needs a dynamic prop"
                    )
                if bool(getattr(asset, "is_fixed_base", False)):
                    raise ValueError(
                        f"isaaclab2 backend: prop {name!r} has a fixed "
                        f"(anchored) root; set_root_state needs a free dynamic "
                        f"prop"
                    )
                return asset
        declared = sorted(self.articulations)
        props = list(self.prop_manager.names) if self.prop_manager is not None else []
        raise ValueError(
            f"isaaclab2 backend: set_root_state names unknown body {name!r}; "
            f"declared articulations: {declared}, props: {props}"
        )

    def _joint_ids(self, art, joint_names: Sequence[str]) -> list[int]:
        """Resolve explicit joint names to the asset's joint indices."""
        index = {name: i for i, name in enumerate(art.asset.joint_names)}
        missing = sorted(set(joint_names) - set(index))
        if missing:
            raise ValueError(
                f"isaaclab2 backend: joint names {missing} are not asset joints "
                f"of articulation {art.name!r} | asset joints: "
                f"{list(art.asset.joint_names)}"
            )
        return [index[name] for name in joint_names]

    def set_root_state(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
        *,
        name: str,
        pos_w: torch.Tensor,
        quat_wxyz: torch.Tensor,
        lin_vel_w: torch.Tensor,
        ang_vel_w: torch.Tensor,
    ) -> torch.Tensor:
        """Write a free root's world pose/velocity through the link writer.

        ``name`` is a declared free-root articulation or a dynamic prop.
        Uses ``write_root_link_pose_to_sim`` / ``write_root_link_velocity_to_sim``
        (link-origin semantics), never the CoM-default convenience writers.
        Does not call :meth:`settle`.
        """
        self._raise_if_closed()
        ids = normalize_state_env_ids(env_ids, self.num_envs, device=self.device)
        rows = ids.numel()
        pos = validate_state_tensor(pos_w, field="pos_w", rows=rows, cols=3, device=self.device)
        quat_in = validate_state_tensor(
            quat_wxyz, field="quat_wxyz", rows=rows, cols=4, device=self.device
        )
        lin = validate_state_tensor(
            lin_vel_w, field="lin_vel_w", rows=rows, cols=3, device=self.device
        )
        ang = validate_state_tensor(
            ang_vel_w, field="ang_vel_w", rows=rows, cols=3, device=self.device
        )
        unit_quat = normalize_state_quat(quat_in, rows=rows, device=self.device)
        asset = self._state_root_asset(name)
        pose = torch.cat((pos, unit_quat), dim=-1).to(dtype=torch.float32)
        velocity = torch.cat((lin, ang), dim=-1).to(dtype=torch.float32)
        asset.write_root_link_pose_to_sim(root_pose=pose, env_ids=ids)
        asset.write_root_link_velocity_to_sim(root_velocity=velocity, env_ids=ids)
        return ids

    def set_joint_state(
        self,
        env_ids: Sequence[int] | torch.Tensor | None,
        *,
        articulation: str,
        joint_names: Sequence[str],
        position: torch.Tensor,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write absolute joint positions (and optional velocities) by name.

        ``joint_names`` fixes the ``position``/``velocity`` columns.
        ``velocity=None`` writes positions only (velocities stay unchanged).
        Does not call :meth:`settle`.
        """
        self._raise_if_closed()
        art = self.articulations.get(articulation)
        if art is None:
            raise ValueError(
                f"isaaclab2 backend: set_joint_state names unknown articulation "
                f"{articulation!r}; declared: {sorted(self.articulations)}"
            )
        if isinstance(joint_names, (str, bytes)):
            raise TypeError("joint_names must be a sequence of joint names, not a string")
        names = tuple(str(name) for name in joint_names)
        if not names:
            raise ValueError("joint_names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError(f"joint_names carries duplicate names: {names}")
        ids = normalize_state_env_ids(env_ids, self.num_envs, device=self.device)
        rows = ids.numel()
        pos = validate_state_tensor(
            position, field="position", rows=rows, cols=len(names), device=self.device
        )
        vel = (
            None
            if velocity is None
            else validate_state_tensor(
                velocity, field="velocity", rows=rows, cols=len(names), device=self.device
            )
        )
        joint_ids = self._joint_ids(art, names)
        pos = pos.to(dtype=torch.float32)
        if vel is None:
            art.asset.write_joint_position_to_sim(position=pos, joint_ids=joint_ids, env_ids=ids)
        else:
            art.asset.write_joint_state_to_sim(
                position=pos,
                velocity=vel.to(dtype=torch.float32),
                joint_ids=joint_ids,
                env_ids=ids,
            )
        return ids

    def settle(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        """Flush pending target data, forward the view, refresh the cache.

        Mirrors the reset chain ordering: ``scene.write_data_to_sim()`` ->
        ``sim.forward()`` -> camera re-pose -> observation refresh.  Callers
        must settle after writes before reading observations or
        contact/camera-dependent state.
        """
        self._raise_if_closed()
        ids = normalize_state_env_ids(env_ids, self.num_envs, device=self.device)
        self.scene.write_data_to_sim()
        self.sim.forward()
        if self.camera_manager is not None:
            self.camera_manager.on_reset(ids)
        self._refresh_obs()
        return ids

    # ------------------------------------------------------------------ #
    # episode reset (per-env subset; also covers the batch case)
    # ------------------------------------------------------------------ #
    @property
    def last_reset_outcomes(self) -> dict[int, object]:
        """No structured reset outcomes: isaaclab2 has no initializer face.

        The public runner-facing contract is still present (empty mapping) so
        episode facts stay uniform across backends.
        """
        return {}

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
        self._raise_if_closed()
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device)
        if env_ids.numel() == 0:
            return env_ids
        self._reset_idx(env_ids)
        if self.prop_manager is not None:  # re-place props + close joints
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
        if not self._partitioned and (
            self._init_pose is not None or "reset" in self.event_manager.available_modes
        ):
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
                        n_pick,
                        ip["roll_range"],
                        ip["pitch_range"],
                        ip["yaw_range"],
                        ip["height"],
                        self.device,
                    )
                    drm = dr[pick].clone()
                    drm[:, :3] = pos
                    drm[:, 3:7] = quat
                    drm[:, 7:13] = 0.0  # at rest before first contact
                    dr[pick] = drm
            robot.write_root_pose_to_sim(root_pose=dr[:, :7], env_ids=env_ids)
            robot.write_root_velocity_to_sim(root_velocity=dr[:, 7:13], env_ids=env_ids)
            robot.write_joint_state_to_sim(
                position=dj, velocity=torch.zeros_like(dj), env_ids=env_ids
            )
            robot.data.root_state_w[env_ids] = dr
        self.scene.write_data_to_sim()
        self.sim.forward()
        self._episode_id[env_ids] += 1
        self._time_in_episode[env_ids] = 0.0
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
        self._refresh_obs()  # registry now reflects spawn state
        return env_ids

    def reset_all(self) -> torch.Tensor:
        """Batch-synchronous episode reset (the shorthand)."""
        self._raise_if_closed()
        return self.reset_envs(torch.arange(self.num_envs, device=self.device))

    # ------------------------------------------------------------------ #
    # native escape hatch (external packs only; unstable)
    # ------------------------------------------------------------------ #
    def native(self, backend: str) -> NativeHandle:
        """Return identity references to this backend's native objects.

        External-pack escape hatch only (``docs/native_handle.md``): the
        handle is unstable and never copies, settles, or writes state.
        Partitioned multi-articulation runs expose ``robot=None`` and carry
        their assets in ``scene``/``robots``.
        """
        if backend != "isaaclab2":
            raise ValueError(
                f"isaaclab2 backend native() expects backend='isaaclab2', got {backend!r}"
            )
        return NativeHandle(
            "isaaclab2",
            {
                "sim": self.sim,
                "scene": self.scene,
                "robot": getattr(self, "robot", None),
            },
        )

    # ------------------------------------------------------------------ #
    # optional runtime-query face (trait-gated by run.query)
    # ------------------------------------------------------------------ #
    @property
    def query(self) -> Query:
        """The live on-demand runtime-query facade (one cached instance).

        The object is built lazily on first access and reused for this live
        env instance, so ``run.query`` and ``env.query`` return the same
        handle.  Query is not an observation channel: it never publishes to
        ``env.obs`` and never records.

        Raises:
            QueryStaleError: The environment is already closed.
        """
        if self._closed or self._closing:
            raise QueryStaleError(
                "env.query for backend 'isaaclab2' is stale: the environment is closed"
            )
        if self._query_handle is None:
            from unigym.backends.isaaclab2.query import Isaaclab2Query

            self._query_handle = Isaaclab2Query(self)
        return self._query_handle

    # ------------------------------------------------------------------ #
    # teardown
    # ------------------------------------------------------------------ #
    def _raise_if_closed(self) -> None:
        """Reject use during or after teardown with the backend closed error."""
        if self._closed or self._closing:
            raise RuntimeError("isaaclab2 backend: environment is closed")

    def __del__(self) -> None:
        """Skip the parent finalizer when construction never reached ``sim``.

        A refused build can leave an instance without the parent-initialized
        ``sim`` attribute.  ``ManagerBasedEnv.__del__`` calls ``close()``
        unconditionally, which touches ``self.sim``; returning early here only
        covers that unconstructed case.  Any other finalizer error still
        propagates unchanged, and this never closes the Kit application.
        """
        if not hasattr(self, "sim"):
            return
        super().__del__()

    def close(self) -> None:
        """Release the inherited managers/sim (idempotent, not retryable).

        The inherited ``ManagerBasedEnv.close()`` deletes managers and clears
        the singleton; a teardown failure therefore leaves the environment
        unusable and is propagated unchanged.  ``_closed`` is set in
        ``finally`` so a failed close cannot be retried as if nothing had
        been released.

        The public ``pause()`` runs first so close leaves the Kit timeline
        non-playing.  A subsequent build in the same process then constructs a
        clean ``SimulationContext``: the new context registers its STOP
        callback before the parent constructor, and the parent only stops a
        playing timeline, so a non-playing timeline never re-enters that
        callback.  Pausing before the inherited teardown also covers a close
        that raises after the pause.
        """
        if self._closed:
            return
        self._closing = True
        try:
            if self._query_handle is not None:
                self._query_handle.close()
            self.sim.pause()
            super().close()
        finally:
            self._closed = True
            self._closing = False

    # ------------------------------------------------------------------ #
    # snapshot / restore (Round B rewind): DECLARED ABSENT
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict:
        """Not implemented on isaaclab v1 (rewind is mujoco-only, Round B)."""
        raise NotImplementedError(
            "isaaclab2 backend does not implement snapshot()/restore() "
            "(rewind is mujoco-only); run the "
            "rewind face against env.backend: mujoco"
        )

    def restore(self, state) -> int:
        """Not implemented on isaaclab v1 (rewind is mujoco-only, Round B)."""
        raise NotImplementedError(
            "isaaclab2 backend does not implement snapshot()/restore() "
            "(rewind is mujoco-only); run the "
            "rewind face against env.backend: mujoco"
        )
