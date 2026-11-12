"""MuJoCo Warp backend.

Independent copy of the mujoco env orchestration: the same
``MjModel``/``MjData`` contract, driven by ``mujoco_warp`` instead of the
CPU ``mujoco.mj_step``.  There is exactly ONE real ``MjData`` per world, used
as the CPU **mirror** read face:
``mjw.step(wm, wd)`` advances every device world in one kernel pass, then
``mjw.get_data_into(mirror, mjm, wd, e)`` refreshes every derived quantity
(props / contacts / director / state readout consume the mirror unchanged;
cameras at N>1 render from the warp batch directly, N=1 from the
mirror's CPU ``mujoco.Renderer``).

Batch scope: ``nworld=N`` batched worlds on ONE warp ``Model`` + ONE ``Data``;
the N CPU ``MjData`` mirrors stay the read face.  N=1 keeps the single-world path
(one-element batch).  Device choice is warp's (``cpu`` or ``cuda:N``; default
``cuda:0`` when available).  Torch caches stay CPU, exactly like the mujoco
backend.

Divergences from the CPU mujoco backend (documented, not bugs):
* Physics is single precision (warp float32) while the CPU lane is float64;
  contact-free short horizons agree to ~5e-7, contact-heavy trajectories are
  chaotic and are NOT expected to agree.
* Snapshot/restore is mirror-faithful (``mj_getState``/``mj_setState`` plus a
  per-world device re-seed), not CPU-bitwise: the warp device is rebuilt from
  the mirrors on restore.  Batched worlds are independent (the test pins
  N>1 against N=1 runs).
* Contact sets are warp's (``nacon``/``nefc`` + ``get_data_into`` contacts),
  not CPU mujoco's; ``ncon``/force values are mirror reads of warp's solve.
* A CPU ``mj_step`` must NEVER replace a warp step (including
  ``_settle_substeps``): that would desync the device.

Highlights inherited from the mujoco backend (unchanged semantics):
* 43 torque motors, software PD at sim rate (standard_pd default), named
  action routing, the actuator seam's ``step1 -> actuator -> step2`` split.
* Joint order: the controller's ``asset_joint_names`` (43 hinges) /
  ``joint_names`` (29)
  built by name and cross-checked against ``MUJOCO_TO_ISAACLAB``.
* Reset writes q0 explicitly and runs ``mj_forward`` on the mirror, then
  re-seeds the device with ``mjw.put_data``.
* Foot contacts filter by geom -> foot body membership and
  ``mj_contactForce`` on the mirror.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from unigym.env.backends.base import BackendCapabilities
from unigym.cameras.manager import validate_image_frame
from unigym._optional import MissingExtraError

# The camera backend needs a GL context (mujoco.Renderer); EGL is the
# verified headless backend on the humanoid_1 host (~0.5 ms/frame @224²).
# MUST be set before the first `import mujoco` (gl_context reads it at
# import time); an explicit MUJOCO_GL in the environment always wins.
os.environ.setdefault("MUJOCO_GL", "egl")

_MUJOCO_IMPORT_ERROR: BaseException | None = None
_WARP_MISSING: str = "mujoco_warp"

if TYPE_CHECKING:  # pragma: no cover - one canonical mixin for the analyzer
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    from unigym.backends.mjwarp.sizing import Capacity
    from unigym.backends.mjwarp.warp_core import WarpBatchMixin
    from unigym.backends.shared.mjfamily.name_map import (
        MjArticulationBinding,
        MjControlBinding,
    )
    from unigym.runtime.controllers import ArticulationRuntime, ControlRuntime
else:
    try:  # mujoco is only importable in the host .venv_sim environment
        import mujoco
    except ImportError as _e:  # pragma: no cover - gated by importorskip
        mujoco = None
        _MUJOCO_IMPORT_ERROR = _e

    try:  # the GPU stack is an optional extra (pyproject `warp`)
        import warp as wp
        import mujoco_warp as mjw
    except ModuleNotFoundError as _e:  # pragma: no cover - construction gate
        wp = None
        mjw = None
        _WARP_MISSING = _e.name or "mujoco_warp"

    if wp is not None and mjw is not None:
        from unigym.backends.mjwarp.warp_core import WarpBatchMixin
    else:  # pragma: no cover - construction fails before the mixin is used

        class WarpBatchMixin:
            """Stand-in base when the warp stack is absent (lane gate raises)."""


from unigym.env.episode import parse_episode_length, sample_durations
from unigym.env.obs_catalog import (
    build_obs_catalog,
    canonicalize_obs_requests,
    split_articulation_key,
)
from unigym.backends.mjwarp.model import (
    MjwarpModelProvider,
    body_id,
    site_id,
)
from unigym.backends.shared.mjfamily.articulation_obs import (
    articulation_extra,
    read_articulation_state,
    refresh_articulation_obs,
)
from unigym.backends.shared.mjfamily.articulation_runtime import (
    apply_control_substep,
    init_tables,
    reset_articulations,
    route_actions,
)
from unigym.backends.shared.mjfamily.props.reset_face import resolve_base_height, wire_reset_face
from unigym.backends.shared.mjfamily.props.multienv import reject_model_global_unsupported
from unigym.backends import collect_core
from unigym.backends.mjwarp import sizing
from unigym.backends.mjwarp.support import (  # noqa: F401 (re-exported)
    _mj_enum,
    _policy_gains,
    _put_warp_model,
    _quat_to_rotmat,
    _resolve_warp_device,
)
from unigym.backends.shared.mjfamily.scene.scene_spawn import (
    apply_spawn_base_pose,
    needs_spec,
)
from unigym.backends.shared.mjfamily.name_map import MjNameMap
from unigym.backends.shared.mjfamily.scene.articulation_compose import (
    compose_articulation_spec,
)
from unigym.backends.shared.mjfamily.scene.scene_artifacts import (
    _fill_scene_content,
    _require_content_scene,
    capture_composed_xml,
    write_scene_artifacts,
)
from unigym.backends.shared.mjfamily.world_base import (
    reject_legacy_mj_model,
    resolve_world_base_mjcf,
)
from unigym.runtime.actions import MujocoJointResolver
from unigym.actuators import StandardPdActuator, build_actuator

BASE_BODY = "pelvis"

# capability bits: runs N>1 as ONE batched warp Data (nworld=N) with N CPU
# mirrors.  contacts/props/recorder ride the CPU mirror; cameras are per-world
# (N=1 CPU mirror renderer, N>1 native warp batch renderer); the
# actuator seam is the same software ctrl face as the mujoco backend.
# bits must stay a CHECKED subset of MJWARP_TRAITS.  Both paths are
# proved by tests/backends/mjwarp/test_recorder.py and
# test_actuator_seam.py; the matching traits are declared.
BACKEND_CAPABILITIES = BackendCapabilities(
    recorder=True, cameras=True, contacts=True, props=True, actuator_seam=True, multi_env=True
)


class _ActuatorTargetFeed:
    """Compose one actuator's joint-ordered target vector from controls."""

    def __init__(self, act, sources):
        self.act = act
        self._sources = list(sources)

    def __call__(self, targets: dict, e: int) -> np.ndarray:
        out = np.zeros(len(self._sources), dtype=np.float64)
        for i, (control_name, index) in enumerate(self._sources):
            out[i] = targets[control_name][e, index]
        return out


class MjwarpEnv(WarpBatchMixin):
    """MuJoCo Warp backend (N batched worlds, N CPU mirror read faces).

    Director dispatch is injected by the run layer (``BackendInfo.
    director_ops``); this env advertises no backend name.
    ``MjwarpDirectorOps`` adds the mirror->device push per write.

    Args (``build_env_backend`` kwargs; backend-specific keys come from the
    yaml ``env.params`` section via collect.py's ``backend_params``):
        robot_profile: RobotRecipe (pure constants; ``articulation_cfg``
            unused — the simulator model is the MJ XML).
        adapter: Adapter (declares obs requests / orders).
        num_envs: native N>1 batched worlds (one warp ``nworld=N`` Data; N
            CPU mirrors).  Cameras are per-world; director ops /
            anchored articulation props still fail fast at N>1.
        episode_length_s: scalar or [lo, hi] (``env.episode_length_s``).
        decimation / sim_dt: policy/sim rate chain (4 x 0.005 -> 50 Hz).
            The XML default timestep (0.002) is overridden to ``sim_dt``.
        world_base: the run-level world declaration from the config's
            top-level ``world_base`` (``{mjcf: <path>}``, or a bare path
            string for direct construction).  ``mjcf`` is required; there
            is no default model (``DEFAULT_MJ_MODEL`` and
            ``env.params.mj_model`` were removed).
        articulations: the run's ordered neutral articulation specs.
            Entries declaring ``model.mjcf`` are composed into
            the world base before the scene/props/camera injections;
            ``None``/no declared asset keeps the historical fused load
            path byte-identically.
        asset_plan: the composition boundary that resolves the declared
            scene's assets; the backend never resolves asset references.
        backend_params: yaml ``env.params`` dict.  Keys:
            mj_scene_params: optional slot dict passed through to the
                registered ``mj_scene`` provider's dispatch verbatim
                (e.g. ``{seed: 2026}`` for
                robocasa_kitchen_random).  ABSENT -> ``{}`` -- the
                provider's own fail-closed slot gates stay the single
                truth.  Ignored (and validated nowhere) when
                ``mj_scene`` is not declared.
            spawn_base_pose: optional fixed-base spawn declaration
                -- ``{body: <name>, pos: [x, y, z], yaw: rad}``
                rewrites the declared body's SPEC pose before compile
                (the declared yaw IS the base's world yaw; the baked
                quat is replaced).  For fixed-base models whose base
                pose is baked into the XML (the GR1 tabletop robot:
                no freejoint, spawn_xy/spawn_yaw have no qpos to land
                in).  ABSENT -> the XML-baked pose (bit-identical).
                Fail-closed: the body must exist, pos finite 3-vector,
                yaw finite radians.
            spawn_height: pelvis z at reset (default 0.78, g1_constants).
            spawn_xy: planar spawn position [x, y] in metres, finite 2-vector
                (default [0, 0]); written into the freejoint qpos every reset.
            spawn_yaw: planar spawn heading in radians, finite scalar
                (default 0.0); reset writes the (w,x,y,z) freejoint quat.
                mujoco-only capability: the isaaclab backend manages its
                spawn through the Isaac Lab init system and has no
                spawn_height/spawn_xy/spawn_yaw params either.
            contact_force_threshold: foot-contact threshold in N (default
                1.0; UNCALIBRATED across engines — see module docstring).
            kp / kd: optional per-joint-name-or-regex PD gain overrides.
            hand_kp / hand_kd: optional hand-DOF soft-PD gain overrides
                (defaults G1_KP_HANDS/G1_KD_HANDS = 20/2).
            actuators: optional ctrl-law map.  ABSENT -> the legacy path
                (standard_pd on the policy + hand joints), bit-identical.
                Present: {group: {type, regex | joint_names, ...law
                fields}}; every robot joint (policy + hands) must be
                covered exactly once (fail-fast gaps/overlaps, and prop
                servo slots are off limits).  Laws (unigym/actuators
                registry): standard_pd = the legacy formula (kp/kd
                optional -- the gain channel above pre-fills them);
                computed_torque_pd = tau = M_part@(kp*e - kd*dq) +
                qfrc_bias recomputed per substep, per-part blocks via
                'parts' (regex list partitioning the group; kp/kd
                required); native_servo = ctrl = clip(target, ctrlrange)
                passthrough (model-side affine servo).  Any non-standard
                law switches the decimation loop into the mj_step1 ->
                actuator.step -> mj_step2 phase split (fresh per-substep
                M/bias).  The isaaclab backends reject non-standard
                declarations at the capability gate (error + hint).
            reset: optional reset-protocol block (legacy face; the
                scene-bundle wiring declares ``initializer`` instead --
                this block transposes to an anonymous inline initializer
                at dispatch, see ``initializer``).  ABSENT -> the legacy
                reset, bit-identical.  Keys: protocol ("gr1_cotrain");
                seed (the stream; same seed -> bit-identical resets);
                settle_substeps (default 250); cotrain_noise (default
                0.2); elbow_joints / elbow_mix (defaults = the audit's
                r/l_elbow_pitch 0.7/0.3); placement (SplitA declarative
                range table, requires wired props).
            initializer: scene-bundle reset face:
                {type: <registry name> | module: "pkg.mod:factory",
                params: {...}} dispatched through INITIALIZER_REGISTRY
                (validate -> resolve -> per-episode apply -> the sampled
                names must all be declared props).  The builtin
                splita_tabletop type carries the SplitA placement
                vocabulary (the legacy ``placement`` fields, flat) plus
                the robot-stage keys; the GR1 stage chain runs after
                apply() in the unchanged order over the SAME stream.
                Declaring both initializer and reset is an error.
            base: scene-bundle base declaration ({joint: fixed}); fixed
                absorbs the legacy spawn_height param (height 0.0, no
                freejoint, no spawn drop).  No declaration -> the
                legacy spawn_height semantics.
            content: optional content-fill spec (``sets`` / ``demands`` /
                ``policy`` / ``provider`` / ``seed``, split out of the
                run's ``objects:`` block).  When present the env fills the
                built scene before composing, merges the returned
                instances and re-runs the four binding checks; the fill
                always runs (the scene must be complete), while
                ``fill.json`` is written only when artifacts are on.
                Build-time fill uses ``spec.seed`` (default 0).
        artifact_dir: optional run artifact directory.  When declared and
            the env composes a scene (``env.params.mj_scene``), the six
            build artifacts land here: ``draw.json`` / ``resolution.json``
            / ``scene.json`` / ``payload_index.json`` / ``composed.xml`` /
            ``compile.json``.  ``None`` -> no artifact IO (the historical
            construction path is unchanged).
        device: warp device: "cpu" or "cuda:N" (default "cuda:0" when a
            CUDA device is visible, else "cpu").  Torch caches stay CPU.
        terrain_cfg / env_spacing / events_cfg / recorder
        / lighting / depth_camera / init_pose: accepted for
        builder-kwargs parity and IGNORED (capabilities are all False, so
        the gate already rejected any non-None request) — EXCEPT
        ``camera_manager``: a wired CameraManager gets its cameras
        injected into the MJ model (MjSpec, pre-compile) and renders at every
        obs refresh (N=1 CPU mirror renderer; N>1 the native warp batch
        renderer, ``backends/mjwarp/cameras.py``); it is the frame source for
        the recorder's ``camera_ref``.
            contact_manager : a wired ContactManager
        (yaml ``contacts``, docs/contracts/contacts_contract.md) binds to the loaded
        model (body-id resolution) and merges ``contact/<name>`` keys into
        ``extra`` every step (``contacts/mujoco_backend.py``).
            prop_manager : a wired PropManager (yaml ``props``).
        Box-family prop bodies are appended to the MjSpec pre-compile
        (``props/mujoco_backend.py``), re-placed on reset, merged as
        ``object/<name>/{pos,quat,held}`` into ``extra`` every step, and
        served as obs (``<name>_pos_b``/``_vel_b``) on adapter request.
    """

    # Backend-private runtime tables (built by ``init_tables``, not in
    # ``__init__``: declared here so the step/obs faces see the contract).
    articulations: dict[str, ArticulationRuntime]
    controls: dict[str, ControlRuntime]
    _articulation_order: list[ArticulationRuntime]
    _control_order: list[ControlRuntime]
    _mj_bindings: dict[str, MjControlBinding]
    _mj_articulation_bindings: dict[str, MjArticulationBinding]
    _act_of_joint: dict[str, int]
    _prop_actuators: dict[str, dict[str, int]]
    _prop_ctrl_mask: np.ndarray
    _prop_targets: np.ndarray
    _joint_limits: tuple[torch.Tensor, torch.Tensor] | None
    _joint_limits_by_articulation: dict[str, tuple[torch.Tensor, torch.Tensor]]

    # Live sim handles: always bound by ``__init__``; deliberately nulled at
    # the explicit ``close()`` teardown boundary (see ``close``).
    _model: mujoco.MjModel
    _data: mujoco.MjData
    _datas: list[mujoco.MjData]
    _active_data: mujoco.MjData
    _capacity: Capacity
    _closed: bool
    _closing: bool

    def __init__(
        self,
        robot_profile,
        adapter,
        num_envs: int = 1,
        episode_length_s: float = 10.0,
        decimation: int = 4,
        sim_dt: float = 0.005,
        backend_params: Optional[dict] = None,
        world_base=None,
        articulations=None,
        device: str = "cpu",
        terrain_cfg=None,
        env_spacing: float = 4.0,
        events_cfg=None,
        recorder=None,
        prop_manager=None,
        lighting=None,
        depth_camera=None,
        init_pose=None,
        camera_manager=None,
        contact_manager=None,
        director_cfg=None,
        artifact_dir: Optional[str] = None,
        controls=None,
        recipes=None,
        backend_options=None,
        asset_plan=None,
    ):
        if mujoco is None:  # pragma: no cover
            raise RuntimeError(
                f"mjwarp backend requires mujoco (host .venv_sim); "
                f"import failed with: {_MUJOCO_IMPORT_ERROR}"
            )
        if mjw is None or wp is None:  # pragma: no cover - lane gate
            raise MissingExtraError(
                _WARP_MISSING if _WARP_MISSING in ("warp", "mujoco_warp") else "mujoco_warp",
                "warp",
                feature="MjwarpEnv (MuJoCo Warp backend)",
            )
        n = int(num_envs)
        if n < 1:
            raise ValueError(f"MjwarpEnv requires num_envs>=1, got {num_envs}")
        # Cameras ARE per-world here (native warp batch renderer at
        # N>1; the N=1 mirror path is unchanged), so the camera branch is not
        # rejected.  Director ops + anchored articulation props still write
        # model-global state with no per-world face.
        reject_model_global_unsupported(n, prop_manager, None, director_cfg)
        if any(x is not None for x in (events_cfg, lighting, depth_camera, init_pose)):
            raise NotImplementedError(
                "mjwarp backend does not implement events/"
                "depth_camera/init_pose; the capability gate should have "
                "rejected this request"
            )
        self.warp_device = _resolve_warp_device(device)
        params = dict(backend_params or {})
        options = dict(backend_options or {})
        auto_prefix = bool((options.get("compose") or {}).get("auto_prefix", False))
        reject_legacy_mj_model(params, source="env.params")

        # ---- model load (flat-ground scene; timestep -> sim_dt) ----
        mj_path = resolve_world_base_mjcf(world_base)
        if not os.path.isfile(mj_path):
            raise FileNotFoundError(f"mujoco backend: MJ XML not found: {mj_path}")
        self._mj_path = mj_path
        self.artifact_dir = None if artifact_dir is None else os.fspath(artifact_dir)
        self.artifact_paths: dict = {}
        self._fill_record = None
        self._closed = False
        self._closing = False
        self.camera_manager = camera_manager
        # feat/object-state : props join the MjSpec BEFORE compile
        # (same injection point as the camera rigs)
        self.prop_manager = prop_manager
        # director: --direct only (director_contract.md §0.3) --
        # gripper mocap body + per-prop disabled weld slots join the spec
        # pre-compile; None -> the model is byte-identical (regression red
        # line)
        self.director_cfg = director_cfg
        # Scene-chain: env.params.mj_scene names a REGISTERED
        # scene definition (build_scene_definition) composed + attached
        # into the base spec (the robot world-part XML -- it owns the
        # world-level option/size/visual settings; attach never
        # transfers child options).  The composed content joins the
        # spec BEFORE the camera/prop/director injections.
        self.mj_scene = params.get("mj_scene")
        self._mj_scene_params = dict(params.get("mj_scene_params") or {})
        self._content_spec = params.get("content")
        self._placement_regions = {}
        _require_content_scene(self._content_spec, self.mj_scene)
        spawn_base = params.get("spawn_base_pose")
        # Compose the declared articulation assets into the world
        # base BEFORE the scene/props/camera/director injections.  None
        # means the base is already fused (no articulation declares
        # model.mjcf) and the historical load path stays byte-identical.
        composition = compose_articulation_spec(mj_path, articulations, auto_prefix=auto_prefix)
        self._mj_composition = composition
        self._mj_name_map = (
            composition.name_map if composition is not None else MjNameMap.identity()
        )
        composed_spec = composition.spec if composition is not None else None
        if composed_spec is not None or needs_spec(
            self.camera_manager, self.prop_manager, self.director_cfg, self.mj_scene, spawn_base
        ):
            from unigym.scenes.registry import build_scene_definition
            from unigym.backends.shared.mjfamily.scene.scene_engine import compose_and_attach

            mjspec = (
                composed_spec if composed_spec is not None else mujoco.MjSpec.from_file(mj_path)
            )
            scene = None
            if self.mj_scene is not None:
                # the mj_scene_params slot dict rides the provider
                # dispatch verbatim; absent -> the provider's own gates
                scene = build_scene_definition(str(self.mj_scene), self._mj_scene_params)
                scene, self._fill_record = _fill_scene_content(
                    scene, self._content_spec, asset_plan
                )
                self._placement_regions = compose_and_attach(mjspec, scene, asset_plan)
            # the fixed-base spawn declaration (None -> XML-baked pose)
            apply_spawn_base_pose(mjspec, spawn_base)
            # the scene artifact face serializes the SAME chain the A/B
            # dump tool pins (base model + scene + spawn rewrite), BEFORE
            # the camera/prop/director layers join the spec and compile.
            composed_xml = capture_composed_xml(mjspec, self.artifact_dir, scene)
            if self.camera_manager is not None:
                # Inject the declared rigs as <camera> elements BEFORE
                # compilation (MjSpec path; see cameras/mujoco_backend.py)
                self.camera_manager.setup_scene(mjspec)
            if self.prop_manager is not None:
                self.prop_manager.inject_spec(mjspec)
            if self.director_cfg is not None:
                from unigym.backends.shared.mjfamily.director.gripper import add_director_bodies

                add_director_bodies(mjspec, self.director_cfg, self.prop_manager)
            self._model = mjspec.compile()
            # The assembled IR is the COMPILED one (addresses/ids
            # filled from MjModel); the conformance digest still builds
            # the declared IR separately in ``scene_engine.describe``.
            self._model_ir = MjwarpModelProvider().describe(self._model)
            self._native_model_ir = self._model_ir
            self.artifact_paths = write_scene_artifacts(
                self.artifact_dir,
                scene,
                self.mj_scene,
                self._model,
                self._mj_scene_params,
                composed_xml,
                fill_record=self._fill_record,
            )
        else:
            self._model = mujoco.MjModel.from_xml_path(mj_path)
            self._model_ir = MjwarpModelProvider().describe(self._model)
            self._native_model_ir = self._model_ir
        if self._placement_regions:
            from unigym.backends.shared.mjfamily.scene.scene_render import bind_placement_regions

            self._placement_regions = bind_placement_regions(self._model, self._placement_regions)
        self._model.opt.timestep = float(sim_dt)  # XML ships 0.002 (500 Hz)
        # SIMPLE-alignment knob (env.params.mj_opt): engine-option overrides
        # applied post-compile, e.g. {impratio: 10, integrator:
        # "mjINT_IMPLICITFAST", cone: "mjCONE_ELLIPTIC", noslip_iterations: 2}
        # (SIMPLE engines/mujoco.py L136-140).  Enum values resolve by member
        # name; numbers pass through.
        for opt_key, opt_val in (params.get("mj_opt") or {}).items():
            if not hasattr(self._model.opt, opt_key):
                raise ValueError(
                    f"mujoco backend: mj_opt key {opt_key!r} is not an " f"mjtOption field"
                )
            cur = getattr(self._model.opt, opt_key)
            val = _mj_enum(opt_val) if isinstance(cur, int) else type(cur)(opt_val)
            setattr(self._model.opt, opt_key, val)
        # SIMPLE-alignment knob (env.params.obj_gravity_boost): persistent
        # Cartesian force on one body, e.g. {body: crate, force_z: -0.981}
        # (SIMPLE BendPick need_gravity=True adds -9.81*0.1 N to the object
        # -- engines/mujoco.py L200-227; an ADDITIONAL load, not an
        # anti-gravity compensation).  Applied after every reset (mj_resetData
        # clears xfrc_applied).
        boost = params.get("obj_gravity_boost") or None
        if boost:
            bname = str(boost.get("body", ""))
            self._boost_body_id = body_id(self._model_ir, bname)
            if self._boost_body_id < 0:
                raise ValueError(
                    f"mujoco backend: obj_gravity_boost body {bname!r} "
                    f"not found in the MJ model"
                )
            self._boost_fz = float(boost.get("force_z", 0.0))
        else:
            self._boost_body_id, self._boost_fz = -1, 0.0
        # EEF site record family (env.params.record_sites):
        # declarative per-step ``site/<name>/pos`` record keys (world
        # frame) -- the pack criteria's far predicates consume the GR1
        # grip sites.  Additive only: absent/empty declaration -> no ids,
        # no keys, zero behavior diff.  Unknown names fail fast here (the
        # site names are the scene XML's, per the pack contract).
        self._record_sites: list = []
        _sites = params.get("record_sites")
        if _sites is not None:
            if not isinstance(_sites, (list, tuple)) or not all(
                isinstance(s, str) and s for s in _sites
            ):
                raise ValueError(
                    f"mujoco backend: env.params.record_sites must be a "
                    f"list of site names, got {_sites!r}"
                )
            if len(set(_sites)) != len(_sites):
                raise ValueError(
                    f"mujoco backend: record_sites carries duplicate " f"names: {sorted(_sites)}"
                )
            for sname in _sites:
                sid = site_id(self._model_ir, sname)
                if sid < 0:
                    raise ValueError(
                        f"mujoco backend: record_sites site {sname!r} not "
                        f"found in the MJ model | hint: site names come "
                        f"from the scene XML ('<site name=...' entries); "
                        f"the GR1 tabletop scene carries "
                        f"gripper0_{{left,right}}_grip_site"
                    )
                self._record_sites.append((sname, sid))
        # ---- warp model + ONE batched device Data (nworld=N) ----
        # put_model snapshots the compiled model; every opt/timestep write
        # above has already landed, and the CPU MjModel stays the read face.
        # put_data broadcasts world 0 to every world -- the initial state.
        # N>1 cameras need per-world cam_pos/cam_quat (batch_sizes);
        # N=1 keeps the unbatched model path bit-identical.
        self._wm = _put_warp_model(self._model, self.camera_manager, n)
        self._capacity = sizing.capacity_for(self._model, steps=max(100, int(decimation) * 10))
        self._datas = [mujoco.MjData(self._model) for _ in range(n)]
        self._data = self._datas[0]  # legacy alias: tests/scene/director read it
        self._active_data = self._datas[0]
        self._wdata = self._new_warp_data(self._datas[0], n)
        m = self._model
        self._decimation = int(decimation)
        self._sim_dt = float(sim_dt)
        self.step_dt = float(decimation) * float(sim_dt)  # policy rate (0.02)
        self.num_envs = n
        # torch caches (obs/extra) stay CPU; ``sim_device`` is warp's world.
        self.device = "cpu"
        self.sim_device = self.warp_device
        # base declaration (scene bundles): ``base.joint: fixed``
        # absorbs the legacy spawn_height param (no freejoint, no drop
        # -> 0.0); no declaration -> the legacy spawn_height semantics
        # (unigym/backends/shared/mjfamily/props/reset_face.py)
        self._spawn_height = resolve_base_height(params)
        # spawn_xy/spawn_yaw (forward-port from the cyx g1_nav fork): planar
        # spawn pose written into the freejoint qpos on EVERY reset.  Fail-fast
        # validation; the defaults keep the origin spawn bit-for-bit identical.
        self._spawn_xy = np.asarray(params.get("spawn_xy", [0.0, 0.0]), dtype=float)
        self._spawn_yaw = float(params.get("spawn_yaw", 0.0))
        if (
            self._spawn_xy.shape != (2,)
            or not np.isfinite(self._spawn_xy).all()
            or not np.isfinite(self._spawn_yaw)
        ):
            raise ValueError("spawn_xy must be finite xy; spawn_yaw must be finite radians")
        # Fixed-base (no freejoint) state-readout constants --
        # the base frame IS the configured spawn pose (the base is welded
        # to the world and never moves), velocities are zero, projected
        # gravity follows the spawn yaw.  The freejoint path never reads
        # these (its base state lives in qpos[0:7]/qvel[0:6]).
        self._fixed_base_pos_w = np.array([*self._spawn_xy, self._spawn_height], dtype=np.float64)
        self._fixed_base_quat_w = np.array(
            [np.cos(self._spawn_yaw / 2), 0.0, 0.0, np.sin(self._spawn_yaw / 2)], dtype=np.float64
        )
        self._contact_force_threshold = float(params.get("contact_force_threshold", 1.0))

        # ---- feat/contact-manager: bind AFTER model/data exist (body ids,
        # geom tables); no scene elements are needed (setup_scene is a no-op)
        self.contact_manager = contact_manager
        if self.contact_manager is not None:
            self.contact_manager.bind_env(self)

        # ---- feat/object-state: props runtime side (body ids + freejoint
        # addresses resolved against the compiled model, then the initial
        # placement sampled from the specs)
        self.prop_backend = None
        if self.prop_manager is not None:
            from unigym.backends.shared.mjfamily.props.props import MujocoPropBackend

            self.prop_backend = MujocoPropBackend(self.prop_manager)
            self.prop_backend.bind_env(self)
            self.prop_backend.on_reset([0], self)

        # Every engine iterates env.articulations / env.controls.
        # The run layer built the neutral control runtimes from RunSpec.controls
        # + the robot recipes; this backend resolves their addresses + drive
        # strategies against the compiled model.
        self._backend_label = "mjwarp backend"
        self._articulation_specs = tuple(articulations or ())
        self._recipes = dict(recipes or {})
        self._planned_controls = tuple(controls or ())
        self._jid0 = 1 if int(m.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE) else 0
        self._has_freejoint = bool(self._jid0)
        init_tables(self, m, params, _policy_gains, label="mjwarp backend")
        self._hand_names = sorted(
            n
            for art in self._articulation_order
            for c in self._control_order
            if c.articulation == art.name and c.action_face != art.name
            for n in c.joint_names
        )
        self._action_resolver = MujocoJointResolver(
            m, [], [], prop_actuators=self._prop_actuators, model_ir=self._model_ir
        )
        self._joint_gathers: dict = {}
        first = next(
            (c for c in self._control_order if not c.prop and c.action_face == c.articulation),
            self._control_order[0] if self._control_order else None,
        )
        if first is not None:
            self._perm_il2pol = torch.tensor(
                [int(i) for i in first.addresses.get("perm_il2pol", ())], dtype=torch.int64
            )
            self._perm_pol2il = torch.argsort(self._perm_il2pol)
            self._restricted_to_policy = torch.argsort(self._perm_pol2il)
        else:  # pragma: no cover - a run always declares controls
            self._perm_il2pol = torch.zeros(0, dtype=torch.int64)
            self._perm_pol2il = torch.zeros(0, dtype=torch.int64)
            self._restricted_to_policy = torch.zeros(0, dtype=torch.int64)

        # ---- actuator seam:
        # explicit per-substep ctrl laws over pluggable actuators.  The
        # default laws ALWAYS exist (the retired inline _apply_pd body);
        # a declared env.params.actuators map replaces the loop's write
        # face and may switch it into the phase-split protocol.
        self._setup_actuator_seam(params)

        # ---- reset dispatch (scene bundles): the declared initializer
        # block resolves through INITIALIZER_REGISTRY (six-step
        # contract); the legacy ``reset`` block rides the compat sugar
        # and the GR1 robot-body stages keep their engine-side call
        # site below (unchanged order, shared stream -- determinism
        # red line).  See unigym/backends/shared/mjfamily/props/reset_face.py.
        (self._initializer, self._reset_spec) = wire_reset_face(self, params, robot_profile)

        # ---- compatibility aliases for the first articulation ----
        if self._articulation_order:
            first_art = self._articulation_order[0]
            self._foot_body_ids = list(first_art.foot_body_ids)
            self._foot_geom_sets = list(first_art.foot_geom_sets)
            self._base_body_id = first_art.addresses.get("base_body_id", -1)
        else:  # pragma: no cover - the run layer requires articulations
            self._foot_body_ids, self._foot_geom_sets = [], []
            self._base_body_id = -1
        self._frc6 = np.zeros(6)

        # ---- obs cache (exactly the adapter's requests) ----
        self._obs_requests = adapter.get_obs_requests()
        if self.contact_manager is not None:
            self.contact_manager.register_obs_requests({r.key for r in self._obs_requests})
        from unigym.env.obs_catalog import ObsCatalog

        entries: dict = {}
        requestable: set = set()
        self.obs_catalogs = {}
        for art in self._articulation_order:
            cat = build_obs_catalog(
                art.recipe,
                camera_manager=self.camera_manager,
                prop_manager=self.prop_manager,
                contact_manager=self.contact_manager,
                articulation_joint_names=list(art.joint_names),
                record_sites=(
                    tuple(name for name, _ in self._record_sites)
                    if art is self._articulation_order[0]
                    else None
                ),
                articulation=art.name,
            )
            art.catalog = cat
            self.obs_catalogs[art.name] = cat
            entries.update(cat.entries)
            requestable |= cat.requestable
        self.obs_catalog = ObsCatalog(entries, requestable)
        self._obs_requests = canonicalize_obs_requests(self._obs_requests)
        for req in self._obs_requests:
            self.obs_catalog.validate_request(req)
            if split_articulation_key(req.key)[1] in (
                "joint_pos_rel",
                "joint_vel",
                "applied_torque",
            ):
                self._joint_gather(req.joint_names, req)
        self.obs_cache: dict = {}

        # ---- episode bookkeeping (per-env) ----
        self._dur_lo, self._dur_hi = parse_episode_length(episode_length_s)
        self._episode_id = torch.zeros(n, dtype=torch.int64)
        self._time_in_episode = torch.zeros(n)
        self._episode_dur = sample_durations(self._dur_lo, self._dur_hi, n, self.device)

        # Cameras bind after the model/data exist (renderers, cam ids)
        if self.camera_manager is not None:
            self.camera_manager.bind_env(self)

        adapter.bind_env(self)
        adapter.on_reset(torch.arange(n))
        self.reset_all()
        self._rebuild_model_set()

    def _rebuild_model_set(self, meta=None) -> None:
        """Rebuild ``model_irs`` / ``model_set`` from the compiled IR.

        Runs once at construction and again after the run builder annotates
        the IR with ``meta["articulations"]``; the per-articulation views are
        filtered from the (annotated) compiled IR, so their row data is
        identical to the whole scene at every point.
        """
        from unigym.backends.shared.mjfamily.model.model_set import (
            build_model_set,
        )

        if len(self._articulation_order) == 1:
            self.model_set, self._model_ir = build_model_set(
                self._native_model_ir,
                single_name=self._articulation_order[0].name,
                meta=meta,
                name_map=self._mj_name_map,
            )
        else:
            allow_merge = not (any(p for p in self._mj_name_map.prefixes.values()))
            self.model_set, self._model_ir = build_model_set(
                self._native_model_ir,
                runtimes=self._articulation_order,
                meta=meta,
                name_map=self._mj_name_map,
                allow_merge=allow_merge,
            )
        self.model_irs = dict(self.model_set.articulations)

    # ------------------------------------------------------------------ #
    # ordering / contract properties
    # ------------------------------------------------------------------ #
    @property
    def perm_il2pol(self) -> torch.Tensor:
        """Gather index: ``data_il[:, perm_il2pol]`` -> policy joint order."""
        return self._perm_il2pol

    @property
    def perm_pol2il(self) -> torch.Tensor:
        """Gather index: ``data_pol[:, perm_pol2il]`` -> MJ hinge order."""
        return self._perm_pol2il

    # The camera rigs consume the raw simulator handles
    @property
    def mj_model(self):
        """The live ``mujoco.MjModel`` (camera rigs + director ops).

        backend-internal; neutral code must use ``BackendInfo.model_handle``
        (``env.mj_model`` is a backend handle, never a neutral surface).
        """
        return self._model

    @property
    def mj_data(self):
        """The live ``mujoco.MjData`` (rendering must NOT solve on it)."""
        return self._data

    @property
    def mj_datas(self):
        """All per-env ``MjData`` handles (index 0 is ``mj_data``)."""
        return self._datas

    def sync_writes(self, e: int = 0) -> None:
        """Push CPU-mirror + model-level director writes to the warp device.

        Warp steps consume the device ``Data``/``Model``; director ops edit
        the CPU mirror (qpos/qvel/mocap/eq_active) and the single ``MjModel``
        (articulation anchor body_pos/body_quat, weld eq_data).  The mjwarp
        director ops call this after every mutating op so the NEXT
        ``step`` observes the write.  N=1 scope (the trait table scopes
        director/mocap to ``max_envs=1``).
        """
        self._push_director_writes(int(e))

    @property
    def policy_joint_pos_limits(self):
        """Soft joint limits (policy order) from the model, or ``None``."""
        return self._joint_limits

    @property
    def joint_pos_limits_by_articulation(self):
        """``{articulation: (lower, upper)}`` for joint-limit terminations."""
        return getattr(self, "_joint_limits_by_articulation", None)

    def to_policy_order(self, x_il: torch.Tensor) -> torch.Tensor:
        """(N, len(il)) full-IL gather, or (N, J) restricted-IL -> policy.

        The ACTION interface is the adapter's restricted order (policy
        joints in simulator relative order — the isaaclab action-term
        convention, what ``CollectSink`` records as ``action_target``):
        ``x[..., _restricted_to_policy]`` maps it to policy order, the
        exact behaviour isaaclab's ``x[:, perm_il2pol]`` exhibits on that
        same input (its asset happens to keep the policy joints at indices
        0..J-1; the MJ hinge order interleaves the fingers, so the full-IL
        gather would be out of range AND semantically wrong there).
        """
        if self._control_order and x_il.shape[-1] == len(self._control_order[0].joint_names):
            return x_il[..., self._restricted_to_policy]
        return x_il[:, self._perm_il2pol]

    def to_il_order(self, x_pol: torch.Tensor) -> torch.Tensor:
        """Gather ``(N, ...)`` policy-order joint tensors into MJ hinge order."""
        return x_pol[:, self._perm_pol2il]

    # ------------------------------------------------------------------ #
    # obs cache
    # ------------------------------------------------------------------ #
    def _articulation_for_key(self, key):
        name, _ = split_articulation_key(key)
        if name is None:
            if len(self._articulation_order) != 1:
                raise ValueError(
                    f"mjwarp backend: obs key {key!r} does not name an "
                    f"articulation; this run has "
                    f"{sorted(self.articulations)}"
                )
            return self._articulation_order[0]
        try:
            return self.articulations[name]
        except KeyError:
            raise ValueError(
                f"mjwarp backend: obs key {key!r} names unknown "
                f"articulation {name!r}; declared: "
                f"{sorted(self.articulations)}"
            ) from None

    def _joint_gather(self, names, req=None) -> tuple:
        """Cached name->asset-index gather for one joint obs request."""
        art = (
            self._articulation_for_key(req.key) if req is not None else self._articulation_order[0]
        )
        key = (art.name, tuple(names))
        cached = self._joint_gathers.get(key)
        if cached is None:
            index = {n: i for i, n in enumerate(art.joint_names)}
            missing = sorted(set(names) - set(index))
            if missing:
                raise ValueError(
                    f"mjwarp backend: obs joint names {missing} are not "
                    f"asset joints of articulation {art.name!r}"
                )
            np_idx = np.asarray([index[n] for n in names], dtype=np.int64)
            cached = (np_idx, torch.tensor(np_idx, dtype=torch.int64))
            self._joint_gathers[key] = cached
        return cached

    def _joint_obs(self, req) -> torch.Tensor:
        """Serve a joint-vector request by name gather over the MJ hinges."""
        art = self._articulation_for_key(req.key)
        base = split_articulation_key(req.key)[1]
        np_idx, t_idx = self._joint_gather(req.joint_names, req)
        q0 = art.q0.to(dtype=torch.float32)
        rows = []
        for d in self._datas:
            if base == "joint_pos_rel":
                q = torch.tensor(d.qpos[art.jadr_art][np_idx], dtype=torch.float32)
                rows.append(q - q0[t_idx])
                continue
            if base == "applied_torque":
                rows.append(torch.tensor(d.qfrc_actuator[art.jdr_art][np_idx], dtype=torch.float32))
                continue
            rows.append(torch.tensor(d.qvel[art.jdr_art][np_idx], dtype=torch.float32))
        return torch.stack(rows, dim=0)

    def _refresh_obs_cache(self, raw_action=None, cmd=None) -> None:
        if len(self._articulation_order) > 1:
            refresh_articulation_obs(self)
        # Image keys come from the CameraManager — render + read at the
        # fixed point of the tick-read contract (docs/contracts/camera_contract.md §6):
        # the chase pose written at the PREVIOUS step tail is what this
        # render captures; sync_pose at THIS step tail writes the next one.
        if self.camera_manager is not None:
            for k, v in self.camera_manager.tick_and_read().items():
                self.obs_cache[k] = v
        # feat/contact-manager: adapter-requested contact/<name> keys are
        # injected every step; NOT requested -> never enters obs_cache,
        # recording unaffected
        if self.contact_manager is not None and self.contact_manager.obs_keys():
            for k, v in self.contact_manager.read().items():
                if k in self.contact_manager.obs_keys():
                    self.obs_cache[k] = v
        # Ops detail face: adapter-requested contact_ops/<name> keys
        # carry the per-contact records (force/normal/v_t) for the
        # decision package + slip criterion; backend without the detail
        # face -> keys stay absent (the adapter degrades honestly)
        if self.contact_manager is not None:
            for req in self._obs_requests:
                if req.key.startswith("contact_ops/"):
                    name = req.key.split("/", 1)[1]
                    if name in {s.name for s in self.contact_manager.specs}:
                        self.obs_cache[req.key] = self.contact_manager.read_detail(name)
        # feat/object-state: adapter-requested prop keys read live values
        # (no props -> the fetch loop zero-fills as before)
        if self.prop_backend is not None:
            requested = {r.key for r in self._obs_requests}
            for k, v in self.prop_backend.get_state(self).items():
                if k in requested:
                    self.obs_cache[k] = v
        fetch = collect_core.fetch_table(self, cmd, raw_action)
        for req in self._obs_requests:
            self._serve_obs_request(req, fetch)

    def _serve_obs_request(self, req, fetch) -> None:
        """Validate/serve ONE adapter observation request."""
        # camera-served keys (ImageObsRequest or depth alias): validate
        # by FULL SHAPE + dtype (the manager contract), not last-dim
        if self.camera_manager is not None and req.key in self.camera_manager.obs_keys():
            val = self.obs_cache.get(req.key)
            if val is not None:
                validate_image_frame(self.camera_manager.frame_request(req.key), val, self.num_envs)
            return
        # contact-served keys: validate last-dim against the request
        if self.contact_manager is not None and req.key in self.contact_manager.obs_keys():
            val = self.obs_cache.get(req.key)
            if val is not None and val.shape[-1] != req.dim:
                raise ValueError(f"obs '{req.key}' dim {val.shape[-1]} != requested " f"{req.dim}")
            return
        # Ops detail keys: dict-record lists served above -- no
        # tensor validation (the adapter consumes the records as-is)
        if req.key.startswith("contact_ops/"):
            return
        base = split_articulation_key(req.key)[1]
        if base in ("joint_pos_rel", "joint_vel", "applied_torque"):
            val = self._joint_obs(req)
        elif base not in fetch:
            # prop-provided key (injected above by the prop backend when
            # present) or zero-fill when no props configured -- isaaclab2/env.py
            # parity; the fetch table itself never carried prop keys
            v = self.obs_cache.get(req.key)
            if v is None:
                v = torch.zeros(self.num_envs, req.dim)
                self.obs_cache[req.key] = v
            elif v.shape[-1] != req.dim:
                raise ValueError(f"obs '{req.key}' dim {v.shape[-1]} != requested " f"{req.dim}")
            return
        else:
            val = fetch[base]()
        if val is None:
            return  # passthrough not provided this call; keep stale
        if val.shape[-1] != req.dim:
            raise ValueError(f"obs '{req.key}' dim {val.shape[-1]} != requested " f"{req.dim}")
        self.obs_cache[req.key] = val

    # ------------------------------------------------------------------ #
    # state readout (post-step, shared by obs cache + extra)
    # ------------------------------------------------------------------ #
    def _read_state(self, e: int = 0) -> dict:
        """Snapshot of the post-step sim state (numpy, policy order).

        Multi-articulation: the historical flat keys describe the FIRST binding;
        each binding's namespaced state is added by
        the shared articulation record helper.
        """
        return read_articulation_state(self, self._articulation_order[0], e)

    def _foot_contact_forces(self, d=None) -> np.ndarray:
        """Per-foot world-frame net contact force (F, 3), contact-filtered.

        Mirrors ``backends/mujoco/multienv.foot_contact_forces`` but reads
        the warp-synced mirror: contacts are matched by geom membership of
        the foot bodies (the sole collision boxes are unnamed), and
        ``mj_contactForce`` yields the contact-frame force whose frame rows
        rotate it to world.
        """
        m = self._model
        if d is None:
            d = self._data
        out = np.zeros((len(self._foot_geom_sets), 3))
        ncon = int(d.ncon)
        # array views (contact[i]/geom1 go stale after get_data_into reallocs
        # the mirror contact buffer under mujoco's cached struct aliases)
        con_geoms = np.asarray(d.contact.geom)[:ncon]
        frames = np.asarray(d.contact.frame)[:ncon].reshape(-1, 9)
        for i in range(ncon):
            g1, g2 = int(con_geoms[i, 0]), int(con_geoms[i, 1])
            for f, geoms in enumerate(self._foot_geom_sets):
                if g1 in geoms or g2 in geoms:
                    mujoco.mj_contactForce(m, d, i, self._frc6)
                    frame = frames[i].reshape(3, 3)  # rows: normal, t1, t2
                    out[f] += frame.T @ self._frc6[:3]
                    break
        return out

    def _build_extra(self, state: dict, e: int = 0) -> dict:
        """The recorded keys (per-articulation state namespaced)."""
        del state, e
        extra = collect_core.build_extra(self, None, include_state=False)
        for art in self._articulation_order:
            extra.update(articulation_extra(self, art))
        return extra

    # ------------------------------------------------------------------ #
    # stepping (software PD @ sim rate, targets held per policy step)
    # ------------------------------------------------------------------ #
    def _damping_stability_scale(
        self, d, dofs: np.ndarray, kp: np.ndarray, kd: np.ndarray
    ) -> np.ndarray:
        """Per-dof scale (<= 1) on the explicit damping term of the PD.

        The damping term ``-kd*dq`` is integrated explicitly at the sim
        timestep.  For a dof with effective inertia ``M`` (the mass-matrix
        diagonal, live via ``mj_fullM``) the velocity update carries the
        factor ``(1 - dt*kd/M)`` — past ``dt*kd/M > 1`` it rings (alternating
        sign, no monotone decay) and past 2 the actuator saturates into a
        period-2 limit cycle: the joint position keeps tracking (zero-mean
        ±jitter) while ``qvel`` aliases to a near-constant ±8 rad/s phantom
        at the policy sample phase, and the recorded
        ``joint_vel``/``applied_torque`` columns then violate the dq/dt and
        PD identities (the low-inertia wrist dofs under heavy yaml kd
        overrides were the observed case).  Where the damping term is the
        destabilizer AND the stiffness term is still resolvable
        (``dt*sqrt(kp/M) < 2``), the scale substitutes the implicit-Euler
        equivalent gain ``kd/(1 + dt*kd/M)`` (A-stable, same steady state);
        everywhere else the scale is 1.0 — dofs already inside the bound are
        bit-identical, and stiffness-unresolvable dofs (the finger group at
        the default gains) keep their legacy behavior untouched.
        """
        dt = self._sim_dt
        Mfull = np.zeros((self._model.nv, self._model.nv))
        try:
            mujoco.mj_fullM(self._model, d, Mfull)  # mujoco >= 3.2
        except TypeError:
            getattr(mujoco, "mj_fullM")(self._model, Mfull, d.qM)  # legacy layout
        M = np.maximum(Mfull[dofs, dofs], 1e-9)
        ratio = dt * np.asarray(kd, dtype=np.float64) / M
        resolvable = dt * np.sqrt(np.maximum(np.asarray(kp, dtype=np.float64), 0.0) / M) < 2.0
        return np.where((ratio > 1.0) & resolvable, 1.0 / (1.0 + ratio), 1.0)

    # ------------------------------------------------------------------ #
    # actuator seam
    # ------------------------------------------------------------------ #
    def _settle_substeps(self, n: int) -> None:
        """Zero-action settle loop.

        Exactly ``step()``'s per-substep physics with the targets PINNED
        to the stage-1 q0 (the audited ``composite_controller.reset()``
        goal semantics: the body holds its reset pose under gravity
        compensation, the hand servos hold 0, re-placed objects fall and
        settle).
        """
        d = self._active_data
        targets = {}
        for control in self._control_order:
            if control.prop:
                targets[control.name] = np.zeros((1, len(control.joint_names)), dtype=np.float64)
            else:
                targets[control.name] = (
                    control.q0.detach().cpu().numpy()[None, :].astype(np.float64)
                )
        if self.num_envs == 1:
            for _ in range(int(n)):
                if self._actuator_seam_declared:
                    for feed in self._actuator_feeds:
                        feed.act.set_targets(feed(targets, 0))
                    if self._actuator_phase_split:
                        self._warp_step1(d)
                    for act in self._actuators:
                        act.step(self._model, d)
                    if self._actuator_phase_split:
                        self._warp_step2(d)
                    else:
                        self._warp_step(d)
                else:
                    apply_control_substep(self, targets, 0, d)
                    self._warp_step(d)
            return
        # N>1: settle ONE world on a scratch single-world Data so the other
        # worlds stay frozen (a batched step here would advance them too).
        e = next(i for i, dd in enumerate(self._datas) if dd is d)
        scratch = self._new_warp_data(d, nworld=1)
        for _ in range(int(n)):
            if self._actuator_seam_declared:
                for feed in self._actuator_feeds:
                    feed.act.set_targets(feed(targets, 0))
                if self._actuator_phase_split:
                    self._warp_step1(d, 0, scratch)
                for act in self._actuators:
                    act.step(self._model, d)
                if self._actuator_phase_split:
                    self._warp_step2(d, 0, scratch)
                else:
                    self._warp_step(d, 0, scratch)
            else:
                apply_control_substep(self, targets, 0, d)
                self._warp_step(d, 0, scratch)
        # the settled world becomes the device truth for world e only
        self._push_mirror_to_device(d, e)

    def _prop_extents(self, name: str) -> tuple:
        """``(half_xy, height)`` of a box prop body.

        The z-stack inputs of the SplitA sampler; the prop origin is
        the bottom-face center.
        """
        import mujoco

        if self.prop_backend is None:
            raise RuntimeError("mjwarp backend: _prop_extents needs a wired prop backend")
        bid, _, _ = self.prop_backend.free_addresses(name)
        m = self._model
        half_xy, height = 0.0, 0.0
        for g in range(m.body_geomadr[bid], m.body_geomadr[bid] + m.body_geomnum[bid]):
            if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            half_xy = max(half_xy, float(max(m.geom_size[g][0], m.geom_size[g][1])))
            height = max(height, float(m.geom_size[g][2]) * 2)
        return half_xy, height

    def _apply_reset_protocol(self, e: int = 0) -> None:
        """The declared reset chain: placement apply -> robot stages.

        The initializer samples + writes the prop freejoints (its own
        stream); the name-neutrality check gates the sample against the
        declared props.  The GR1 robot-body stages follow in the
        unchanged engine-side order (settle -> elbow -> cotrain over the
        SAME stream: placement draws first, then the noise draws -- the
        determinism red line).  The single post-stage forward is the
        reset tail's own mj_forward (the settle already advanced the
        physics; the stage writes only need derived quantities
        refreshed).
        """
        prev = self._active_data
        self._active_data = self._datas[e]
        try:
            if self._initializer is None:
                raise RuntimeError(
                    "mjwarp backend: _apply_reset_protocol called without a " "declared initializer"
                )
            placements = self._initializer.apply()
            declared = (
                {s.name for s in self.prop_manager.specs}
                if self.prop_manager is not None
                else set()
            )
            unknown = sorted(set(placements) - declared)
            if unknown:
                raise ValueError(
                    f"mujoco backend: initializer sampled undeclared prop "
                    f"names {unknown} | declared props: {sorted(declared)}"
                )
            if self._reset_spec is not None:
                from unigym.robots.gr1_reset import apply_gr1_reset_stages

                apply_gr1_reset_stages(
                    self._model,
                    self._active_data,
                    self._initializer.rng,
                    self._reset_spec,
                    settle_fn=self._settle_substeps,
                    forward=False,
                )
        finally:
            self._active_data = prev

    def _setup_actuator_seam(self, params: dict) -> None:
        """Wire the ctrl laws for the decimation loop.

        ABSENT ``actuators`` key -> the legacy body: the two default
        standard_pd laws bound to the exact policy/hand slot tables;
        ``_apply_pd`` (the retired inline formula) keeps its signature and
        the default code path stays bit-identical.  A DECLARED map
        must cover every robot joint (policy + hands) exactly once
        (fail-fast gaps/duplicates/prop-slot collisions); a non-standard
        law in the map switches the loop into the
        ``mj_step1 -> actuator.step -> mj_step2`` phase split (computed
        -torque needs the CURRENT substep's M/qfrc_bias).
        """
        self._actuators = []
        self._actuator_feeds: list = []
        self._actuator_phase_split = False
        self._actuator_seam_declared = False

        cfg = params.get("actuators")
        if not cfg:
            self._actuator_default_path = True
            return
        if not isinstance(cfg, dict):
            raise ValueError(
                f"env.params.actuators must be a mapping of group name -> "
                f"group spec, got {type(cfg).__name__}"
            )
        acts, claimed = [], {}
        for gname, gcfg in cfg.items():
            if not isinstance(gcfg, dict):
                raise ValueError(
                    f"env.params.actuators.{gname} must be a group spec "
                    f"mapping, got {type(gcfg).__name__}"
                )
            act = build_actuator(gcfg)
            act.bind(self, gcfg)
            slot_list = act.bound_slots.tolist()
            dup = sorted(set(slot_list) & set(claimed))
            if dup:
                owners = sorted({claimed[s] for s in dup})
                raise ValueError(
                    f"env.params.actuators.{gname} re-addresses ctrl slots "
                    f"{dup} already owned by group(s) {owners} (each "
                    f"actuator slot must have exactly one ctrl law)"
                )
            prop_hit = sorted(set(slot_list) & set(np.nonzero(self._prop_ctrl_mask)[0].tolist()))
            if prop_hit:
                raise ValueError(
                    f"env.params.actuators.{gname} claims prop-servo ctrl "
                    f"slots {prop_hit} (props own their position-hold "
                    f"write face; actuator groups address ROBOT joints "
                    f"only)"
                )
            for s in slot_list:
                claimed[s] = gname
            acts.append(act)
        # The coverage universe is the ACTUATED robot joints
        # (policy + hands) -- a fixed-base scene carries unactuated device
        # hinges / baked freejoints that no ctrl law may or must address
        # (for G1 this is exactly the non-Prop hinge set, unchanged)
        controls = [c for c in self._control_order if not c.prop]
        robot_universe = [n for c in controls for n in c.joint_names]
        bound = [n for act in acts for n in act.joint_names]
        if sorted(bound) != sorted(robot_universe):
            missing = sorted(set(robot_universe) - set(bound))
            extra = sorted(set(bound) - set(robot_universe))
            raise ValueError(
                f"env.params.actuators must cover every robot joint "
                f"exactly once; uncovered: {missing}"
                + (f"; over-covered: {extra}" if extra else "")
                + f" | hint: group regexes fullmatch against the robot "
                f"hinge names, e.g. {robot_universe[:4]} ..."
            )
        owner = {}
        for control in controls:
            for index, name in enumerate(control.joint_names):
                owner[name] = (control.name, index)
        feeds = []
        for act in acts:
            feeds.append(_ActuatorTargetFeed(act, [owner[n] for n in act.joint_names]))
        self._actuators = acts
        self._actuator_feeds = feeds
        self._actuator_phase_split = any(not isinstance(a, StandardPdActuator) for a in acts)
        self._actuator_seam_declared = True
        self._actuator_default_path = False

    def _actuator_gain_fallback(self, joint_name: str, which: str):
        """Legacy gain-channel lookup for declared standard_pd groups."""
        for control in self._control_order:
            if control.prop or joint_name not in control.joint_names:
                continue
            gains = control.addresses.get(which)
            if gains is not None:
                return float(gains[control.joint_names.index(joint_name)])
        return None

    def step(
        self, actions: dict, raw_action: torch.Tensor | None = None, cmd: torch.Tensor | None = None
    ) -> tuple[dict, dict]:
        """Advance one policy tick: decimation x mj_step, PD every sim step.

        ``actions`` is a ``dict[str, NamedAction]`` (the adapter's
        ``postprocess_action`` output); entry joint names resolve by name to
        policy-order PD targets (the "g1" body face).  Read AFTER the
        physics as in the isaaclab backend (``step -> refresh obs/extra``).
        The ctrl write face is the actuator seam (``_setup_actuator_seam``):
        default = the legacy explicit-PD body; a declared
        ``env.params.actuators`` map with a non-standard law runs the
        mj_step1/step2 phase split.
        """
        if self._closed or self._closing:
            raise RuntimeError("mjwarp backend: environment is closed")
        targets = route_actions(self, actions, label="mjwarp backend")
        for e, d in enumerate(self._datas):
            # prop position servos: ctrl IS the position target ;
            # written once per policy step, held across the decimation loop
            d.ctrl[self._prop_ctrl_mask] = self._prop_targets[e][self._prop_ctrl_mask]
        if self._actuator_seam_declared:
            # declared ctrl laws (fresh-forward): targets fed once
            # per control step (held across the substeps); laws recompute
            # per substep.  Non-standard laws run the phase split
            # mj_step1 -> actuator.step -> mj_step2: mj_step alone would
            # hand a computed-torque law LAST substep's M/qfrc_bias.
            for e, d in enumerate(self._datas):
                for feed in self._actuator_feeds:
                    feed.act.set_targets(feed(targets, e))
            for _ in range(self._decimation):
                if self._actuator_phase_split:
                    self._warp_step1_all()
                for e, d in enumerate(self._datas):
                    for act in self._actuators:
                        act.step(self._model, d)
                    if not self._actuator_phase_split:
                        self._warp_put_ctrl(d, e)
                if self._actuator_phase_split:
                    self._warp_step2_all()
                else:
                    self._warp_step_all()
        else:
            for _ in range(self._decimation):
                for e, d in enumerate(self._datas):
                    apply_control_substep(self, targets, e, d)
                    self._warp_put_ctrl(d, e)
                self._warp_step_all()
        self._time_in_episode += self.step_dt
        state = self._read_state()
        self._refresh_obs_cache(raw_action, cmd)
        extra = self._build_extra(state)
        due = self._time_in_episode >= self._episode_dur
        extra["reset_env_ids"] = torch.nonzero(due).squeeze(-1).to(torch.int64)
        extra["should_reset"] = bool(due.any())
        if self.camera_manager is not None:
            # step-tail posing (tick-read contract): the NEXT render (inside
            # the next obs refresh) captures this pose
            self.camera_manager.sync_pose()
        return dict(self.obs_cache), extra

    # ------------------------------------------------------------------ #
    # reset (per-env subset)
    # ------------------------------------------------------------------ #
    def reset_envs(self, env_ids) -> torch.Tensor:
        """Full reset chain for the given ids (given order preserved).

        Args:
            env_ids: id subset to reset (empty -> no-op; ids outside
                ``[0, num_envs)`` fail fast).

        Returns:
            The reset ids (int64).
        """
        if self._closed or self._closing:
            raise RuntimeError("mjwarp backend: environment is closed")
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids)
        if env_ids.numel() == 0:
            return env_ids.to(torch.int64)
        env_ids = env_ids.to(torch.int64)
        if bool(((env_ids < 0) | (env_ids >= self.num_envs)).any()):
            raise ValueError(
                f"mjwarp backend: num_envs={self.num_envs}, cannot reset "
                f"env ids {env_ids.tolist()}"
            )
        m = self._model
        # batched device reset for the subset (Warp resets only masked
        # worlds; the mirror writes + per-world push below are authoritative)
        mask = np.zeros(self.num_envs, dtype=bool)
        mask[env_ids.detach().cpu().numpy()] = True
        mjw.reset_data(
            self._wm,
            self._wdata,
            reset=wp.array(mask, dtype=wp.bool, device=self._wdata.qpos.device),
        )
        for e in env_ids.tolist():
            d = self._datas[e]
            mujoco.mj_resetData(m, d)  # time/ctrl/xfrc/contacts cleared
            reset_articulations(self, d)
            d.qvel[:] = 0.0
            d.ctrl[:] = 0.0
            if self.prop_backend is not None:
                # Parity: re-place props BEFORE the forward (reset chain
                # mirrors isaaclab2/env.py: robot reset -> props placement ->
                # forward)
                self.prop_backend.on_reset([e], self)
            if self._initializer is not None:
                # scene-bundle initializer: placement apply -> the GR1
                # robot-body stages (unchanged order), all inside one
                # shared stream; the tail forward below covers the writes.
                # The settle chain advances the WARP world, so the device
                # must first receive the just-written mirror state.
                self._push_mirror_to_device(d, e)
                self._apply_reset_protocol(e)
            if self._boost_body_id >= 0:
                d.xfrc_applied[self._boost_body_id][2] += self._boost_fz
            # Derived read quantities on the mirror, then re-seed THIS device
            # world only (the reset's authoritative state is the mirror).
            mujoco.mj_forward(m, d)
            self._push_mirror_to_device(d, e)
        if self.prop_backend is not None:
            # servo targets -> 0 for the RESET subset only (other envs keep
            # their held position-servo targets)
            rows = env_ids.detach().cpu().numpy()[:, None]
            cols = np.flatnonzero(self._prop_ctrl_mask)[None, :]
            self._prop_targets[rows, cols] = 0.0
        self._episode_id[env_ids] += 1
        self._time_in_episode[env_ids] = 0.0
        self._episode_dur[env_ids] = sample_durations(
            self._dur_lo, self._dur_hi, env_ids.numel(), self.device
        )
        if self.camera_manager is not None:
            # re-pose the rigs to the spawn state BEFORE the refresh, so the
            # post-reset render/read sees the spawn view (pose first, read after)
            self.camera_manager.on_reset(env_ids)
        self._refresh_obs_cache()  # cache reflects the spawn state
        return env_ids

    def reset_all(self) -> torch.Tensor:
        """Reset every env (``reset_envs(arange(num_envs))``)."""
        if self._closed or self._closing:
            raise RuntimeError("mjwarp backend: environment is closed")
        return self.reset_envs(torch.arange(self.num_envs, dtype=torch.int64))

    # ------------------------------------------------------------------ #
    # snapshot / restore (Round B B1 rewind)
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict:
        """Mirror-faithful engine snapshot for the rewind face.

        Payload = ``mjSTATE_INTEGRATION`` full bit set read from the CPU
        mirror (which is synced after every warp step).  This is NOT
        CPU-bitwise: the warp device is rebuilt from the mirror on
        ``restore`` (``mj_setState`` -> ``mj_forward`` -> re-pin ->
        ``mjw.put_data``), so the rewind replays deterministically but not
        against a CPU ``mj_step`` reference.  ``tick`` is None (the rewind
        controller stamps the engine tick); ``episode_id`` deliberately
        stays out of the payload (process truth, never rewinds).
        """
        m = self._model
        if self.num_envs == 1:
            self._warp_sync_mirror(self._datas[0])
        else:
            self._sync_all_mirrors()
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        size = mujoco.mj_stateSize(m, spec)
        bits = {
            1: "time",
            2: "qpos",
            4: "qvel",
            8: "act",
            16: "history",
            32: "warmstart",
            64: "ctrl",
            128: "qfrc_applied",
            256: "xfrc_applied",
            512: "eq_active",
            1024: "mocap_pos",
            2048: "mocap_quat",
            4096: "userdata",
            8192: "plugin",
        }
        # warp's next step seeds from the previous qacc (not part of
        # mjSTATE): carry it so a restored timeline is bit-reproducible.
        warp_state = {}
        for name in ("qacc", "qacc_smooth"):
            array = getattr(self._wdata, name, None)
            if array is not None:
                warp_state[name] = array.numpy().copy()
        if self.num_envs == 1:
            d = self._data
            buf = np.zeros(size, dtype=np.float64)
            mujoco.mj_getState(m, d, buf, spec)
            return {
                "state": buf,
                "tick": None,
                "warp": warp_state,
                "meta": {
                    "spec": "mjSTATE_INTEGRATION",
                    "spec_bits": spec,
                    "bits": [n for b, n in sorted(bits.items()) if spec & b],
                    "mj_time": float(d.time),
                },
                # backend bookkeeping OUTSIDE mjSTATE: the episode clock
                # and the sampled episode duration are sim truth and roll
                # back with the rewind (episode_id is process truth and
                # deliberately stays out -- counter dichotomy)
                "extra": {
                    "time_in_episode": self._time_in_episode.clone(),
                    "episode_dur": self._episode_dur.clone(),
                },
            }
        bufs = []
        for d in self._datas:
            buf = np.zeros(size, dtype=np.float64)
            mujoco.mj_getState(m, d, buf, spec)
            bufs.append(buf)
        return {
            "states": bufs,
            "tick": None,
            "warp": warp_state,
            "meta": {
                "spec": "mjSTATE_INTEGRATION",
                "spec_bits": spec,
                "bits": [n for b, n in sorted(bits.items()) if spec & b],
                "mj_time": float(self._datas[0].time),
            },
            "extra": {
                "time_in_episode": self._time_in_episode.clone(),
                "episode_dur": self._episode_dur.clone(),
            },
        }

    def restore(self, state) -> int:
        """Restore a :meth:`snapshot` payload mirror-faithfully; 0 tick.

        ``mj_setState`` (all bits, warmstart included) -> ``mj_forward``
        (derived read quantities) -> ``mj_setState`` again (re-pin the
        warm-start the forward overwrote) -> ``mjw.put_data`` re-seeds the
        warp world from the restored mirror.  The episode bookkeeping rides
        the payload's ``extra`` (sim truth) -- ``episode_id`` never rewinds.
        """
        m = self._model
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        if self.num_envs == 1:
            d = self._data
            buf = np.ascontiguousarray(state["state"], dtype=np.float64)
            if buf.shape != (mujoco.mj_stateSize(m, spec),):
                raise ValueError(
                    f"mjwarp backend: snapshot state shape {buf.shape} does "
                    f"not match mjSTATE_INTEGRATION size "
                    f"{mujoco.mj_stateSize(m, spec)} (model changed since the "
                    f"snapshot?)"
                )
            mujoco.mj_setState(m, d, buf, spec)
            mujoco.mj_forward(m, d)
            mujoco.mj_setState(m, d, buf, spec)  # re-pin the warm-start
            self._push_mirror_to_device(d)
            # refresh warp's DERIVED fields (xpos/xmat/...) from the restored
            # qpos with warp's own kernels, then re-pin the continuation
            # arrays (forward touches qacc/warmstart).
            mjw.forward(self._wm, self._wdata)
            self._restore_warp_state(state.get("warp") or {})
            self._copy_warmstart(d)
            self._warp_sync_mirror(d)
        else:
            bufs = state.get("states")
            if not isinstance(bufs, (list, tuple)) or len(bufs) != self.num_envs:
                raise ValueError(
                    f"mjwarp backend: multi-env snapshot carries "
                    f"{len(bufs) if isinstance(bufs, (list, tuple)) else 0} "
                    f"states, need {self.num_envs}"
                )
            want = mujoco.mj_stateSize(m, spec)
            for e, (d, raw) in enumerate(zip(self._datas, bufs)):
                buf = np.ascontiguousarray(raw, dtype=np.float64)
                if buf.shape != (want,):
                    raise ValueError(
                        f"mjwarp backend: snapshot state shape {buf.shape} "
                        f"does not match mjSTATE_INTEGRATION size {want} "
                        f"(model changed since the snapshot?)"
                    )
                mujoco.mj_setState(m, d, buf, spec)
                mujoco.mj_forward(m, d)
                mujoco.mj_setState(m, d, buf, spec)  # re-pin the warm-start
                self._push_mirror_to_device(d, e)
            # refresh every restored world's derived fields with warp's own
            # kernels, then re-pin the continuation arrays + warm-start.
            mjw.forward(self._wm, self._wdata)
            self._restore_warp_state(state.get("warp") or {})
            for e, d in enumerate(self._datas):
                self._copy_warmstart(d, e)
            self._sync_all_mirrors()
        extra = state.get("extra") or {}
        if "time_in_episode" in extra:
            self._time_in_episode = extra["time_in_episode"].clone()
        if "episode_dur" in extra:
            self._episode_dur = extra["episode_dur"].clone()
        return 0

    # ------------------------------------------------------------------ #
    # teardown
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Release sim state + per-rig render resources (idempotent).

        Teardown is retryable: ``_closing`` guards the window between the
        first manager release and the final handle invalidation.  A manager
        exception makes the backend unusable but retryable: managers may have
        partially released, the handles stay live, and ``step``/``reset_*``
        keep rejecting until a later ``close()`` finishes the teardown.
        """
        if self._closed:
            return
        self._closing = True
        if self.camera_manager is not None:
            self.camera_manager.close()
        if self.prop_backend is not None:
            self.prop_backend.close()
        # Teardown boundary: the live handles are deliberately invalidated.
        # They are annotated non-Optional above so the live step/obs paths
        # never carry a spurious Optional; writing through ``__dict__``
        # records the released state without re-typing the class.
        for handle in ("_wdata", "_wm", "_data", "_model", "_datas", "_active_data"):
            self.__dict__[handle] = None
        # Mark closed only after every manager released successfully: a
        # manager failure leaves the backend retryable, not half-closed.
        self._closed = True
        self._closing = False
