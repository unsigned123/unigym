"""MuJoCo camera backend: MjSpec injection + convention bridge + rigs.

This is the mujoco counterpart of ``cameras/isaaclab_backend.py`` + the
scene-side halves of ``cameras/rigs.py``: it injects the declared camera
rigs into the MJ model (``<camera>`` elements via ``MjSpec`` before
compilation), re-poses chase cameras per step, and renders rgb/depth frames
through ``mujoco.Renderer``.  Only this module (plus ``env/backends/
mujoco_env.py``) may import mujoco in the cameras package; the module is
import-safe without mujoco (lazy import inside functions).

Scope: N=1 (the mujoco backend is single-env; every frame is ``(1, ...)``).

⚠️ Camera convention bridge (the module's core hazard)
=====================================================
MuJoCo cameras look along their local **-Z** with local **+Y** up; every
rig parameter in this repo follows the Isaac convention (forward **+X**,
up **+Z** — see ``cameras/rigs.lookat_quat``).  Every injected/written
camera orientation is therefore the Isaac-convention quaternion composed
with the FIXED bridge rotation::

    q_bridge (w,x,y,z) = (0.5, 0.5, -0.5, -0.5)
    # 120 deg about (1,-1,-1)/sqrt(3): R(q_bridge) @ (-Z) = +X (forward)
    # and R(q_bridge) @ (+Y) = +Z (up)
    q_mj = q_isaac ⊗ q_bridge     (Hamilton product, MuJoCo wxyz)

so the MJ camera faces exactly where the Isaac-convention camera would.
This is NOT verified by "it renders something": the rendered-pixel
orientation tests (MJ-C2*, known asymmetric scene) lock the forward /
left-right / up-down axes empirically.

Depth semantics
===============
``Renderer.enable_depth_rendering()`` yields metric **z-depth** (distance
along the camera optical axis to the image plane), NOT the ray distance of
the isaaclab ``distance_to_camera`` annotator.  On-axis the two coincide;
off-axis ``ray = z / cos(alpha)`` (alpha = pixel angle off axis).  The
calibration tests (MJ-C4/MJ-C5) lock the vertical case (reading == known
distance) and record the oblique numbers in docs/contracts/camera_contract.md.
Per the key-space contract, pixels beyond ``far_clip`` are masked to
``inf`` ("nothing observed").

Frame ownership
===============
``render()`` output is ALWAYS ``.copy()``-ed before wrapping (the render-buffer
aliasing lesson applies to any recycled render buffer); locked by the
manager-level two-tick ownership test (MJ-C6).

GL backend
==========
``MUJOCO_GL=egl`` must be set before the first mujoco import (done by
``env/backends/mujoco_env.py`` via ``os.environ.setdefault``; verified
on the humanoid_1 host at ~0.5 ms/frame @224²).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - typing-only (mujoco imported lazily)
    import mujoco

from unigym.cameras import geomflags
from unigym.cameras.appearance import appearance_plan
from unigym.cameras.base import CameraSpec, depth_key, rgb_key, rig_hide_geom_groups
from unigym.cameras.state import check_single_env, make_state_table
from unigym.cameras.base import rig_type
from unigym.cameras.decor import DecorContext, run_generators
from unigym.cameras.rigs import DEFAULT_CHASE_OFFSET, lookat_quat, yaw_pitch_roll_quat

# injected MJCF camera names are namespaced to avoid collisions with
# cameras shipped inside the robot/scene XML (scene_43dof has "tracking"
# and "head_camera" already).
from unigym.cameras.spec import NAMED_VIEW_PARAMS, DEFAULT_ORBIT, spherical_camera_keys

from unigym.backends.shared.mjfamily.cameras.helpers import (  # noqa: F401
    DEFAULT_BODY_MOUNT,
    MJ_CAMERA_PREFIX,
    _quat_conj,
    _quat_mul_np,
    _rig_offset_quat,
    bridge_quat,
    fovy_from_spec,
    inject_spec_camera,
    mj_camera_quat,
    quat_to_rotmat,
    rotmat_to_quat,
)


# --------------------------------------------------------------------------- #
# rigs (mujoco flavour of cameras/rigs.py)
# --------------------------------------------------------------------------- #
class _MjRigBase:
    """Shared MJ rig plumbing: inject, bind, render-on-read.

    Pipeline: inject (pre-compile) -> bind + Renderer
    (post-construction) -> render-on-read with owned frames.

    Live camera properties (whitelist): ``pose_offset`` (mount
    frame), ``yaw_deg``/``pitch_deg``/``roll_deg`` (mount-frame
    orientation offset; roll about the view axis) and ``target_env`` are
    folded into ``model.cam_pos/cam_quat`` on
    ``apply_properties`` (immediate: the next render already sees them).
    N=1: env_ids must be None/[0] and target_env must stay 0.
    """

    def __init__(self, spec: CameraSpec, render_groups=None):
        self.spec = spec
        self.name = spec.name
        self._mj_name = MJ_CAMERA_PREFIX + spec.name
        self._env = None
        self._renderer = None
        self._cam_id = None
        # the plain-render MjvOption (visual group ON); built at bind by
        # the reclassification step -- None before bind_env
        self._plain_opt = None
        # capture-flags geom-class override (camera_contract §14): None =
        # the contype/conaffinity audit rule; a dict = per-config group
        # declaration (validated at classify time)
        self.render_groups = render_groups
        # the publish-path live render state (§8.3):
        # the COMPLETE RenderSpec when non-default, ``None`` = the plain
        # publish path (the default view is bitwise the historical
        # render).  Written ONLY by the manager's apply_properties
        # ("render" key, full-write semantics); the capture face never
        # reads it (stateless).
        self.render_spec = None
        self.render_source = None
        rig = spec.rig or {}
        self.target_env = int(rig.get("target_env", 0))
        # per-camera MJ geom-group visibility override (mujoco
        # only; the GR1 tabletop scene renders like the omniforge reference
        # with group 0 hidden -- robosuite render_collision_mesh=False
        # parity).  Opt-in only: absent -> (), the historical plain option
        # is untouched (G1 default camera behavior stays bitwise).
        self._hide_groups = rig_hide_geom_groups(rig, spec.name)
        self._live_offset = np.zeros(3)
        self._live_yaw = 0.0  # rad
        self._live_pitch = 0.0  # rad
        self._live_roll = 0.0  # rad (view-axis roll)
        # single-world pose state (mujoco has exactly ONE world;
        # the table's shared None row is the authoritative pose -- see
        # ``_apply_live_pose`` / ``get_world_pose``).
        self.state_table = make_state_table(self.name, "shared")
        # base mount (body/world): the pre-offset body-local/world pose
        self._mount_pos = np.array(
            [float(x) for x in rig.get("pos", (0.0, 0.0, 0.0))], dtype=np.float64
        )
        self._mount_quat = _rig_offset_quat(spec)  # isaac convention (4,)
        # the render-side mjData copy (lazily allocated at first render;
        # refreshed per render via mj_copyData -- see _render_copy())
        self._render_buf = None

    # Lazily-bound MJ handles (None before ``bind_env`` / set to None by
    # ``close``); every live path resolves them through the ``_require_*``
    # helpers so the Optional never leaks into the render code.
    _env: Any
    _renderer: mujoco.Renderer | None
    _cam_id: int | None
    _plain_opt: mujoco.MjvOption | None
    _render_buf: mujoco.MjData | None

    def _require_env(self) -> Any:
        if self._env is None:
            raise RuntimeError(f"mujoco camera {self.name!r}: not bound " "(bind_env)")
        return self._env

    def _require_renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            raise RuntimeError(f"mujoco camera {self.name!r}: no renderer " "(bind_env)")
        return self._renderer

    def _require_cam_id(self) -> int:
        if self._cam_id is None:
            raise RuntimeError(f"mujoco camera {self.name!r}: no camera id " "(bind_env)")
        return self._cam_id

    # -- lifecycle -------------------------------------------------------- #
    def setup_scene(self, mjspec) -> None:
        """Inject the ``<camera>`` element into the MjSpec (pre-compile)."""
        inject_spec_camera(self.spec, mjspec)

    def bind_env(self, env) -> None:
        """Resolve the compiled camera id and create this rig's Renderer.

        Also applies the ONE-TIME geom-class reclassification (§8.1).
        """
        import mujoco  # lazy

        from unigym.backends.shared.mjfamily.model.ir_access import camera_id, model_ir_of

        self._env = env
        cid = camera_id(model_ir_of(env), self._mj_name)
        if cid < 0:
            raise RuntimeError(
                f"mujoco camera '{self._mj_name}' missing from the compiled "
                f"model (injection lost?)"
            )
        self._cam_id = cid
        self._renderer = mujoco.Renderer(
            env.mj_model, height=int(self.spec.height), width=int(self.spec.width)
        )
        self._reclassify_groups()

    def _reclassify_groups(self) -> None:
        """Load-time reclassification (flags v2, §8.1).

        Visual-only geoms
        move PERMANENTLY into the reserved visual group; the render face
        never touches ``geom_group`` again (a pure scene_option switch per
        capture).  Idempotent: every rig binding the same shared model
        recomputes the identical layout from contype/conaffinity, which
        the reclassification does not modify.

        ⚠️ MuJoCo's default MjvOption runs with geom groups 3-5 DISABLED
        (``[1, 1, 1, 0, 0, 0]``) -- after the reclassification every
        render path MUST carry an explicit option that enables the visual
        group (:meth:`_plain_option`), or plain renders would silently
        drop every visual-only geom (obs/recorder pixels included).
        """
        m = self._require_env().mj_model
        m.geom_group[:] = geomflags.reclassify_geom_groups(
            m.geom_contype, m.geom_conaffinity, m.geom_group, self.render_groups
        )
        self._plain_opt = self._build_plain_option()

    def _build_plain_option(self):
        """The plain-render MjvOption: default groups PLUS the visual group ON.

        Bitwise-identical pixels to the historical option-less render
        for any model that keeps the reserved group 5 empty and no
        ``hide_geom_groups`` declared -- which the reclassification
        guarantees post-bind.
        """
        import mujoco  # lazy

        opt = mujoco.MjvOption()
        for g in self._hide_groups:
            opt.geomgroup[g] = 0
        opt.geomgroup[geomflags.VISUAL_GROUP] = 1
        return opt

    def close(self) -> None:
        """Free the GL context (no-op when never bound)."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -- posing ------------------------------------------------------------ #
    def sync_pose(self, env_ids=None) -> None:
        """Per-step-tail hook.

        body rigs ride the mount body and world rigs
        never move, so only chase re-poses (override below).
        """
        check_single_env(self.name, env_ids)

    def on_reset(self, env_ids) -> None:
        """Post-reset hook: default = re-pose (no-op for body/world)."""
        self.sync_pose(env_ids=env_ids)

    # -- live properties  ------------------------------------------ #
    def apply_properties(
        self,
        pose_offset=None,
        yaw_deg=None,
        pitch_deg=None,
        roll_deg=None,
        target_env=None,
        env_ids: list[int] | None = None,
    ):
        """Update the live pose properties (manager pre-validates keys)."""
        if env_ids is not None and list(env_ids) != [0]:
            raise ValueError(
                f"camera '{self.name}': mujoco is single-env; env_ids must "
                f"be None or [0], got {env_ids}"
            )
        if target_env is not None:
            if int(target_env) != 0:
                raise ValueError(
                    f"camera '{self.name}': mujoco is single-env; target_env "
                    f"must be 0, got {target_env}"
                )
            self.target_env = 0
        if pose_offset is not None:
            off = np.asarray([float(x) for x in pose_offset], dtype=np.float64)
            if off.shape != (3,):
                raise ValueError(
                    f"camera '{self.name}': pose_offset must be " f"[dx, dy, dz], got {list(off)}"
                )
            self._live_offset = off
        if yaw_deg is not None:
            self._live_yaw = math.radians(float(yaw_deg))
        if pitch_deg is not None:
            self._live_pitch = math.radians(float(pitch_deg))
        if roll_deg is not None:
            self._live_roll = math.radians(float(roll_deg))
        self._apply_live_pose()

    # -- transient world pose (ux-7; world rigs only) ----------------------- #
    def begin_world_pose(self, pos, quat_wxyz):
        raise NotImplementedError(
            f"camera '{self.name}': transient world pose is supported on "
            f"world rigs only (this rig: {rig_type(self.spec)!r})"
        )

    def end_world_pose(self, saved) -> None:
        raise NotImplementedError(
            f"camera '{self.name}': transient world pose is supported on " f"world rigs only"
        )

    def set_world_pose(self, pos, quat_wxyz):
        raise NotImplementedError(
            f"camera '{self.name}': world-pose set is supported on "
            f"world rigs only (this rig: {rig_type(self.spec)!r})"
        )

    def get_world_pose(self):
        """Read-back: the rig's CURRENT world pose (Isaac convention wxyz).

        body mounts read ``data.cam_xpos/cam_xmat`` (FK-composed at the
        live state); world/chase read the state-table row (C2).  The
        engine-side resolution lives in ``camera_pose.read_world_pose``.
        """
        from .camera_pose import read_world_pose

        return read_world_pose(self)

    def _apply_live_pose(self) -> None:
        """Write the state table, then model.cam_pos/cam_quat from it.

        Base mount ∘ live offsets resolves the single-world pose; the
        table is the authoritative row (shared None row) and the model
        slots are written from ``state_table.get`` -- no pose scalar.
        """
        if self._env is None or self._cam_id is None:
            return
        q = self._mount_quat
        if self._live_yaw or self._live_pitch or self._live_roll:
            q = _quat_mul_np(
                yaw_pitch_roll_quat(self._live_yaw, self._live_pitch, self._live_roll), q
            )
        self.state_table.set(
            self.name, None, self._mount_pos + self._live_offset, np.asarray(q, dtype=np.float64)
        )
        row = self.state_table.get(self.name)
        if row is None:
            raise RuntimeError(f"camera {self.name!r}: pose table row missing after write")
        m = self._env.mj_model
        m.cam_pos[self._cam_id] = row.pos
        m.cam_quat[self._cam_id] = mj_camera_quat(row.quat)

    # -- rendering ---------------------------------------------------------- #
    def _render_copy(self):
        """A PRIVATE mjData copy to render from (the physics-neutrality rule).

        Solving on the LIVE data rewrites ``qacc`` and
        ``qacc_warmstart`` (a redundant solve takes a different
        iterative path -> sub-rounding deltas), and the next
        ``mj_step`` consumes the poisoned warmstart -- contact chaos
        amplifies from there, so the perturbation is a deterministic
        function of the render schedule (same schedule replays bitwise,
        any camera-dependent schedule drifts the trajectory).  Here the
        copy is allocated once per rig and refreshed per render
        (``mj_copyData`` into the pre-allocated buffer -- microseconds,
        noise next to an EGL render); ``mj_forward`` then runs on the
        copy and the live mjData stays a pure observer, so rendering is
        physically neutral: any camera schedule produces the exact
        no-camera trajectory.  Regression gate:
        ``tests/test_camera_neutral.py``.
        """
        import mujoco  # lazy

        env = self._require_env()
        if self._render_buf is None:
            self._render_buf = mujoco.MjData(env.mj_model)
        mujoco.mj_copyData(self._render_buf, env.mj_model, env.mj_data)
        return self._render_buf

    def _update_scene(self) -> None:
        """Refresh mj-derived state on the PRIVATE data copy.

        Then stages this camera's scene.
        ``update_scene`` needs current derived quantities (xpos/xmat);
        the env runs ``mj_forward`` before ``tick_and_read`` -- but be
        defensive and make the render self-sufficient when called
        directly.  The refresh runs on :meth:`_render_copy` (a copy):
        solving on the live data would rewrite its warmstart and leak
        into the next physics step.  The staged option is
        :meth:`_build_plain_option` (visual group ON -- the
        reclassification moved visual-only geoms there; the default
        option would silently hide them).
        """
        import mujoco  # lazy

        d2 = self._render_copy()
        mujoco.mj_forward(self._require_env().mj_model, d2)
        self._require_renderer().update_scene(
            d2,
            camera=self._require_cam_id(),
            scene_option=(
                self._plain_opt if self._plain_opt is not None else self._build_plain_option()
            ),
        )

    def _read_rgb(self) -> torch.Tensor:
        """(1, H, W, 3) uint8 — OWNED copy (aliasing rule)."""
        renderer = self._require_renderer()
        renderer.disable_depth_rendering()
        self._update_scene()
        arr = np.asarray(renderer.render(), dtype=np.uint8).copy()
        return torch.from_numpy(arr).unsqueeze(0)

    def _read_depth(self) -> torch.Tensor:
        """(1, 1, H, W) float32 metric z-depth; >far_clip masked to inf."""
        renderer = self._require_renderer()
        renderer.enable_depth_rendering()
        self._update_scene()
        arr = np.asarray(renderer.render(), dtype=np.float32).copy()
        renderer.disable_depth_rendering()
        arr = arr.reshape(int(self.spec.height), int(self.spec.width))
        arr[arr > float(self.spec.far_clip)] = np.inf  # "nothing observed"
        return torch.from_numpy(np.ascontiguousarray(arr)).reshape(1, 1, arr.shape[0], arr.shape[1])

    def read(self, env_ids=None) -> dict:
        """Render + read this rig's streams: ``{key: tensor}``.

        Never None:
        the MJ render is synchronous, placeholders cannot occur.
        ``env_ids`` only scopes the refresh (N=1: None/[0]); the returned
        block stays ``(1, ·)``.
        """
        check_single_env(self.name, env_ids)
        out = {}
        for key in self.spec.keys():
            if key == rgb_key(self.name):
                out[key] = self._read_rgb()
            elif key == depth_key(self.name):
                out[key] = self._read_depth()
            else:  # pragma: no cover - CameraSpec.keys() is closed
                raise KeyError(key)
        return out

    # -- RenderSpec capture (camera_contract §14.9) --------------------- #
    def _build_decor_context(self, d) -> DecorContext:
        """Engine-side context for the decor generators.

        The mujoco calls
        live HERE -- generators stay engine-free: active contacts with
        frame normals and linear forces (mj_contactForce).  Reads the
        render-side COPY handed in by :meth:`capture_render` so the
        decor geometry and the frame pixels describe the same solved
        state (and the live mjData is never touched).
        """
        import mujoco  # lazy

        m = self._require_env().mj_model
        n = int(d.ncon)
        pos = np.zeros((n, 3))
        normal = np.zeros((n, 3))
        force = np.zeros((n, 3))
        geoms = np.zeros((n, 2), dtype=int)
        f6 = np.zeros(6)
        # array views: contact[i] aliases go stale after a contact-buffer
        # realloc (mjwarp's get_data_into) while the arrays re-read live
        all_pos = np.asarray(d.contact.pos)[:n]
        all_frames = np.asarray(d.contact.frame)[:n].reshape(-1, 9)
        all_geoms = np.asarray(d.contact.geom)[:n]
        for i in range(n):
            pos[i] = all_pos[i]
            normal[i] = all_frames[i].reshape(3, 3)[0]
            mujoco.mj_contactForce(m, d, i, f6)
            force[i] = f6[:3]
            geoms[i] = (int(all_geoms[i, 0]), int(all_geoms[i, 1]))
        return DecorContext(
            model=m,
            data=d,
            contact_pos=pos,
            contact_normal=normal,
            contact_force=force,
            contact_geoms=geoms,
            body_names=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)],
            geom_bodyid=np.asarray(m.geom_bodyid),
            geom_contype=np.asarray(m.geom_contype),
            geom_conaffinity=np.asarray(m.geom_conaffinity),
            box_type=int(mujoco.mjtGeom.mjGEOM_BOX),
            mesh_type=int(mujoco.mjtGeom.mjGEOM_MESH),
        )

    def _capture_option(self, spec):
        """Per-render MjvOption for the spec path: explicit ALWAYS.

        The
        default option runs with groups 3-5 disabled -- §14.2; the
        scene.geom_class switch mirrors the flags face's collision mode
        (hide the reserved visual group); decor option toggles
        (contact_mjvis) apply here.
        """
        import mujoco  # lazy

        opt = mujoco.MjvOption()
        for g in self._hide_groups:
            opt.geomgroup[g] = 0
        hide_visual = spec["scene"].get("geom_class") == "collision"
        opt.geomgroup[geomflags.VISUAL_GROUP] = 0 if hide_visual else 1
        for name, on in spec["_opt_flags"].items():
            opt.flags[int(getattr(mujoco.mjtVisFlag, name))] = bool(on)
        return opt

    @staticmethod
    def _apply_scene_flags(scn, spec) -> None:
        """scn.flags overrides (probe5 recipe).

        None = engine default
        (view ``visual`` stays bitwise); a bool = explicit per-render
        override.  Never a model write.
        """
        for key, flag in (
            ("shadow", "mjRND_SHADOW"),
            ("reflection", "mjRND_REFLECTION"),
            ("skybox", "mjRND_SKYBOX"),
        ):
            val = spec["scene"].get(key)
            if val is not None:
                import mujoco  # lazy

                scn.flags[int(getattr(mujoco.mjtRndFlag, flag))] = val

    @staticmethod
    def _resolve_lookat(camc: dict, ctx: DecorContext) -> np.ndarray:
        t = camc["target"]
        if t == "contact_centroid":
            return ctx.centroid(camc["contact_select"])
        return np.asarray(t, dtype=np.float64)

    def _free_camera(self, lookat, dist, az, el):
        import mujoco  # lazy

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self._require_env().mj_model, cam)
        cam.lookat[:] = lookat
        cam.distance = float(dist)
        cam.azimuth = float(az)
        cam.elevation = float(el)
        return cam

    def _paint_scene(self, plan) -> None:
        """Scene-buffer recolor (L1.5).

        Overwrites scn.geoms rows for
        MODEL geoms per the appearance plan (probe1(f) route --
        bitwise-equal to load-time material stripping).  Zero model
        writes.
        """
        import mujoco  # lazy

        obj_geom = int(mujoco.mjtObj.mjOBJ_GEOM)
        rgba_tab, mat_tab = plan["rgba"], plan["matid"]
        painted = plan["painted"]
        scn = self._require_renderer().scene
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.objtype != obj_geom or g.objid < 0:
                continue
            oid = int(g.objid)
            if not painted[oid]:
                continue
            g.matid = int(mat_tab[oid])
            g.texid = -1
            g.rgba[:] = np.asarray(rgba_tab[oid], dtype=np.float32)

    def _inject_decor(self, descs) -> int:
        """Append decor geoms into the CURRENT scene (single pass).

        Follows
        the probe5 scrub discipline: untouched scn slots carry garbage
        (measured camdist=nan / matid=21951) and initGeom writes only
        size/pos/mat/rgba -- every field mjr_render may consult is set
        explicitly (camdist=nan poisons the transparent sort).
        """
        import mujoco  # lazy

        scn = self._require_renderer().scene
        start = scn.ngeom
        n = 0
        for k, dsc in enumerate(descs):
            if start + k >= scn.maxgeom:
                break
            g = scn.geoms[start + k]
            gtype = getattr(mujoco.mjtGeom, "mjGEOM_" + str(dsc["type"]).upper())
            rgba = np.asarray(dsc["rgba"], dtype=np.float32)
            mujoco.mjv_initGeom(g, gtype, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
            mujoco.mjv_connector(
                g,
                gtype,
                float(dsc["radius"]),
                np.asarray(dsc["from"], dtype=np.float64),
                np.asarray(dsc["to"], dtype=np.float64),
            )
            g.rgba[:] = rgba
            for attr, val in (
                ("objtype", int(mujoco.mjtObj.mjOBJ_UNKNOWN)),
                ("objid", -1),
                ("category", int(mujoco.mjtCatBit.mjCAT_DECOR)),
                ("matid", -1),
                ("texid", -1),
                ("segid", -1),
                ("dataid", -1),
                ("camdist", 0.5),
                ("modelrbound", 0.0),
                ("emission", 0.0),
                ("specular", 0.0),
                ("shininess", 0.0),
                ("reflectance", 0.0),
                ("transparent", 0),
            ):
                try:
                    setattr(g, attr, val)
                except (AttributeError, TypeError):
                    pass
            n += 1
        scn.ngeom = start + n
        return n

    def _render_one(self, data, option, spec, plan, descs, cam=None) -> np.ndarray:
        """THE single-pass frame: update -> flags -> paint -> inject -> render.

        ONE mjr_render (GL depth occludes natively; probe5).
        """
        renderer = self._require_renderer()
        if cam is None:
            renderer.update_scene(data, camera=self._require_cam_id(), scene_option=option)
        else:
            renderer.update_scene(data, camera=cam, scene_option=option)
        scn = renderer.scene
        self._apply_scene_flags(scn, spec)
        if plan is not None:
            self._paint_scene(plan)
        if descs:
            self._inject_decor(descs)
        return np.asarray(renderer.render()).copy()

    @staticmethod
    def _caption(img, title, sub=None) -> np.ndarray:
        """Burn a PIL caption (probe5 style).

        output.caption opts in --
        the plain-equivalent default never touches the pixels.
        """
        from PIL import Image, ImageDraw, ImageFont

        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
            small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        except Exception:
            font = small = ImageFont.load_default()
        dr.text(
            (10, 8), title, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0)
        )
        if sub:
            dr.text(
                (10, 30),
                sub,
                font=small,
                fill=(205, 220, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
        return np.asarray(im)

    def capture_render(self, spec: dict) -> dict:
        """Single-pass RenderSpec capture (design §6 inner layer).

        ``spec`` is the COMPLETE normalized spec (manager ran
        ``spec.resolve_render`` -- view name or inline spec).
        Pipeline per frame: ``mjv_updateScene -> scn.flags -> appearance
        paint (L1.5) -> decor inject (scrubbed) -> ONE mjr_render``.
        ZERO model writes (orbit/closeup ride an MjvCamera, paint rides
        the scene buffer) and ZERO live-mjData writes: the solve runs on
        :meth:`_render_copy`'s private copy -- rendering is physically
        neutral.  Returns ``{rgb, frames?, gif?, seg?,
        legend?, effective_render}`` -- ``frames`` = named/closeup
        non-primary stills, ``gif`` = PIL GIF bytes for orbit.
        """
        import mujoco  # lazy

        m = self._require_env().mj_model
        renderer = self._require_renderer()
        d = self._render_copy()
        mujoco.mj_forward(m, d)
        spec = dict(spec)
        need_ctx = (
            bool(spec["decor"])
            or spec["camera"]["mode"] in ("orbit", "closeup")
            or any(k in spec["camera"] for k in ("d", "el", "az"))
        )
        ctx = self._build_decor_context(d) if need_ctx else None
        if spec["decor"]:
            assert ctx is not None
            descs, opt_flags = run_generators(spec["decor"], ctx)
        else:
            descs, opt_flags = [], {}
        spec["_opt_flags"] = opt_flags
        option = self._capture_option(spec)
        plan = None
        if spec["appearance"]:
            plan = appearance_plan(
                spec["appearance"],
                [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)],
                m.geom_bodyid,
                m.geom_contype,
                m.geom_conaffinity,
                m.geom_group,
                self.render_groups,
            )
        camc = spec["camera"]
        if "pos" in camc or "lookat" in camc:
            # the absolute-pose free-camera form (camera_contract
            # §14 -- the camera(aim) semantics) -- converted to the
            # spherical keys the render loop consumes BEFORE anything
            # reads them (exact round-trip: el = asin(-vz/d),
            # az = atan2(-vy, -vx), d = |v|, v = pos - lookat).  Never a
            # mutation of the caller's dict: the effective_render echo
            # keeps the declared form.
            if "pos" not in camc or "lookat" not in camc:
                raise ValueError(
                    "camera: pos and lookat go TOGETHER (world metres "
                    "-- the absolute free-camera pose)"
                )
            camc = dict(camc)
            camc.update(spherical_camera_keys(camc.pop("pos"), camc.pop("lookat")))
        out: dict = {}
        az0 = float(camc.get("az", 0.0))
        label = spec.get("_label", "spec")
        out_cfg = spec["output"]

        def frame_title(extra: str) -> str:
            return f"[{label}] {camc['mode']}" + (f" {extra}" if extra else "")

        if camc["mode"] == "fixed" and not any(k in camc for k in ("d", "el", "az")):
            # historical rig-camera path (view ``visual`` == the plain
            # capture bitwise: same camera, same option discipline)
            img = self._render_one(d, option, spec, plan, descs)
            if out_cfg["caption"]:
                img = self._caption(img, frame_title(""))
            out["rgb"] = img.astype(np.uint8)
        else:
            if ctx is None:
                raise RuntimeError("camera: decor context missing for the " "requested capture")
            lookat = self._resolve_lookat(camc, ctx)
            frames: list[tuple[str, np.ndarray]] = []

            def az_frame(az: float, el: float, dist: float) -> np.ndarray:
                return self._render_one(
                    d, option, spec, plan, descs, cam=self._free_camera(lookat, dist, az, el)
                )

            if camc["mode"] == "fixed":
                frames.append(
                    (
                        "fixed",
                        az_frame(camc.get("az", 0.0), camc.get("el", -45.0), camc.get("d", 0.5)),
                    )
                )
            else:
                orbit = camc.get("orbit", dict(DEFAULT_ORBIT))
                n, step = int(orbit["n"]), float(orbit["step_deg"])
                orbit_frames = []
                for i in range(n):
                    az = az0 + i * step
                    img = az_frame(az, camc.get("el", -32.0), camc.get("d", 0.20))
                    if out_cfg["caption"]:
                        img = self._caption(
                            img,
                            frame_title(
                                f"az={az % 360:.0f} ({i + 1}/{n})"
                                f" d={camc.get('d', 0.2):.2f}"
                                f" el={camc.get('el', -32):.0f}"
                            ),
                        )
                    orbit_frames.append(img)
                out["rgb"] = orbit_frames[0].astype(np.uint8)
                if out_cfg["gif"]:
                    out["gif"] = self._pack_gif(orbit_frames, out_cfg["gif_ms"])
                if camc["mode"] == "closeup":
                    frames = [
                        (f"az{int(round(az0 + i * step)) % 360}", f)
                        for i, f in enumerate(orbit_frames)
                    ]
            for name in camc.get("views", ()):
                p = NAMED_VIEW_PARAMS[name]
                img = az_frame(az0 + p["az_off"], p["el"], p["d"])
                if out_cfg["caption"]:
                    img = self._caption(
                        img, frame_title(f"{name} d={p['d']:.2f}" f" el={p['el']:.0f}")
                    )
                frames.append((name, img))
            if frames:
                primary = out.get("rgb")
                extra = {}
                for k, v in frames:
                    v8 = v.astype(np.uint8)
                    if primary is None:
                        out["rgb"] = primary = v8
                    elif not np.array_equal(v8, primary):
                        extra[k] = v8
                if extra:
                    out["frames"] = extra
        if "rgb" not in out:  # pragma: no cover - every branch assigns
            raise RuntimeError("render produced no primary frame")
        out["rgb"] = out["rgb"].astype(np.uint8)

        # seg buffer (primary frame only, model-only scene: decor is
        # segid=-1 -> honest background) + legend
        if "seg" in spec["buffers"]:
            renderer.enable_segmentation_rendering()
            renderer.update_scene(d, camera=self._require_cam_id(), scene_option=option)
            seg_raw = np.asarray(renderer.render()).copy()
            renderer.disable_segmentation_rendering()
            seg = np.asarray(seg_raw, dtype=np.int64)
            miss = seg[..., 0] < 0
            ids = np.where(miss, -1, seg[..., 0])
            types = seg[..., 1]
            out["seg"] = geomflags.colorize_seg(ids)
            if out_cfg["legend"]:
                out["legend"] = self._legend(
                    ids, types, spec["scene"].get("geom_class", "all"), contacts=bool(opt_flags)
                )
        return out

    @staticmethod
    def _pack_gif(frames, ms: int) -> bytes:
        """Orbit frames -> GIF bytes (probe4 assembly).

        PIL save_all,
        loop forever, optimized.
        """
        import io

        from PIL import Image

        ims = [Image.fromarray(f) for f in frames]
        buf = io.BytesIO()
        ims[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=ims[1:],
            duration=int(ms),
            loop=0,
            optimize=True,
        )
        return buf.getvalue()

    def _legend(self, ids: np.ndarray, types: np.ndarray, mode: str, contacts: bool) -> dict:
        """id->name table for the hit pixels -> the legend document."""
        import mujoco  # lazy

        m = self._env.mj_model
        hits = ids[ids >= 0]
        entries: dict[int, dict] = {}
        for oid in np.unique(hits):
            oid = int(oid)
            otype = int(types[ids == oid][0])
            label = geomflags.legend_type_name(otype)
            name = mujoco.mj_id2name(m, otype, oid)
            body = None
            if otype == int(mujoco.mjtObj.mjOBJ_GEOM):
                bid = int(m.geom_bodyid[oid])
                body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
            rows = int((ids == oid).sum())
            color = geomflags.seg_palette(oid + 1)[oid].tolist()
            entries[oid] = {
                "id": oid,
                "type": label,
                "name": name or f"{label}_{oid}",
                "body": body,
                "color": color,
                "pixels": rows,
            }
        return geomflags.build_legend(
            list(entries.values()), {"geom": mode, "contacts": contacts, "segmentation": True}
        )


class MjBodyRig(_MjRigBase):
    """Base-body mount: the MJCF camera rides the mount body.

    The render
    is the exact bolted-on first-person view at the render-time state.

    NOTE vs isaaclab: the isaaclab standard Camera is a scene-level prim
    re-posed at the step tail, so its frame lags one policy step; a body-
    attached MJ camera has NO lag (documented in camera_contract.md).
    """


class MjWorldRig(_MjRigBase):
    """World-anchored worldbody tripod.

    Injected once, never re-posed
    by the physics loop (live properties + the transient/freecam pose
    faces re-pose it on demand).
    """

    # -- transient world pose (ux-7) ---------------------------------------- #
    def begin_world_pose(self, pos, quat_wxyz):
        """Overwrite the model camera slots with the forensic pose.

        The
        saved token is the raw (cam_pos, cam_quat) copies -- restore is
        bitwise (no convention round-trip).
        """
        if self._env is None or self._cam_id is None:
            raise RuntimeError(f"camera '{self.name}': not bound (bind_env)")
        m = self._env.mj_model
        saved = (m.cam_pos[self._cam_id].copy(), m.cam_quat[self._cam_id].copy())
        self.state_table.set(
            self.name,
            None,
            np.asarray(pos, dtype=np.float64),
            np.asarray(quat_wxyz, dtype=np.float64),
        )
        row = self.state_table.get(self.name)
        if row is None:
            raise RuntimeError(f"camera {self.name!r}: pose table row missing after write")
        m.cam_pos[self._cam_id] = row.pos
        m.cam_quat[self._cam_id] = mj_camera_quat(row.quat)
        return saved

    def end_world_pose(self, saved) -> None:
        """Restore the saved (cam_pos, cam_quat) model slots bitwise."""
        m = self._env.mj_model
        m.cam_pos[self._cam_id], m.cam_quat[self._cam_id] = saved
        # the table read-back is the world-convention mirror of the slots
        q_isaac = _quat_mul_np(np.asarray(saved[1], dtype=np.float64), _quat_conj(bridge_quat()))
        self.state_table.set(
            self.name, None, np.asarray(saved[0], dtype=np.float64).copy(), q_isaac
        )

    def set_world_pose(self, pos, quat_wxyz, env_idx: int | None = None):
        """Persistent absolute world pose (freecam face).

        The
        exact :meth:`begin_world_pose` write minus the save/restore
        protocol -- model slots stick until the next pose write (a
        later offset-keyed ``apply_properties`` recomputes from the
        mount, last write wins).  mujoco is single-env: ``env_idx``
        must be None or 0.
        """
        if env_idx not in (None, 0):
            raise ValueError(
                f"camera '{self.name}': mujoco is single-env; env_idx "
                f"must be None or 0, got {env_idx}"
            )
        if self._env is None or self._cam_id is None:
            raise RuntimeError(f"camera '{self.name}': not bound (bind_env)")
        self.state_table.set(
            self.name,
            None,
            np.asarray(pos, dtype=np.float64),
            np.asarray(quat_wxyz, dtype=np.float64),
        )
        row = self.state_table.get(self.name)
        if row is None:
            raise RuntimeError(f"camera {self.name!r}: pose table row missing after write")
        m = self._env.mj_model
        m.cam_pos[self._cam_id] = row.pos
        m.cam_quat[self._cam_id] = mj_camera_quat(row.quat)


class MjChaseRig(_MjRigBase):
    """World-frame chase mount: per-step write of ``model.cam_pos/cam_quat``.

    Exact port of the isaaclab ``ChaseRig`` math (robot-YAW-rotated offsets
    around the root + ``lookat_quat`` aiming, target_env semantics) with the
    convention bridge applied to the written quaternion.  N=1: target_env
    must be 0.
    """

    def __init__(self, spec: CameraSpec, render_groups=None):
        super().__init__(spec, render_groups=render_groups)
        rig = spec.rig or {}
        pos = rig.get("pos", DEFAULT_CHASE_OFFSET["pos"])
        lookat = rig.get("lookat", DEFAULT_CHASE_OFFSET["lookat"])
        self._pos_off = np.asarray([float(x) for x in pos], dtype=np.float64)
        self._look_off = np.asarray([float(x) for x in lookat], dtype=np.float64)

    def _apply_live_pose(self) -> None:
        # chase recomputes its world pose from the root every sync; just
        # re-run it (writes model.cam_pos/cam_quat from _pos_off + offsets)
        self.sync_pose()

    def bind_env(self, env) -> None:
        """Bind + the single-env gate (chase is N=1 by construction)."""
        super().bind_env(env)
        if self.target_env != 0 or int(env.num_envs) != 1:
            raise ValueError(
                f"chase camera '{self.name}': mujoco backend is single-env "
                f"(num_envs={env.num_envs}); target_env must be 0, got "
                f"{self.target_env}"
            )

    def sync_pose(self, env_ids=None) -> None:
        """Re-pose the chase camera from the root (per-step tail).

        World-frame chase write of ``model.cam_pos/cam_quat``: robot-YAW-
        rotated offsets around the root + ``lookat_quat`` aiming.
        """
        check_single_env(self.name, env_ids)
        if self._env is None or self._cam_id is None:
            return
        env = self._env
        if not getattr(env, "_has_freejoint", True):
            # fixed base -- qpos[0:7] belong to the robot's
            # own hinges here (GR1 waist/arms), NOT a base frame; reading
            # them was stale handoff garbage.  The base frame IS
            # the configured spawn pose (construction-time constants,
            # backends/mujoco/env.py) and it never moves.
            root = env._fixed_base_pos_w
            w, x, y, z = env._fixed_base_quat_w
        else:
            d = env.mj_data
            root = d.qpos[0:3]
            w, x, y, z = d.qpos[3:7]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, s = math.cos(yaw), math.sin(yaw)

        def _rot_xy(off: np.ndarray) -> np.ndarray:
            return np.array([off[0] * c - off[1] * s, off[0] * s + off[1] * c, off[2]])

        pos = root + _rot_xy(self._pos_off + self._live_offset)
        lookat = root + _rot_xy(self._look_off)
        q_isaac = lookat_quat(pos, lookat)
        if self._live_yaw or self._live_pitch or self._live_roll:
            q_isaac = _quat_mul_np(
                yaw_pitch_roll_quat(self._live_yaw, self._live_pitch, self._live_roll), q_isaac
            )
        self.state_table.set(
            self.name,
            None,
            np.asarray(pos, dtype=np.float64),
            np.asarray(q_isaac, dtype=np.float64),
        )
        row = self.state_table.get(self.name)
        if row is None:
            raise RuntimeError(f"camera {self.name!r}: pose table row missing after write")
        m = self._env.mj_model
        m.cam_pos[self._cam_id] = row.pos
        m.cam_quat[self._cam_id] = mj_camera_quat(row.quat)


def build_mj_rig(spec: CameraSpec, render_groups=None) -> _MjRigBase:
    """Factory: mujoco rig instance from the spec's ``rig.type``.

    ``render_groups``: optional capture-flags geom-class override
    (camera_contract §14; passed through to the rig base).
    """
    rtype = rig_type(spec)
    if rtype == "body":
        return MjBodyRig(spec, render_groups=render_groups)
    if rtype == "world":
        return MjWorldRig(spec, render_groups=render_groups)
    if rtype == "fixed":
        raise ValueError(
            f"camera {spec.name!r}: rig type 'fixed' was renamed 'world' "
            f"(camera unification) -- update the config; there is no alias"
        )
    if rtype == "chase":
        return MjChaseRig(spec, render_groups=render_groups)
    raise ValueError(f"camera {spec.name!r}: unknown rig type {rtype!r} " "(body|world|chase)")
