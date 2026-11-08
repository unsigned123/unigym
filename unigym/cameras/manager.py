"""CameraManager: multi-rig combination with request-driven instantiation.

Responsibilities (design doc §1.1, owner rulings ①②③):

* **Key space** — ``cam/<name>/<rgb|depth>`` obs keys plus the optional depth
  aliases (``depth_image`` / ``depth_image_raw`` sugar shim, docs/
  camera_contract.md §depth migration).
* **Request-driven instantiation** — of the yaml-declared cameras, those
  actually consumed (adapter image requests or recorder ``camera_ref``)
  are instantiated; unknown requested keys fail with the list of
  available ones.  mujoco binds the FULL declared set instead (injected
  cameras cost nothing until rendered) and splits the obs tick: only
  requested rigs render per tick, the rest stay addressable by the
  camera faces (declared-camera availability, camera unification).
* **Raw-value caching** — uint8 RGB / metric depth (owner ruling ②);
  normalization stays with the adapter except the depth-alias
  ``depth_image`` alias which reuses ``env/depth.py::preprocess_depth``
  verbatim.
* **Placeholder fallback** — before the first successful render the
  annotator output is degenerate; the manager serves an all-inf frame
  (depth) / all-zero frame (RGB) instead (generalization of the classic
  ``_raw_depth_frame`` all-far mechanism), warning ONCE per key.
* **Render scheduling** — per-camera wall-clock fps (v069; tier defaults
  by name, declared intervals fold to ``target_fps = policy_hz/N``);
  between due times the last cached frame is served
  (stale-until-next-read).
* **Viewer / live-property API**  — ``get_view(name, env_id)`` is
  the single-frame primitive (isaaclab: most recent sensor output, no extra
  render; mujoco: on-demand render); ``apply_properties(name, params)``
  applies the live whitelist (pose_offset / yaw_deg / pitch_deg / roll_deg /
  target_env / fps / interval-deprecated) — everything else raises
  ValueError "requires restart".  Both are instance-aware: shared cameras
  address only their ``target_env`` instance; per_env cameras slice by
  ``env_id``.  The obs path stays batched (N, ·) — these APIs never touch
  the obs cache.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from contextlib import contextmanager

import numpy as np
import torch

from unigym.adapters.base import ImageObsRequest
from unigym.backends.registry import BackendSelectionError, backend_info
from unigym.env.depth import (DEPTH_CROP_COLS, DEPTH_CROP_ROWS,
                                    preprocess_depth)

from .base import (DEPTH_ALIAS_KEYS, apply_rate_keys, CameraRig,
                   CameraSpec, depth_key, effective_target_fps,
                   fold_declared_intervals, render_due, rgb_key)
from .capabilities import check_backend as check_camera_capabilities
from .rigs import build_rig, lookat_quat
from .state import apply_declared_state, read_env_ids, state_is_shared
from .spec import complete_spec, resolve_render

_DEFAULT_RENDER = complete_spec({})
"""The default COMPLETE spec (== the resolved ``{"view": "visual"}``):
the plain publish path by construction (X2: bitwise-identical to the
historical flagless render).  A rig stores ``None`` for this state, so
the publish/WYSIWYG paths never re-render through the capture face for
it."""

# the ``depth_image`` alias reproduces the TSC 106x60 -> 58x87 crop
# geometry; any other shim resolution would silently mis-crop.
_PREPROCESS_SIZE = (60, 106)

POSE_INLINE_RIG_TYPES = ("world",)
"""ux-7 (owner-narrowed S2): rigs eligible for the transient pose-inline
snapshot.  ``world`` only -- body/chase rigs are re-posed from the robot
at every step, so a world-pose override there would be misleading (the
free debug fly camera is a world rig by the console convention)."""

LIVE_PROPERTY_KEYS = ("fps", "interval", "pitch_deg", "pose_offset",
                      "render", "roll_deg", "target_env", "yaw_deg")
"""camera_set_properties live whitelist (owner ruling): pose offset,
yaw-pitch, target_env, interval -- plus the v054 ``render`` live key
(the publish loop's ONLY render state: ``{"view": name}`` or a RenderSpec
fragment, resolved to a COMPLETE spec on write -- full-write semantics)
and the v069 ``fps`` wall-rate key (target_fps; the wall scheduler's live
knob; ``interval`` is the deprecated v069 alias -- v070 removes it).
EVERY other property (resolution, kind, types, hfov, depth_type, clips,
...) is construction-frozen and must go through a restart."""


def validate_image_frame(req: ImageObsRequest, val: torch.Tensor,
                         num_envs: int) -> None:
    """Shape/dtype validation for one camera frame (offline-testable).

    Raises ``ValueError`` unless ``val`` is ``(num_envs, *req.shape)`` with
    exactly ``req.dtype`` — the ImageObsRequest counterpart of the
    RawObsRequest last-dim check.
    """
    expected = (num_envs, *req.shape)
    if tuple(val.shape) != expected:
        raise ValueError(
            f"camera obs '{req.key}' shape {tuple(val.shape)} != expected "
            f"{expected}")
    if val.dtype != req.dtype:
        raise ValueError(
            f"camera obs '{req.key}' dtype {val.dtype} != expected "
            f"{req.dtype}")


class CameraManager:
    """World-component manager for the declared camera rigs.

    Lifecycle mirrors the prop manager: ``setup_scene`` (before
    env construction) -> ``bind_env`` (after) -> per-step ``sync_pose`` at
    the step tail + ``tick_and_read`` inside the obs-cache refresh ->
    ``on_reset`` after each reset chain.
    """

    def __init__(self, specs: list[CameraSpec], rigs: list[CameraRig],
                 aliases: dict[str, tuple[str, bool]] | None = None,
                 obs_served: set[str] | None = None,
                 policy_hz: float | None = None):
        self.specs = list(specs)
        self.rigs = list(rigs)
        # the policy control rate (1 / adapter.control_dt) -- the v069
        # wall-clock render scheduler's clock reference: declared
        # intervals fold to target_fps = policy_hz / N and tier defaults
        # need it.  None (probe/test managers) keeps the legacy per-tick
        # counter semantics (camera_contract §16.1).
        self.policy_hz = (float(policy_hz) if policy_hz else None)
        # alias key -> (manager key, preprocess?)
        self.aliases: dict[str, tuple[str, bool]] = dict(aliases or {})
        # obs-served rig split (declared-camera availability vs the obs
        # tick): a rig whose keys were REQUESTED (adapter obs / recorder
        # / depth aliases) rides ``tick_and_read``; the other declared
        # rigs stay addressable by the camera faces (get_view / capture
        # / properties) but never render on the obs tick.  ``None`` =
        # every rig is obs-served (the pre-split behavior).
        served = None if obs_served is None else set(obs_served)
        if served is not None:
            served |= {mkey for mkey, _ in self.aliases.values()}
        self._obs_rigs = [r for r in self.rigs
                          if served is None
                          or bool(set(r.spec.keys()) & served)]
        self._counter = 0
        self._last: dict[str, torch.Tensor] = {}
        self._warned: set[str] = set()
        # v069 wall scheduler: last render wall-clock (monotonic) per rig
        # name -- the tick_and_read due check
        self._last_read: dict[str, float] = {}
        self.num_envs: int | None = None
        self.device: str | torch.device | None = None
        self._requests: dict[str, ImageObsRequest] = {}
        for spec in self.specs:
            for key in spec.keys():
                self._requests[key] = ImageObsRequest(
                    key, spec.frame_shape(key), spec.frame_dtype(key),
                    "camera", spec.name)
        for alias, (mkey, pre) in self.aliases.items():
            spec = self._spec_serving(mkey)
            shape = spec.frame_shape(mkey)
            if pre:
                # preprocess_depth crops rows/cols -> shrink the locked shape
                shape = (1,
                         shape[1] + DEPTH_CROP_ROWS[1] - DEPTH_CROP_ROWS[0],
                         shape[2] + DEPTH_CROP_COLS[1] - DEPTH_CROP_COLS[0])
            self._requests[alias] = ImageObsRequest(
                alias, shape, torch.float32, "camera", spec.name)

    # ------------------------------------------------------------------ #
    # key space
    # ------------------------------------------------------------------ #
    @property
    def names(self) -> list[str]:
        """Instantiated rig names."""
        return [s.name for s in self.specs]

    def obs_keys(self) -> set:
        """All obs-cache keys served (``cam/*`` + depth aliases)."""
        return set(self._requests)

    def camera_keys(self) -> set:
        """Manager-native keys only (``cam/<name>/<type>``)."""
        return {k for k in self._requests if k.startswith("cam/")}

    def frame_request(self, key: str) -> ImageObsRequest | None:
        """The ImageObsRequest describing one served key (None if absent)."""
        return self._requests.get(key)

    def _spec_serving(self, key: str) -> CameraSpec:
        for spec in self.specs:
            if key in spec.keys():
                return spec
        raise KeyError(f"no instantiated camera serves {key!r} "
                       f"(cameras: {self.names})")

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def setup_scene(self, scene_cfg) -> None:
        """Add every rig's sensor cfg to the scene cfg (pre-construction)."""
        for rig in self.rigs:
            rig.setup_scene(scene_cfg)

    def bind_env(self, env) -> None:
        """Bind every rig to the constructed env (grabs sensor handles).

        v095 C2: validate ``env_poses`` against the REAL ``num_envs`` and
        apply the declared pose table (single pose -> shared row;
        ``env_poses`` -> per-instance rows) right after binding.
        """
        self.num_envs = int(env.num_envs)
        self.device = env.device
        for spec, rig in zip(self.specs, self.rigs):
            rig.bind_env(env)
            apply_declared_state(spec, rig, self.num_envs)

    def sync_pose(self) -> None:
        """Per-policy-step tail hook: re-pose body/chase rigs (fixed: no-op).

        The pose written here is captured by the NEXT step's render inside
        ``sim.step`` (the classic depth-camera re-pose semantics).
        """
        for rig in self.rigs:
            rig.sync_pose()

    def on_reset(self, env_ids) -> None:
        """Post-reset hook: re-pose the rigs to the spawn state."""
        for rig in self.rigs:
            rig.on_reset(env_ids)

    def close(self) -> None:
        """Release per-rig render resources (mujoco renderers; isaaclab
        sensors die with the kit)."""
        for rig in self.rigs:
            close = getattr(rig, "close", None)
            if callable(close):
                close()

    # ------------------------------------------------------------------ #
    # frame serving
    # ------------------------------------------------------------------ #
    def tick_and_read(self) -> dict[str, torch.Tensor]:
        """Advance the step counter and return ``{obs_key: frame}``.

        Frames of cameras whose render schedule is not due are served from
        the last read (stale); pre-render placeholders are substituted per
        the module docstring.  Every returned tensor is validated against
        its :class:`ImageObsRequest` (shape + dtype) — fail-fast on
        annotator surprises.  v069: the due check is WALL-CLOCK per
        camera (§16.1); no policy clock keeps the legacy counter bitwise.
        """
        if self.num_envs is None:
            raise RuntimeError("CameraManager.tick_and_read before bind_env")
        self._counter += 1
        now = time.monotonic()
        for rig in self._obs_rigs:
            keys = rig.spec.keys()
            if not self._read_due(rig, now) \
                    and all(k in self._last for k in keys):
                continue
            frames = rig.read(env_ids=read_env_ids(rig.spec, rig))
            self._last_read[rig.spec.name] = now
            for key in keys:
                val = frames.get(key)
                if val is None:
                    val = self._placeholder(key)
                    if key not in self._warned:
                        self._warned.add(key)
                        print(f"[cameras] placeholder frame for '{key}': "
                              f"annotator output not ready -> "
                              f"{'all-inf' if val.dtype == torch.float32 else 'all-zero'} "
                              f"fallback", flush=True)
                validate_image_frame(self._requests[key], val, self.num_envs)
                self._last[key] = val
        out = dict(self._last)
        for alias, (mkey, pre) in self.aliases.items():
            out[alias] = preprocess_depth(self._last[mkey]) if pre \
                else self._last[mkey]
        return out

    def _read_due(self, rig: CameraRig, now: float) -> bool:
        """Wall-clock due for one rig (§16.1; pure logic in base)."""
        return render_due(rig.spec, self.policy_hz, self._counter,
                          self._last_read.get(rig.spec.name), now)


    def effective_fps(self, name: str) -> float | None:
        """One camera's effective wall render fps (§16.1); None = the
        caller keeps legacy per-tick pacing.  Unknown cameras raise."""
        spec, _ = self._spec_and_rig(name)
        return effective_target_fps(spec, self.policy_hz)

    def video_rgb(self, name: str, env_idx: int = 0):
        """Latest cached RGB frame of camera *name* for one env, as an
        ``(H, W, 3)`` uint8 numpy array (recorder ``camera_ref`` consumer).

        Returns ``None`` when nothing has been rendered yet; the rig's
        follow/pose semantics apply through ``sync_pose`` (chase = video
        follow).  NOTE the known trade-off: the video shares the obs
        camera's resolution/interval (docs/contracts/camera_contract.md).
        WYSIWYG (v054): when the rig carries a NON-default live render
        state (the ``render`` live key), the frame is re-rendered through
        the RenderSpec capture face with that state, so the recording
        shows what the live stream shows -- the policy obs cache stays
        untouched (the render is a pure read under the v2 layout).
        """
        rspec = None
        rig = None
        for spec, rig in zip(self.specs, self.rigs):
            if spec.name == name:
                rspec = getattr(rig, "render_spec", None)
                break
        if (rspec is not None and rig is not None
                and callable(getattr(rig, "capture_render", None))):
            return rig.capture_render(rspec)["rgb"]
        val = self._last.get(rgb_key(name))
        if val is None:
            return None
        return val[env_idx].detach().cpu().numpy()

    def render_state(self, name: str) -> dict | None:
        """One rig's current live render state (the v054 ``render`` live
        key; the publish loop's ONLY render state).  ``None`` = the plain
        publish path (unset / explicitly reset / non-mujoco rig).  The
        capture face never reads this -- the publish loop, status faces
        and the WYSIWYG sidecar do."""
        for spec, rig in zip(self.specs, self.rigs):
            if spec.name == name:
                rspec = getattr(rig, "render_spec", None)
                if rspec is None:
                    return None
                return {"render": rspec,
                        "render_source": getattr(rig, "render_source",
                                                 "spec")}
        raise ValueError(f"unknown camera {name!r}; instantiated cameras: "
                         f"{self.names}")

    @property
    def capture_backend(self) -> str | None:
        """``"mujoco"`` when any rig serves the RenderSpec capture face,
        else None (isaaclab: the honest auto-disable signal for the
        webui)."""
        for rig in self.rigs:
            if callable(getattr(rig, "capture_render", None)):
                return "mujoco"
        return None

    # ------------------------------------------------------------------ #
    # viewer / live properties (future bridge-command call face)
    # ------------------------------------------------------------------ #
    def _spec_and_rig(self, name: str) -> tuple[CameraSpec, CameraRig]:
        for spec, rig in zip(self.specs, self.rigs):
            if spec.name == name:
                return spec, rig
        raise ValueError(f"unknown camera {name!r}; instantiated cameras: "
                         f"{self.names}")

    @staticmethod
    def _target_env(rig: CameraRig) -> int:
        """The shared camera's meaningful instance (chase/fixed/body: 0
        unless ``rig.target_env`` says otherwise)."""
        return int(getattr(rig, "target_env", 0))

    def _view_env(self, spec: CameraSpec, rig: CameraRig,
                  env_id: int | None) -> int:
        """Resolve the get_view instance: shared -> target_env (env_id must
        agree or be omitted); per_instance -> env_id (default 0)."""
        te = self._target_env(rig)
        if self.num_envs is None:
            raise RuntimeError("CameraManager._view_env before bind_env")
        if state_is_shared(spec):
            if env_id is not None and int(env_id) != te:
                raise ValueError(
                    f"camera '{spec.name}' is instance=shared: it has a "
                    f"single meaningful view (env {te} = target_env); "
                    f"address it without env_id or with env_id={te}")
            return te
        idx = 0 if env_id is None else int(env_id)
        if not 0 <= idx < int(self.num_envs):
            raise ValueError(f"camera '{spec.name}': env_id {idx} out of "
                             f"range for {self.num_envs} env(s)")
        return idx

    def get_view(self, name: str, env_id: int | None = None,
                 pos=None, lookat=None) -> np.ndarray:
        """Single-frame primitive (SHM/stream/grab all sit on this).

        isaaclab: the most recent sensor output for that env slice (no
        extra render); mujoco: an on-demand render (owned-copy discipline).
        RGB frame when the camera renders one, else its depth frame.
        Never touches the obs cache; pre-render placeholders apply.

        ux-7: ``pos`` + ``lookat`` (given TOGETHER, world frame) render
        this one frame from a TRANSIENT world pose -- the rig pose is
        restored before the call returns, so the broadcast loop can never
        observe the forensic pose.  Fixed rigs only (body/chase are
        re-posed from the robot every step and reject the override).
        """
        if pos is None and lookat is None:
            return self._get_view(name, env_id)
        if pos is None or lookat is None:
            raise ValueError(
                f"camera {name!r}: pose-inline view needs BOTH 'pos' and "
                f"'lookat' (world frame), got pos={pos!r} "
                f"lookat={lookat!r}")
        with self.transient_world_pose(name, pos, lookat):
            return self._get_view(name, env_id)

    def _get_view(self, name: str, env_id: int | None = None) -> np.ndarray:
        if self.num_envs is None:
            raise RuntimeError("CameraManager.get_view before bind_env")
        spec, rig = self._spec_and_rig(name)
        idx = self._view_env(spec, rig, env_id)
        frames = rig.read(env_ids=read_env_ids(rig.spec, rig))
        key = rgb_key(name) if "rgb" in spec.types else depth_key(name)
        val = frames.get(key)
        if val is None:
            val = self._placeholder(key)
        return np.array(val[idx].detach().cpu().numpy())

    # ------------------------------------------------------------------ #
    # RenderSpec capture (v054; camera_contract §14)
    # ------------------------------------------------------------------ #
    def capture(self, name: str, env_id: int | None = None,
                pos=None, lookat=None, *, render) -> dict:
        """Per-request RenderSpec capture (the ONLY capture face).

        ``render``: ``{"view": name}`` (optionally + complementary
        overrides) or a full inline spec -- resolved by
        ``spec.resolve_render`` (fail-fast registry + aspect-merge
        compiler) -> ``{rgb, effective_render, render_source, frames?,
        gif?, seg?, legend?}``.  mujoco-only (the isaaclab rig answers
        the honest ValueError -- v1 degradation unchanged).

        The ``pos``/``lookat`` transient-pose protocol (ux-7) composes
        with the face (fixed rigs, both keys together)."""
        if self.num_envs is None:
            raise RuntimeError("CameraManager.capture before bind_env")
        if render is None:
            raise ValueError(
                f"camera {name!r}: capture requires render= "
                f"({{'view': name}} or a RenderSpec dict; camera_contract "
                f"§14) -- a plain frame is get_view's job")
        spec, rig = self._spec_and_rig(name)
        self._view_env(spec, rig, env_id)     # env_id discipline (§12.2)
        rspec, label = resolve_render(render)
        cap_r: "Callable[[dict], dict] | None" = getattr(
            rig, "capture_render", None)
        if not callable(cap_r):
            raise ValueError(
                f"camera {name!r}: the render= face is mujoco-only "
                f"(this rig: {type(rig).__name__}); the isaaclab "
                f"face gains it with the kit pipeline, not before")
        if pos is None and lookat is None:
            doc = cap_r(rspec)
        elif pos is None or lookat is None:
            raise ValueError(
                f"camera {name!r}: pose-inline capture needs BOTH "
                f"'pos' and 'lookat' (world frame)")
        else:
            with self.transient_world_pose(name, pos, lookat):
                doc = cap_r(rspec)
        doc["effective_render"] = rspec
        doc["render_source"] = label
        return doc

    # ------------------------------------------------------------------ #
    # transient world pose (ux-7, owner-narrowed S2)
    # ------------------------------------------------------------------ #
    @contextmanager
    def transient_world_pose(self, name: str, pos, lookat):
        """One-shot world-frame pose override around a single render.

        Saves the rig's current pose, applies ``pos`` with the camera
        aimed at ``lookat`` (world convention, :func:`lookat_quat` --
        the same semantics as a fixed rig's ``pos``/``lookat`` yaml),
        yields, and ALWAYS restores the saved pose -- even when the body
        raises.  Runs on the engine thread inside one command-slot
        dispatch, so the broadcast loop cannot interleave: the forensic
        pose is never broadcast.  Fixed rigs only; anything else raises.
        """
        spec, rig = self._spec_and_rig(name)
        rtype = str((spec.rig or {}).get("type", "")).strip().lower()
        if rtype not in POSE_INLINE_RIG_TYPES:
            raise ValueError(
                f"camera {name!r}: pose-inline snapshot supports world "
                f"rigs only (this rig: {rtype!r}); follow cams move every "
                f"step -- use camera_set_properties, or declare a world "
                f"rig (the debug fly camera 'free' is one by convention)")
        p = self._check_pose_vec(name, pos)
        look = self._check_pose_vec(name, lookat)
        quat = lookat_quat(p, look)
        saved = rig.begin_world_pose(p, quat)
        try:
            self._refresh_render(rig)
            yield
        finally:
            # restore FIRST (unconditional), then flush the renderer so a
            # cached annotator frame from the forensic pose cannot leak
            # into the broadcast path (isaaclab; mujoco is stateless here)
            rig.end_world_pose(saved)
            self._refresh_render(rig)

    @staticmethod
    def _check_pose_vec(name: str, v) -> list[float]:
        if not isinstance(v, (list, tuple)) or len(v) != 3 or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool)
                and math.isfinite(float(x)) for x in v):
            raise ValueError(
                f"camera {name!r}: pose must be [x, y, z] finite numbers, "
                f"got {v!r}")
        return [float(x) for x in v]

    @staticmethod
    def _refresh_render(rig) -> None:
        """Make a pose write visible / flush it (isaaclab freeze-render
        fix, console contract §frozen-phase rendering: sim.render() + scene.update(0);
        mujoco has neither attribute -- renders on demand, natural no-op).
        Fail-fast: a broken render must not silently leave a stale or
        forensic frame in the annotator."""
        env = getattr(rig, "_env", None)
        render = getattr(getattr(env, "sim", None), "render", None)
        update = getattr(getattr(env, "scene", None), "update", None)
        if render is not None:
            render()
        if update is not None:
            update(0)

    def get_world_pose(self, name: str) -> dict:
        """Read-back face (ux-7 acceptance/forensics; freecam +
        ``camera get`` op face): one rig's current world pose
        (Isaac convention, wxyz quat) plus ``lookat = pos + R(q) @ X̂``
        (forward = +X convention) and the state facts (rig type,
        interval, live render style)."""
        spec, rig = self._spec_and_rig(name)
        pos, quat = rig.get_world_pose()
        w, x, y, z = (float(v) for v in quat)
        fwd = (1.0 - 2.0 * (y * y + z * z),
               2.0 * (x * y + w * z),
               2.0 * (x * z - w * y))
        state = self.render_state(name)
        fps = self.effective_fps(name)
        return {"camera": name, "rig": str(
                    (spec.rig or {}).get("type", "")).strip().lower(),
                "pos": [float(v) for v in pos],
                "quat_wxyz": [float(v) for v in quat],
                "lookat": [float(pos[i] + fwd[i]) for i in range(3)],
                "interval": int(spec.interval),
                "target_fps": fps,
                "render_state": state}

    def apply_properties(self, name: str, params: dict,
                         env_id: int | None = None) -> None:
        """Live-update one camera's whitelisted properties.

        Whitelist : ``pose_offset`` (mount-frame ``[dx, dy, dz]``),
        ``yaw_deg`` / ``pitch_deg`` (mount-frame orientation offset,
        positive pitch = down), ``roll_deg`` (rotation about the view
        axis), ``target_env``, ``interval`` (DEPRECATED
        v069: folds to target_fps when the policy clock is known) and
        ``fps`` (v069 wall render rate; wins over interval) -- plus the
        Freecam face ``pos`` + ``lookat`` (absolute world-frame
        pose, fixed rigs only, both keys together and alone in the call).
        Any other key raises ``ValueError`` "...requires restart".
        ``env_id``: shared cameras take none (or the target_env value);
        per_env ``None`` = every instance, an int = that instance only.
        Pose changes re-pose immediately (isaaclab set_world_poses /
        mujoco cam_pos/cam_quat).
        """
        if self.num_envs is None:
            raise RuntimeError("CameraManager.apply_properties before "
                               "bind_env")
        spec, rig = self._spec_and_rig(name)
        params = dict(params or {})
        if "pos" in params or "lookat" in params:
            self._apply_absolute_pose(name, spec, rig, params, env_id)
            return
        frozen = sorted(set(params) - set(LIVE_PROPERTY_KEYS))
        if frozen:
            raise ValueError(
                f"camera '{name}': property key(s) {frozen} are not "
                f"live-adjustable and requires restart; live keys: "
                f"{sorted(LIVE_PROPERTY_KEYS)} "
                f"(docs/contracts/camera_contract.md §instance 与寻址)")
        te = self._target_env(rig)
        if state_is_shared(spec):
            if env_id is not None and int(env_id) != te:
                raise ValueError(
                    f"camera '{name}' is state=shared: it has a single "
                    f"meaningful instance (env {te} = target_env); apply "
                    f"without env_id or with env_id={te}")
            apply_env_ids = None          # the rig routes to target_env
        else:
            if "target_env" in params:
                raise ValueError(
                    f"camera '{name}': 'target_env' applies to shared "
                    f"cameras only (this camera is per-instance)")
            apply_env_ids = None
            if env_id is not None:
                idx = int(env_id)
                if not 0 <= idx < int(self.num_envs):
                    raise ValueError(f"camera '{name}': env_id {idx} out of "
                                     f"range for {self.num_envs} env(s)")
                apply_env_ids = [idx]
        if "interval" in params or "fps" in params:
            self._apply_rate_keys(name, spec, params)
        if "render" in params:
            # v054 live render key: the publish loop's ONLY render
            # state.  The value is a RenderSpec ({"view": name} or a
            # fragment) -- resolved ONCE here to the COMPLETE spec and
            # stored on the rig (full-write semantics; a partial update
            # would silently inherit, the exact thing the merge compiler
            # kills); the capture face stays stateless.  ``None`` state
            # (= the default view) keeps the plain publish path bitwise.
            if getattr(rig, "capture_render", None) is None:
                raise ValueError(
                    f"camera '{name}': render is mujoco-only "
                    f"(this rig: {type(rig).__name__}); the isaaclab "
                    f"face gains the render face with the kit pipeline")
            rspec, label = resolve_render(params["render"])
            default = rspec == _DEFAULT_RENDER
            rig.render_spec = None if default else rspec
            rig.render_source = None if default else label
        pose_kwargs = {}
        if "pose_offset" in params:
            off = [float(x) for x in params["pose_offset"]]
            if len(off) != 3:
                raise ValueError(f"camera '{name}': pose_offset must be "
                                 f"[dx, dy, dz], got {off}")
            pose_kwargs["pose_offset"] = off
        for k in ("yaw_deg", "pitch_deg", "roll_deg"):
            if k in params:
                try:
                    pose_kwargs[k] = float(params[k])
                except (TypeError, ValueError):
                    raise ValueError(f"camera '{name}': {k} must be a "
                                     f"number, got {params[k]!r}") from None
        if "target_env" in params:
            t = int(params["target_env"])
            if not 0 <= t < int(self.num_envs):
                raise ValueError(f"camera '{name}': target_env {t} out of "
                                 f"range for {self.num_envs} env(s)")
            pose_kwargs["target_env"] = t
        if pose_kwargs:
            rig.apply_properties(env_ids=apply_env_ids, **pose_kwargs)

    def _apply_rate_keys(self, name: str, spec: CameraSpec,
                         params: dict) -> None:
        """The v069 rate keys of one live write (§16; pure in base)."""
        apply_rate_keys(spec, name, params, self.policy_hz)

    def _apply_absolute_pose(self, name: str, spec: CameraSpec,
                             rig: CameraRig, params: dict,
                             env_id: int | None) -> None:
        """Freecam face: persistent absolute world-frame pose.

        ``pos`` + ``lookat`` (world frame, BOTH required, alone in the
        call -- mixing with the offset whitelist keys raises), fixed
        rigs only (follow cams reject the override exactly like the
        pose-inline snapshot, ``POSE_INLINE_RIG_TYPES``).  The write
        targets the resolved instance (shared -> target_env) and is
        flushed through :meth:`_refresh_render` (isaaclab freeze-period
        render fix; mujoco renders on demand, natural no-op).  A later
        offset-keyed apply recomputes from the mount -- last write wins.
        """
        others = set(params) - {"pos", "lookat"}
        if others or "pos" not in params or "lookat" not in params:
            raise ValueError(
                f"camera '{name}': 'pos'/'lookat' is an absolute-pose set "
                f"(freecam face): send BOTH together and alone "
                f"in the call, got keys {sorted(params)}")
        rtype = str((spec.rig or {}).get("type", "")).strip().lower()
        if rtype not in POSE_INLINE_RIG_TYPES:
            raise ValueError(
                f"camera {name!r}: absolute world-pose set supports world "
                f"rigs only (this rig: {rtype!r}); follow cams move every "
                f"step (the debug fly camera 'free' is a world rig by "
                f"convention)")
        p = self._check_pose_vec(name, params["pos"])
        look = self._check_pose_vec(name, params["lookat"])
        idx = self._view_env(spec, rig, env_id)
        rig.set_world_pose(p, lookat_quat(p, look), idx)
        self._refresh_render(rig)

    # ------------------------------------------------------------------ #
    # placeholders
    # ------------------------------------------------------------------ #
    def _placeholder(self, key: str) -> torch.Tensor:
        """Fallback frame for a not-yet-rendered key.

        depth -> all ``inf`` (float32; "nothing observed", preprocesses to
        the far-plane value exactly like the classic all-far mechanism),
        rgb -> all zeros (uint8).
        """
        req = self._requests[key]
        if self.num_envs is None:
            raise RuntimeError("CameraManager._placeholder before bind_env")
        shape = (self.num_envs, *req.shape)
        if req.dtype == torch.uint8:
            return torch.zeros(shape, dtype=torch.uint8, device=self.device)
        return torch.full(shape, float("inf"), dtype=torch.float32,
                          device=self.device)


# --------------------------------------------------------------------------- #
# factory (collect.py entry point)
# --------------------------------------------------------------------------- #
def build_camera_manager(cameras_cfg, obs_requests,
                         enable_cameras: bool = False,
                         refs=None,
                         depth_overrides: dict | None = None,
                         backend: str = "isaaclab",
                         render_groups=None,
                         policy_hz: float | None = None,
                         num_envs: int | None = None) \
        -> CameraManager | None:
    """Build a :class:`CameraManager` from yaml + adapter requests.

    Args:
        cameras_cfg: yaml ``cameras:`` list (None/[] = none declared).
        obs_requests: ``adapter.get_obs_requests()`` (Raw and/or Image).
        enable_cameras: the ``--enable_cameras`` CLI flag; isaaclab camera
            sensors need the kit camera pipeline, so a request without the
            flag is a hard error (fail-fast, not silent degradation).
            mujoco renders synchronously in-process — the flag is NOT
            required for that backend.
        refs: recorder ``camera_ref`` names — cameras referenced only for
            video are instantiated too (zero extra sensors: the same
            rig serves obs and video).
        depth_overrides: yaml ``camera_depth`` dict — the classic-parameter
            overrides applied to the IMPLICIT shim camera only (verbatim
            semantics; a user-declared ``depth`` camera wins instead).
        backend: ``"isaaclab"`` (default; kit sensors), ``"mujoco"``
            (M2b; MjSpec-injected cameras + in-process renderers) or
            ``"mjwarp"`` (C3; N=1 mirror renderer, N>1 native warp batch).
        render_groups: optional per-config geom-class declaration for the
            capture flags face (``env.params.render_groups``; §14).  Only
            consumed by mujoco rigs (v1).
        policy_hz: the policy control rate (1 / adapter.control_dt) --
            the v069 wall scheduler's clock (§16); None keeps the legacy
            per-tick counter semantics (probe scripts).
        num_envs: the run's environment count for the v096 ``TraitScope``
            availability check (``None`` = no scope check).

    Returns ``None`` when nothing is requested (no-camera runs are
    untouched); raises ``ValueError`` on unknown keys, duplicate names,
    a non-``--enable_cameras`` isaaclab request, or a shim/alias geometry
    mismatch.
    """
    try:
        info = backend_info(backend)
    except BackendSelectionError as e:
        raise ValueError(f"unknown camera backend {backend!r} "
                         "(isaaclab|mujoco|mjwarp)") from e
    reqs = list(obs_requests or [])
    req_keys = {r.key for r in reqs}
    declared = fold_declared_intervals(cameras_cfg, policy_hz)
    dup = sorted({n for n in (s.name for s in declared)
                  if [x.name for x in declared].count(n) > 1})
    if dup:
        raise ValueError(f"duplicate camera names in yaml cameras: {dup}")

    available = {k for s in declared for k in s.keys()}
    cam_req = {k for k in req_keys if k.startswith("cam/")}
    image_req = {r.key for r in reqs if isinstance(r, ImageObsRequest)}
    unknown = (cam_req | image_req) - available
    if unknown:
        raise ValueError(
            f"adapter requests unknown camera obs keys {sorted(unknown)}; "
            f"declared cameras serve: {sorted(available)}")

    alias_keys = [k for k in DEPTH_ALIAS_KEYS if k in req_keys]
    shim = False
    if alias_keys and not any(s.name == "depth" for s in declared):
        shim_spec = CameraSpec.depth_body_default(depth_overrides)
        if policy_hz and shim_spec.target_fps is None:
            # shim obs-rate parity pin (camera_contract §16.2): the
            # implicit depth shim historically read every policy tick
            # (interval=1); pin it to the policy rate so the tier
            # defaults never silently change a consumer's obs rate.
            shim_spec.target_fps = float(policy_hz)
        declared.append(shim_spec)
        shim = True

    refs = set(refs or [])
    # obs-served key set: the rigs whose keys were actually requested
    # (adapter obs + image obs) or referenced (recorder camera_ref --
    # the recorder reads the obs CACHE, so a referenced camera must
    # ride the obs tick); aliases resolve to their manager keys inside
    # the manager.  Declared-but-unrequested cameras stay addressable
    # by the camera faces without riding the obs tick.
    ref_keys: set = set()
    for spec in declared:
        if spec.name in refs:
            ref_keys |= set(spec.keys())
    obs_served = cam_req | image_req | ref_keys
    included: list[CameraSpec] = []
    if info.bind_all_declared_cameras:
        # declared-camera availability (camera unification): every
        # declared rig is bound -- an injected camera costs nothing
        # until rendered (render-on-demand), so the camera faces can
        # address the full declared set while only requested rigs ride
        # the obs tick (the manager's obs-served split).
        included = list(declared)
    else:
        # request-driven instantiation (every sensor prim is a
        # real render product -- unrequested cameras stay unconstructed)
        for spec in declared:
            wanted = (bool(set(spec.keys()) & cam_req)
                      or spec.name in refs
                      or spec.name == "free"     # debug-console fly
                      or (shim and spec.name == "depth")  # camera (contract:
                      or (alias_keys and spec.name == "depth"))  # console §free)
            if wanted:
                included.append(spec)
            else:
                print(f"[cameras] camera '{spec.name}' declared but not "
                      f"requested by adapter/recorder -> not instantiated",
                      flush=True)
    if alias_keys and not shim and depth_overrides:
        print("[cameras] note: yaml 'camera_depth' overrides ignored — a "
              "user-declared 'depth' camera wins over the implicit shim",
              flush=True)

    aliases: dict[str, tuple[str, bool]] = {}
    if alias_keys:
        depth_spec = next((s for s in included if s.name == "depth"), None)
        if depth_spec is None:
            raise ValueError(
                "adapter requests depth alias obs but no 'depth' camera is "
                "instantiated")
        if depth_key("depth") not in depth_spec.keys():
            raise ValueError(
                f"camera 'depth' must render the depth stream to serve the "
                f"alias keys (types={list(depth_spec.types)})")
        if "depth_image" in alias_keys and \
                (int(depth_spec.height), int(depth_spec.width)) != \
                _PREPROCESS_SIZE:
            raise ValueError(
                "the 'depth_image' alias applies preprocess_depth "
                f"(crop {_PREPROCESS_SIZE[0]}x{_PREPROCESS_SIZE[1]} -> "
                f"58x87); camera 'depth' renders "
                f"{depth_spec.width}x{depth_spec.height}. Use "
                f"'depth_image_raw' + adapter-side preprocessing, or render "
                f"the 106x60 geometry")
        if "depth_image_raw" in alias_keys:
            aliases["depth_image_raw"] = (depth_key("depth"), False)
        if "depth_image" in alias_keys:
            aliases["depth_image"] = (depth_key("depth"), True)

    if not included:
        return None
    if info.camera_enable_required and not enable_cameras:
        raise ValueError(
            "cameras requested (yaml cameras/adapter image obs) but "
            "--enable_cameras was not passed: the backend camera sensors "
            "require the kit camera pipeline (AppLauncher --enable_cameras)")
    if render_groups is not None:
        from .geomflags import validate_render_groups
        validate_render_groups(render_groups)

    check_camera_capabilities(backend, included, num_envs=num_envs)

    rigs = [build_rig(s, backend=backend, render_groups=render_groups)
            for s in included]
    return CameraManager(included, rigs, aliases, obs_served=obs_served,
                         policy_hz=policy_hz)
