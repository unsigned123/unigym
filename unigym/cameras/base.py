"""cameras package contract: key space, CameraSpec, CameraRig protocol.

This module is **numpy/torch-only** (no isaaclab) so the key space and the
spec parsing are unit-testable offline; isaaclab imports live exclusively in
``backends/isaaclab2/cameras.py`` (and inside rig methods that touch the scene,
lazy-imported), mirroring the ``robots/`` package convention.

Key space
===================================
Every camera *rig* named ``<name>`` serves up to two obs keys::

    cam/<name>/rgb    ->  (N, H, W, 3) uint8   raw RGB
    cam/<name>/depth  ->  (N, 1, H, W) float32 metric depth (meters; inf beyond
                          the far clip = "nothing observed")

Depth is served only by an explicitly declared camera.  When the run's
``cameras:`` list contains a camera named ``depth``, the manager also maps the
adapter-facing ``depth_image`` / ``depth_image_raw`` aliases onto that camera's
``cam/depth/depth`` stream; without a declared ``depth`` camera the alias
request fails with a teaching error (there is no implicit shim camera).

Caching stores RAW values: uint8 RGB / metric depth; any
normalization (crop/clip/normalize) belongs to the adapter — the only
exception is the ``depth_image`` alias, which reproduces the original
``env/depth.py::preprocess_depth`` behavior verbatim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Literal, Protocol, runtime_checkable

import torch

from .state import CameraStateRow, CameraStateTable  # noqa: F401 (re-export)


# --------------------------------------------------------------------------- #
# key space helpers
# --------------------------------------------------------------------------- #
def rgb_key(name: str) -> str:
    """Obs key of the RGB stream of camera *name* (``cam/<name>/rgb``)."""
    return f"cam/{name}/rgb"


def depth_key(name: str) -> str:
    """Obs key of the depth stream of camera *name* (``cam/<name>/depth``)."""
    return f"cam/{name}/depth"


DEPTH_ALIAS_KEYS = ("depth_image", "depth_image_raw")
"""Adapter-facing depth obs aliases served by an explicitly declared ``depth`` camera."""


# --------------------------------------------------------------------------- #
# wall-clock render scheduling (docs/contracts/camera_contract.md §16)
# --------------------------------------------------------------------------- #
CAMERA_TIER_FPS = {
    "ego": 30.0,
    "wrist": 30.0,
    "zoom": 30.0,  # close observation tier
    "closeup": 15.0,
    "chase": 15.0,
    "free": 15.0,  # panoramic / debug tier
    "top": 15.0,
    "side": 15.0,
}
"""render fps tiers (design §2.3): 30 fps for the close observation
surfaces, 15 fps for the panoramic / debug surfaces.  A camera name not in
the table defaults to 30 (the "默认 30" rule)."""


def default_target_fps(name: str) -> float:
    """The tier default render fps for camera *name* (30 when unlisted)."""
    return float(CAMERA_TIER_FPS.get(str(name).strip().lower(), 30.0))


def render_due(
    spec, policy_hz: float | None, counter: int, last_read: float | None, now: float
) -> bool:
    """The wall-clock due check for one read tick (§16.1).

    A resolvable fps renders on the wall schedule (``now`` past
    ``last_read`` by one fps period); without one (no policy clock) the
    legacy per-tick interval counter applies.  First read is always due.
    Pure -- the manager supplies its counter/last-read state.
    """
    fps = effective_target_fps(spec, policy_hz)
    if fps is None:
        return (counter - 1) % int(spec.interval) == 0
    return last_read is None or (now - last_read) >= (1.0 / fps)


def apply_rate_keys(spec, name: str, params: dict, policy_hz: float | None) -> None:
    """The rate keys of one live ``apply_properties`` write (§16).

    ``interval`` (DEPRECATED, removed v070) folds to
    ``target_fps = policy_hz / N`` under a policy clock (the counter
    semantics survive without one); ``fps`` is the wall render rate and
    wins over interval (positive finite number).  Raises ValueError on
    malformed values.  Pure spec mutation.
    """
    if "interval" in params:
        try:
            iv = int(params["interval"])
        except (TypeError, ValueError):
            raise ValueError(
                f"camera '{name}': interval must be an "
                f"integer >= 1, got "
                f"{params['interval']!r}"
            ) from None
        if iv < 1:
            raise ValueError(f"camera '{name}': interval must be >= 1, " f"got {iv}")
        spec.interval = iv  # rigs/manager read it live
        if policy_hz:
            spec.target_fps = float(policy_hz) / iv
    if "fps" in params:
        v = params["fps"]
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(float(v))
            or float(v) <= 0
        ):
            raise ValueError(
                f"camera '{name}': fps must be a positive " f"finite number (fps), got {v!r}"
            )
        spec.target_fps = float(v)


def fold_declared_intervals(cameras_cfg, policy_hz: float | None) -> list:
    """Parse the yaml camera list with the interval migration
    (§16.2): a DECLARED interval folds to ``target_fps = policy_hz/N``
    (same average rate, wall scheduling; v070 removes the key) with the
    deprecation notice printed.  An explicit ``target_fps`` wins -- the
    declared marker is only set when none was given.
    """
    declared = []
    for entry in cameras_cfg or []:
        spec = CameraSpec.from_yaml(entry)
        if getattr(spec, "_interval_declared", False):
            folded = ""
            if policy_hz:
                spec.target_fps = float(policy_hz) / max(1, spec.interval)
                folded = f" -> target_fps {spec.target_fps:.5g} fps"
            print(
                f"[cameras] camera '{spec.name}': 'interval' is "
                f"DEPRECATED (v069, removed v070) -- interval "
                f"{spec.interval}{folded}; declare 'target_fps:' "
                f"instead",
                flush=True,
            )
        declared.append(spec)
    return declared


def effective_target_fps(spec, policy_hz: float | None = None) -> float | None:
    """One camera's effective wall render fps (scheduler).

    Resolution order (docs/contracts/camera_contract.md §16.1):

    1. ``spec.target_fps`` -- explicit (yaml ``target_fps:`` or the live
       ``fps`` property) always wins;
    2. a DECLARED ``interval`` folds to ``policy_hz / interval`` (the
       migration: same average rate, wall semantics);
    3. otherwise the tier default (§2.3) -- needs a policy clock;
    4. ``None`` -- no policy clock available: the caller keeps the legacy
       per-tick counter semantics (probe/test doubles without a run
       context; the migration story stays bitwise for them).

    ``policy_hz`` is the policy control rate (1 / adapter.control_dt).
    """
    tf = getattr(spec, "target_fps", None)
    if tf is not None:
        return float(tf)
    if getattr(spec, "_interval_declared", False):
        iv = max(1, int(getattr(spec, "interval", 1)))
        if policy_hz:
            return float(policy_hz) / iv
        return None
    if policy_hz:
        return default_target_fps(getattr(spec, "name", ""))
    return None


def rig_hide_geom_groups(rig, name: str) -> tuple[int, ...]:
    """Parse the mujoco-only ``hide_geom_groups`` rig key.

    Value: geom-group indices force-hidden in this camera's MjvOption
    (BOTH the plain render and the RenderSpec capture path).  The GR1
    tabletop assets carry robosuite-style collision geoms in group 0;
    the omniforge reference stack renders every view with
    ``geomgroup[0] = 0`` (robosuite ``render_collision_mesh=False``) --
    this key is the declarative per-camera equivalent, opt-in only (the
    historical G1 plain render keeps ``[1, 1, 1, 0, 0, 0]`` + the visual
    group: no default behavior moves).

    Fail-fast validation: list/tuple of ints (bool is not an int here),
    each in 0..4 -- index 5 is the reserved visual-only group the plain
    render exists to SHOW (hiding it would blind every camera); index 0
    is legal (that is the whole point).  The isaaclab backend rejects a
    non-empty value at build (no kit-side equivalent -- honest failure
    over silent divergence).
    """
    val = (rig or {}).get("hide_geom_groups")
    if val is None:
        return ()
    if isinstance(val, (str, bytes)) or not isinstance(val, (list, tuple)):
        raise ValueError(
            f"camera {name!r}: rig.hide_geom_groups must be a list of "
            f"geom-group ints, got {val!r}"
        )
    out: list[int] = []
    for g in val:
        if isinstance(g, bool) or not isinstance(g, int):
            kind = "bool" if isinstance(g, bool) else repr(g)
            raise ValueError(
                f"camera {name!r}: rig.hide_geom_groups entries must be "
                f"ints ({kind} is not an int here), got {g!r}"
            )
        if not 0 <= g <= 4:
            raise ValueError(
                f"camera {name!r}: rig.hide_geom_groups index {g} out of "
                f"range 0..4 (group 5 is the reserved visual-only group "
                f"-- hiding it would blind the plain render)"
            )
        if g in out:
            raise ValueError(f"camera {name!r}: rig.hide_geom_groups duplicate entry " f"{g}")
        out.append(g)
    return tuple(out)


# --------------------------------------------------------------------------- #
# camera spec (yaml-declarable, backend-agnostic)
# --------------------------------------------------------------------------- #
@dataclass
class CameraSpec:
    """One declared camera rig (yaml ``cameras:`` list entry).

    Attributes:
        name: rig name (unique; part of the obs key).
        rig: rig description dict; ``type`` in ``body|world|chase`` plus
            type-specific params (see ``cameras/rigs.py``):
            body  -> ``attach`` (explicit mount link name; the run layer
                     fills it from the effective profile ``base_body`` when
                     omitted, and a body rig with no mount fails),
                     ``pos`` (mount-frame offset), ``pitch_down_deg`` or
                     ``rot`` (w,x,y,z world-convention override, wins),
                     ``prim_path`` override, ``update_latest_camera_pose``;
            world -> ``pos``, ``lookat`` / ``rot``, ``convention``;
            chase -> ``pos``, ``lookat`` (robot-yaw-relative follow
                     offsets), ``target_env`` (which env's base the chase
                     re-poses against, default 0).
        kind: sensor backend, ``standard`` (Camera) or ``tiled``
            (TiledCamera); multi-env + tiled for throughput, standard is the
            RTX-5090-safe default (TiledCamera deterministically hangs on
            Blackwell, isaac-sim/IsaacLab#4951).
        types: which streams to render, subset of ``("rgb", "depth")``.
        width / height: render resolution (px).
        hfov_deg: horizontal field of view; focal length is solved from it
            (``focal = aperture / (2 tan(hfov/2))``, matching the old
            ``_build_depth_camera_cfg`` math exactly).
        horizontal_aperture: 35 mm standard (20.955).
        depth_type: isaaclab depth annotator, ``distance_to_camera`` (ray
            distance) or ``distance_to_image_plane`` (z-depth).
        interval: DEPRECATED (v069, removed v070): read the sensor every N
            policy steps; between reads the manager serves the last cached
            frame (stale-until-next-read).  Declared values fold to
            ``target_fps = policy_hz / interval`` at manager build (same
            average rate, wall-clock scheduling); declare ``target_fps``
            instead.
        target_fps: wall-clock render rate (fps; scheduler).  ``None``
            = tier default by camera name (§16.1: ego/wrist/zoom 30,
            closeup/chase/free/top/side 15, else 30).  Wins over a
            declared ``interval`` (new key priority).
        near_clip / far_clip: camera clipping range (m); far-clip pixels come
            out as inf (depth) — the "nothing observed" sentinel.
        focus_distance: pinhole focus distance (isaaclab default parity 400).
        update_latest_camera_pose: sensor-level data.pos_w refresh flag
            (render posing is rig-driven; see BodyRig.sync_pose).
        instance: addressing scope, ``per_env`` (one meaningful instance per
            env; get_view slices by env_id) or ``shared`` (a single
            meaningful instance — the ``target_env`` clone; get_view ignores
            env_id).  Default resolves by rig: body -> per_env,
            world/chase -> shared.  NOTE (honesty): isaaclab still clones a
            physical sensor prim per env; ``shared`` is a SEMANTIC scoping
            (pose-sync + read only touch the target_env instance), not a
            single physical sensor.  The obs path stays batched (N, ·) in
            every case.
        count: request axis, ``global`` (one scene-wide instance) or
            ``per_env`` (one instance per env).  Default derived from
            ``instance`` (legacy mapping below): ``per_env`` -> per_env.
        state: request axis, ``shared`` (one pose broadcast) or
            ``per_instance`` (each ``(camera, env)`` has its own state).
            Default derived from ``instance``: ``per_env`` -> per_instance,
            ``shared`` -> shared (NOT global).
        env_poses: optional per-env DECLARED world poses, a list
            of ``{"pos": [x, y, z], "quat": [w, x, y, z]}`` dicts (world
            convention).  Length must be 1 (one pose broadcast over the N
            instances) or exactly ``num_envs`` (validated at manager bind);
            forbidden when ``state == "shared"`` (use the rig's single
            ``pos``/``lookat``/``rot`` declaration instead).
    """

    name: str
    rig: dict = field(default_factory=dict)
    kind: str = "standard"
    types: tuple = ("rgb", "depth")
    width: int = 640
    height: int = 360
    hfov_deg: float = 87.0
    horizontal_aperture: float = 20.955
    depth_type: str = "distance_to_camera"
    interval: int = 1
    target_fps: float | None = None
    near_clip: float = 0.1
    far_clip: float = 20.0
    focus_distance: float = 400.0
    update_latest_camera_pose: bool = True
    instance: str | None = None
    count: Literal["global", "per_env"] | None = None
    state: Literal["shared", "per_instance"] | None = None
    env_poses: list | None = None

    def __post_init__(self):
        if not self.name or "/" in self.name or self.name.strip() != self.name:
            raise ValueError(
                f"camera name {self.name!r} must be non-empty, trimmed and "
                f"free of '/' (it is part of the cam/<name>/<type> key)"
            )
        rtype = str((self.rig or {}).get("type", "")).strip().lower()
        if rtype == "fixed":
            raise ValueError(
                f"camera {self.name!r}: rig type 'fixed' was renamed "
                f"'world' (camera unification) -- update the config; "
                f"there is no alias (legal: body|world|chase)"
            )
        if "body" in (self.rig or {}):
            raise ValueError(
                f"camera {self.name!r}: rig key 'body' was renamed "
                f"'attach' (BodyRig link mount) -- update the config; "
                f"there is no alias"
            )
        if self.kind not in ("standard", "tiled"):
            raise ValueError(
                f"camera {self.name!r}: unknown kind " f"{self.kind!r} (standard|tiled)"
            )
        bad = set(self.types) - {"rgb", "depth"}
        if not self.types or bad:
            raise ValueError(
                f"camera {self.name!r}: types must be a "
                f"non-empty subset of ['rgb','depth'], got "
                f"{self.types}"
            )
        if self.depth_type not in ("distance_to_camera", "distance_to_image_plane"):
            raise ValueError(
                f"camera {self.name!r}: depth_type must be "
                f"distance_to_camera|distance_to_image_plane, "
                f"got {self.depth_type!r}"
            )
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError(f"camera {self.name!r}: bad resolution " f"{self.width}x{self.height}")
        if int(self.interval) < 1:
            raise ValueError(f"camera {self.name!r}: interval must be >= 1")
        if self.target_fps is not None:
            if (
                isinstance(self.target_fps, bool)
                or not isinstance(self.target_fps, (int, float))
                or not math.isfinite(float(self.target_fps))
                or float(self.target_fps) <= 0
            ):
                raise ValueError(
                    f"camera {self.name!r}: target_fps must be a positive "
                    f"finite number, got {self.target_fps!r}"
                )
            self.target_fps = float(self.target_fps)
        if self.instance is None:
            self.instance = "per_env" if rtype == "body" else "shared"
        if self.instance not in ("per_env", "shared"):
            raise ValueError(
                f"camera {self.name!r}: instance must be per_env|shared, " f"got {self.instance!r}"
            )
        self._resolve_request_axes()
        self._validate_env_poses_declaration()

    def _validate_env_poses_declaration(self) -> None:
        """Structural validation of ``env_poses``.

        Length checks live at bind, where ``num_envs`` is known -- see
        :meth:`validate_env_poses`.
        """
        if self.env_poses is None:
            return
        if self.state == "shared":
            raise ValueError(
                f"camera {self.name!r}: env_poses is forbidden when "
                f"state='shared' (one broadcast pose); declare per-env "
                f"poses only with state='per_instance'"
            )
        if isinstance(self.env_poses, (str, bytes)) or not isinstance(
            self.env_poses, (list, tuple)
        ):
            raise ValueError(
                f"camera {self.name!r}: env_poses must be a list of "
                f"{{'pos','quat'}} dicts, got {self.env_poses!r}"
            )
        if len(self.env_poses) == 0:
            raise ValueError(f"camera {self.name!r}: env_poses must not be empty")
        out = []
        for i, row in enumerate(self.env_poses):
            if not isinstance(row, dict) or set(row) != {"pos", "quat"}:
                raise ValueError(
                    f"camera {self.name!r}: env_poses[{i}] must be a dict "
                    f"with exactly 'pos' and 'quat', got {row!r}"
                )
            p, q = row["pos"], row["quat"]
            if (
                isinstance(p, (str, bytes))
                or not isinstance(p, (list, tuple))
                or len(p) != 3
                or not all(
                    isinstance(x, (int, float))
                    and not isinstance(x, bool)
                    and math.isfinite(float(x))
                    for x in p
                )
            ):
                raise ValueError(
                    f"camera {self.name!r}: env_poses[{i}].pos must be "
                    f"[x, y, z] finite numbers, got {p!r}"
                )
            if (
                isinstance(q, (str, bytes))
                or not isinstance(q, (list, tuple))
                or len(q) != 4
                or not all(
                    isinstance(x, (int, float))
                    and not isinstance(x, bool)
                    and math.isfinite(float(x))
                    for x in q
                )
            ):
                raise ValueError(
                    f"camera {self.name!r}: env_poses[{i}].quat must be "
                    f"[w, x, y, z] finite numbers, got {q!r}"
                )
            out.append({"pos": [float(x) for x in p], "quat": [float(x) for x in q]})
        self.env_poses = out

    def validate_env_poses(self, num_envs: int) -> None:
        """Length discipline: ``len(env_poses) in (1, num_envs)`` (bind)."""
        if self.env_poses is None:
            return
        n = len(self.env_poses)
        if n not in (1, int(num_envs)):
            raise ValueError(
                f"camera {self.name!r}: env_poses has {n} row(s); legal "
                f"lengths are 1 (broadcast) or num_envs={int(num_envs)}"
            )

    def _resolve_request_axes(self) -> None:
        """Derive/validate the ``count × state`` request axes."""
        if self.count is None:
            self.count = "per_env"
        if self.state is None:
            self.state = (
                "shared"
                if self.count == "global"
                else ("per_instance" if self.instance == "per_env" else "shared")
            )
        if self.count not in ("global", "per_env"):
            raise ValueError(
                f"camera {self.name!r}: count must be global|per_env, " f"got {self.count!r}"
            )
        if self.state not in ("shared", "per_instance"):
            raise ValueError(
                f"camera {self.name!r}: state must be shared|per_instance, " f"got {self.state!r}"
            )
        if self.count == "global" and self.state == "per_instance":
            raise ValueError(
                f"camera {self.name!r}: count='global' cannot combine with "
                f"state='per_instance'; legal (count, state) combinations: "
                f"(global, shared), (per_env, shared), "
                f"(per_env, per_instance)"
            )

    # ---- parsing -------------------------------------------------------- #
    @classmethod
    def from_yaml(cls, entry: dict) -> CameraSpec:
        """Build a spec from one yaml ``cameras:`` list entry.

        Unknown keys raise (fail-fast against typos).  ``rig.type`` is
        required (there is no meaningful default mounting).
        """
        allowed = {f.name for f in fields(cls)} | {"rig"}
        unknown = set(entry) - allowed
        if unknown:
            raise ValueError(
                f"camera spec unknown keys {sorted(unknown)}; " f"allowed: {sorted(allowed)}"
            )
        rig = dict(entry.get("rig") or {})
        if not rig.get("type"):
            raise ValueError(
                f"camera {entry.get('name')!r}: rig.type is " f"required (body|world|chase)"
            )
        kwargs = {k: v for k, v in entry.items() if k != "rig"}
        if "types" in kwargs:
            kwargs["types"] = tuple(kwargs["types"])
        spec = cls(rig=rig, **kwargs)
        if "interval" in kwargs and "target_fps" not in kwargs:
            # migration marker (camera_contract §16.2): a DECLARED
            # interval folds to policy_hz/interval at manager build.  The
            # deprecation notice prints there (it knows policy_hz); this
            # marker only distinguishes "declared interval" from the
            # field default (which moves to the tier fps instead).
            # runtime migration marker (not a declared dataclass field);
            # setattr keeps it out of the serialized schema surface
            setattr(spec, "_interval_declared", True)
        return spec

    # ---- key space ------------------------------------------------------ #
    def keys(self) -> list[str]:
        """Obs keys this spec serves, in canonical order (rgb first)."""
        out = []
        if "rgb" in self.types:
            out.append(rgb_key(self.name))
        if "depth" in self.types:
            out.append(depth_key(self.name))
        return out

    def frame_shape(self, key: str) -> tuple:
        """Per-env frame shape (batch excluded) for one of ``keys()``."""
        if key == rgb_key(self.name):
            return (int(self.height), int(self.width), 3)
        if key == depth_key(self.name):
            return (1, int(self.height), int(self.width))
        raise KeyError(f"{key!r} is not served by camera {self.name!r}")

    def frame_dtype(self, key: str) -> torch.dtype:
        """Raw-value dtype: uint8 RGB, float32 metric depth."""
        if key == rgb_key(self.name):
            return torch.uint8
        if key == depth_key(self.name):
            return torch.float32
        raise KeyError(f"{key!r} is not served by camera {self.name!r}")

    def focal_length(self) -> float:
        """Pinhole focal length (mm) solved from ``hfov_deg`` + aperture."""
        return self.horizontal_aperture / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))

    def sensor_key(self) -> str:
        """Scene key of the cloned sensor prim (last prim path component)."""
        return f"{self.name}_camera"


# --------------------------------------------------------------------------- #
# rig protocol
# --------------------------------------------------------------------------- #
def rig_type(spec: CameraSpec) -> str:
    """Normalized rig ``type`` of *spec* (``body|world|chase``).

    Pure spec classification — lives on the capability side so both
    backend rig stacks (and the rigs) share one definition.
    """
    return str((spec.rig or {}).get("type", "")).strip().lower()


@runtime_checkable
class CameraRig(Protocol):
    """Structural protocol for one camera mounting (world-component style).

    Capability trait: :attr:`unigym.backends.registry.Trait.CAMERAS_RGB`
    (rig-level depth is ``CAMERAS_DEPTH_RIG``).  Per-backend rig
    implementations: ``cameras/rigs.py`` (isaaclab sensors) +
    ``backends/mujoco/cameras.build_mj_rig`` (MjSpec-injected cameras).

    Lifecycle (driven by :class:`cameras.manager.CameraManager`, mirroring
    the prop manager conventions):

    * ``setup_scene(scene_cfg)`` — before env construction; adds the sensor
      cfg under the rig's prim path (isaaclab import inside).
    * ``bind_env(env)`` — after construction; grabs the sensor handle from
      ``env.scene``.
    * ``sync_pose(env_ids=None)`` — every policy step, at the step TAIL (the
      render inside the NEXT step's ``sim.step`` then captures this pose —
      the standard Camera render product is pinned to the prim's world
      transform, so body/chase mountings must re-pose explicitly; this
      reproduces the classic ``_sync_depth_camera_pose`` semantics
      verbatim).  ``env_ids`` scopes WHICH instances are re-posed
      (``None`` = every served instance: all envs for ``per_instance``,
      the single ``target_env`` for ``shared``); the addressing rule is
      the rig's, never the caller's guess.
    * ``on_reset(env_ids)`` — after the reset chain's ``sim.forward``
      (re-pose to the spawn state, same hook the classic depth camera used).
    * ``read(env_ids=None)`` — read the current sensor output; returns
      ``{key: tensor | None}`` (``None`` = not rendered yet / placeholder ->
      the manager substitutes its all-inf / all-zero fallback frame).
      ``env_ids`` only scopes which instances are REFRESHED; the returned
      block stays ``(N, ·)`` (the obs/record key space never changes).
    """

    name: str
    spec: CameraSpec

    # Live render state (camera_contract §14): ``None`` means the plain
    # publish path.  Only capture-capable rigs (mujoco/mjwarp) set these, but
    # the manager reads them on every rig, so they belong to the boundary.
    render_spec: dict | None
    render_source: str | None
    target_env: int

    def setup_scene(self, scene_cfg) -> None: ...

    def bind_env(self, env) -> None: ...

    def sync_pose(self, env_ids=None) -> None: ...

    def on_reset(self, env_ids) -> None: ...

    def read(self, env_ids=None) -> dict: ...

    def apply_properties(
        self,
        pose_offset=None,
        yaw_deg=None,
        pitch_deg=None,
        roll_deg=None,
        target_env=None,
        env_ids=None,
    ) -> None: ...

    def begin_world_pose(self, pos, quat_wxyz): ...

    def end_world_pose(self, saved) -> None: ...

    def set_world_pose(self, pos, quat_wxyz, env_idx: int | None = None) -> None: ...

    def get_world_pose(self) -> tuple: ...

    def capture_render(self, spec: dict) -> dict: ...
