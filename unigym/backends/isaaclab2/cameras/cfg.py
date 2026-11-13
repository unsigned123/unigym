"""isaaclab camera sensor construction + frame reading.

Every function here lazily imports isaaclab; the module itself is
import-safe without the kit (unit tests only touch the pure helpers).
This sensor module and the sibling ``rigs`` module are the only camera
modules that import isaaclab, and both do so inside function bodies.

Sensor construction reproduces the classic depth-camera math verbatim
(``isaaclab2/env.py._build_depth_camera_cfg``): pinhole focal solved from the
horizontal FOV (``focal = aperture / (2 tan(hfov/2))``), ``convention=
"world"`` offsets, ray-distance depth annotator by default.

Frame reading normalizes annotator layouts to the manager contract:

* RGB  -> ``(N, H, W, 3)`` uint8 (RGBA input dropped to RGB; float input
  clipped/scaled to uint8 — the defensive conversion the classic recorder
  path used before it was removed).
* depth -> ``(N, 1, H, W)`` float32 meters (annotator layout may be
  ``(N, H, W)``, ``(N, 1, H, W)`` or ``(N, H, W, 1)`` — all flatten to
  ``N*H*W`` first; anything else returns ``None`` = "not rendered yet",
  which the manager replaces with its placeholder frame).
"""

from __future__ import annotations

import math

import torch

from unigym.cameras.base import CameraSpec, depth_key, rgb_key, rig_type, rig_hide_geom_groups


def build_sensor_cfg(spec: CameraSpec):
    """Build the isaaclab sensor cfg (CameraCfg / TiledCameraCfg) for *spec*.

    Prim path by rig type: ``body`` requires an explicit mount -- either
    ``rig.attach`` (a robot-child link prim) or an explicit ``rig.prim_path``
    override; the run layer fills ``attach`` from the effective profile
    ``base_body`` when the camera omits both.  There is no implicit body
    mount.  ``world``/``chase`` sit at the env origin level
    (``{ENV_REGEX_NS}/<name>_camera``).

    Offset resolution: explicit ``rig.rot`` (w,x,y,z, world convention)
    wins; ``body`` defaults to the pitch-down quaternion
    ``(cos(p/2), 0, sin(p/2), 0)`` (reference parity); ``world``/``chase``
    derive the quaternion from ``pos -> lookat`` (recorder look-at math,
    P8 black-frame regression safe).

    Link mount (``rig.attach`` set, e.g. G1's ``d435_link``): the sensor
    prim is created as a CHILD of the robot link
    (``{ENV_REGEX_NS}/Robot/<attach>/<name>_camera``) so the USD kinematic
    tree carries the pose (exactly how upstream HA mounts its front cam as
    ``Robot/d435_link/front_cam``); ``rig.pos``/``rig.mount_rot``/
    ``rig.mount_convention`` become the prim offset verbatim and BodyRig
    skips per-step re-posing.  ``rig.focal_length`` (mm) overrides the
    hfov-solved focal when the upstream aperture/focal pair must be
    replicated exactly.
    """
    from unigym.cameras.rigs import lookat_quat
    from isaaclab.sensors import CameraCfg, TiledCameraCfg

    import isaaclab.sim as sim_utils  # noqa: F811 (2.3.2: module alias form)

    rtype = rig_type(spec)
    rig = spec.rig or {}
    _hide = rig_hide_geom_groups(rig, spec.name)
    if _hide:
        raise ValueError(
            f"camera {spec.name!r}: rig.hide_geom_groups is mujoco-only "
            f"(MJ geom-group visibility); the isaaclab backend has no "
            f"kit-side equivalent -- remove the key (got {list(_hide)})"
        )
    mount_body = str(rig.get("attach", "") or "").strip().strip("/")
    if rtype == "body":
        if mount_body:
            prim_path = f"{{ENV_REGEX_NS}}/Robot/{mount_body}/" f"{spec.sensor_key()}"
        elif rig.get("prim_path") is not None:
            prim_path = str(rig["prim_path"])
        else:
            raise ValueError(
                f"camera {spec.name!r}: a body rig requires an explicit "
                f"`attach` link or an explicit `prim_path`; the effective "
                f"profile base_body is filled before this boundary (there is "
                f"no implicit body mount)"
            )
    elif rtype in ("world", "chase"):
        prim_path = str(rig.get("prim_path") or f"{{ENV_REGEX_NS}}/{spec.sensor_key()}")
    elif rtype == "fixed":
        raise ValueError(
            f"camera {spec.name!r}: rig type 'fixed' was renamed 'world' "
            f"(camera unification) -- update the config; there is no alias"
        )
    else:
        raise ValueError(f"camera {spec.name!r}: unknown rig type {rtype!r} " "(body|world|chase)")

    pos = tuple(float(x) for x in rig.get("pos", (0.0, 0.0, 0.0)))
    convention = "world"
    if rig.get("mount_rot") is not None:
        quat = tuple(float(x) for x in rig["mount_rot"])
        convention = str(rig.get("mount_convention", "world"))
    elif rig.get("rot") is not None:
        quat = tuple(float(x) for x in rig["rot"])
    elif rtype == "body":
        pitch = math.radians(float(rig.get("pitch_down_deg", 0.0)))
        quat = (math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0)
    else:
        lookat = rig.get("lookat", (1.0, 0.0, 0.25))
        quat = tuple(float(x) for x in lookat_quat(pos, lookat))

    focal_override = rig.get("focal_length")
    focal_length = (
        float(focal_override) if focal_override is not None else float(spec.focal_length())
    )

    data_types = []
    if "rgb" in spec.types:
        data_types.append("rgb")
    if "depth" in spec.types:
        data_types.append(spec.depth_type)

    spawn = sim_utils.PinholeCameraCfg(
        focal_length=focal_length,
        horizontal_aperture=float(spec.horizontal_aperture),
        focus_distance=float(spec.focus_distance),
        clipping_range=(float(spec.near_clip), float(spec.far_clip)),
    )
    cfg_cls = TiledCameraCfg if spec.kind == "tiled" else CameraCfg
    return cfg_cls(
        prim_path=prim_path,
        offset=cfg_cls.OffsetCfg(pos=pos, rot=quat, convention=convention),
        data_types=data_types,
        spawn=spawn,
        width=int(spec.width),
        height=int(spec.height),
        update_latest_camera_pose=bool(spec.update_latest_camera_pose),
    )


def push_state_table_poses(
    sensor, table, camera: str, env_ids, *, num_envs: int | None = None, force_env_ids: bool = False
) -> list[int]:
    """Drive ``sensor.set_world_poses`` from the ``(camera, env)`` state table.

    The rig computes/writes resolved world poses into
    :class:`unigym.cameras.base.CameraStateTable` rows; this backend helper
    is the ONLY sensor write for world/chase rigs.  One prim already
    exists per env (``{ENV_REGEX_NS}``), so ``env_ids`` selects the prims
    written -- ``env_ids=[i]`` moves only env ``i`` (the container test
    ``tests/backends/isaaclab2/test_camera_per_env.py``).

    ``force_env_ids`` keeps the historical explicit-index call (chase
    always passed ``env_ids=[target_env]``); otherwise a full-range write
    goes through ``env_ids=None`` (the historical call shape, bit-identical).
    Returns the env ids written.  Pure torch -- import-safe without the kit.
    """
    ids = [int(i) for i in env_ids]
    if not ids:
        return ids
    positions = []
    orientations = []
    for i in ids:
        row = table.get(camera, i)
        if row is None:
            raise KeyError(
                f"camera {camera!r}: no state row for env {i} " f"(mode={table.mode(camera)!r})"
            )
        positions.append(torch.as_tensor(row.pos, dtype=torch.float32).reshape(3))
        orientations.append(torch.as_tensor(row.quat, dtype=torch.float32).reshape(4))
    pos = torch.stack(positions)
    quat = torch.stack(orientations)
    # The live sensor's pose buffers live on its own device; stacking builds
    # CPU tensors, so move them before set_world_poses (which validates the
    # device and otherwise raises "array_positions is on device=cpu").
    dev = getattr(sensor, "_device", None)
    if dev is not None:
        pos = pos.to(dev)
        quat = quat.to(dev)
    full = num_envs is not None and ids == list(range(int(num_envs)))
    if full and not force_env_ids:
        sensor.set_world_poses(positions=pos, orientations=quat, convention="world")
    else:
        sensor.set_world_poses(
            positions=pos,
            orientations=quat,
            env_ids=torch.tensor(ids, dtype=torch.long, device=pos.device),
            convention="world",
        )
    return ids


def read_frames(sensor, spec: CameraSpec, num_envs: int) -> dict:
    """Read one sensor's annotator outputs; ``{key: tensor | None}``.

    ``None`` means the annotator has not produced a full-resolution frame
    yet (pre-first-render placeholders come out degenerate); the manager
    substitutes its fallback frame.
    """
    out = sensor.data.output
    frames = {}
    for key in spec.keys():
        if key == rgb_key(spec.name):
            frames[key] = _read_rgb(out.get("rgb"), spec, num_envs)
        elif key == depth_key(spec.name):
            frames[key] = _read_depth(out.get(spec.depth_type), spec, num_envs)
        else:  # pragma: no cover - CameraSpec.keys() is closed
            raise KeyError(key)
    return frames


def _read_rgb(t, spec: CameraSpec, num_envs: int):
    if t is None:
        return None
    h, w = int(spec.height), int(spec.width)
    try:
        arr = t.reshape(num_envs, h, w, -1)
    except RuntimeError:
        return None
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        return None
    if arr.dtype != torch.uint8:  # defensive: clip->scale reference conversion
        arr = torch.clip(arr, 0.0, 255.0).to(torch.uint8)
    # OWN the buffer: annotator output memory is recycled in-place by the
    # render graph, so a view here would let every cached frame alias (and
    # retroactively mutate) the sensor's staging buffer.
    return arr.clone()


def _read_depth(t, spec: CameraSpec, num_envs: int):
    if t is None:
        return None
    h, w = int(spec.height), int(spec.width)
    flat = t.reshape(num_envs, -1)
    if flat.shape[1] != h * w:
        return None
    # .clone() even when .to() is a no-op (float32 input): the annotator
    # buffer is reused in-place by the render graph (see _read_rgb note).
    return flat.reshape(num_envs, 1, h, w).to(torch.float32).clone()
