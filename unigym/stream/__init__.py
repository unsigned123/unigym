"""Streaming control schema  -- pure logic, bridge-protocol-free.

One JSON object per instruction, POSTed to the daemon media gateway's
``/run/<id>/control`` endpoint (formerly the run bridge's
``/control``; same schema, same command-slot mapping), or sent over the
WebRTC DataChannel.  This module owns
parse/validate/normalize + the mapping onto the SHM bridge command set
(docs/contracts/runtime_contract.md §4 command ids); it never touches SHM, sockets
or torch, so it is unit-testable offline.

Instruction types::

    {type: camera_set_properties, camera, pose_offset?[3], yaw_deg?,
     pitch_deg?, target_env?, interval?, env_id?}      # env_id: per_env addressing extension
    {type: stream_bind, camera, env_id?}      # L2 producer rebind (S2)
    {type: pause} {type: resume} {type: shutdown}
    {type: snapshot, camera?, env_id?, pos?[3], lookat?[3]}
    # ux-7 (S2 narrowed scope): snapshot camera OPTIONAL (default "free", the
    # debug fly camera); pos+lookat (given TOGETHER) request a TRANSIENT
    # world-pose override for exactly that one frame -- fixed/free rigs
    # only, the rig pose is restored before the ack (see
    # cameras.manager.CameraManager.transient_world_pose).

Validation rules:

* envelope keys are whitelisted per type -- an unknown key is an explicit
  ``ControlError`` (never silently ignored);
* the ``camera_set_properties`` property whitelist IS the established live set
  (pose_offset / yaw_deg / pitch_deg / target_env / interval) -- frozen
  keys are rejected here before they can reach the manager;
* types/ranges: pose_offset = 3 finite numbers, angles = finite numbers,
  target_env/interval/env_id = non-negative ints (interval >= 1),
  camera = non-empty string (<= 64 B, the SHM slot name field width).

``stream_bind`` maps onto the bridge ``set_camera`` command (S2):
the broadcast producer rebinds, which the WebRTC single encoder follows;
the MJPEG path never needs it (L1 = the player changes its stream URL).
"""

from __future__ import annotations

import math

__all__ = [
    "ControlError",
    "CONTROL_TYPES",
    "LIVE_PROPERTY_KEYS",
    "CAMERA_SET_KEYS",
    "STREAM_BIND_KEYS",
    "SNAPSHOT_KEYS",
    "SNAPSHOT_DEFAULT_CAMERA",
    "parse_control",
    "to_bridge_command",
    "parse_control_json",
]

CONTROL_TYPES = ("camera_set_properties", "stream_bind", "pause", "resume", "snapshot", "shutdown")

# The live whitelist (unigym.cameras.manager.LIVE_PROPERTY_KEYS),
# restated here verbatim: this module must stay torch-free (bridge.py
# imports it for POST /control) and the cameras package imports torch at
# module level.  tests/test_stream.py locks the two tuples together
# (parity assertion) -- change them only as a pair.
LIVE_PROPERTY_KEYS = (
    "fps",
    "interval",
    "pitch_deg",
    "pose_offset",
    "render",
    "roll_deg",
    "target_env",
    "yaw_deg",
)

# per-type envelope whitelists
CAMERA_SET_KEYS = ("type", "camera", "env_id") + LIVE_PROPERTY_KEYS
STREAM_BIND_KEYS = ("type", "camera", "env_id")
SNAPSHOT_KEYS = ("type", "camera", "env_id", "pos", "lookat")
_BARE_KEYS = ("type",)  # pause / resume / shutdown

SNAPSHOT_DEFAULT_CAMERA = "free"  # ux-7: pose-inline snapshots default

MAX_NAME = 64  # SHM frame-slot camera name field width


class ControlError(ValueError):
    """Invalid streaming control instruction (explicit, user-facing)."""


def _fail(msg: str) -> ControlError:
    return ControlError(msg)


def _check_camera(ctrl: dict, type_name: str) -> str:
    cam = ctrl.get("camera")
    if cam is None:
        raise _fail(
            f"{type_name}: 'camera' is required " f"(instantiated cameras: see GET /status)"
        )
    if not isinstance(cam, str) or not cam.strip():
        raise _fail(f"{type_name}: 'camera' must be a non-empty string, " f"got {cam!r}")
    if len(cam.encode()) > MAX_NAME - 1:
        raise _fail(
            f"{type_name}: 'camera' name longer than "
            f"{MAX_NAME - 1} bytes (SHM slot field width)"
        )
    return cam


def _check_int(ctrl: dict, key: str, type_name: str, minimum: int = 0) -> int:
    v = ctrl[key]
    if isinstance(v, bool) or not isinstance(v, int):
        raise _fail(f"{type_name}: '{key}' must be an integer, " f"got {v!r}")
    if v < minimum:
        raise _fail(f"{type_name}: '{key}' must be >= {minimum}, got {v}")
    return v


def _check_number(ctrl: dict, key: str, type_name: str) -> float:
    v = ctrl[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _fail(f"{type_name}: '{key}' must be a number, got {v!r}")
    v = float(v)
    if not math.isfinite(v):
        raise _fail(f"{type_name}: '{key}' must be finite, got {v!r}")
    return v


def _check_vec3(ctrl: dict, key: str, type_name: str) -> list[float]:
    """`[x, y, z]` of three finite numbers -> normalized float list."""
    v = ctrl[key]
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise _fail(f"{type_name}: '{key}' must be [x, y, z], got {v!r}")
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
            raise _fail(
                f"{type_name}: '{key}' elements must be finite " f"numbers, got {x!r} in {v!r}"
            )
    return [float(x) for x in v]


def parse_control(obj) -> dict:
    """Validate + normalize one control instruction (returns a new dict).

    Raises :class:`ControlError` with an explicit message on any shape /
    key / type / range violation.  Normalization: numbers become floats,
    ints stay ints; unknown keys never pass.
    """
    if not isinstance(obj, dict):
        raise _fail(f"control instruction must be a JSON object, " f"got {type(obj).__name__}")
    t = obj.get("type")
    if not isinstance(t, str):
        raise _fail(f"control instruction needs a string 'type', " f"got {t!r}")
    if t not in CONTROL_TYPES:
        raise _fail(f"unknown control type {t!r} " f"(valid: {', '.join(CONTROL_TYPES)})")

    if t in ("pause", "resume", "shutdown"):
        extra = set(obj) - set(_BARE_KEYS)
        if extra:
            raise _fail(
                f"{t}: unexpected key(s) {sorted(extra)} " f"(expected exactly {_BARE_KEYS})"
            )
        return {"type": t}

    if t == "camera_set_properties":
        extra = set(obj) - set(CAMERA_SET_KEYS)
        if extra:
            raise _fail(
                f"camera_set_properties: unknown key(s) {sorted(extra)}; "
                f"live property whitelist : "
                f"{sorted(LIVE_PROPERTY_KEYS)} -- frozen properties "
                f"(resolution, hfov, types, ...) require a restart"
            )
        out: dict = {"type": t, "camera": _check_camera(obj, t)}
        if "env_id" in obj:
            out["env_id"] = _check_int(obj, "env_id", t)
        n_props = 0
        if "pose_offset" in obj:
            off = obj["pose_offset"]
            if not isinstance(off, (list, tuple)) or len(off) != 3:
                raise _fail(
                    f"camera_set_properties: 'pose_offset' must " f"be [dx, dy, dz], got {off!r}"
                )
            for x in off:
                if (
                    isinstance(x, bool)
                    or not isinstance(x, (int, float))
                    or not math.isfinite(float(x))
                ):
                    raise _fail(
                        f"camera_set_properties: 'pose_offset' "
                        f"elements must be finite numbers, got "
                        f"{x!r} in {off!r}"
                    )
            out["pose_offset"] = [float(x) for x in off]
            n_props += 1
        for k in ("yaw_deg", "pitch_deg"):
            if k in obj:
                out[k] = _check_number(obj, k, t)
                n_props += 1
        if "interval" in obj:
            out["interval"] = _check_int(obj, "interval", t, minimum=1)
            n_props += 1
        if "target_env" in obj:
            out["target_env"] = _check_int(obj, "target_env", t)
            n_props += 1
        if "render" in obj:
            rd = obj["render"]
            if not isinstance(rd, dict):
                raise _fail(
                    f"camera_set_properties: 'render' must be a "
                    f"RenderSpec dict ({{'view': name}} or a spec "
                    f"fragment; camera_contract §14), got {rd!r} "
                    f"(the manager resolves and validates it)"
                )
            out["render"] = rd
            n_props += 1
        if n_props == 0:
            raise _fail(
                "camera_set_properties: no property given "
                f"(live keys: {sorted(LIVE_PROPERTY_KEYS)})"
            )
        return out

    if t == "stream_bind":
        extra = set(obj) - set(STREAM_BIND_KEYS)
        if extra:
            raise _fail(
                f"stream_bind: unknown key(s) {sorted(extra)} " f"(expected {STREAM_BIND_KEYS})"
            )
        out = {"type": t, "camera": _check_camera(obj, t)}
        if "env_id" in obj:
            out["env_id"] = _check_int(obj, "env_id", t)
        return out

    # snapshot (ux-7: camera optional -> SNAPSHOT_DEFAULT_CAMERA;
    # pos+lookat together = transient world-pose override)
    extra = set(obj) - set(SNAPSHOT_KEYS)
    if extra:
        raise _fail(f"snapshot: unknown key(s) {sorted(extra)} " f"(expected {SNAPSHOT_KEYS})")
    out = {"type": t}
    if "camera" in obj:
        out["camera"] = _check_camera(obj, t)
    else:
        out["camera"] = SNAPSHOT_DEFAULT_CAMERA
    if "env_id" in obj:
        out["env_id"] = _check_int(obj, "env_id", t)
    has_pos, has_look = "pos" in obj, "lookat" in obj
    if has_pos != has_look:
        raise _fail(
            "snapshot: 'pos' and 'lookat' must be given together "
            "(world-frame pose: pos=[x,y,z] lookat=[x,y,z]); a "
            "position without an orientation is ambiguous"
        )
    if has_pos:
        out["pos"] = _check_vec3(obj, "pos", t)
        out["lookat"] = _check_vec3(obj, "lookat", t)
    return out


def parse_control_json(raw: bytes | str) -> dict:
    """JSON-decode then :func:`parse_control` (ValueError on bad JSON)."""
    import json

    try:
        obj = json.loads(raw)
    except ValueError as e:
        raise ControlError(f"body is not valid JSON: {e}") from None
    return parse_control(obj)


def to_bridge_command(ctrl: dict) -> tuple[str, dict]:
    """Map a parsed instruction onto the SHM command set (§4 command ids).

    Returns ``(command_name, payload)`` -- names are the stable
    ``CMD_NAMES`` strings ("set_props" / "get_view" / "pause" / ...).
    Raises :class:`ControlError` for instructions whose mechanism is not
    available yet (stream_bind before the S2 media pipeline).
    """
    t = ctrl.get("type")
    if t == "camera_set_properties":
        params = {
            k: ctrl[k]
            for k in ("pose_offset", "yaw_deg", "pitch_deg", "target_env", "interval", "render")
            if k in ctrl
        }
        payload: dict = {"camera": ctrl["camera"], "params": params, "env_id": ctrl.get("env_id")}
        return "set_props", payload
    if t == "snapshot":
        payload: dict = {"camera": ctrl["camera"], "env_id": ctrl.get("env_id"), "jpeg": True}
        if "pos" in ctrl:  # ux-7 transient pose override
            payload["pos"] = ctrl["pos"]
            payload["lookat"] = ctrl["lookat"]
        return "get_view", payload
    if t in ("pause", "resume", "shutdown"):
        return t, {}
    if t == "stream_bind":
        # S2: the bridge's `set_camera` command rebinds the
        # broadcast producer (get_view source) -- the WebRTC single-
        # encoder path.  MJPEG L1 switching needs no command at all
        # (the player just changes its stream URL).
        return "set_camera", {"camera": ctrl["camera"]}
    raise ControlError(f"not a parsed control instruction: {ctrl!r}")
