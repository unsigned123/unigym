"""Camera inspect face: one console eval -> spec + world pose (toolkit D).

`unigym camera inspect` (design doc §5.3) aggregates, for each
instantiated rig: mount chain (the yaml rig dict verbatim), resolved world
pose (fixed/free rigs; follow rigs report why not), intrinsics solved from
the spec (pinhole focal from hfov + aperture), and the convention line
(the repo's world convention: +X forward, +Z up, quat wxyz -- plus the
mount's composition story).  Implementation = ONE console-eval expression
against the injected ``env`` namespace (no new bridge command, no daemon
change); the same expression feeds the photo sidecar (contract
camera_contract §sidecar) with intrinsics + pose in one roundtrip.

Offline-testable: :func:`build_info_expr` (string), :func:`unwrap`
(deterministic Inspector formatting envelope), :func:`quat_axes`,
:func:`convention_line`, :func:`render` (pure doc -> text).
"""
from __future__ import annotations

import numpy as np

# repo camera convention (cameras/rigs.py: set_world_poses convention="world",
# lookat_quat forward=+X up=+Z, Hamilton wxyz) -- one line, single source here
CONVENTION_WRITE = "world (forward=+X, up=+Z, quat wxyz)"

_MOUNT_CONV = {
    "body": ("pose = root_pose ∘ mount offset (set_world_poses world) "
             "each step-tail"),
    "world": ("static world-anchored tripod; lookat composed by "
              "yaw-about-Z then pitch-about-Y (lookat_quat)"),
    "chase": ("yaw-rotated offset behind target_env's root, lookat=roi "
              "(lookat_quat), re-posed each step-tail"),
}

_INFO_EXPR = """\
_cm = env.camera_manager
_doc = {'cameras': {}}
if _cm is not None:
    _want = __NAMES__
    for _s in _cm.specs:
        _n = str(_s.name)
        if _want and _n not in _want:
            continue
        _rig = _s.rig or {}
        _e = {'rig_type': str(_rig.get('type', '')).lower(),
              'mount': {}, 'world': None, 'world_error': None}
        for _k, _v in _rig.items():
            if isinstance(_v, (list, tuple)):
                _e['mount'][str(_k)] = [float(x) for x in _v]
            elif _v is None or isinstance(_v, (int, float, str, bool)):
                _e['mount'][str(_k)] = _v
            else:
                _e['mount'][str(_k)] = str(_v)
        _e['width'] = int(_s.width)
        _e['height'] = int(_s.height)
        _e['hfov_deg'] = float(_s.hfov_deg)
        _e['aperture_mm'] = float(_s.horizontal_aperture)
        _e['focal_mm'] = float(_s.focal_length())
        _e['types'] = [str(t) for t in _s.types]
        _e['interval'] = int(_s.interval)
        _tf = getattr(_s, 'target_fps', None)
        _e['target_fps'] = None if _tf is None else float(_tf)
        _e['near_clip'] = float(_s.near_clip)
        _e['far_clip'] = float(_s.far_clip)
        _e['instance'] = str(_s.instance)
        _e['kind'] = str(_s.kind)
        try:
            _r = _cm.get_world_pose(_n)
            _e['world'] = {'pos': [float(x) for x in _r['pos']],
                           'quat_wxyz': [float(x) for x in
                                         _r['quat_wxyz']]}
        except Exception as _ex:
            _e['world_error'] = type(_ex).__name__ + ': ' + str(_ex)[:90]
        _doc['cameras'][_n] = _e
_doc['instantiated'] = sorted(_doc['cameras'])
_doc
"""


def build_info_expr(names: list[str] | None = None) -> str:
    """The console-eval expression (unit-locked against a fake manager)."""
    return _INFO_EXPR.replace("__NAMES__", repr(list(names or [])))


def unwrap(doc):
    """Strip the Inspector's deterministic {"dict": ...}/{"inline": ...}
    formatting envelope into plain python values."""
    if isinstance(doc, dict):
        if len(doc) == 1 and "dict" in doc:
            return unwrap(doc["dict"])
        if len(doc) == 1 and "inline" in doc:
            return doc["inline"]
        return {k: unwrap(v) for k, v in doc.items()}
    return doc


def quat_axes(quat_wxyz) -> tuple[list, list]:
    """(fwd, up) unit vectors of a world-convention (w,x,y,z) quat
    (the same algebra as CameraManager.get_world_pose's fwd)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    fwd = (1.0 - 2.0 * (y * y + z * z),
           2.0 * (x * y + w * z),
           2.0 * (x * z - w * y))
    up = (2.0 * (x * y - w * z),
          1.0 - 2.0 * (x * x + z * z),
          2.0 * (y * z + w * x))
    n = np.linalg.norm(fwd)
    fwd = [float(v) / n for v in fwd]
    n = np.linalg.norm(up)
    up = [float(v) / n for v in up]
    return fwd, up


def convention_line(cam: dict) -> str:
    """The convention story for one camera doc (design §5.3 required fields)."""
    rig = cam.get("rig_type", "?")
    chain = _MOUNT_CONV.get(rig, "rig-specific composition")
    if rig == "body" and cam.get("mount", {}).get("body"):
        chain = ("kinematic-tree mount: prim parented under "
                 f"'{cam['mount']['body']}' -- the USD/link chain owns the "
                 f"pose (upstream-authored offsets ride along)")
    return f"write={CONVENTION_WRITE}; mount: {chain}"


def _fmt_vec(v, nd=3) -> str:
    return "[" + ",".join(f"{x:.{nd}f}" for x in v) + "]"


def render(doc: dict, name: str | None = None) -> str:
    """One-screen report for one camera (or all when name is None)."""
    cams = doc.get("cameras") or {}
    lines = []
    for cam_name, cam in cams.items():
        if name and cam_name != name:
            continue
        lines.append(f"camera '{cam_name}'  rig={cam.get('rig_type')}  "
                     f"(instance={cam.get('instance')}, "
                     f"kind={cam.get('kind')}, "
                     f"fps={cam.get('target_fps')}, "
                     f"interval={cam.get('interval')} (deprecated), "
                     f"types={'+'.join(cam.get('types') or [])})")
        mount = cam.get("mount") or {}
        mtxt = "  ".join(f"{k}={_fmt_vec(v) if isinstance(v, list) else v}"
                         for k, v in mount.items()) or "-"
        lines.append(f"  mount:    {mtxt}")
        world = cam.get("world")
        if world:
            fwd, up = quat_axes(world["quat_wxyz"])
            lines.append(f"  world:    pos={_fmt_vec(world['pos'])} "
                         f"quat_wxyz="
                         f"{_fmt_vec(world['quat_wxyz'], 4)} "
                         f"fwd={_fmt_vec(fwd)} up={_fmt_vec(up)}")
        else:
            lines.append(f"  world:    (no read-back: "
                         f"{cam.get('world_error')})")
        lines.append(f"  intrins:  {cam.get('width')}x{cam.get('height')} "
                     f"hfov={cam.get('hfov_deg')}deg "
                     f"(f={cam.get('focal_mm'):.2f}mm, "
                     f"aperture={cam.get('aperture_mm'):.3f}mm) "
                     f"near={cam.get('near_clip')} far={cam.get('far_clip')}")
        lines.append(f"  convention: {convention_line(cam)}")
    return "\n".join(lines)


def doc_to_sidecar_facts(doc: dict, cam_name: str) -> dict:
    """Extract (intrinsics, world, convention, rig) for one camera from an
    :func:`build_info_expr` doc -- the sidecar's photo-metadata payload
    (camera_contract §sidecar)."""
    cam = (doc.get("cameras") or {}).get(cam_name) or {}
    world = cam.get("world")
    if world:
        fwd, up = quat_axes(world["quat_wxyz"])
        world = {**world, "fwd": fwd, "up": up}
    return {
        "rig": cam.get("rig_type"),
        "mount": cam.get("mount"),
        "world": world,
        "world_error": cam.get("world_error"),
        "convention": convention_line(cam) if cam else None,
        "intrinsics": ({"w": cam.get("width"), "h": cam.get("height"),
                        "hfov_deg": cam.get("hfov_deg"),
                        "model": "pinhole"} if cam else None),
    }
