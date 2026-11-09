"""Region-geometry term: keypoint-in-region (tiered spawn criteria).

The tiered-container criteria primitive behind
``keypoint_in_region_fraction`` (whitelisted in benchmarks/terms.py,
contract row benchmark_contract.md §9.2).  The official semantics this
module reproduces exactly, per frame:

* keypoints = the moving body's 8 bbox corners + 6 face centers + body
  center (MJCFObject._get_bbox_points, objects.py:174-222 +
  obj_in_region_with_keypoints, object_utils.py:247-276), built from
  the body's LOCAL bbox model -- the scaled bottom_site / top_site /
  horizontal_radius_site readings;
* the cloud is eroded toward the body center ``pts*0.8 + center*0.2``
  (object_utils.py:281);
* exactly ONE spawn region is tested per frame -- the geom whose world
  origin z is highest (get_highest_spawn_region, object_utils.py:393-402;
  the generated _check_success passes it as the single-element
  ``spawn_regions`` list, tabletop_24dc.py:631-641);
* the region is the spawn geom's world box/cylinder/sphere
  (calculate_spawn_region, object_utils.py:405-451), tested with
  inclusive bounds (obj_in_box/cylinder/sphere_region,
  object_utils.py:560-646);
* a frame succeeds when >= min_points keypoints are inside (the official
  min_num_points=2, object_utils.py:484).

Everything is structured data (no eval): recorded-key strings, numeric
vectors, whitelist enums.  numpy-only so the term stays offline-
scoreable.  The recorded-key readers mirror terms.py's ``_frames``/
``_check_frames`` semantics verbatim (kept local to avoid an import
cycle -- terms.py imports this module).
"""
from __future__ import annotations

import numpy as np

from .base import resolve_episode_key

# region geometry whitelist (the robocasa spawn-geom types)
REGION_TYPES = ("box", "cylinder", "sphere")
# per-frame region selection among a geom list (whitelist; the official
# criteria test exactly ONE spawn region per frame)
REGION_SELECTORS = ("highest_z",)

# official erosion constant (object_utils.py:281); the spec field pins
# it explicitly -- no silent default.
DEFAULT_ERODE = 0.8

_GEOM_KEYS = {"type", "pos", "quat", "size"}


# --------------------------------------------------------------------------- #
# recorded-key readers (terms.py _frames/_check_frames semantics)
# --------------------------------------------------------------------------- #
def _frames(ep, key, articulation=None) -> np.ndarray:
    """Episode key -> float ndarray (T, .); missing key -> ValueError."""
    if not isinstance(key, str):
        raise ValueError(f"key must be a recorded-key string, got {key!r}")
    key = resolve_episode_key(ep.data, key, articulation)
    if key not in ep.data:
        raise ValueError(
            f"episode data has no key '{key}' (available: {sorted(ep.data)})")
    return np.asarray(ep.data[key], dtype=float)


def _check_frames(x: np.ndarray, what: str) -> np.ndarray:
    """Empty episode (T=0) is a definition error, not a 0/NaN metric."""
    if x.ndim < 1 or x.shape[0] == 0:
        raise ValueError(f"{what}: empty episode (T=0) cannot be scored")
    return x


# --------------------------------------------------------------------------- #
# geometry (official transcription, vectorized)
# --------------------------------------------------------------------------- #
def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz quats -> (..., 3, 3) rotation matrices."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = (q[..., 0], q[..., 1], q[..., 2], q[..., 3])
    row0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                     2 * (x * z + w * y)], axis=-1)
    row1 = np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                     2 * (y * z - w * x)], axis=-1)
    row2 = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x),
                     1 - 2 * (x * x + y * y)], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def bbox_keypoints(bottom: np.ndarray, top: np.ndarray,
                   radius: np.ndarray) -> np.ndarray:
    """Local (15, 3) keypoint cloud of an object bbox model.

    Official derivation (MJCFObject._get_bbox_points, objects.py:194-222
    + obj_in_region_with_keypoints, object_utils.py:247-276): center =
    (bottom + top) / 2, half = [radius[0], radius[1],
    top[2] - center[2]], the eight corners in the fixed p0/px/py/pz
    order, then the six face centers built on the p0 edges plus the body
    center (mean of the corners) -- 15 keypoints total.
    """
    center = (bottom + top) / 2.0
    half = np.array([radius[0], radius[1], top[2] - center[2]])
    signs = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
                      [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]],
                     dtype=float)
    pts = center + signs * half
    u = pts[1] - pts[0]
    v = pts[2] - pts[0]
    w = pts[3] - pts[0]
    centers = [
        pts[0] + 0.5 * u + 0.5 * v,          # xy front
        pts[0] + 0.5 * u + 0.5 * v + w,      # xy back
        pts[0] + 0.5 * u + 0.5 * w,          # xz left
        pts[0] + 0.5 * u + 0.5 * w + v,      # xz right
        pts[0] + 0.5 * v + 0.5 * w,          # yz top
        pts[0] + 0.5 * v + 0.5 * w + u,      # yz bottom
        pts.mean(axis=0),                    # body center
    ]
    return np.concatenate([pts, np.asarray(centers)], axis=0)


def region_points_in(points: np.ndarray, rtype: str, r_pos: np.ndarray,
                     r_mat: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Official region-inside tests (object_utils.py:405-451, 560-646).

    calculate_spawn_region + obj_in_{box,cylinder,sphere}_region:
    batched per-point boolean, ``points`` (B, N, 3), ``r_pos`` (B, 3),
    ``r_mat`` (B, 3, 3) -> (B, N).
    """
    if rtype == "box":
        return _points_in_box(points, r_pos, r_mat, size)
    if rtype == "cylinder":
        return _points_in_cylinder(points, r_pos, r_mat, size)
    return np.linalg.norm(points - r_pos[:, None, :], axis=-1) <= size[0]


def _points_in_box(points, r_pos, r_mat, size):
    """The official box test (object_utils.py:560-592).

    Inclusive dot-product bounds on the three p0-anchored edge axes.
    """
    p0 = r_pos - size[0] * r_mat[:, 0] - size[1] * r_mat[:, 1] \
        - size[2] * r_mat[:, 2]
    axes = [p0 + 2 * size[k] * r_mat[:, k] - p0 for k in range(3)]
    dp = points - p0[:, None, :]
    terms = []
    for j, ax in enumerate(axes):
        d = np.einsum("bnj,bj->bn", dp, ax)
        bound = np.einsum("bj,bj->b", ax, ax)[:, None]
        terms.append((d >= 0.0) & (d <= bound))
    inside = terms[0]
    for term in terms[1:]:
        inside = inside & term
    return inside


def _points_in_cylinder(points, r_pos, r_mat, size):
    """The official cylinder test (object_utils.py:595-624).

    Base center, height vector along the geom z axis, inclusive bounds.
    """
    axis = r_mat[:, :, 2] * (2.0 * size[1])              # height vector
    height = np.linalg.norm(axis, axis=-1)
    axis_hat = axis / height[:, None]
    rel = points - r_pos[:, None, :]
    proj = np.einsum("bnj,bj->bn", rel, axis_hat)
    radial = np.linalg.norm(rel - proj[..., None] * axis_hat[:, None, :],
                            axis=-1)
    return ((proj >= 0.0) & (proj <= height[:, None])
            & (radial <= size[0]))


# --------------------------------------------------------------------------- #
# structural validation (construction gate; score-time reads stay at
# score time)
# --------------------------------------------------------------------------- #
def _num_vec(value, n):
    ok = (isinstance(value, (list, tuple)) and len(value) == n
          and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                  for x in value))
    return ok


def _validate_points(spec, where, errors):
    points = spec.get("points")
    if not isinstance(points, dict):
        errors.append(f"{where}.points: must be a dict {{pos, quat}}, got "
                      f"{points!r}")
        return
    extra = set(points) - {"pos", "quat"}
    if extra:
        errors.append(f"{where}.points: unknown keys {sorted(extra)}")
    for k in ("pos", "quat"):
        if not isinstance(points.get(k), str) or not points.get(k):
            errors.append(f"{where}.points.{k}: must be a recorded-key "
                          f"string, got {points.get(k)!r}")


def _validate_bbox(spec, where, errors):
    bbox = spec.get("bbox")
    if not isinstance(bbox, dict):
        errors.append(f"{where}.bbox: must be a dict {{bottom, top, "
                      f"radius}}, got {bbox!r}")
        return
    extra = set(bbox) - {"bottom", "top", "radius"}
    if extra:
        errors.append(f"{where}.bbox: unknown keys {sorted(extra)}")
    for key, n in (("bottom", 3), ("top", 3), ("radius", 2)):
        if not _num_vec(bbox.get(key), n):
            errors.append(f"{where}.bbox.{key}: must be {n} numbers, got "
                          f"{bbox.get(key)!r}")
    if _num_vec(bbox.get("radius"), 2) \
            and any(x < 0 for x in bbox["radius"]):
        errors.append(f"{where}.bbox.radius: must be non-negative, got "
                      f"{bbox['radius']!r}")


def _validate_geom(g, gi, errors):
    if not isinstance(g, dict):
        errors.append(f"{gi}: must be a dict, got {g!r}")
        return
    extra = set(g) - _GEOM_KEYS
    if extra:
        errors.append(f"{gi}: unknown keys {sorted(extra)} (geom form "
                      "allows type/pos/quat/size)")
    if g.get("type") not in REGION_TYPES:
        errors.append(f"{gi}.type: unknown region type {g.get('type')!r} "
                      f"(whitelist: {', '.join(REGION_TYPES)})")
    if not _num_vec(g.get("pos"), 3):
        errors.append(f"{gi}.pos: must be [x, y, z] numbers, got "
                      f"{g.get('pos')!r}")
    if g.get("quat") is not None and not _num_vec(g.get("quat"), 4):
        errors.append(f"{gi}.quat: must be 4 wxyz numbers, got "
                      f"{g.get('quat')!r}")
    if not _num_vec(g.get("size"), 3):
        errors.append(f"{gi}.size: must be 3 numbers, got "
                      f"{g.get('size')!r}")


def _validate_region(spec, where, errors):
    region = spec.get("region")
    if not isinstance(region, dict):
        errors.append(f"{where}.region: must be a dict {{selector, body, "
                      f"geoms}}, got {region!r}")
        return
    extra = set(region) - {"selector", "body", "geoms"}
    if extra:
        errors.append(f"{where}.region: unknown keys {sorted(extra)}")
    if region.get("selector", "highest_z") not in REGION_SELECTORS:
        errors.append(f"{where}.region.selector: unknown selector "
                      f"{region.get('selector')!r} (whitelist: "
                      f"{', '.join(REGION_SELECTORS)})")
    body = region.get("body")
    if not isinstance(body, dict):
        errors.append(f"{where}.region.body: must be a dict {{pos, quat}},"
                      f" got {body!r}")
    else:
        for k in ("pos", "quat"):
            if not isinstance(body.get(k), str) or not body.get(k):
                errors.append(f"{where}.region.body.{k}: must be a "
                              "recorded-key string, got "
                              f"{body.get(k)!r}")
    geoms = region.get("geoms")
    if not isinstance(geoms, list) or not geoms:
        errors.append(f"{where}.region.geoms: must be a non-empty list of "
                      f"geom dicts, got {geoms!r}")
        return
    for i, g in enumerate(geoms):
        _validate_geom(g, f"{where}.region.geoms[{i}]", errors)


def validate_spec(spec: dict, where: str) -> list[str]:
    """Structural checks for one keypoint-in-region spec.

    Every error names the offending field (score-time frame reads stay
    at score time).
    """
    errors: list[str] = []
    _validate_points(spec, where, errors)
    _validate_bbox(spec, where, errors)
    _validate_region(spec, where, errors)
    min_points = spec.get("min_points")
    if isinstance(min_points, bool) or not isinstance(min_points, int) \
            or min_points < 1:
        errors.append(f"{where}.min_points: must be an int >= 1, got "
                      f"{min_points!r}")
    erode = spec.get("erode")
    if isinstance(erode, bool) or not isinstance(erode, (int, float)) \
            or not (0.0 < float(erode) <= 1.0):
        errors.append(f"{where}.erode: must be a number in (0, 1], got "
                      f"{erode!r}")
    return errors


# --------------------------------------------------------------------------- #
# the per-frame series
# --------------------------------------------------------------------------- #
def _read_frames(ep, spec, what, articulation=None):
    """Frame-aligned recorded keys of one validated spec."""
    points = spec.get("points")
    body = spec["region"]["body"]
    p = _check_frames(_frames(ep, points["pos"], articulation), what)
    pq = _check_frames(_frames(ep, points["quat"], articulation), what)
    r_pos = _check_frames(_frames(ep, body["pos"], articulation), what)
    r_quat = _check_frames(_frames(ep, body["quat"], articulation), what)
    if p.shape[1] != 3 or pq.shape[1] != 4 or r_pos.shape[1] != 3 \
            or r_quat.shape[1] != 4:
        raise ValueError(
            f"{what}: expected pos (T, 3) and quat (T, 4) keys, got "
            f"{p.shape}/{pq.shape}/{r_pos.shape}/{r_quat.shape}")
    if p.shape[0] != pq.shape[0] or p.shape[0] != r_pos.shape[0] \
            or p.shape[0] != r_quat.shape[0]:
        raise ValueError(
            f"{what}: frame counts must align (points {p.shape[0]}, "
            f"quat {pq.shape[0]}, body {r_pos.shape[0]}/"
            f"{r_quat.shape[0]})")
    return p, pq, r_pos, r_quat


def _pick_regions(geom_world: np.ndarray, selector: str) -> np.ndarray:
    """Per-frame region index (the official selection rule).

    ``highest_z`` = get_highest_spawn_region: max world z of the geom
    frame origin (object_utils.py:393-402).
    """
    if selector == "highest_z":
        return np.argmax(geom_world[:, :, 2], axis=1)
    raise ValueError(f"unhandled selector {selector!r}")


def _test_selected(world, picked, geoms, geom_world, r_mat, min_points):
    """Boolean per frame: >= min_points keypoints inside the selection.

    Exactly ONE region is tested per frame -- the official
    spawn_regions=[highest] form (object_utils.py:472-488).
    """
    out = np.zeros(world.shape[0], dtype=bool)
    for k, (rtype, _gpos, gmat, gsize) in enumerate(geoms):
        rows = np.flatnonzero(picked == k)
        if rows.size == 0:
            continue
        rmat_k = r_mat[rows] @ gmat[None, :, :]
        inside = region_points_in(world[rows], rtype,
                                  geom_world[rows, k, :], rmat_k, gsize)
        out[rows] = (inside.sum(axis=1) >= min_points)
    return out


def keypoint_region_series(ep, spec: dict, what: str,
                           articulation=None) -> np.ndarray:
    """Per-frame boolean series behind ``keypoint_in_region_fraction``.

    Raises ValueError naming the offending spec field on definition
    errors (unknown keys, bad geometry, missing recorded keys, empty
    episode).  See the module docstring for the official semantics with
    citations.
    """
    errors = validate_spec(spec, what)
    if errors:
        raise ValueError("; ".join(errors))
    p, pq, r_pos, r_quat = _read_frames(ep, spec, what, articulation)
    local = bbox_keypoints(np.asarray(spec["bbox"]["bottom"], dtype=float),
                           np.asarray(spec["bbox"]["top"], dtype=float),
                           np.asarray(spec["bbox"]["radius"], dtype=float))
    r_mat = quat_wxyz_to_mat(r_quat)                        # (T, 3, 3)
    p_mat = quat_wxyz_to_mat(pq)
    erode = float(spec["erode"])
    world = np.einsum("tij,kj->tki", p_mat, local) + p[:, None, :]
    body_center = world.mean(axis=1, keepdims=True)         # (T, 1, 3)
    world = world * erode + body_center * (1.0 - erode)
    geoms = [(g["type"], np.asarray(g["pos"], dtype=float),
              quat_wxyz_to_mat(np.asarray(
                  g.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=float)),
              np.asarray(g["size"], dtype=float))
             for g in spec["region"]["geoms"]]
    gp = np.stack([g[1] for g in geoms])                    # (K, 3)
    geom_world = np.einsum("tij,kj->tki", r_mat, gp) + r_pos[:, None, :]
    picked = _pick_regions(geom_world, spec["region"].get(
        "selector", "highest_z"))
    min_points = int(spec["min_points"])
    return _test_selected(world, picked, geoms, geom_world, r_mat,
                          min_points)
