"""cage_grid generator (probe4 B4 + probe5 single-pass, ported verbatim).

Three stacked encoding layers (probe4 verdict: color > density >
structure; depth is the only evidence-grade channel):

* density: per-face quadtree (base 2 levels) refined 2 extra levels
  inside the ``r_dense`` ball around the contact centroid;
* color: per-segment lerp cold (far) -> warm (near) by the midpoint's
  distance to the centroid, normalized by the target half-diagonal;
* depth tint: segments whose midpoint is INSIDE a reference body's
  collision mesh (two-level SDF) shade shallow->deep magenta.

The layout derives from the target's BOX ``geom_size`` (probe4 lesson:
never from the body AABB -- rotated boxes lie).  Deterministic: same
context -> same segments (the quadtree and endpoint dedup are stable).

v056 freeze-paradigm fallback (design §3): with NO matching active
contact the encoding's meaning (distance to the live bite) is gone, so
the generator returns a NEUTRAL ruler (uniform gray, anchored at the
body center) instead of raising -- one behavior branch, zero new
params.  Contact present -> the full encoding, bit-identical to
pre-v056.
"""
from __future__ import annotations

import numpy as np

from ..appearance import parse_color
from . import (_check_keys, _pos_float, _norm_contact_select,
               DecorContext, register_gen)

__all__ = ["CageGrid"]

# probe5 color constants (findings.txt §D): the acceptance replay compares
# against probe5 finals, so the defaults ARE the probe values.
CAGE_PARAM_KEYS = ("r_dense", "seg", "radius", "color_near", "color_far",
                   "depth_tint", "contact_select")
WARM = np.array([1.00, 0.30, 0.08])    # default color_near  (#ff4d14)
COLD = np.array([0.02, 0.38, 0.50])    # default color_far   (#056180)
MAG_D = np.array([0.85, 0.05, 0.55])   # depth tint, shallow (#d90d8c)
MAG_L = np.array([1.00, 0.72, 1.00])   # depth tint, deep    (#ffb8ff)
NEUTRAL = np.array([0.55, 0.55, 0.55])  # no-contact fallback ruler color

_AXJ = {0: 1, 1: 2, 2: 0}
_AXK = {0: 2, 1: 0, 2: 1}


def _face_point(axis, sgn, half, pos, rot, u, v):
    loc = np.zeros(3)
    loc[axis] = sgn * half[axis]
    loc[_AXJ[axis]] = u * half[_AXJ[axis]]
    loc[_AXK[axis]] = v * half[_AXK[axis]]
    return pos + rot @ loc


def _quadtree_face(axis, sgn, half, pos, rot, center, r_dense,
                   base_depth=2, extra=2):
    """One face's quadtree leaves -> corner segments (probe5
    cage_segments; shared-edge dedup by rounded endpoints)."""
    cells = [((-1.0, -1.0, 1.0, 1.0), 0)]
    leaves = []
    while cells:
        (u0, v0, u1, v1), dep = cells.pop()
        cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
        ctr = _face_point(axis, sgn, half, pos, rot, cu, cv)
        near = np.linalg.norm(ctr - center) < r_dense
        if dep < base_depth or (near and dep < base_depth + extra):
            cells += [((u0, v0, (u0 + u1) / 2, (v0 + v1) / 2), dep + 1),
                      (((u0 + u1) / 2, v0, u1, (v0 + v1) / 2), dep + 1),
                      ((u0, (v0 + v1) / 2, (u0 + u1) / 2, v1), dep + 1),
                      (((u0 + u1) / 2, (v0 + v1) / 2, u1, v1), dep + 1)]
        else:
            leaves.append((u0, v0, u1, v1))
    segs = []
    seen = set()
    for (u0, v0, u1, v1) in leaves:
        corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        for a in range(4):
            p1 = _face_point(axis, sgn, half, pos, rot, *corners[a])
            p2 = _face_point(axis, sgn, half, pos, rot,
                             *corners[(a + 1) % 4])
            key = (tuple(np.round(p1, 4)), tuple(np.round(p2, 4)))
            if key in seen or (key[1], key[0]) in seen:
                continue
            seen.add(key)
            segs.append((p1, p2))
    return segs


def _discretize(segs, seg_len):
    out = []
    for p1, p2 in segs:
        length = float(np.linalg.norm(p2 - p1))
        n = max(1, int(round(length / seg_len)))
        for i in range(n):
            out.append((p1 + (p2 - p1) * i / n,
                        p1 + (p2 - p1) * (i + 1) / n))
    return out


@register_gen("cage_grid")
class CageGrid:
    """Wireframe-cage decor over a BOX target (probe4/5 recipe)."""

    @staticmethod
    def validate(params, label="cage_grid"):
        _check_keys(params, CAGE_PARAM_KEYS, label)
        r_dense = _pos_float(params, "r_dense", 0.030, label)
        seg = _pos_float(params, "seg", 0.004, label)
        radius = _pos_float(params, "radius", 0.0005, label)
        near = parse_color(params.get("color_near", WARM.tolist()))
        far = parse_color(params.get("color_far", COLD.tolist()))
        tint = params.get("depth_tint")
        norm_tint = None
        if tint is not None:
            if not isinstance(tint, dict):
                raise ValueError(f"{label}: depth_tint must be a dict "
                                 f"{{mesh_body, band, tint_shallow, "
                                 f"tint_deep}}, got {tint!r}")
            unknown = sorted(set(tint) - {"mesh_body", "band",
                                          "tint_shallow", "tint_deep"})
            if unknown:
                raise ValueError(f"{label}: unknown depth_tint key(s) "
                                 f"{unknown}; legal: ['band', 'mesh_body', "
                                 f"'tint_deep', 'tint_shallow']")
            if not isinstance(tint.get("mesh_body"), str) or \
                    not tint["mesh_body"]:
                raise ValueError(
                    f"{label}: depth_tint.mesh_body must be a body-name "
                    f"prefix (the SDF reference mesh), got "
                    f"{tint.get('mesh_body')!r}")
            norm_tint = {
                "mesh_body": tint["mesh_body"],
                "band": _pos_float(tint, "band", 0.006, label),
                "tint_shallow": parse_color(tint.get("tint_shallow",
                                                     MAG_D.tolist())),
                "tint_deep": parse_color(tint.get("tint_deep",
                                                  MAG_L.tolist())),
            }
        return {"r_dense": r_dense, "seg": seg, "radius": radius,
                "color_near": near, "color_far": far,
                "depth_tint": norm_tint,
                "contact_select": _norm_contact_select(params, label)}

    @staticmethod
    def generate(params, target, ctx: DecorContext):
        from ._sdf import GeomSDF
        body, geoms = ctx.target_geoms(
            target, collision_only=True,
            types={int(ctx.box_type)})   # cage layout derives from size
        g = geoms[0]
        model = ctx.require_model()
        data = ctx.require_data()
        half = np.asarray(model.geom_size[g][:3], dtype=np.float64)
        pos = np.asarray(data.geom_xpos[g])
        rot = np.asarray(data.geom_xmat[g]).reshape(3, 3)
        # v056 pure-ruler fallback (design v056-freeze-paradigm §3): the
        # full warm/cold/tint encoding MEANS "distance to the live bite"
        # -- with no matching contact that meaning is gone, so the cage
        # degrades to a NEUTRAL ruler anchored at the body center.  One
        # behavior branch, zero new params; the contact-present path is
        # bit-identical to the pre-v056 generator.
        idx = ctx.contacts_where(params["contact_select"])
        if len(idx):
            center = ctx.contact_pos[idx].mean(axis=0)
        else:
            segs = []
            for axis in range(3):
                for sgn in (1.0, -1.0):
                    segs += _quadtree_face(axis, sgn, half, pos, rot,
                                           pos, params["r_dense"])
            pieces = _discretize(segs, params["seg"])
            rgba = np.tile(np.concatenate([NEUTRAL, [1.0]]),
                           (len(pieces), 1))
            radius = params["radius"]
            return [{"type": "capsule", "from": a, "to": b,
                     "radius": radius, "rgba": rgba[k]}
                    for k, (a, b) in enumerate(pieces)]
        segs = []
        for axis in range(3):
            for sgn in (1.0, -1.0):
                segs += _quadtree_face(axis, sgn, half, pos, rot, center,
                                       params["r_dense"])
        pieces = _discretize(segs, params["seg"])
        mids = np.asarray([(a + b) / 2 for a, b in pieces])
        dmax = float(np.linalg.norm(half))
        t = np.clip(np.linalg.norm(mids - center, axis=1) / dmax, 0, 1)
        warm = np.asarray(params["color_near"])
        cold = np.asarray(params["color_far"])
        cols = warm[None] * (1 - t[:, None] ** 1.5) \
            + cold[None] * t[:, None] ** 1.5
        rgba = np.concatenate([cols, np.ones((len(mids), 1))], axis=1)
        tint = params["depth_tint"]
        if tint is not None:
            _, ref = ctx.target_geoms(tint["mesh_body"], collision_only=True)
            sdf = GeomSDF.from_geoms(model, data, ref,
                                     int(ctx.box_type), int(ctx.mesh_type))
            dep = sdf.depth(mids)
            inside = dep > 0
            u = np.clip(dep[inside] / tint["band"], 0, 1)
            shallow = np.asarray(tint["tint_shallow"])
            deep = np.asarray(tint["tint_deep"])
            mix = shallow[None] * (1 - u[:, None]) + deep[None] * u[:, None]
            rgba[inside, :3] = mix
        radius = params["radius"]
        return [{"type": "capsule", "from": a, "to": b, "radius": radius,
                 "rgba": rgba[k]} for k, (a, b) in enumerate(pieces)]
