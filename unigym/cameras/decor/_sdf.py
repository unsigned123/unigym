"""Two-level triangle SDF against a set of collision geoms (probe4/5).

mujoco 3.11 ships no ``mju_rayMesh`` binding usable for signed depth, so
the cage generator measures "segment midpoint inside the gripper mesh"
with a numpy port of the probe's two-level scheme:

1. AABB coarse pass: distance to each geom's world AABB; points farther
   than the coarse radius never touch the exact triangles (168k-tri
   G1 hand mesh stays interactive).
2. Exact pass: point-triangle distance (region-based, Ericson) with the
   sign taken from the closest feature's face normal.

Sign bookkeeping (probe4 lesson -- mixing conventions silently marks 92%
of segments "inside"): this module's contract is ``depth > 0 == point is
INSIDE``; the exact pass returns ``-b * sgn`` with ``sgn`` the side of
the closest triangle, and unresolved points keep the AABB distance
negated (outside = negative).
"""
from __future__ import annotations

import itertools

import numpy as np

__all__ = ["GeomSDF", "geom_tris"]

_BOX_QUADS = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
              (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))


def geom_tris(model, data, geom_ids, box_type: int, mesh_type: int) -> np.ndarray:
    """World-frame triangles of the given geoms: meshes contribute their
    faces, boxes their 12 face triangles (probe5 collect_tris)."""
    tris: list[np.ndarray] = []
    for g in geom_ids:
        g = int(g)
        rot = np.asarray(data.geom_xmat[g]).reshape(3, 3)
        pos = np.asarray(data.geom_xpos[g])
        gtype = int(model.geom_type[g])
        if gtype == mesh_type:
            di = int(model.geom_dataid[g])
            va, vn = int(model.mesh_vertadr[di]), int(model.mesh_vertnum[di])
            fa, fn = int(model.mesh_faceadr[di]), int(model.mesh_facenum[di])
            verts = (model.mesh_vert[va:va + vn].astype(np.float64) @ rot.T
                     + pos)
            faces = model.mesh_face[fa:fa + fn].astype(int)
            tris.append(verts[faces])
        elif gtype == box_type:
            half = np.asarray(model.geom_size[g][:3], dtype=np.float64)
            grid = np.array(list(itertools.product([-1, 1], repeat=3))) * half
            corners = grid @ rot.T + pos
            for q in _BOX_QUADS:
                tris.append(corners[[q[0], q[1], q[2]]][None])
                tris.append(corners[[q[0], q[2], q[3]]][None])
    if not tris:
        raise ValueError("GeomSDF: no mesh/box geoms among the given ids")
    return np.concatenate(tris, axis=0)


class GeomSDF:
    """Signed distance to a geom set: ``depth(P) > 0`` == inside."""

    COARSE_RADIUS = 0.015   # AABB pre-filter (probe5: 15mm)
    EXACT_RADIUS = 0.010    # per-geom exact pass gate
    MIN_DEPTH = 3e-4        # |d| below this snaps to -MIN (sign stability)

    def __init__(self, tris: np.ndarray, boxes: list[tuple],
                 slices: list[tuple[int, int]]):
        """``tris`` (n, 3, 3) world triangles; ``boxes`` = per-geom
        ``(center, half)`` coarse level; ``slices`` = per-geom ``(lo, hi)``
        triangle ranges into ``tris`` (same order as ``boxes``)."""
        self.tris = np.asarray(tris, dtype=np.float64)
        self._ca = np.asarray([c for c, _ in boxes], dtype=np.float64)
        self._ha = np.asarray([h for _, h in boxes], dtype=np.float64)
        ta, tb, tc = (self.tris[:, 0], self.tris[:, 1], self.tris[:, 2])
        n = np.cross(tb - ta, tc - ta)
        self._tn = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
        self._slices = list(slices)

    @classmethod
    def from_geoms(cls, model, data, geom_ids, box_type: int,
                   mesh_type: int) -> GeomSDF:
        tris = geom_tris(model, data, geom_ids, box_type, mesh_type)
        boxes = []
        off = 0
        for g in geom_ids:
            g = int(g)
            gtype = int(model.geom_type[g])
            if gtype == box_type:
                n_t = 12
                rot = np.asarray(data.geom_xmat[g]).reshape(3, 3)
                half = np.abs(rot) @ np.asarray(model.geom_size[g][:3])
                center = np.asarray(data.geom_xpos[g])
            else:
                # mesh: slice its triangles out of `tris` by face count
                di = int(model.geom_dataid[g])
                n_t = int(model.mesh_facenum[di])
                sub = tris[off:off + n_t].reshape(-1, 3)
                lo, hi = sub.min(axis=0), sub.max(axis=0)
                center = (lo + hi) / 2
                half = (hi - lo) / 2
            boxes.append((center, half, n_t))
            off += n_t
        slices = []
        off = 0
        for _, _, n_t in boxes:
            slices.append((off, off + n_t))
            off += n_t
        return cls(tris, [(c, h) for c, h, _ in boxes], slices)

    # -- internals -------------------------------------------------------- #
    def _aabb_dist(self, points: np.ndarray) -> np.ndarray:
        d = (np.abs(points[:, None, :] - self._ca[None])
             - self._ha[None])
        return np.linalg.norm(np.maximum(d, 0), axis=2).min(axis=1)

    def _exact(self, points: np.ndarray, tri_idx: np.ndarray,
               chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Closest distance + side sign for each point against the given
        triangle subset (vectorized Ericson closest-point)."""
        n = len(points)
        best = np.full(n, np.inf)
        sign = np.ones(n)
        ta = self.tris[tri_idx, 0]
        tb = self.tris[tri_idx, 1]
        tc = self.tris[tri_idx, 2]
        tn = self._tn[tri_idx]
        for s in range(0, len(tri_idx), chunk):
            a, b, c = (ta[s:s + chunk][None], tb[s:s + chunk][None],
                       tc[s:s + chunk][None])
            ab, ac = b - a, c - a
            ap = points[:, None] - a
            d1 = (ab * ap).sum(-1)
            d2 = (ac * ap).sum(-1)
            bp = points[:, None] - b
            d3 = (ab * bp).sum(-1)
            d4 = (ac * bp).sum(-1)
            cp = points[:, None] - c
            d5 = (ab * cp).sum(-1)
            d6 = (ac * cp).sum(-1)
            va = d3 * d6 - d5 * d4
            vb = d5 * d2 - d1 * d6
            vc = d1 * d4 - d3 * d2
            den = va + vb + vc
            with np.errstate(divide="ignore", invalid="ignore"):
                v = vb / den
                w = vc / den
            inside = ((np.abs(den) > 1e-12) & (v >= 0) & (w >= 0)
                      & (v + w <= 1))
            q = a + ab * v[..., None] + ac * w[..., None]
            din = np.linalg.norm(points[:, None] - q, axis=-1)

            def seg(pa, pb):
                e = pb - pa
                t = np.clip(((points[:, None] - pa) * e).sum(-1)
                            / ((e * e).sum(-1) + 1e-18), 0, 1)
                proj = pa + t[..., None] * e
                return np.linalg.norm(points[:, None] - proj, axis=-1), proj

            dab, qab = seg(a, b)
            dac, qac = seg(a, c)
            dbc, qbc = seg(b, c)
            dout = np.minimum(np.minimum(dab, dac), dbc)
            dist = np.where(inside, din, dout)
            closest = np.where(inside[..., None], q, qab)
            pick_ac = (~inside) & (dac < np.minimum(dab, dbc))
            closest = np.where(pick_ac[..., None], qac, closest)
            pick_bc = (~inside) & (dbc <= np.minimum(dab, dac))
            closest = np.where(pick_bc[..., None], qbc, closest)
            flat = np.argmin(dist, axis=1)
            dmin = dist[np.arange(n), flat]
            upd = dmin < best
            best[upd] = dmin[upd]
            qi = closest[np.arange(n), flat]
            sg = np.sign(((points - qi) * tn[s:s + chunk][flat]).sum(-1))
            sign[upd] = np.where(np.abs(sg[upd]) > 0, sg[upd], 1.0)
        return best, sign

    # -- public ------------------------------------------------------------ #
    def depth(self, points: np.ndarray) -> np.ndarray:
        """Signed depth, positive inside (the module contract)."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        da = self._aabb_dist(points)
        depth = -da
        near = np.where(da < self.COARSE_RADIUS)[0]
        # resolved flag (probe4 lesson): a point INSIDE an AABB has
        # da == 0 -> depth starts at -0, so "|d_new| < |depth|" is never
        # true and the exact result would be dropped forever; the FIRST
        # geom must always take.
        resolved = np.zeros(len(points), dtype=bool)
        for gi, (c, h) in enumerate(zip(self._ca, self._ha)):
            if not len(near):
                break
            dd = np.abs(points[near] - c) - h
            sel = near[np.linalg.norm(np.maximum(dd, 0), axis=1)
                       < self.EXACT_RADIUS]
            if not len(sel):
                continue
            lo, hi = self._slices[gi]
            b, sg = self._exact(points[sel], np.arange(lo, hi))
            d_new = -b * sg
            take = (np.abs(d_new) < np.abs(depth[sel])) | ~resolved[sel]
            depth[sel[take]] = d_new[take]
            resolved[sel[take]] = True
        depth[np.abs(depth) < self.MIN_DEPTH] = -self.MIN_DEPTH
        return depth
