"""mujoco contact backend: data.ncon scan + per-(sensor, filter) aggregation.

This module is the ONLY module in the contacts package allowed to touch the
mujoco API; it is imported lazily by ``build_contact_manager(backend="mujoco")``
so the package stays host-import-safe.

Semantics (measured, probe ``scripts/probe_contact_sign.py``):

* ``mj_contactForce`` returns the contact force acting on the **geom2
  body**, expressed in the CONTACT frame (rows of ``con.frame``:
  normal, tangent1, tangent2, world-oriented).  The force on the geom1
  body is the negation (Newton's third law).  ``contact_force_on_body``
  implements exactly this sign rule — the single place in the package
  where the convention lives.
* contact-frame -> world rotation: ``f_world = frame.T @ f_contact[:3]``
  (frame rows are the contact axes; verified by unit test CU-4 and the
  probe).
* ``general`` groups: every contact touching ANY sensor body accumulates
  into that body's slot (net force over all counterparts + all contact
  points).  ``pairwise`` groups: a contact contributes iff one side is a
  sensor body and the other side is a filter body; it accumulates into the
  (sensor, filter) slot (multiple contact points between the same pair
  sum — the net pair force).
* B ordering: MJ model body order filtered by the sensor_bodies regexes.
  F ordering: ``filter_bodies`` entry order, model body order within an
  entry.
* ``frame: body``: the world force is rotated into the SENSOR body frame
  (``R_b^T @ f_world``, ``R_b`` from ``data.xquat[body_id]``, wxyz).
  ``frame: world`` is direct.
* N is always 1 (MujocoEnv is single-env); read_raw returns the
  batched (1, B, ...) array so the manager layer is backend-agnostic.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing-only (mujoco is imported lazily)
    import mujoco

from unigym.contacts.base import ContactGroupSpec
from unigym.contacts.discovery import (contact_candidate_error,
                                        mujoco_body_inventory)

# measured probe result (scripts/probe_contact_sign.py): the 6D force
# returned by mj_contactForce acts on the geom2 body.
MJ_FORCE_ON_GEOM2 = True


# --------------------------------------------------------------------------- #
# pure conversion helpers (unit-testable without a simulator)
# --------------------------------------------------------------------------- #
def contact_frame_to_world(frame3x3: np.ndarray,
                           f_contact: np.ndarray) -> np.ndarray:
    """Contact-frame force -> world frame.

    ``frame3x3`` rows are the contact axes (normal, t1, t2) in world
    coordinates, so the world vector is ``frame.T @ f``.
    """
    return np.asarray(frame3x3).reshape(3, 3).T @ np.asarray(f_contact)[:3]


def rotate_to_frame(vec_w: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """World vector -> body frame via ``R^T @ v`` (quat wxyz, mj convention).

    Same quaternion convention as ``env/backends/mujoco_env.py`` (freejoint
    qpos[3:7]); the matrix rows map body axes to world, so body-frame
    components are ``R^T @ v_world``.
    """
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    rot = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return rot.T @ np.asarray(vec_w, dtype=np.float64)


def contact_force_on_body(f_world: np.ndarray, body1: int, body2: int,
                          body_id: int) -> np.ndarray:
    """Force applied to ``body_id`` from one contact (sign convention).

    ``f_world`` is the raw mj_contactForce world rotation (the geom2-body
    force).  Sensor on the geom2 side -> as-is; geom1 side -> negation;
    body not involved -> zeros.
    """
    if body_id == body2 and MJ_FORCE_ON_GEOM2:
        return f_world
    if body_id == body1 and MJ_FORCE_ON_GEOM2:
        return -f_world
    return np.zeros(3)


# --------------------------------------------------------------------------- #
# backend object
# --------------------------------------------------------------------------- #
class MujocoContactBackend:
    """Backend object for all groups of one run (mujoco, N=1)."""

    backend_key = "mujoco"

    @classmethod
    def group_bodies(cls, gval, path: str):
        """Select this backend's value from a body-parts group declaration.

        A flat list (or a non-mapping) is shared by every backend; a
        per-backend mapping is keyed by :attr:`backend_key`.
        """
        if not isinstance(gval, dict):
            return gval
        if cls.backend_key not in gval:
            group = path.rsplit(".", 1)[-1]
            raise ValueError(
                f"body_parts path {path!r}: group {group!r} declares no "
                f"{cls.backend_key!r} bodies (available: {sorted(gval)})")
        return gval[cls.backend_key]

    @classmethod
    def prop_filter_paths(cls, name: str, articulated: bool) -> list[str]:
        """Native contact-filter address for a declared prop."""
        if articulated:
            return [f"Prop_{name}_.*"]
        return [f"Prop_{name}"]

    def __init__(self, specs: list[ContactGroupSpec]):
        self.specs = list(specs)
        self._resolved: dict[str, dict] = {}

    # typed internal handles: bound by ``bind_env`` before any read.
    _model: mujoco.MjModel
    _data: mujoco.MjData
    _env: Any

    # ---------------------------------------------------------------- #
    # lifecycle
    # ---------------------------------------------------------------- #
    def setup_scene(self) -> dict:
        """No scene elements needed — contacts come from data.ncon scans."""
        return {}

    def bind_env(self, env) -> None:
        """Index the MJ model: resolve sensor/filter body ids per group.

        Unresolvable declarations refuse with a
        ``ContactCandidateError`` carrying the model's discovered body
        inventory (contacts/discovery.py) -- a typo never fails bare.
        """
        m = env.mj_model
        self._model = m
        self._data = env.mj_data
        self._env = env

        def inventory():
            return mujoco_body_inventory(m)

        body_names = [m.body(i).name for i in range(m.nbody)]
        for spec in self.specs:
            sensor_ids = _resolve_bodies(m, body_names, spec.sensor_bodies,
                                         spec.name, "sensor_bodies",
                                         require_geoms=True,
                                         inventory_fn=inventory)
            entry: dict = {"sensor_ids": sensor_ids}
            if spec.mode == "pairwise":
                filter_ids: list[int] = []
                for j, entry_regex in enumerate(spec.filter_bodies or ()):
                    ids = _resolve_bodies(
                        m, body_names, [entry_regex],
                        spec.name, f"filter_bodies[{j}]", require_geoms=True,
                        inventory_fn=inventory)
                    filter_ids.extend(ids)
                entry["filter_ids"] = filter_ids
            self._resolved[spec.name] = entry

    def group_axes(self, spec: ContactGroupSpec) -> tuple[int, int]:
        """Resolved ``(B, F)`` sensor/filter counts (post bind_env).

        Feeds the obs catalog's per-key shape resolution; ``_resolved``
        entries are exactly what ``read_raw`` indexes, so the shapes can
        never drift from what a step actually produces.
        """
        res = self._resolved[spec.name]
        return len(res["sensor_ids"]), len(res.get("filter_ids", ()))

    def read_raw(self, spec: ContactGroupSpec) -> np.ndarray:
        """Raw forces in ``spec.frame`` (float64 numpy, freshly allocated).

        Shapes: ``(N, B, 3)`` general, ``(N, B, F, 3)`` pairwise.
        """
        import mujoco

        m = self._model
        datas = getattr(self._env, "mj_datas", None) or [self._data]
        n = len(datas)
        res = self._resolved[spec.name]
        sensor_ids = res["sensor_ids"]
        B = len(sensor_ids)
        if spec.mode == "pairwise":
            filter_ids = res["filter_ids"]
            raw = np.zeros((n, B, len(filter_ids), 3), dtype=np.float64)
        else:
            filter_ids = []
            raw = np.zeros((n, B, 3), dtype=np.float64)
        frc6 = np.zeros(6, dtype=np.float64)
        for e, d in enumerate(datas):
            # Array views (not the per-struct ``contact[i]``/``geom1`` aliases):
            # mujoco's cached struct aliases go stale when a backend reallocates
            # the contact buffer (mjwarp's ``get_data_into`` -> _realloc_con_efc);
            # the ``geom``/``pos``/``frame``/``dist`` arrays are re-read live.
            ncon = int(d.ncon)
            geoms = np.asarray(d.contact.geom)[:ncon]
            frames = np.asarray(d.contact.frame)[:ncon].reshape(-1, 9)
            for i in range(ncon):
                geom1, geom2 = int(geoms[i, 0]), int(geoms[i, 1])
                b1 = int(m.geom_bodyid[geom1])
                b2 = int(m.geom_bodyid[geom2])
                # which side is the sensor (each contact has at most one slot;
                # sensor bodies never overlap filter bodies by construction)
                s_idx = None
                other = None
                if b1 in sensor_ids:
                    s_idx, other = sensor_ids.index(b1), b2
                elif b2 in sensor_ids:
                    s_idx, other = sensor_ids.index(b2), b1
                if s_idx is None:
                    continue
                if spec.mode == "pairwise":
                    if other not in filter_ids:
                        continue
                    f_idx = filter_ids.index(other)
                else:
                    if other in sensor_ids:
                        continue  # self-pair would double count; not expected
                    f_idx = None
                mujoco.mj_contactForce(m, d, i, frc6)
                f_world = contact_frame_to_world(
                    frames[i].reshape(3, 3), frc6[:3])
                f_on_sensor = contact_force_on_body(f_world, b1, b2,
                                                    sensor_ids[s_idx])
                if spec.frame == "body":
                    f_on_sensor = rotate_to_frame(
                        f_on_sensor, d.xquat[sensor_ids[s_idx]])
                if f_idx is None:
                    raw[e, s_idx] += f_on_sensor
                else:
                    raw[e, s_idx, f_idx] += f_on_sensor
        return raw

    def read_detail(self, spec: ContactGroupSpec) -> list[dict]:
        """Per-contact records for the decision package (design §2/§3).

        One dict per ACTIVE contact point matching the group,
        each with the bodies, the world contact position, the normal
        (geom1 -> geom2, world), the contact force on the SENSOR body
        (``spec.frame``) with its magnitude and the tangential relative
        velocity ``v_t`` (m/s, computed here from the live mjData --
        the engine-side readout path; the slip criterion's source).

        JSON-ready floats; freshly allocated.  ``v_rel`` is body2 w.r.t.
        body1 at the contact point (``mj_objectVelocity`` linear + the
        omega lever arm), tangential = the normal projected out.
        """
        import mujoco

        m = self._model
        datas = getattr(self._env, "mj_datas", None) or [self._data]
        multi = len(datas) > 1
        res = self._resolved[spec.name]
        sensor_ids = res["sensor_ids"]
        filter_ids = res.get("filter_ids", [])
        out: list[dict] = []
        frc6 = np.zeros(6, dtype=np.float64)
        vel = np.zeros(6, dtype=np.float64)
        name_of = lambda b: m.body(b).name  # noqa: E731
        for e, d in enumerate(datas):
            ncon = int(d.ncon)
            geoms = np.asarray(d.contact.geom)[:ncon]
            frames = np.asarray(d.contact.frame)[:ncon].reshape(-1, 9)
            poss = np.asarray(d.contact.pos)[:ncon]
            dists = np.asarray(d.contact.dist)[:ncon]
            for i in range(ncon):
                geom1, geom2 = int(geoms[i, 0]), int(geoms[i, 1])
                b1 = int(m.geom_bodyid[geom1])
                b2 = int(m.geom_bodyid[geom2])
                s_idx = other = None
                if b1 in sensor_ids:
                    s_idx, other = sensor_ids.index(b1), b2
                elif b2 in sensor_ids:
                    s_idx, other = sensor_ids.index(b2), b1
                if s_idx is None:
                    continue
                if spec.mode == "pairwise":
                    if other not in filter_ids:
                        continue
                elif other in sensor_ids:
                    continue
                mujoco.mj_contactForce(m, d, i, frc6)
                frame = frames[i].reshape(3, 3)
                pos = poss[i]
                f_world = contact_frame_to_world(frame, frc6[:3])
                f_on_sensor = contact_force_on_body(f_world, b1, b2,
                                                    sensor_ids[s_idx])
                if spec.frame == "body":
                    f_on_sensor = rotate_to_frame(
                        f_on_sensor, d.xquat[sensor_ids[s_idx]])
                # tangential relative velocity at the contact point (world)
                v1 = self._body_point_velocity(mj=m, d=d, body=b1,
                                               point=pos, vel_buf=vel)
                v2 = self._body_point_velocity(mj=m, d=d, body=b2,
                                               point=pos, vel_buf=vel)
                v_rel = v2 - v1
                n = np.asarray(frame[0], dtype=np.float64)
                v_t_vec = v_rel - float(v_rel @ n) * n
                rec = {
                    "sensor": name_of(sensor_ids[s_idx]),
                    "other": name_of(other),
                    "bodies": [name_of(b1), name_of(b2)],
                    "pos": [float(v) for v in pos],
                    "normal": [float(v) for v in n],
                    "force": [float(v) for v in f_on_sensor],
                    "force_mag": float(np.linalg.norm(f_on_sensor)),
                    "v_t": float(np.linalg.norm(v_t_vec)),
                    "dist": float(dists[i]),
                }
                if multi:
                    rec["env"] = e
                out.append(rec)
        return out

    @staticmethod
    def _body_point_velocity(mj, d, body: int, point,
                             vel_buf) -> np.ndarray:
        """World velocity of *body*'s frame origin + the omega lever to point.

        ``mj_objectVelocity`` world frame: [ang(3), lin(3)].
        """
        import mujoco

        # module-level mj function (an MjModel carries no mj_* methods --
        # the e2e caught this the first time a real contact went active)
        mujoco.mj_objectVelocity(mj, d, mujoco.mjtObj.mjOBJ_BODY, body,
                                 vel_buf, 0)
        ang = np.asarray(vel_buf[0:3])
        lin = np.asarray(vel_buf[3:6])
        r = np.asarray(point) - np.asarray(d.xpos[body])
        return lin + np.cross(ang, r)

    def close(self) -> None:
        """Release the model/data handles (ContactBackend lifecycle)."""
        # Teardown boundary: deliberately invalidate the live handles (the
        # class annotations stay non-Optional for the read paths).
        for handle in ("_model", "_data", "_env"):
            self.__dict__[handle] = None
        self._resolved.clear()


def _resolve_bodies(m, body_names: list[str], patterns: Sequence[str],
                    group: str, arg: str, require_geoms: bool,
                    inventory_fn=None) -> list[int]:
    """Body ids (model order) fullmatching ANY pattern; fail-fast empty.

    ``require_geoms`` rejects bodies without COLLISION geoms (zero geoms,
    or geoms with ``contype == 0`` -- visual-only bodies can never appear
    in data.ncon, a silent empty group would be a config bug).
    Both failure modes refuse with a ``ContactCandidateError`` carrying the
    discovered body inventory (contacts/discovery.py) instead of
    a bare list-dump error.
    """
    compiled = [re.compile(p) for p in patterns]
    ids = [i for i, n in enumerate(body_names)
           if any(c.fullmatch(n) for c in compiled)]
    if not ids:
        inventory = (inventory_fn or (lambda: []))()
        raise contact_candidate_error(
            group, arg, patterns, inventory, "mujoco",
            reason="matched no MJ body")

    def _no_collision(i: int) -> bool:
        adr, n = int(m.body_geomadr[i]), int(m.body_geomnum[i])
        return not any(m.geom_contype[g] != 0
                       for g in range(adr, adr + n) if g >= 0)

    empty = [body_names[i] for i in ids
             if require_geoms and
             (m.body_geomnum[i] == 0 or _no_collision(i))]
    if empty:
        inventory = (inventory_fn or (lambda: []))()
        raise contact_candidate_error(
            group, arg, empty, inventory, "mujoco",
            reason="have no collision geoms (contacts can never involve "
                   "them)")
    return ids
