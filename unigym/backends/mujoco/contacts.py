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

from unigym.contacts.base import ContactGroupSpec, is_all_bodies
from unigym.contacts.discovery import contact_candidate_error, mujoco_body_inventory

# measured probe result (scripts/probe_contact_sign.py): the 6D force
# returned by mj_contactForce acts on the geom2 body.
MJ_FORCE_ON_GEOM2 = True


# --------------------------------------------------------------------------- #
# pure conversion helpers (unit-testable without a simulator)
# --------------------------------------------------------------------------- #
def contact_frame_to_world(frame3x3: np.ndarray, f_contact: np.ndarray) -> np.ndarray:
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
    rot = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rot.T @ np.asarray(vec_w, dtype=np.float64)


def contact_force_on_body(f_world: np.ndarray, body1: int, body2: int, body_id: int) -> np.ndarray:
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
    def prop_filter_paths(cls, name: str, articulated: bool) -> list[str]:
        """Native contact-filter address for a declared prop."""
        if articulated:
            return [f"Prop_{name}_.*"]
        return [f"Prop_{name}"]

    def __init__(self, specs: list[ContactGroupSpec], contact_assets=None, articulation_names=None):
        self.specs = list(specs)
        self._contact_assets = dict(contact_assets or {})
        self._articulation_names = (
            None if articulation_names is None else tuple(str(n) for n in articulation_names)
        )
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
            allowed_ids = self._articulation_body_ids(env, spec, body_names)
            filter_allowed_ids = self._filter_body_ids(env, allowed_ids)
            if is_all_bodies(spec.sensor_bodies):
                sensor_ids = self._all_body_ids(env, spec)
            else:
                sensor_ids = _resolve_bodies(
                    m,
                    body_names,
                    spec.sensor_bodies,
                    spec.name,
                    "sensor_bodies",
                    require_geoms=True,
                    inventory_fn=inventory,
                    allowed_ids=allowed_ids,
                )
            entry: dict = {"sensor_ids": sensor_ids}
            if spec.mode == "pairwise":
                filter_ids: list[int] = []
                for j, entry_regex in enumerate(spec.filter_bodies or ()):
                    ids = _resolve_bodies(
                        m,
                        body_names,
                        [entry_regex],
                        spec.name,
                        f"filter_bodies[{j}]",
                        require_geoms=True,
                        inventory_fn=inventory,
                        allowed_ids=filter_allowed_ids,
                    )
                    filter_ids.extend(ids)
                duplicates = sorted({i for i in filter_ids if filter_ids.count(i) > 1})
                if duplicates:
                    names = [body_names[i] for i in duplicates]
                    raise ValueError(
                        f"contact group {spec.name!r}: filter_bodies resolve "
                        f"to duplicate bodies {names} in articulation "
                        f"{spec.articulation or '<single>'!r}; each resolved "
                        f"filter body must be distinct"
                    )
                entry["filter_ids"] = filter_ids
            self._resolved[spec.name] = entry

    def _articulation_roots(self, env) -> dict[int, str]:
        """Model body id -> articulation name for every bound articulation."""
        m = self._model
        import mujoco

        bindings = getattr(env, "_mj_articulation_bindings", {}) or {}
        roots: dict[int, str] = {}
        for articulation, binding in bindings.items():
            base = str(getattr(binding, "backend_base_body", "") or "")
            bid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base)) if base else -1
            if bid > 0:
                roots[bid] = articulation
        if not roots:
            fallback = int(getattr(env, "_base_body_id", -1))
            if fallback > 0:
                name = next(iter(self._articulation_names or ()), None)
                if name is None:
                    name = next(iter(self._contact_assets), "robot")
                roots[fallback] = name
        return roots

    def _body_ids_under(self, roots: dict[int, str], articulation: str) -> set[int]:
        """Model body ids attributed to one articulation root subtree."""
        m = self._model
        selected = {bid for bid, name in roots.items() if name == articulation}
        ids: set[int] = set()
        for i in range(1, m.nbody):
            p = i
            while p > 0 and p not in roots:
                p = int(m.body_parentid[p])
            if p in selected:
                ids.add(i)
        return ids

    def _articulation_body_ids(
        self, env, spec: ContactGroupSpec, body_names: list[str]
    ) -> set[int] | None:
        """Body-id scope for an explicitly owned group (``None`` when implicit)."""
        if spec.articulation is None:
            return None
        roots = self._articulation_roots(env)
        if spec.articulation not in set(roots.values()):
            raise ValueError(
                f"contact group {spec.name!r}: articulation "
                f"{spec.articulation!r} has no MJ base-body binding "
                f"(bindings: {sorted(set(roots.values()))})"
            )
        ids = self._body_ids_under(roots, spec.articulation)
        if not ids:
            declared = [name for name in body_names if name]
            raise ValueError(
                f"contact group {spec.name!r}: articulation "
                f"{spec.articulation!r} resolves no model bodies; declared "
                f"bodies: {declared}"
            )
        return ids

    def _filter_body_ids(self, env, allowed_ids: set[int] | None) -> set[int] | None:
        """Filter-body scope for an owned group (``None`` when implicit).

        A pairwise filter may name the owner's bodies (prop-free robot
        contact) or a free body outside every articulation.  Free bodies
        include the world body (id 0: the floor/static geoms belong to it,
        so ``filter_bodies: [world]`` is a legal ground-contact filter) and
        declared props.  It must never silently match a DIFFERENT
        articulation's body; that keeps the owner declaration meaningful
        across a composed MJ model.
        """
        if allowed_ids is None:
            return None
        roots = self._articulation_roots(env)
        attributed: set[int] = set()
        for articulation in set(roots.values()):
            attributed |= self._body_ids_under(roots, articulation)
        free_ids = set(range(self._model.nbody)) - attributed
        return allowed_ids | free_ids

    def _all_body_ids(self, env, spec: ContactGroupSpec) -> list[int]:
        """Model body ids under the referenced articulation, model order.

        ``all`` means every rigid-body link under ONE articulation subtree.
        The scope comes from the backend articulation bindings (their
        compiled base bodies): each body is attributed to its nearest
        articulation root, so world bodies, props and other articulations are
        excluded.  An omitted owner is legal only when exactly one
        articulation is bound; otherwise the group must reference its
        articulation instead of silently unioning inventories.  Collisionless
        bodies stay in the B axis with zero force channels -- they are never
        a bind failure.
        """
        m = self._model
        roots = self._articulation_roots(env)
        names = sorted(set(roots.values()))
        if not roots:
            bodies = [m.body(i).name for i in range(1, m.nbody) if m.body(i).name]
            raise ValueError(
                f"contacts: all: no articulation base_body binding in the MJ "
                f"model; declared bodies: {bodies}"
            )
        if spec.articulation is not None:
            articulation = spec.articulation
            if articulation not in set(roots.values()):
                raise ValueError(
                    f"contact group {spec.name!r}: contacts: all references "
                    f"articulation {articulation!r}, which has no MJ base-body "
                    f"binding (bindings: {names})"
                )
        elif len(names) == 1:
            articulation = names[0]
        else:
            raise ValueError(
                f"contact group {spec.name!r}: contacts: all is ambiguous "
                f"across articulations {names}; reference the owner with "
                f"sensor_bodies: {{contact_group: {{articulation: <name>, "
                f"group: all}}}} (or declare articulation: <name>)"
            )
        ids = sorted(self._body_ids_under(roots, articulation))
        if not ids:
            bodies = [m.body(i).name for i in range(1, m.nbody) if m.body(i).name]
            raise ValueError(
                f"contacts: all: resolved no bodies under articulation "
                f"{articulation!r}; declared bodies: {bodies}"
            )
        return ids

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
                f_world = contact_frame_to_world(frames[i].reshape(3, 3), frc6[:3])
                f_on_sensor = contact_force_on_body(f_world, b1, b2, sensor_ids[s_idx])
                if spec.frame == "body":
                    f_on_sensor = rotate_to_frame(f_on_sensor, d.xquat[sensor_ids[s_idx]])
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
                f_on_sensor = contact_force_on_body(f_world, b1, b2, sensor_ids[s_idx])
                if spec.frame == "body":
                    f_on_sensor = rotate_to_frame(f_on_sensor, d.xquat[sensor_ids[s_idx]])
                # tangential relative velocity at the contact point (world)
                v1 = self._body_point_velocity(mj=m, d=d, body=b1, point=pos, vel_buf=vel)
                v2 = self._body_point_velocity(mj=m, d=d, body=b2, point=pos, vel_buf=vel)
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
    def _body_point_velocity(mj, d, body: int, point, vel_buf) -> np.ndarray:
        """World velocity of *body*'s frame origin + the omega lever to point.

        ``mj_objectVelocity`` world frame: [ang(3), lin(3)].
        """
        import mujoco

        # module-level mj function (an MjModel carries no mj_* methods --
        # the e2e caught this the first time a real contact went active)
        mujoco.mj_objectVelocity(mj, d, mujoco.mjtObj.mjOBJ_BODY, body, vel_buf, 0)
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


def _resolve_bodies(
    m,
    body_names: list[str],
    patterns: Sequence[str],
    group: str,
    arg: str,
    require_geoms: bool,
    inventory_fn=None,
    allowed_ids: set[int] | None = None,
) -> list[int]:
    """Body ids (model order) fullmatching ANY pattern; fail-fast empty.

    ``require_geoms`` rejects bodies without COLLISION geoms (zero geoms,
    or geoms with ``contype == 0`` -- visual-only bodies can never appear
    in data.ncon, a silent empty group would be a config bug).
    ``allowed_ids`` restricts resolution to one articulation's subtree (an
    explicitly owned group); ``None`` keeps the historical whole-model
    resolution.  Both failure modes refuse with a
    ``ContactCandidateError`` carrying the discovered body inventory
    (contacts/discovery.py) instead of a bare list-dump error.
    """
    compiled = [re.compile(p) for p in patterns]
    ids = [
        i
        for i, n in enumerate(body_names)
        if (allowed_ids is None or i in allowed_ids) and any(c.fullmatch(n) for c in compiled)
    ]
    if not ids:
        inventory = (inventory_fn or (lambda: []))()
        reason = (
            "matched no MJ body"
            if allowed_ids is None
            else "matched no MJ body in the owning articulation scope"
        )
        raise contact_candidate_error(group, arg, patterns, inventory, "mujoco", reason=reason)

    def _no_collision(i: int) -> bool:
        adr, n = int(m.body_geomadr[i]), int(m.body_geomnum[i])
        return not any(m.geom_contype[g] != 0 for g in range(adr, adr + n) if g >= 0)

    empty = [
        body_names[i] for i in ids if require_geoms and (m.body_geomnum[i] == 0 or _no_collision(i))
    ]
    if empty:
        inventory = (inventory_fn or (lambda: []))()
        raise contact_candidate_error(
            group,
            arg,
            empty,
            inventory,
            "mujoco",
            reason="have no collision geoms (contacts can never involve " "them)",
        )
    return ids
