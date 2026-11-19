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
* B ordering: sensor refs in declaration order, MJ model body order within
  each ref.  F ordering: filter refs in declaration order, model body order
  within each ref.  No union and no reordering.
* ``frame: body``: the world force is rotated into the SENSOR body frame
  (``R_b^T @ f_world``, ``R_b`` from ``data.xquat[body_id]``, wxyz).
  ``frame: world`` is direct.
* read_raw scans every per-env ``MjData`` (``env.mj_datas``) and returns the
  batched ``(N, B, ...)`` array, so the manager layer is backend-agnostic and
  ``MujocoEnv`` at ``N > 1`` is served per env.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing-only (mujoco is imported lazily)
    import mujoco

from unigym.contacts.base import ContactGroupSpec
from unigym.contacts.inventory import ContactBodyAddress, ContactInventory, resolve_refs

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

    def __init__(
        self,
        specs: list[ContactGroupSpec],
        *,
        articulation_names=None,
        articulation_body_sets=None,
        prop_names=None,
        articulated_props=None,
        contact_assets=None,
    ):
        self.specs = list(specs)
        self._contact_assets = dict(contact_assets or {})
        self._articulation_names = (
            None if articulation_names is None else tuple(str(n) for n in articulation_names)
        )
        self._articulation_body_sets: Mapping[str, Mapping[str, Sequence[str]]] = dict(
            articulation_body_sets or {}
        )
        self._prop_names = [str(name) for name in (prop_names or ())]
        self._articulated_props = {str(name) for name in (articulated_props or ())}
        self._resolved: dict[str, dict] = {}
        self.inventory: Optional[ContactInventory] = None
        self._body_ids: dict[str, int] = {}

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
        """Build the native inventory and resolve every ref independently."""
        m = env.mj_model
        self._model = m
        self._data = env.mj_data
        self._env = env
        self._body_ids = {m.body(i).name: i for i in range(m.nbody) if m.body(i).name}
        self._roots = self._articulation_roots(env)
        self.inventory = self._build_inventory(m)
        self._row_by_id = {}
        for row in self.inventory.rows:
            bid = self._body_ids.get(row.native)
            if bid is not None:
                self._row_by_id[bid] = row
        for spec in self.specs:
            sensors = resolve_refs(
                self.inventory, spec.sensor_refs, axis="sensor_refs", group=spec.name
            )
            filters = resolve_refs(
                self.inventory, spec.filter_refs, axis="filter_refs", group=spec.name
            )
            sensor_ids = [self._body_ids[row.native] for row in sensors]
            filter_ids = [self._body_ids[row.native] for row in filters]
            for axis, ids, rows in (
                ("sensor_refs", sensor_ids, sensors),
                ("filter_refs", filter_ids, filters),
            ):
                duplicates = sorted({bid for bid in ids if ids.count(bid) > 1})
                if duplicates:
                    names = [m.body(bid).name for bid in duplicates]
                    keys = sorted(
                        {row.key for row in rows if self._body_ids[row.native] in duplicates}
                    )
                    raise ValueError(
                        f"contact group {spec.name!r}: {axis} resolve body ids "
                        f"{duplicates} ({names}) from multiple owners/rows "
                        f"{keys}; ownership is ambiguous"
                    )
            self._resolved[spec.name] = {
                "sensor_ids": sensor_ids,
                "filter_ids": filter_ids,
                "sensor_rows": sensors,
                "filter_rows": filters,
            }

    # ---------------------------------------------------------------- #
    # native inventory
    # ---------------------------------------------------------------- #
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

    def _body_ids_under(self, articulation: str) -> list[int]:
        """Model body ids attributed to one articulation root subtree."""
        m = self._model
        selected = {bid for bid, name in self._roots.items() if name == articulation}
        ids: list[int] = []
        for i in range(1, m.nbody):
            p = i
            while p > 0 and p not in self._roots:
                p = int(m.body_parentid[p])
            if p in selected:
                ids.append(i)
        return ids

    @staticmethod
    def _collision(m, bid: int) -> bool:
        adr, n = int(m.body_geomadr[bid]), int(m.body_geomnum[bid])
        return any(m.geom_contype[g] != 0 for g in range(adr, adr + n) if g >= 0)

    def _row(
        self,
        m,
        bid: int,
        owner: str,
        scope: Optional[str],
        body: str,
        *,
        root: bool,
        prim_path: Optional[str] = None,
    ) -> ContactBodyAddress:
        native = m.body(bid).name
        scope_part = f"{scope}/" if scope is not None else ""
        return ContactBodyAddress(
            owner=owner,
            scope=scope,
            body=body,
            native=native,
            prim_path=prim_path,
            collision=bool(m.body_geomnum[bid] > 0),
            contact_capable=self._collision(m, bid),
            key=f"{owner}/{scope_part}{body}",
            root=root,
        )

    def _build_inventory(self, m) -> ContactInventory:
        rows: list[ContactBodyAddress] = []
        # Declared props are classified FIRST: ownership is exclusive.  The
        # LONGEST declared prop name wins so prop "obj" never swallows
        # "Prop_obj_container" (declared prop "obj_container").  A prop nested
        # under an articulation (welded/attached) must not also appear as a
        # robot row, or a ref could resolve the same body twice.
        ordered_props = sorted(self._prop_names, key=len, reverse=True)
        prop_of: dict[int, tuple[str, str]] = {}
        for bid in range(1, m.nbody):
            native = m.body(bid).name
            matched = None
            body_name = None
            for name in ordered_props:
                root_name = f"Prop_{name}"
                if native == root_name:
                    matched, body_name = name, name
                    break
                prefix = root_name + "_"
                if native.startswith(prefix):
                    matched, body_name = name, native[len(prefix) :]
                    break
            if matched is not None:
                prop_of[bid] = (matched, body_name)
        prop_ids = set(prop_of)
        attributed: set[int] = set()
        articulations = list(dict.fromkeys(self._roots.values()))
        if self._articulation_names:
            ordered = [name for name in self._articulation_names if name in articulations]
            ordered += [name for name in articulations if name not in ordered]
            articulations = ordered
        for articulation in articulations:
            ids = self._body_ids_under(articulation)
            attributed.update(ids)
            roots = {bid for bid, name in self._roots.items() if name == articulation}
            for bid in ids:
                if bid in prop_ids:  # the declared prop owns this body
                    continue
                name = m.body(bid).name
                if not name:
                    continue
                rows.append(
                    self._row(
                        m,
                        bid,
                        "robot",
                        articulation,
                        name,
                        root=bid in roots,
                    )
                )
        # Prop rows: the scope ROOT is the top body of that prop's own body
        # set (not necessarily parented to the world), so a prop nested under
        # an articulation still resolves its default/root body.
        for bid in sorted(prop_of):
            matched, body_name = prop_of[bid]
            owner = "fixture" if matched in self._articulated_props else "object"
            rows.append(
                self._row(
                    m,
                    bid,
                    owner,
                    matched,
                    body_name,
                    root=int(m.body_parentid[bid]) not in prop_ids,
                )
            )
        # scene supports: free bodies outside articulations and props
        for bid in range(1, m.nbody):
            if bid in attributed or bid in prop_ids:
                continue
            native = m.body(bid).name
            if not native:
                continue
            rows.append(
                self._row(
                    m,
                    bid,
                    "support",
                    native,
                    native,
                    root=bool(int(m.body_parentid[bid]) == 0),
                )
            )
        # native environment bodies: the world body owns static ground geoms
        world = int(getattr(m, "body_worldid", 0))
        if world == 0:
            rows.append(
                ContactBodyAddress(
                    owner="environment",
                    scope="world",
                    body="world",
                    native="world",
                    prim_path=None,
                    collision=bool(int(m.ngeom) > 0),
                    contact_capable=bool(int(m.ngeom) > 0),
                    key="environment/world/world",
                    root=True,
                )
            )
        triples: list[tuple[str, str, tuple[str, ...]]] = []
        for articulation, sets in self._articulation_body_sets.items():
            for set_name, bodies in sets.items():
                triples.append((articulation, set_name, tuple(str(b) for b in bodies)))
        return ContactInventory(self.backend_key, tuple(rows), tuple(triples))

    def group_axes(self, spec: ContactGroupSpec) -> tuple[int, int]:
        """Resolved ``(B, F)`` sensor/filter counts (post bind_env)."""
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

        One dict per ACTIVE contact point matching the group, each with the
        bodies, canonical owner/scope metadata, the world contact position,
        the normal (geom1 -> geom2, world), the contact force on the SENSOR
        body (``spec.frame``) with its magnitude and the tangential relative
        velocity ``v_t`` (m/s, computed here from the live mjData).

        JSON-ready floats; freshly allocated.
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
                body1_vel = self._body_point_velocity(mj=m, d=d, body=b1, point=pos, vel_buf=vel)
                body2_vel = self._body_point_velocity(mj=m, d=d, body=b2, point=pos, vel_buf=vel)
                v_rel = body2_vel - body1_vel
                n = np.asarray(frame[0], dtype=np.float64)
                v_t_vec = v_rel - float(v_rel @ n) * n
                sensor_row = self._row_by_id.get(sensor_ids[s_idx])
                other_row = self._row_by_id.get(other)
                rec = {
                    "sensor": name_of(sensor_ids[s_idx]),
                    "other": name_of(other),
                    "bodies": [name_of(b1), name_of(b2)],
                    "sensor_key": sensor_row.key if sensor_row is not None else None,
                    "other_key": other_row.key if other_row is not None else None,
                    "sensor_owner": sensor_row.owner if sensor_row is not None else None,
                    "other_owner": other_row.owner if other_row is not None else None,
                    "sensor_scope": sensor_row.scope if sensor_row is not None else None,
                    "other_scope": other_row.scope if other_row is not None else None,
                    "sensor_native": sensor_row.native if sensor_row is not None else None,
                    "other_native": other_row.native if other_row is not None else None,
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
        self.inventory = None
        self._body_ids.clear()
