"""IsaacLab 3 contact backend: ContactSensorCfg construction + raw reading.

This module is the ONLY module in the shared contacts package allowed to
import isaaclab (imports live inside the functions that need it, mirroring
``cameras/cfg.py``); the module itself is import-safe without the kit.  It
never imports a concrete backend.

Scheme A semantics (body resolution matches the IsaacLab2 face; sensor-scope
expansion does not yet):

* A group's bodies come from explicit ``ContactBodyRef`` records only.  Robot
  refs are anchored at the owning articulation's prim leaf (``Robot`` for the
  first declared articulation, ``Robot_<name>`` otherwise), with
  ``"scope/body_name"`` entries addressing nested Robot prim scopes.
  Object/fixture refs address the declared prop's root prim
  ``{ENV}/Prop_<name>``.
* ``general`` groups -> one ContactSensorCfg per sensor ref; the raw array is
  ``data.net_normal_forces_w`` (N, B, 3), world frame natively.  The B axis is
  the concatenation of per-ref slices in declaration order.
* Known gap (B4 ledger): unlike the IsaacLab2 backend this module does not yet
  expand a robot ref into per-parent sensor scopes (``Robot/(.*)``,
  ``Robot/left_hand/(.*)``, ...).  IsaacLab's prim matcher is single-segment, so
  a nested body scope needs its own sensor; until that expansion lands, a
  nested robot body ref must be declared as an explicit nested-scope ref.  The
  runtime inventory still comes from each articulation's live
  ``data.body_names``.
* ``pairwise`` groups -> one ContactSensorCfg PER sensor ref with
  ``filter_prim_paths_expr`` flattened from the filter refs in declaration
  order (exact literals are emitted without regex grouping).  Each pairwise
  sensor/filter ref must resolve exactly one body (PhysX filtered reporting is
  one-to-many only).  read_raw asserts the sensor F axis equals the resolved
  filter count -- no silent F drift.
* Static support/environment refs are NOT representable by the rigid-prop
  ContactSensor surface: they are refused at build with a clear limitation.
* ``frame: body`` remains a v1 NotImplementedError for this backend.
* Sensor buffers are recycled in-place by the sensor update — read_raw ALWAYS
  clones (cameras package lesson, annotator-buffer aliasing).

IsaacLab 3 wraps sensor data in ``ProxyArray``; every force read uses the
explicit ``.torch`` accessor (never ``torch.as_tensor(proxy_array)``).  The
sensor's internal buffer limits are left to IsaacLab defaults for force-only
groups; a group that opts into ``manifold: true`` explicitly asks the engine to
track per-contact points with a declared
``max_contact_data_count_per_prim`` (see ``MANIFOLD_MAX_CONTACT_DATA_COUNT_PER_PRIM``).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch

from unigym.backends.shared.isaaclab3_common.naming import (
    articulation_prim_leaf,
    articulation_scene_field,
)
from unigym.contacts.base import ENV_PREFIX_SENTINEL, ContactGroupSpec
from unigym.contacts.inventory import ContactBodyAddress, ContactInventory, resolve_refs
from unigym.contacts.refs import ContactBodyRef, is_wildcard

_REGEX_META = re.compile(r"[.^$*+?{}\[\]\\|()]")

#: Contact-point slots per sensor body and env reserved in the PhysX contact
#: view when a group declares ``manifold: true``.  The value is passed to
#: ``ContactSensorCfg.max_contact_data_count_per_prim`` explicitly instead of
#: relying on the engine's internal default; raising it costs device memory and
#: is a future per-group option, not a hidden default.
MANIFOLD_MAX_CONTACT_DATA_COUNT_PER_PRIM = 4


@dataclass(frozen=True)
class ManifoldContact:
    """One engine contact point from a filtered PhysX contact view.

    ``body1`` is the filtered partner and ``body2`` is the sensor body; the
    order is the group's declaration order only, not a proven force/normal
    direction.  ``pos_w`` and ``separation_m`` are the engine's per-contact
    values.  The engine normal direction is not exposed by this record because
    the filtered view does not prove its one-sided/two-sided orientation;
    ``normal_force_n`` is the non-negative magnitude of the engine's normal
    force scalar.  ``contact_id`` is unique within one env of a manifold read
    and is not a stable cross-step identifier.
    """

    env_id: int
    body1: str
    body2: str
    pos_w: torch.Tensor
    separation_m: float
    normal_force_n: float
    contact_id: int


def _union(names: Sequence[str]) -> str:
    uniq = list(dict.fromkeys(names))
    if len(uniq) == 1:
        return uniq[0]
    return "(" + "|".join(uniq) + ")"


class IsaacLabContactBackend:
    """Backend object for all groups of one run (shared IsaacLab 3 face).

    ``backend_key`` is not an engine decision: it stays empty here and the
    owning backend identity / caller sets it.  This module never hardcodes a
    concrete backend name.
    """

    backend_key = ""

    def __init__(
        self,
        specs: list[ContactGroupSpec],
        *,
        articulation_names=None,
        articulation_body_sets=None,
        prop_names=None,
        articulated_props=None,
        contact_assets=None,
        backend_key: str | None = None,
    ):
        if backend_key is not None:
            self.backend_key = str(backend_key)
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
        self._sensors: dict[str, dict] = {}
        self._perm: dict[str, list[list[int]]] = {}
        self._filters: dict[str, list[ContactBodyAddress]] = {}
        self._fields: dict[str, list[str]] = {}
        self.inventory: Optional[ContactInventory] = None
        self._bound = False

    # ---------------------------------------------------------------- #
    # articulation scope
    # ---------------------------------------------------------------- #
    def _articulation_index(self, articulation: str) -> int:
        if self._articulation_names is None:
            raise ValueError(
                f"contact ref names articulation {articulation!r} but the "
                f"articulation order is unknown; construct the manager through "
                f"build_contact_manager(articulation_names=...)"
            )
        try:
            return self._articulation_names.index(articulation)
        except ValueError:
            raise ValueError(
                f"contact ref names articulation {articulation!r}, which is "
                f"not declared by the run (declared: {list(self._articulation_names)})"
            ) from None

    def _leaf(self, ref: ContactBodyRef) -> str:
        if ref.articulation is None:
            return "Robot"
        return articulation_prim_leaf(ref.articulation, self._articulation_index(ref.articulation))

    def _articulation_names_used(self) -> list[str]:
        out: list[str] = []
        for spec in self.specs:
            for name in spec.articulation_names():
                if name not in out:
                    out.append(name)
        if not out and self._articulation_names:
            out.append(self._articulation_names[0])
        return out

    # ---------------------------------------------------------------- #
    # pre-clone prim paths from declared refs
    # ---------------------------------------------------------------- #
    def _robot_path(self, ref: ContactBodyRef) -> str:
        leaf = self._leaf(ref)
        if ref.body_set is not None:
            scope = self._body_set_scope(ref)
            sets = self._articulation_body_sets.get(scope)
            bodies = None if sets is None else sets.get(ref.body_set)
            if bodies is None:
                raise ValueError(
                    f"contact ref {ref.label()!r}: body_set {ref.body_set!r} is "
                    f"not declared for articulation {scope!r}; declare it in the "
                    f"profile body_sets or the articulation contacts inventory"
                )
            union = _union([str(body) for body in bodies])
            return f"{ENV_PREFIX_SENTINEL}/{leaf}/{union}"
        body = ref.body
        assert body is not None
        if is_wildcard(body, ref.match):
            return f"{ENV_PREFIX_SENTINEL}/{leaf}/(.*)"
        if "/" in body:
            scope, name = body.rsplit("/", 1)
            return f"{ENV_PREFIX_SENTINEL}/{leaf}/{scope}/{name}"
        if ref.match == "exact" or not _REGEX_META.search(body):
            return f"{ENV_PREFIX_SENTINEL}/{leaf}/{body}"
        return f"{ENV_PREFIX_SENTINEL}/{leaf}/({body})"

    def _body_set_scope(self, ref: ContactBodyRef) -> str:
        """Articulation scope owning a ``body_set`` ref.

        An explicit ref articulation wins.  A single-articulation run makes an
        omitted owner unambiguous (the historical implicit ``Robot`` scope);
        otherwise an omitted owner is legal only when exactly one declared
        body-set scope exists.
        """
        if ref.articulation is not None:
            return ref.articulation
        if self._articulation_names is not None and len(self._articulation_names) == 1:
            return self._articulation_names[0]
        scopes = [
            scope for scope, sets in self._articulation_body_sets.items() if ref.body_set in sets
        ]
        if len(scopes) == 1:
            return scopes[0]
        raise ValueError(
            f"contact ref {ref.label()!r}: body_set {ref.body_set!r} must name "
            f"articulation when the run declares {list(self._articulation_names or ())} "
            f"or several body-set scopes; declared scopes: "
            f"{sorted(self._articulation_body_sets)}"
        )

    def _validate_declared_prop_refs(self, spec: ContactGroupSpec) -> None:
        """Object/fixture refs must name declared props (no synthesized rows)."""
        declared = set(self._prop_names)
        for ref in (*spec.sensor_refs, *spec.filter_refs):
            if ref.owner in ("object", "fixture") and ref.name not in declared:
                raise ValueError(
                    f"contact group {spec.name!r}: ref {ref.label()!r} names an "
                    f"undeclared {ref.owner} prop {ref.name!r}; declared props "
                    f"are {sorted(declared)}"
                )

    def _prop_path(self, ref: ContactBodyRef) -> str:
        if ref.owner in ("support", "environment"):
            raise NotImplementedError(
                f"contact ref {ref.label()!r}: static support/environment "
                f"contacts are not representable on the IsaacLab 3 rigid-prop "
                f"ContactSensor surface (it cannot report static "
                f"ground/terrain pairs); run the group on the mujoco family "
                f"or drop the ref"
            )
        body = ref.body
        if body is not None and body != ref.name:
            raise NotImplementedError(
                f"contact ref {ref.label()!r}: non-root articulated prop link "
                f"sensors are not supported on the IsaacLab 3 contact face "
                f"without runtime inventory evidence; address the prop root "
                f"(omit body or body: {ref.name}) or run the group on the "
                f"mujoco family"
            )
        return f"{ENV_PREFIX_SENTINEL}/Prop_{ref.name}"

    def _ref_path(self, ref: ContactBodyRef) -> str:
        if ref.owner in ("support", "environment"):
            # Static scopes must fail at BUILD, before ref.path (an
            # environment path) can bypass the static contact limitation.
            return self._prop_path(ref)
        if ref.path is not None:
            return ref.path
        if ref.owner == "robot":
            return self._robot_path(ref)
        return self._prop_path(ref)

    # ---------------------------------------------------------------- #
    # scene contribution (before cloning)
    # ---------------------------------------------------------------- #
    @staticmethod
    def _add_field(fields: dict, field: str, value, group: str) -> None:
        """Add one scene sensor field; derived-name collisions fail fast."""
        if field in fields:
            raise ValueError(
                f"contact group {group!r}: derived scene field {field!r} "
                f"collides with another contact group's sensor field; choose "
                f"group names whose derived sensor fields are distinct"
            )
        fields[field] = value

    def setup_scene(self) -> dict:
        """``{scene_field: ContactSensorCfg}`` for every group.

        One cfg per sensor ref (``contact_<name>_s<i>``); pairwise cfgs carry
        the flattened filter prim paths in F order.
        """
        from isaaclab.sensors import ContactSensorCfg

        fields_cfg: dict = {}
        for spec in self.specs:
            self._validate_declared_prop_refs(spec)
            names: list[str] = []
            filter_exprs = [self._ref_path(ref) for ref in spec.filter_refs]
            for i, ref in enumerate(spec.sensor_refs):
                field = f"contact_{spec.name}_s{i}"
                kwargs = {"prim_path": self._ref_path(ref), "history_length": 1}
                if spec.mode == "pairwise":
                    kwargs["filter_prim_paths_expr"] = filter_exprs
                if spec.manifold:
                    kwargs["track_contact_points"] = True
                    kwargs["max_contact_data_count_per_prim"] = (
                        MANIFOLD_MAX_CONTACT_DATA_COUNT_PER_PRIM
                    )
                self._add_field(
                    fields_cfg,
                    field,
                    ContactSensorCfg(**kwargs),
                    spec.name,
                )
                names.append(field)
            self._fields[spec.name] = names
        return fields_cfg

    # ---------------------------------------------------------------- #
    # bind (after construction)
    # ---------------------------------------------------------------- #
    def bind_env(self, env) -> None:
        """Resolve refs against the live scene and cache sensor handles."""
        if spec_body_frame_unsupported(self.specs):
            name = next(s.name for s in self.specs if s.frame == "body")
            raise NotImplementedError(
                f"contact group {name!r}: isaaclab frame='body' transform "
                f"is not implemented in v1 (plan §6 allowed list); use "
                f"frame: world"
            )
        self.inventory = self._runtime_inventory(env)
        for spec in self.specs:
            sensor_rows = resolve_refs(
                self.inventory, spec.sensor_refs, axis="sensor_refs", group=spec.name
            )
            filter_rows = resolve_refs(
                self.inventory, spec.filter_refs, axis="filter_refs", group=spec.name
            )
            for ref in spec.filter_refs:
                resolved = self.inventory.resolve(ref)
                if len(resolved) != 1:
                    raise ValueError(
                        f"contact group {spec.name!r}: filter ref {ref.label()!r} "
                        f"resolved {len(resolved)} bodies; the IsaacLab 3 filter "
                        f"axis is one column per explicit ref, so each filter "
                        f"ref must address exactly one body"
                    )
            self._filters[spec.name] = filter_rows
            fields = self._fields.get(spec.name) or [
                f"contact_{spec.name}_s{i}" for i in range(len(sensor_rows))
            ]
            pieces, perms = [], []
            for index, field in enumerate(fields):
                sensor = env.scene[field]
                ref_rows = self.inventory.resolve(spec.sensor_refs[index])
                names = [str(n) for n in sensor.body_names]
                idx: list[int] = []
                for row in ref_rows:
                    native = row.native if row.native in names else row.body
                    if native not in names:
                        raise ValueError(
                            f"contact group {spec.name!r}: sensor ref "
                            f"{spec.sensor_refs[index].label()!r} resolved "
                            f"{row.key!r} but the spawned sensor reports "
                            f"bodies {names}"
                        )
                    idx.append(names.index(native))
                if spec.mode == "pairwise" and len(idx) != 1:
                    raise ValueError(
                        f"contact group {spec.name!r}: pairwise sensor ref "
                        f"{spec.sensor_refs[index].label()!r} resolved "
                        f"{len(idx)} bodies; PhysX filtered reporting is "
                        f"one-to-many only (use one body per sensor ref)"
                    )
                pieces.append(sensor)
                perms.append(idx)
            self._sensors[spec.name] = {"mode": spec.mode, "sensors": pieces}
            self._perm[spec.name] = perms
        self._bound = True

    def _runtime_inventory(self, env) -> ContactInventory:
        rows: list[ContactBodyAddress] = []
        names = self._articulation_names_used()
        for slot, articulation in enumerate(names):
            if self._articulation_names and articulation in self._articulation_names:
                index = self._articulation_index(articulation)
                field = articulation_scene_field(articulation, index)
            else:
                field = "robot" if slot == 0 else f"robot_{articulation}"
            robot = env.scene[field]
            for name in robot.data.body_names:
                native = str(name)
                rows.append(
                    ContactBodyAddress(
                        owner="robot",
                        scope=articulation,
                        body=native,
                        native=native,
                        prim_path=None,
                        collision=True,
                        contact_capable=True,
                        key=f"robot/{articulation}/{native}",
                    )
                )
        for name in self._all_prop_names():
            owner = "fixture" if name in self._articulated_props else "object"
            try:
                asset = env.scene[f"prop_{name}"]
            except (KeyError, TypeError):
                asset = None
            body_names = []
            if asset is not None:
                data = getattr(asset, "data", None)
                body_names = [str(n) for n in (getattr(data, "body_names", None) or ())]
            if not body_names:
                body_names = [name]
            for i, native in enumerate(body_names):
                rows.append(
                    ContactBodyAddress(
                        owner=owner,
                        scope=name,
                        body=native,
                        native=native,
                        prim_path=f"{ENV_PREFIX_SENTINEL}/Prop_{name}",
                        collision=True,
                        contact_capable=True,
                        key=f"{owner}/{name}/{native}",
                        root=i == 0,
                    )
                )
        triples: list[tuple[str, str, tuple[str, ...]]] = []
        for articulation, sets in self._articulation_body_sets.items():
            for set_name, bodies in sets.items():
                triples.append((articulation, set_name, tuple(str(b) for b in bodies)))
        return ContactInventory(self.backend_key, tuple(rows), tuple(triples))

    def _all_prop_names(self) -> list[str]:
        """Declared prop names only (inventories never synthesize ref rows)."""
        return list(self._prop_names)

    def group_axes(self, spec: ContactGroupSpec) -> tuple[int, int]:
        """Resolved ``(B, F)`` sensor/filter counts (post bind_env).

        Feeds the obs catalog's per-key shape resolution.  general: B =
        concatenated per-ref body counts, F = 0; pairwise: one body per sensor
        ref, F = the number of resolved filter bodies.
        """
        perms = self._perm.get(spec.name, [])
        if spec.mode == "general":
            return sum(len(p) for p in perms), 0
        return len(perms), len(self._filters.get(spec.name, ()))

    # ---------------------------------------------------------------- #
    # raw reading
    # ---------------------------------------------------------------- #
    def read_raw(self, spec: ContactGroupSpec):
        """Raw forces in ``spec.frame`` (torch, env device, freshly cloned).

        Shapes: ``(N, B, 3)`` general, ``(N, B, F, 3)`` pairwise.
        """
        import torch

        entry = self._sensors[spec.name]
        perm = self._perm[spec.name]
        if entry["mode"] == "general":
            pieces = [
                sensor.data.net_normal_forces_w.torch[:, p, :]
                for sensor, p in zip(entry["sensors"], perm)
            ]
            if len(pieces) == 1:
                return pieces[0].clone()
            return torch.cat(pieces, dim=1).clone()
        expected_f = len(self._filters.get(spec.name, ()))
        pieces = []
        for sensor, p in zip(entry["sensors"], perm):
            matrix = sensor.data.normal_force_matrix_w.torch[:, p, :, :]
            if int(matrix.shape[2]) != expected_f:
                raise AssertionError(
                    f"contact group {spec.name!r}: sensor F axis "
                    f"{int(matrix.shape[2])} != resolved filter refs "
                    f"{expected_f}; filter_prim_paths_expr and the declared "
                    f"filter_refs drifted"
                )
            pieces.append(matrix)
        return torch.cat(pieces, dim=1).clone()

    def read_manifold(
        self,
        spec: ContactGroupSpec,
        env_ids: torch.Tensor,
        *,
        dt: float,
    ) -> tuple[ManifoldContact, ...]:
        """Read per-contact rows for one ``manifold: true`` pairwise group.

        The engine data comes from the sensor's public
        ``contact_view.get_contact_data``: world contact points, the engine's
        signed contact separation and a normal force scalar.  PhysX filtered
        reporting is one sensor body to many filter bodies, so ``body1`` is the
        filtered partner and ``body2`` is the sensor body; that order is the
        group declaration order, not a proven contact-normal direction.  The
        engine normal is intentionally not exposed: this view does not prove
        its one-sided/two-sided orientation, so the record carries no direction
        and ``normal_force_n`` is the force scalar's magnitude.

        ``env_ids`` is the query's normalized 1-D int64 CPU env selection.
        Points are emitted env-major: env selection order first, then sensor
        declaration order, then filter declaration order, then engine contact
        order.  Every returned tensor is a detached CPU float64 copy; the
        engine buffers are reused in place, so nothing borrowed escapes this
        method.
        """
        if not spec.manifold:
            raise ValueError(f"contact group {spec.name!r} is not declared manifold: true")
        if spec.mode != "pairwise":
            raise ValueError(
                f"contact group {spec.name!r} is not pairwise; a filtered "
                f"contact manifold requires a non-empty filter_refs"
            )
        entry = self._sensors.get(spec.name)
        if entry is None or entry.get("mode") != "pairwise":
            raise ValueError(
                f"contact group {spec.name!r} is not bound as a pairwise "
                f"sensor group; bind_env must run before a manifold read"
            )
        filter_names = [str(row.body) for row in self._filters.get(spec.name, ())]
        selected = [int(value) for value in env_ids.tolist()]
        readings = []
        for sensor, perm in zip(entry["sensors"], self._perm[spec.name]):
            body_names = [str(name) for name in sensor.body_names]
            sensor_index = int(perm[0])
            sensor_name = body_names[sensor_index]
            forces, positions, _normals, separations, counts, starts = self._engine_contact_buffers(
                sensor, spec.name, dt
            )
            env_count = self._env_count(len(body_names), counts, spec.name)
            if int(counts.shape[1]) != len(filter_names):
                raise ValueError(
                    f"contact group {spec.name!r}: engine filter axis "
                    f"{int(counts.shape[1])} != resolved filter refs "
                    f"{len(filter_names)}; filter_prim_paths_expr and the "
                    f"declared filter_refs drifted"
                )
            readings.append(
                (
                    sensor_index,
                    sensor_name,
                    forces,
                    positions,
                    separations,
                    counts,
                    starts,
                    env_count,
                )
            )
        points: list[ManifoldContact] = []
        counters: dict[int, int] = {}
        for env_id in selected:
            for (
                sensor_index,
                sensor_name,
                forces,
                positions,
                separations,
                counts,
                starts,
                env_count,
            ) in readings:
                if env_id >= env_count:
                    raise ValueError(
                        f"contact group {spec.name!r}: env id {env_id} is "
                        f"outside the sensor's {env_count} env slots"
                    )
                flat = sensor_index * env_count + env_id
                for filter_index, filter_name in enumerate(filter_names):
                    count = int(counts[flat, filter_index])
                    if count <= 0:
                        continue
                    start = int(starts[flat, filter_index])
                    for offset in range(count):
                        index = start + offset
                        contact_id = counters.get(env_id, 0)
                        counters[env_id] = contact_id + 1
                        points.append(
                            ManifoldContact(
                                env_id=env_id,
                                body1=filter_name,
                                body2=sensor_name,
                                pos_w=_cpu_vector(positions[index], "contact point"),
                                separation_m=_cpu_scalar(separations[index], "contact separation"),
                                normal_force_n=abs(_cpu_scalar(forces[index], "contact force")),
                                contact_id=contact_id,
                            )
                        )
        return tuple(points)

    def _engine_contact_buffers(
        self,
        sensor,
        group: str,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one sensor's per-contact engine buffers as device tensors."""
        view = getattr(sensor, "contact_view", None)
        getter = getattr(view, "get_contact_data", None)
        if not callable(getter):
            raise NotImplementedError(
                f"contact group {group!r}: the IsaacLab 3 contact sensor "
                f"exposes no contact_view.get_contact_data(); the filtered "
                f"contact manifold is unavailable on this stack"
            )
        raw = getter(dt=dt)
        if not isinstance(raw, tuple) or len(raw) != 6:
            raise NotImplementedError(
                f"contact group {group!r}: get_contact_data returned "
                f"{type(raw).__name__}, expected the 6-tuple (forces, points, "
                f"normals, separations, counts, start_indices); this stack "
                f"cannot supply a contact manifold"
            )
        forces, positions, normals, separations, counts, starts = (
            _engine_tensor(value, group) for value in raw
        )
        if counts.shape != starts.shape or counts.ndim != 2:
            raise ValueError(
                f"contact group {group!r}: engine counts/start_indices shapes "
                f"{tuple(counts.shape)} / {tuple(starts.shape)} are not the "
                f"expected matching (sensor_count, filter_count) pair"
            )
        if (
            not _is_contact_column(forces)
            or not _is_contact_column(separations)
            or positions.ndim != 2
            or normals.ndim != 2
            or int(positions.shape[1]) != 3
            or int(normals.shape[1]) != 3
            or int(positions.shape[0]) != int(forces.shape[0])
            or int(normals.shape[0]) != int(forces.shape[0])
            or int(separations.shape[0]) != int(forces.shape[0])
        ):
            raise ValueError(
                f"contact group {group!r}: engine contact buffers have shapes "
                f"forces={tuple(forces.shape)} points={tuple(positions.shape)} "
                f"normals={tuple(normals.shape)} "
                f"separations={tuple(separations.shape)}; expected "
                f"(max, 1) forces/separations and (max, 3) points/normals"
            )
        return forces, positions, normals, separations, counts, starts

    @staticmethod
    def _env_count(bodies: int, counts: torch.Tensor, group: str) -> int:
        """Derive the env count from the body-major engine sensor axis."""
        if bodies <= 0 or int(counts.shape[0]) % bodies != 0:
            raise ValueError(
                f"contact group {group!r}: engine sensor axis "
                f"{int(counts.shape[0])} is not a positive multiple of the "
                f"{bodies} sensor bodies"
            )
        return int(counts.shape[0]) // bodies

    def close(self) -> None:
        """Release the cached sensor handles (ContactBackend lifecycle)."""
        self._sensors.clear()
        self._perm.clear()
        self._filters.clear()
        self._fields.clear()
        self.inventory = None
        self._bound = False


def spec_body_frame_unsupported(specs: list[ContactGroupSpec]) -> bool:
    """True when any isaaclab spec asks for the v1-unsupported body frame."""
    return any(s.frame == "body" for s in specs)


def _is_contact_column(value: torch.Tensor) -> bool:
    """True for the engine's ``(max,)`` or ``(max, 1)`` scalar-per-contact axes."""
    return value.ndim == 1 or (value.ndim == 2 and int(value.shape[1]) == 1)


def _engine_tensor(value, group: str) -> torch.Tensor:
    """Return a torch view over one engine contact buffer (no copy yet)."""
    import torch

    if isinstance(value, torch.Tensor):
        return value
    try:
        import warp as wp
    except ImportError as exc:  # pragma: no cover - the lane always ships warp
        raise NotImplementedError(
            f"contact group {group!r}: the engine contact buffers are not torch "
            f"tensors and warp is unavailable"
        ) from exc
    tensor = wp.to_torch(value)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"contact group {group!r}: warp contact buffer did not convert to "
            f"a torch.Tensor, got {type(tensor).__name__}"
        )
    return tensor


def _cpu_vector(value: torch.Tensor, field: str) -> torch.Tensor:
    """Return a detached float64 CPU copy of one ``(3,)`` contact vector."""
    import torch

    vector = value.detach().to("cpu").to(torch.float64).reshape(-1)
    if int(vector.numel()) != 3:
        raise ValueError(f"{field}: expected 3 components, got {int(vector.numel())}")
    return vector


def _cpu_scalar(value: torch.Tensor, field: str) -> float:
    """Return one detached finite contact scalar as a plain float."""
    scalar = float(value.detach().to("cpu").reshape(-1)[0])
    if not math.isfinite(scalar):
        raise ValueError(f"{field}: expected a finite value, got {scalar!r}")
    return scalar
