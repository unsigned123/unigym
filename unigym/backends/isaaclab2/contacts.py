"""isaaclab contact backend: ContactSensorCfg construction + raw reading.

This module is the ONLY module in the contacts package allowed to import
isaaclab (imports live inside the functions that need it, mirroring
``cameras/isaaclab_backend.py``); the module itself is import-safe without
the kit.

Scheme A semantics:

* A group's bodies come from explicit ``ContactBodyRef`` records only.  Robot
  refs are anchored at the owning articulation's prim leaf (``Robot`` for the
  first declared articulation, ``Robot_<name>`` otherwise).  Object/fixture
  refs address the declared prop's root prim ``{ENV}/Prop_<name>``.
* ``general`` groups -> one ContactSensorCfg per sensor ref; the raw array is
  ``data.net_forces_w`` (N, B, 3), world frame natively.  The B axis is the
  concatenation of per-ref slices in declaration order.
* ``pairwise`` groups -> one ContactSensorCfg PER sensor ref with
  ``filter_prim_paths_expr`` flattened from the filter refs in declaration
  order (exact literals are emitted without regex grouping).  Each sensor ref
  must resolve exactly one body (PhysX filtered reporting is one-to-many
  only); an empty/multi-body sensor ref fails at bind.  read_raw asserts the
  sensor's F axis equals the resolved filter count -- no silent F drift.
* Static ground/support filters are NOT representable by the rigid-prop
  ContactSensor surface: a support/environment ref is refused at build with a
  clear limitation (use the MuJoCo family for static support contacts).
* ``frame: body`` remains a v1 NotImplementedError for this backend.
* Sensor buffers are recycled in-place by the sensor update — read_raw ALWAYS
  clones (cameras package lesson, annotator-buffer aliasing).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Optional

from unigym.contacts.base import ENV_PREFIX_SENTINEL, ContactGroupSpec
from unigym.contacts.inventory import ContactBodyAddress, ContactInventory, resolve_refs
from unigym.contacts.refs import ContactBodyRef, is_wildcard

from .naming import articulation_prim_leaf, articulation_scene_field

_REGEX_META = re.compile(r"[.^$*+?{}\[\]\\|()]")


class _StaticContactUnsupported(NotImplementedError):
    """A support/environment ref cannot be represented on IsaaccLab2."""


def _is_explicit_prim_path(value: str) -> bool:
    return value.startswith("/") or value.startswith(ENV_PREFIX_SENTINEL)


def _union(names: Sequence[str]) -> str:
    uniq = list(dict.fromkeys(names))
    if len(uniq) == 1:
        return uniq[0]
    return "(" + "|".join(uniq) + ")"


class IsaacLabContactBackend:
    """Backend object for all groups of one run (isaaclab2)."""

    backend_key = "isaaclab2"

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
            raise _StaticContactUnsupported(
                f"contact ref {ref.label()!r}: static support/environment "
                f"contacts are not representable on isaaclab2 (the rigid-prop "
                f"ContactSensor surface cannot report static ground/terrain "
                f"pairs); run the group on the mujoco family or drop the ref"
            )
        body = ref.body
        if body is not None and body != ref.name:
            raise _StaticContactUnsupported(
                f"contact ref {ref.label()!r}: non-root articulated prop link "
                f"sensors are not supported on isaaclab2 in this lane; address "
                f"the prop root (omit body or body: {ref.name}) on the "
                f"isaaclab lane, or run the group on the mujoco family"
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
                        f"resolved {len(resolved)} bodies; the IsaaccLab2 filter "
                        f"axis is one column per explicit ref, so each filter "
                        f"ref must address exactly one body"
                    )
            self._filters[spec.name] = filter_rows
            fields = self._fields.get(spec.name) or [
                f"contact_{spec.name}_s{i}" for i in range(len(sensor_rows))
            ]
            pieces, perms = [], []
            cursor = 0
            for index, field in enumerate(fields):
                sensor = env.scene[field]
                ref_rows = self.inventory.resolve(spec.sensor_refs[index])
                cursor += len(ref_rows)
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
        for index, articulation in enumerate(names):
            if self._articulation_names and articulation in self._articulation_names:
                index = self._articulation_index(articulation)
                field = articulation_scene_field(articulation, index)
            else:
                field = "robot" if index == 0 else f"robot_{articulation}"
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
        prop_names = self._all_prop_names()
        for name in prop_names:
            owner = "fixture" if name in self._articulated_props else "object"
            scene_field = f"prop_{name}"
            try:
                asset = env.scene[scene_field]
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
        """Resolved ``(B, F)`` sensor/filter counts (post bind_env)."""
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
            pieces = [s.data.net_forces_w[:, p, :] for s, p in zip(entry["sensors"], perm)]
            if len(pieces) == 1:
                return pieces[0].clone()
            return torch.cat(pieces, dim=1).clone()
        expected_f = len(self._filters.get(spec.name, ()))
        pieces = []
        for sensor, p in zip(entry["sensors"], perm):
            matrix = sensor.data.force_matrix_w[:, p, :, :]
            if int(matrix.shape[2]) != expected_f:
                raise AssertionError(
                    f"contact group {spec.name!r}: sensor F axis "
                    f"{int(matrix.shape[2])} != resolved filter refs "
                    f"{expected_f}; filter_prim_paths_expr and the declared "
                    f"filter_refs drifted"
                )
            pieces.append(matrix)
        return torch.cat(pieces, dim=1).clone()

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
