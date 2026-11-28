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
* ``general`` groups -> one ContactSensorCfg per sensor ref, except robot refs
  whose coverage needs the authored inventory: those expand to one sensor per
  parent prim scope (``Robot/(.*)``, ``Robot/left_hand/(.*)``, ...) because the
  IsaacLab prim matcher is single-segment.  The raw array is
  ``data.net_forces_w`` (N, B, 3), world frame natively.  The B axis is the
  concatenation of per-ref slices in declaration order; each ref slice follows
  the measured live ``robot.data.body_names`` order through an explicit
  sensor-name -> concat-axis permutation (sensor order is PhysX-owned).
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
from pathlib import Path
from typing import Optional

from unigym.contacts.base import ENV_PREFIX_SENTINEL, ContactGroupSpec
from unigym.contacts.inventory import ContactBodyAddress, ContactInventory, resolve_refs
from unigym.contacts.refs import ContactBodyRef, is_wildcard

from .body_inventory import RigidBody, RobotBodyInventory, robot_body_inventory
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


def _dedupe_bodies(bodies: Sequence[RigidBody]) -> list[RigidBody]:
    """De-duplicate authored bodies by durable path, keeping first occurrence."""
    seen: dict[str, RigidBody] = {}
    for body in bodies:
        seen.setdefault(body.path, body)
    return list(seen.values())


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
        self._fields: dict[str, list[list[str]]] = {}
        self._authored: dict[str, RobotBodyInventory] = {}
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

    def _ref_articulation(self, ref: ContactBodyRef) -> Optional[str]:
        """Resolved articulation name for a robot ref (or ``None``)."""
        if ref.articulation is not None:
            return ref.articulation
        if self._articulation_names is not None and len(self._articulation_names) == 1:
            return self._articulation_names[0]
        return None

    def _authored_inventory(self, articulation: str) -> RobotBodyInventory:
        """Authored asset inventory used only for pre-clone prim scopes.

        Contact semantics still use the measured live PhysX body set; the USD
        inventory contributes the parent prim paths needed to address nested
        bodies before the scene exists.
        """
        cached = self._authored.get(articulation)
        if cached is not None:
            return cached
        asset_path = self._contact_assets.get(articulation)
        if not asset_path:
            raise ValueError(
                f"contact coverage for articulation {articulation!r} needs its "
                f"authored robot asset path, but contact_assets has no entry; "
                f"pass contact_assets={{{articulation!r}: '<robot.usd>'}} to "
                f"build_contact_manager. Contact bodies are always resolved "
                f"from the runtime/simulated body set; the asset is only used "
                f"to discover per-parent sensor prim scopes"
            )
        suffix = Path(str(asset_path)).suffix.lower()
        if suffix not in (".usd", ".usda", ".usdc", ".usdz"):
            raise ValueError(
                f"contact coverage for articulation {articulation!r} requires "
                f"a USD robot asset, but contact_assets[{articulation!r}] is "
                f"{asset_path!r} ({suffix or 'no extension'}); the isaaclab2 "
                f"contact sensor prim scopes are USD-only. Supply the "
                f"articulation's USD asset in contact_assets"
            )
        inventory = robot_body_inventory(asset_path)
        self._authored[articulation] = inventory
        return inventory

    def _authored_bodies(
        self,
        inventory: RobotBodyInventory,
        ref: ContactBodyRef,
        articulation: str,
    ) -> list[RigidBody]:
        """Authored bodies selected by one robot ref (for sensor prim paths)."""
        if ref.body_set is not None:
            sets = self._articulation_body_sets.get(articulation)
            bodies = None if sets is None else sets.get(ref.body_set)
            if bodies is None:
                raise ValueError(
                    f"contact ref {ref.label()!r}: body_set {ref.body_set!r} is "
                    f"not declared for articulation {articulation!r}; declare it "
                    f"in the profile body_sets or the articulation contacts inventory"
                )
            selected: list[RigidBody] = []
            for body in bodies:
                selected.extend(self._resolve_authored(inventory, str(body), ref))
            return _dedupe_bodies(selected)
        body = ref.body
        assert body is not None
        if is_wildcard(body, ref.match):
            return list(inventory.bodies)
        return self._resolve_authored(inventory, body, ref)

    @staticmethod
    def _resolve_authored(
        inventory: RobotBodyInventory,
        body: str,
        ref: ContactBodyRef,
    ) -> list[RigidBody]:
        if "/" in body or ref.match == "exact" or not _REGEX_META.search(body):
            try:
                return [inventory.resolve(body)]
            except KeyError:
                raise ValueError(
                    f"contact ref {ref.label()!r}: authored robot inventory has "
                    f"no body at path or name {body!r}; available paths: "
                    f"{list(inventory.paths())}"
                ) from None
        try:
            matches = [row for row in inventory.bodies if re.fullmatch(body, row.name)]
        except re.error as exc:
            raise ValueError(f"invalid contact body regex {body!r}: {exc}") from exc
        if not matches:
            raise ValueError(
                f"contact ref {ref.label()!r}: authored robot inventory has no "
                f"body matching {body!r}; available paths: {list(inventory.paths())}"
            )
        return matches

    def _authored_sensor_paths(self, ref: ContactBodyRef) -> list[str]:
        """One sensor prim path per parent scope selected by a robot ref.

        IsaacLab's sensor prim matcher is single-segment, so a root wildcard
        cannot cover nested hand bodies.  Grouping the authored bodies by their
        parent scope produces one cfg per scope; bind_env later asserts that
        the union is exactly the measured live body set.
        """
        articulation = self._ref_articulation(ref)
        if articulation is None:
            raise ValueError(
                f"contact ref {ref.label()!r}: robot coverage requires an "
                f"articulation name; declare articulation on the ref"
            )
        inventory = self._authored_inventory(articulation)
        bodies = self._authored_bodies(inventory, ref, articulation)
        groups: dict[str, list[RigidBody]] = {}
        for body in bodies:
            scope = body.path.rsplit("/", 1)[0] if "/" in body.path else ""
            groups.setdefault(scope, []).append(body)
        wildcard = is_wildcard(ref.body, ref.match)
        leaf = self._leaf(ref)
        paths: list[str] = []
        for scope in sorted(groups):
            prefix = f"{ENV_PREFIX_SENTINEL}/{leaf}"
            if scope:
                prefix = f"{prefix}/{scope}"
            expr = "(.*)" if wildcard else _union([body.name for body in groups[scope]])
            paths.append(f"{prefix}/{expr}")
        return paths

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

    def _validate_robot_assets(self, spec: ContactGroupSpec) -> None:
        """Every robot ref requires its articulation's authored USD asset.

        Contact semantics use the measured live body set, but the pre-clone
        sensor prim paths cannot be discovered without the articulation asset.
        Validate this before any Kit/env construction, regardless of whether
        the ref is wildcard, single-body or pairwise.
        """
        for ref in (*spec.sensor_refs, *spec.filter_refs):
            if ref.owner != "robot":
                continue
            articulation = self._ref_articulation(ref)
            if articulation is None:
                raise ValueError(
                    f"contact group {spec.name!r}: robot ref {ref.label()!r} "
                    f"must name an articulation when the run declares several; "
                    f"declared: {list(self._articulation_names or ())}"
                )
            asset_path = self._contact_assets.get(articulation)
            if not asset_path:
                raise ValueError(
                    f"contact group {spec.name!r}: robot ref {ref.label()!r} "
                    f"requires the authored USD asset path for articulation "
                    f"{articulation!r}, but contact_assets has no entry; pass "
                    f"contact_assets={{{articulation!r}: '<robot.usd>'}} to "
                    f"build_contact_manager. The asset is required before "
                    f"cloning to discover sensor prim scopes; contact bodies "
                    f"are still resolved from the runtime/simulated body set"
                )
            suffix = Path(str(asset_path)).suffix.lower()
            if suffix not in (".usd", ".usda", ".usdc", ".usdz"):
                raise ValueError(
                    f"contact group {spec.name!r}: robot ref {ref.label()!r} "
                    f"requires a USD asset for articulation {articulation!r}, "
                    f"but contact_assets[{articulation!r}] is {asset_path!r} "
                    f"({suffix or 'no extension'}); the isaaclab2 contact "
                    f"sensor prim scopes are USD-only"
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

        General robot wildcard/body_set refs expand to one cfg per parent prim
        scope (``contact_<name>_s<i>_p<j>``).  Pairwise and explicit
        single-path refs keep one cfg per sensor ref
        (``contact_<name>_s<i>``); pairwise cfgs carry the flattened filter
        prim paths in F order.
        """
        for spec in self.specs:
            self._validate_declared_prop_refs(spec)
            self._validate_robot_assets(spec)

        from isaaclab.sensors import ContactSensorCfg

        fields_cfg: dict = {}
        for spec in self.specs:
            ref_fields: list[list[str]] = []
            filter_exprs = [self._ref_path(ref) for ref in spec.filter_refs]
            for i, ref in enumerate(spec.sensor_refs):
                paths = self._general_sensor_paths(ref, spec.mode)
                if paths is None:
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
                    ref_fields.append([field])
                    continue
                fields: list[str] = []
                for j, prim_path in enumerate(paths):
                    field = f"contact_{spec.name}_s{i}_p{j}"
                    self._add_field(
                        fields_cfg,
                        field,
                        ContactSensorCfg(prim_path=prim_path, history_length=1),
                        spec.name,
                    )
                    fields.append(field)
                ref_fields.append(fields)
            self._fields[spec.name] = ref_fields
        return fields_cfg

    def _general_sensor_paths(self, ref: ContactBodyRef, mode: str) -> Optional[list[str]]:
        """Per-parent prim paths for a general robot ref (``None``: old path)."""
        if mode != "general" or ref.owner != "robot" or ref.path is not None:
            return None
        if is_wildcard(ref.body, ref.match) or ref.body_set is not None:
            return self._authored_sensor_paths(ref)
        return None

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
            ref_fields = self._fields.get(spec.name)
            if ref_fields is None:
                ref_fields = [[f"contact_{spec.name}_s{i}"] for i in range(len(sensor_rows))]
            sensors_per_ref, perms = [], []
            for index, ref in enumerate(spec.sensor_refs):
                fields = ref_fields[index]
                sensors = [env.scene[field] for field in fields]
                ref_rows = self.inventory.resolve(ref)
                if spec.mode == "pairwise":
                    if len(sensors) != 1:
                        raise ValueError(
                            f"contact group {spec.name!r}: pairwise sensor ref "
                            f"{ref.label()!r} expanded to {len(sensors)} scene "
                            f"sensors; pairwise refs keep exactly one sensor"
                        )
                    sensor = sensors[0]
                    names = [str(n) for n in sensor.body_names]
                    idx: list[int] = []
                    for row in ref_rows:
                        native = row.native if row.native in names else row.body
                        if native not in names:
                            raise ValueError(
                                f"contact group {spec.name!r}: sensor ref "
                                f"{ref.label()!r} resolved {row.key!r} but the "
                                f"spawned sensor reports bodies {names}"
                            )
                        idx.append(names.index(native))
                    if len(idx) != 1:
                        raise ValueError(
                            f"contact group {spec.name!r}: pairwise sensor ref "
                            f"{ref.label()!r} resolved {len(idx)} bodies; PhysX "
                            f"filtered reporting is one-to-many only (use one "
                            f"body per sensor ref)"
                        )
                    sensors_per_ref.append(sensors)
                    perms.append(idx)
                    continue
                perm = self._general_permutation(spec.name, ref, fields, sensors, ref_rows)
                sensors_per_ref.append(sensors)
                perms.append(perm)
            self._sensors[spec.name] = {
                "mode": spec.mode,
                "sensors_by_ref": sensors_per_ref,
            }
            self._perm[spec.name] = perms
        self._bound = True

    def _general_permutation(
        self,
        group: str,
        ref: ContactBodyRef,
        fields: Sequence[str],
        sensors: Sequence,
        ref_rows: Sequence[ContactBodyAddress],
    ) -> list[int]:
        """Map concatenated sensor axes to the live body order for one ref.

        The sensor axis order is PhysX-owned and per-parent scopes concatenate
        in field order, so the B axis must be permuted by body name into the
        measured ``robot.data.body_names`` order.  Robot refs require exact
        set equality (no silent subset or duplicate); object/fixture refs keep
        the historical subset mapping (their root body may not own every prop
        body).
        """
        concat_names: list[str] = []
        for field, sensor in zip(fields, sensors):
            names = [str(n) for n in sensor.body_names]
            prim_path = getattr(getattr(sensor, "cfg", None), "prim_path", field)
            if not names:
                raise ValueError(
                    f"contact group {group!r}: sensor field {field!r} "
                    f"({prim_path!r}) resolved no bodies; the authored prim "
                    f"path must match at least one live PhysX body"
                )
            concat_names.extend(names)
        name_to_index: dict[str, int] = {}
        for position, name in enumerate(concat_names):
            if name in name_to_index:
                raise ValueError(
                    f"contact group {group!r}: sensor ref {ref.label()!r} "
                    f"reports duplicate body {name!r} across its parent-scope "
                    f"sensors {list(fields)}"
                )
            name_to_index[name] = position
        expected = [row.body for row in ref_rows]
        if len(set(expected)) != len(expected):
            raise ValueError(
                f"contact group {group!r}: sensor ref {ref.label()!r} resolved "
                f"duplicate live bodies {expected}"
            )
        expected_set = set(expected)
        missing = [name for name in expected if name not in name_to_index]
        extra = [name for name in concat_names if name not in expected_set]
        if ref.owner == "robot" and (missing or extra):
            raise ValueError(
                f"contact group {group!r}: sensor ref {ref.label()!r} does not "
                f"cover the measured live robot body set exactly; missing from "
                f"sensors={missing}, unexpected in sensors={extra}, sensor "
                f"fields={list(fields)}, live bodies={expected}"
            )
        if missing:
            raise ValueError(
                f"contact group {group!r}: sensor ref {ref.label()!r} resolved "
                f"bodies {missing} that the spawned sensors {list(fields)} do "
                f"not report (sensor bodies={concat_names})"
            )
        return [name_to_index[name] for name in expected]

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
            live_names: list[str] = []
            for name in robot.data.body_names:
                native = str(name)
                if native in live_names:
                    raise ValueError(
                        f"contact runtime inventory: articulation "
                        f"{articulation!r} reports duplicate live body name "
                        f"{native!r} in robot.data.body_names={live_names}"
                    )
                live_names.append(native)
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
        perms = self._perm[spec.name]
        if entry["mode"] == "general":
            pieces = []
            for sensors, perm in zip(entry["sensors_by_ref"], perms):
                concat = torch.cat([s.data.net_forces_w for s in sensors], dim=1)
                pieces.append(concat[:, perm, :])
            if len(pieces) == 1:
                return pieces[0].clone()
            return torch.cat(pieces, dim=1).clone()
        expected_f = len(self._filters.get(spec.name, ()))
        pieces = []
        for sensors, perm in zip(entry["sensors_by_ref"], perms):
            sensor = sensors[0]
            matrix = sensor.data.force_matrix_w[:, perm, :, :]
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
        self._authored.clear()
        self.inventory = None
        self._bound = False


def spec_body_frame_unsupported(specs: list[ContactGroupSpec]) -> bool:
    """True when any isaaclab spec asks for the v1-unsupported body frame."""
    return any(s.frame == "body" for s in specs)
