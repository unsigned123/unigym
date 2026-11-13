"""isaaclab contact backend: ContactSensorCfg construction + raw reading.

This module is the ONLY module in the contacts package allowed to import
isaaclab (imports live inside the functions that need it, mirroring
``cameras/isaaclab_backend.py``); the module itself is import-safe without
the kit.

Semantics (authoritative local source: Isaac Lab 2.3.2
``sensors/contact_sensor/contact_sensor.py``):

* Every group owns ONE articulation (``ContactGroupSpec.articulation``).
  The owning articulation's prim leaf is ``Robot`` for the first declared
  articulation and ``Robot_<name>`` otherwise; the run layer's articulation
  order is the only source of that slot (no global union, no cross-robot
  body matching).
* ``general`` groups -> one ContactSensorCfg per scope bucket with
  ``prim_path = {ENV_REGEX_NS}/<leaf>/(<union of sensor_bodies>)``; the raw
  array is ``data.net_forces_w`` (N, B, 3), world frame natively.  A
  sensor_bodies entry containing ``/`` is an explicit Robot-relative prim
  sub-path (nested scopes, e.g. the localized G1 finger links under
  ``left_hand/``): such entries are bucketed per scope, each bucket gets
  its own cfg (fields ``contact_<name>``, then ``contact_<name>__s1``...)
  and the raw slices concatenate along B in bucket order.
* ``pairwise`` groups -> one ContactSensorCfg PER sensor_bodies ENTRY with
  ``filter_prim_paths_expr`` built from ``filter_bodies``.  Rationale: the
  PhysX filtered reporting is one-to-many ONLY (one sensor body against
  many filter bodies) — a single cfg with multiple matched sensor bodies is
  not valid (Isaac Lab ContactSensor docstring); per-entry sensors keep
  each (sensor, filter) matrix well-defined.  Each entry must resolve to
  exactly one body (fail-fast at bind).  The raw array is the F-axis
  concatenation of the per-entry ``data.force_matrix_w`` (N, 1, F, 3) ->
  (N, B, F, 3), world frame natively.  Scoped entries (``/``) build
  ``{ENV}/<leaf>/{scope}/{name}`` paths; body resolution uses the last
  segment.
* Bare filter bodies are resolved in the owning articulation's body list
  and must resolve exactly one body; explicit prim paths / env-prefixed
  expressions (props, world) pass through unchanged.
* B ordering: owning robot asset body order filtered by the entry regexes
  (``pairwise``: entry order, one body each).  F ordering:
  ``filter_bodies`` entry order (PhysX preserves pattern order).
* ``frame: body`` is a v1 NotImplementedError for this backend (plan §6
  allowed list); ``frame: world`` is direct.
* Sensor buffers are recycled in-place by the sensor update — read_raw
  ALWAYS clones (cameras package lesson, annotator-buffer aliasing).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Optional

from unigym.contacts.base import ENV_PREFIX_SENTINEL, ContactGroupSpec, is_all_bodies
from unigym.contacts.discovery import (
    contact_candidate_error,
    resolve_bucket_names,
    resolve_entry_name,
    usd_robot_inventory,
)

from .naming import articulation_prim_leaf, articulation_scene_field


def _is_explicit_prim_path(entry: str) -> bool:
    return entry.startswith("/") or entry.startswith(ENV_PREFIX_SENTINEL)


def split_scope(entry: str) -> tuple[Optional[str], str]:
    """Split a sensor_bodies entry into ``(scope, body_name)``.

    A ``/`` inside the entry marks an explicit Robot-relative prim
    sub-path (nested scopes): the localized G1 USD keeps the finger and
    palm links under ``left_hand/`` / ``left_wrist_yaw_link/`` subtrees,
    and the ContactSensor prim-path matching is SEGMENT-wise (each path
    token is matched independently, Isaac Lab ``find_matching_prims``),
    so a flat ``Robot/<name>`` expression cannot reach them.  The body
    name used for resolution is the LAST segment.  Plain entries are
    ``(None, entry)``.
    """
    if "/" in entry:
        scope, name = entry.rsplit("/", 1)
        return scope, name
    return None, entry


def entry_prim_path(entry: str, robot_leaf: str = "Robot") -> str:
    """Robot-anchored ``prim_path`` for ONE sensor entry (per-entry cfgs)."""
    scope, name = split_scope(entry)
    if scope is not None:
        return f"{{ENV_REGEX_NS}}/{robot_leaf}/{scope}/{name}"
    return f"{{ENV_REGEX_NS}}/{robot_leaf}/({name})"


def _general_buckets(entries: list[str]) -> list[list[str]]:
    """General-mode body names bucketed by scope (first-appearance order)."""
    order: list[Optional[str]] = []
    buckets: dict[Optional[str], list[str]] = {}
    for e in entries:
        scope, name = split_scope(e)
        if scope not in buckets:
            buckets[scope] = []
            order.append(scope)
        buckets[scope].append(name)
    return [buckets[s] for s in order]


def general_prim_paths(entries: list[str], robot_leaf: str = "Robot") -> list[str]:
    """Union prim-path exprs for a general group, one per scope bucket."""
    order: list[Optional[str]] = []
    buckets: dict[Optional[str], list[str]] = {}
    for e in entries:
        scope, name = split_scope(e)
        if scope not in buckets:
            buckets[scope] = []
            order.append(scope)
        buckets[scope].append(name)
    out = []
    for scope in order:
        union = f"({'|'.join(buckets[scope])})"
        out.append(
            f"{{ENV_REGEX_NS}}/{robot_leaf}/{scope}/{union}"
            if scope
            else f"{{ENV_REGEX_NS}}/{robot_leaf}/{union}"
        )
    return out


def general_field_names(name: str, entries: list[str]) -> list[str]:
    """Scene-cfg field names for a general group (``__sN`` beyond the 1st)."""
    n = len(_general_buckets(entries))
    return [name if i == 0 else f"{name}__s{i}" for i in range(n)]


def sensor_prim_path(
    spec: ContactGroupSpec,
    entries: Optional[list[str]] = None,
    robot_leaf: str = "Robot",
) -> str:
    """Robot-anchored ``prim_path`` for a scope-less (union) group.

    Supports flat body sets: ``({e1}|{e2}|...)``.
    Scoped entries (containing ``/``) are not representable here -- use
    :func:`entry_prim_path` / :func:`general_prim_paths`.
    """
    alts = entries if entries is not None else list(spec.sensor_bodies)
    if any("/" in e for e in alts):
        raise ValueError(
            f"scoped sensor entry {alts} cannot build a flat union path; "
            f"use entry_prim_path/general_prim_paths"
        )
    return f"{{ENV_REGEX_NS}}/{robot_leaf}/({'|'.join(alts)})"


def filter_prim_exprs(spec: ContactGroupSpec, robot_leaf: str = "Robot") -> list[str]:
    """``filter_prim_paths_expr`` for a pairwise group.

    Bare entries are robot-body regexes in the group's articulation scope;
    entries starting with ``/`` or ``{ENV_REGEX_NS}`` pass through verbatim
    (props ``{ENV_REGEX_NS}/Prop_*``).
    """
    return [
        e if _is_explicit_prim_path(e) else f"{{ENV_REGEX_NS}}/{robot_leaf}/{e}"
        for e in spec.filter_bodies or ()
    ]


def resolve_articulation_leaf(
    spec: ContactGroupSpec, articulation_names: Optional[Sequence[str]]
) -> str:
    """Prim leaf of *spec*'s owning articulation (``Robot`` when implicit)."""
    if spec.articulation is None:
        return "Robot"
    if articulation_names is None:
        raise ValueError(
            f"contact group {spec.name!r}: articulation "
            f"{spec.articulation!r} is declared but the articulation order "
            f"is unknown; construct the manager through "
            f"build_contact_manager(articulation_names=...)"
        )
    try:
        index = tuple(articulation_names).index(spec.articulation)
    except ValueError:
        raise ValueError(
            f"contact group {spec.name!r}: articulation "
            f"{spec.articulation!r} is not declared by the run "
            f"(declared: {list(articulation_names)})"
        ) from None
    return articulation_prim_leaf(spec.articulation, index)


def resolve_all_articulation(spec: ContactGroupSpec, contact_assets: Mapping[str, str]) -> str:
    """Articulation whose pre-clone USD inventory expands ``contacts: all``.

    An explicit owner wins.  An omitted owner is legal only when exactly one
    all-mode asset exists (the single-articulation case); otherwise the
    group is ambiguous and the error names the legal reference form instead
    of silently unioning inventories.
    """
    assets = sorted(contact_assets)
    if not assets:
        raise ValueError(
            f"contact group {spec.name!r}: contacts: all requires an "
            f"articulation's resolved USD asset path; the run resolved no "
            f"all-mode asset (explicit contact groups remain the URDF path)"
        )
    if spec.articulation is not None:
        if spec.articulation not in contact_assets:
            raise ValueError(
                f"contact group {spec.name!r}: contacts: all references "
                f"articulation {spec.articulation!r}, but the run resolved "
                f"no USD asset for it (all-mode assets: {assets}); declare "
                f"explicit contact groups for a non-USD asset"
            )
        return spec.articulation
    if len(assets) == 1:
        return assets[0]
    raise ValueError(
        f"contact group {spec.name!r}: contacts: all is ambiguous across "
        f"articulations {assets}; reference the owner with "
        f"sensor_bodies: {{contact_group: {{articulation: <name>, group: "
        f"all}}}} (or declare articulation: <name>)"
    )


class IsaacLabContactBackend:
    """Backend object for all groups of one run (isaaclab)."""

    backend_key = "isaaclab"

    @classmethod
    def prop_filter_paths(cls, name: str, articulated: bool) -> list[str]:
        """Native contact-filter address for a declared prop."""
        if articulated:
            return [f"{{ENV_REGEX_NS}}/Prop_{name}/.*"]
        return [f"{{ENV_REGEX_NS}}/Prop_{name}"]

    def __init__(self, specs: list[ContactGroupSpec], contact_assets=None, articulation_names=None):
        self.specs = list(specs)
        self._contact_assets = dict(contact_assets or {})
        self._articulation_names = (
            None if articulation_names is None else tuple(str(n) for n in articulation_names)
        )
        self._sensors: dict[str, dict] = {}  # name -> pieces
        self._perm: dict[str, list[list[int]]] = {}  # name -> per-piece perm
        self._all_bodies: dict[str, tuple[str, list]] = {}  # name -> (owner, bodies)
        self._all_inventories: dict[str, list] = {}  # articulation -> inventory
        self._bound = False

    # ---------------------------------------------------------------- #
    # articulation scope
    # ---------------------------------------------------------------- #
    def _leaf(self, spec: ContactGroupSpec) -> str:
        """Prim leaf of this group's owning articulation."""
        return resolve_articulation_leaf(spec, self._articulation_names)

    def _articulation_index(self, spec: ContactGroupSpec) -> int:
        if self._articulation_names is None:
            raise ValueError(
                f"contact group {spec.name!r}: articulation "
                f"{spec.articulation!r} is declared but the articulation "
                f"order is unknown; construct the manager through "
                f"build_contact_manager(articulation_names=...)"
            )
        try:
            return self._articulation_names.index(spec.articulation)
        except ValueError:
            raise ValueError(
                f"contact group {spec.name!r}: articulation "
                f"{spec.articulation!r} is not declared by the run "
                f"(declared: {list(self._articulation_names)})"
            ) from None

    def _robot_asset(self, env, spec: ContactGroupSpec):
        """Scene asset of the group's articulation (``env.robot`` when implicit)."""
        if spec.articulation is None:
            return env.robot
        field = articulation_scene_field(spec.articulation, self._articulation_index(spec))
        return env.scene[field]

    # ---------------------------------------------------------------- #
    # scene contribution (before cloning)
    # ---------------------------------------------------------------- #
    def _all_inventory(self, articulation: str) -> list:
        """The pre-clone rigid-body inventory for one articulation.

        Materialized from that articulation's registered USD asset through
        :mod:`unigym.backends.isaaclab2.usd` -- the source of truth before
        cloning (post-spawn body names cannot create sensors).  Fixed-joint
        merged links are excluded by the inventory owner.  Never a union
        across articulations.
        """
        if articulation in self._all_inventories:
            return self._all_inventories[articulation]
        from .usd import robot_rigid_bodies

        asset_path = self._contact_assets[articulation]
        try:
            inventory = robot_rigid_bodies(asset_path)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"contacts: all: cannot inventory articulation "
                f"{articulation!r} asset {asset_path!r}: {exc}"
            ) from exc
        self._all_inventories[articulation] = inventory
        return inventory

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

        Fields: ``contact_<name>`` (general, single scope bucket) and
        ``contact_<name>__sN`` (general, further scope buckets),
        ``contact_<name>_b<i>`` (ONE sensor per body for ``all``) and
        ``contact_<name>_<j>`` (pairwise entry j).  The prim path is anchored
        at the owning articulation's leaf.  ``history_length=1`` matches the
        manager's v1 history support; ``track_air_time`` stays off (we serve
        forces, not timings).
        """
        from isaaclab.sensors import ContactSensorCfg

        fields_cfg: dict = {}
        for spec in self.specs:
            leaf = self._leaf(spec)
            if is_all_bodies(spec.sensor_bodies):
                articulation = resolve_all_articulation(spec, self._contact_assets)
                bodies = self._all_inventory(articulation)
                self._all_bodies[spec.name] = (articulation, bodies)
                for i, body in enumerate(bodies):
                    self._add_field(
                        fields_cfg,
                        f"contact_{spec.name}_b{i}",
                        ContactSensorCfg(
                            prim_path=f"{{ENV_REGEX_NS}}/{leaf}/{body.path}",
                            history_length=1,
                        ),
                        spec.name,
                    )
                continue
            if spec.mode == "general":
                paths = general_prim_paths(list(spec.sensor_bodies), robot_leaf=leaf)
                fields = general_field_names(spec.name, list(spec.sensor_bodies))
                for field, path in zip(fields, paths):
                    self._add_field(
                        fields_cfg,
                        f"contact_{field}",
                        ContactSensorCfg(prim_path=path, history_length=1),
                        spec.name,
                    )
            else:
                exprs = filter_prim_exprs(spec, robot_leaf=leaf)
                for j, entry in enumerate(spec.sensor_bodies):
                    self._add_field(
                        fields_cfg,
                        f"contact_{spec.name}_{j}",
                        ContactSensorCfg(
                            prim_path=entry_prim_path(entry, robot_leaf=leaf),
                            filter_prim_paths_expr=exprs,
                            history_length=1,
                        ),
                        spec.name,
                    )
        return fields_cfg

    # ---------------------------------------------------------------- #
    # bind (after construction)
    # ---------------------------------------------------------------- #
    def bind_env(self, env) -> None:
        """Resolve bodies + grab sensor handles; fail-fast on mismatches.

        Body names resolve against the group's OWNING articulation
        (``env.robot`` when the group has no explicit owner, the articulation
        scene field otherwise).  Unresolvable sensor declarations refuse with
        a ``ContactCandidateError`` carrying that robot's discovered prim
        inventory (``usd_robot_inventory``: rigid-body prims + collision
        coverage, nested scopes included) -- foreign assets with unfamiliar
        naming become self-describing at the failure site.  Bare filter
        bodies resolve in the same articulation and must match exactly one
        body; duplicate resolved sensor/filter bodies fail fast.
        """
        if spec_body_frame_unsupported(self.specs):
            name = next(s.name for s in self.specs if s.frame == "body")
            raise NotImplementedError(
                f"contact group {name!r}: isaaclab frame='body' transform "
                f"is not implemented in v1 (plan §6 allowed list); use "
                f"frame: world"
            )
        for spec in self.specs:
            robot = self._robot_asset(env, spec)
            robot_bodies = [str(n) for n in robot.data.body_names]

            def inventory(robot=robot):
                return usd_robot_inventory(robot)

            if is_all_bodies(spec.sensor_bodies):
                _, bodies = self._all_bodies[spec.name]
                pieces, perms = [], []
                for i, body in enumerate(bodies):
                    sensor = env.scene[f"contact_{spec.name}_b{i}"]
                    names = [str(n) for n in sensor.body_names]
                    expected = str(body.path).rsplit("/", 1)[-1]
                    if expected in names:
                        perm = [names.index(expected)]
                    elif len(names) == 1:
                        perm = [0]
                    else:
                        raise ValueError(
                            f"contact group {spec.name!r}: all-body sensor "
                            f"{body.path!r} resolves to bodies {names} "
                            f"(articulation {spec.articulation or '<single>'!r})"
                        )
                    pieces.append(sensor)
                    perms.append(perm)
                self._sensors[spec.name] = {"mode": "general", "sensors": pieces}
                self._perm[spec.name] = perms
                continue
            if spec.mode == "general":
                fields = general_field_names(spec.name, list(spec.sensor_bodies))
                buckets = _general_buckets(list(spec.sensor_bodies))
                pieces, perms = [], []
                resolved: list[str] = []
                for field, expected in zip(fields, buckets):
                    sensor = env.scene[f"contact_{field}"]
                    names = resolve_bucket_names(
                        expected, robot_bodies, spec, inventory_fn=inventory
                    )
                    perm = [sensor.body_names.index(n) for n in names]
                    if sorted(perm) != list(range(len(names))):
                        raise ValueError(
                            f"contact group {spec.name!r}: body resolution "
                            f"mismatch sensor={sensor.body_names} vs "
                            f"expected={names}"
                        )
                    pieces.append(sensor)
                    perms.append(perm)
                    resolved.extend(names)
                self._reject_duplicates(spec, "sensor_bodies", resolved)
                self._sensors[spec.name] = {"mode": "general", "sensors": pieces}
                self._perm[spec.name] = perms
            else:
                self._resolve_bare_filters(spec, robot_bodies, inventory)
                pieces, perms = [], []
                resolved = []
                for j, entry in enumerate(spec.sensor_bodies):
                    sensor = env.scene[f"contact_{spec.name}_{j}"]
                    name = resolve_entry_name(entry, robot_bodies, spec, j, inventory_fn=inventory)
                    idx = sensor.body_names.index(name)
                    pieces.append(sensor)
                    perms.append([idx])
                    resolved.append(name)
                self._reject_duplicates(spec, "sensor_bodies", resolved)
                self._sensors[spec.name] = {"mode": "pairwise", "sensors": pieces}
                self._perm[spec.name] = perms
        self._bound = True

    def _resolve_bare_filters(
        self, spec: ContactGroupSpec, robot_bodies: list[str], inventory_fn
    ) -> None:
        """Resolve bare filter bodies in the owning articulation (fail fast).

        A bare filter expression must match exactly one body so the filter
        axis ``F`` stays equal to ``len(filter_bodies)`` (the shape contract).
        Explicit prim paths / env-prefixed prop expressions pass through.
        """
        resolved: list[str] = []
        for j, entry in enumerate(spec.filter_bodies or ()):
            if _is_explicit_prim_path(entry):
                continue
            _, pattern = split_scope(entry)
            compiled = re.compile(pattern)
            names = [n for n in robot_bodies if compiled.fullmatch(n)]
            if len(names) != 1:
                reason = (
                    f"filter entry {j} matched {len(names)} robot bodies "
                    f"(a bare filter body must resolve exactly one body in "
                    f"articulation {spec.articulation or '<single>'!r})"
                )
                raise contact_candidate_error(
                    spec.name,
                    "filter_bodies",
                    [pattern],
                    inventory_fn(),
                    "isaaclab",
                    reason=reason,
                )
            resolved.append(names[0])
        self._reject_duplicates(spec, "filter_bodies", resolved)

    @staticmethod
    def _reject_duplicates(spec: ContactGroupSpec, arg: str, names: list[str]) -> None:
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"contact group {spec.name!r}: {arg} resolve to duplicate "
                f"bodies {duplicates}; each resolved body must be distinct"
            )

    def group_axes(self, spec: ContactGroupSpec) -> tuple[int, int]:
        """Resolved ``(B, F)`` sensor/filter counts (post bind_env).

        Feeds the obs catalog's per-key shape resolution.  general: B =
        summed bucket bodies (the concatenated ``net_forces_w`` slices), F =
        0; pairwise: exactly one body per ``sensor_bodies`` entry (the
        one-to-many limitation), F = ``len(filter_bodies)`` (each entry
        contributes one filter prim expr).
        """
        perms = self._perm.get(spec.name, [])
        if spec.mode == "general":
            return sum(len(p) for p in perms), 0
        return len(perms), len(spec.filter_bodies or ())

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
            # one (N, B_bucket, 3) slice per scope bucket, concatenated
            # along B (single flat bucket == the classic single sensor)
            pieces = [s.data.net_forces_w[:, p, :] for s, p in zip(entry["sensors"], perm)]
            if len(pieces) == 1:
                return pieces[0].clone()
            return torch.cat(pieces, dim=1).clone()
        # force_matrix_w: (N, 1, F, 3) world per entry (contact_sensor.py:386)
        pieces = [s.data.force_matrix_w[:, p, :, :] for s, p in zip(entry["sensors"], perm)]
        return torch.cat(pieces, dim=1).clone()

    def close(self) -> None:
        """Release the cached sensor handles (ContactBackend lifecycle)."""
        self._sensors.clear()
        self._perm.clear()
        self._all_bodies.clear()
        self._all_inventories.clear()
        self._bound = False


def _fullmatch(names: list[str], patterns: list[str]) -> list[str]:
    """Names (input order preserved) fullmatching ANY pattern."""
    compiled = [re.compile(p) for p in patterns]
    return [n for n in names if any(c.fullmatch(n) for c in compiled)]


def spec_body_frame_unsupported(specs: list[ContactGroupSpec]) -> bool:
    """True when any isaaclab spec asks for the v1-unsupported body frame."""
    return any(s.frame == "body" for s in specs)
