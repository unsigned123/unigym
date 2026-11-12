"""isaaclab contact backend: ContactSensorCfg construction + raw reading.

This module is the ONLY module in the contacts package allowed to import
isaaclab (imports live inside the functions that need it, mirroring
``cameras/isaaclab_backend.py``); the module itself is import-safe without
the kit.

Semantics (authoritative local source: Isaac Lab 2.3.2
``sensors/contact_sensor/contact_sensor.py``):

* ``general`` groups -> one ContactSensorCfg with
  ``prim_path = {ENV_REGEX_NS}/Robot/(<union of sensor_bodies>)``; the raw
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
  ``{ENV}/{scope}/{name}`` paths; body resolution uses the last segment.
* B ordering: robot asset body order filtered by the entry regexes
  (``pairwise``: entry order, one body each).  F ordering:
  ``filter_bodies`` entry order (PhysX preserves pattern order).
* ``frame: body`` is a v1 NotImplementedError for this backend (plan §6
  allowed list); ``frame: world`` is direct.
* Sensor buffers are recycled in-place by the sensor update — read_raw
  ALWAYS clones (cameras package lesson, annotator-buffer aliasing).
"""

from __future__ import annotations

import re
from typing import Optional

from unigym.contacts.base import ENV_PREFIX_SENTINEL, ContactGroupSpec
from unigym.contacts.discovery import resolve_bucket_names, resolve_entry_name, usd_robot_inventory


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


def entry_prim_path(entry: str) -> str:
    """Robot-anchored ``prim_path`` for ONE sensor entry (per-entry cfgs)."""
    scope, name = split_scope(entry)
    if scope is not None:
        return f"{{ENV_REGEX_NS}}/Robot/{scope}/{name}"
    return f"{{ENV_REGEX_NS}}/Robot/({name})"


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


def general_prim_paths(entries: list[str]) -> list[str]:
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
            f"{{ENV_REGEX_NS}}/Robot/{scope}/{union}"
            if scope
            else f"{{ENV_REGEX_NS}}/Robot/{union}"
        )
    return out


def general_field_names(name: str, entries: list[str]) -> list[str]:
    """Scene-cfg field names for a general group (``__sN`` beyond the 1st)."""
    n = len(_general_buckets(entries))
    return [name if i == 0 else f"{name}__s{i}" for i in range(n)]


def sensor_prim_path(spec: ContactGroupSpec, entries: Optional[list[str]] = None) -> str:
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
    return f"{{ENV_REGEX_NS}}/Robot/({'|'.join(alts)})"


def filter_prim_exprs(spec: ContactGroupSpec) -> list[str]:
    """``filter_prim_paths_expr`` for a pairwise group.

    Bare entries are robot-body regexes (same namespace as sensor_bodies);
    entries starting with ``/`` or ``{ENV_REGEX_NS}`` pass through verbatim
    (props ``{ENV_REGEX_NS}/Prop_*``).
    """
    return [
        e if _is_explicit_prim_path(e) else f"{{ENV_REGEX_NS}}/Robot/{e}"
        for e in spec.filter_bodies
    ]


class IsaacLabContactBackend:
    """Backend object for all groups of one run (isaaclab)."""

    backend_key = "isaaclab"

    @classmethod
    def prop_filter_paths(cls, name: str, articulated: bool) -> list[str]:
        """Native contact-filter address for a declared prop."""
        if articulated:
            return [f"{{ENV_REGEX_NS}}/Prop_{name}/.*"]
        return [f"{{ENV_REGEX_NS}}/Prop_{name}"]

    def __init__(self, specs: list[ContactGroupSpec]):
        self.specs = list(specs)
        self._sensors: dict[str, dict] = {}  # name -> pieces
        self._perm: dict[str, list[list[int]]] = {}  # name -> per-piece perm
        self._bound = False

    # ---------------------------------------------------------------- #
    # scene contribution (before cloning)
    # ---------------------------------------------------------------- #
    def setup_scene(self) -> dict:
        """``{scene_field: ContactSensorCfg}`` for every group.

        Fields: ``contact_<name>`` (general, single scope bucket) and
        ``contact_<name>__sN`` (general, further scope buckets) and
        ``contact_<name>_<j>`` (pairwise entry j).  ``history_length=1``
        matches the manager's v1 history support; ``track_air_time`` stays
        off (we serve forces, not timings).
        """
        from isaaclab.sensors import ContactSensorCfg

        fields_cfg: dict = {}
        for spec in self.specs:
            if spec.mode == "general":
                paths = general_prim_paths(list(spec.sensor_bodies))
                fields = general_field_names(spec.name, list(spec.sensor_bodies))
                for field, path in zip(fields, paths):
                    fields_cfg[f"contact_{field}"] = ContactSensorCfg(
                        prim_path=path, history_length=1
                    )
            else:
                exprs = filter_prim_exprs(spec)
                for j, entry in enumerate(spec.sensor_bodies):
                    fields_cfg[f"contact_{spec.name}_{j}"] = ContactSensorCfg(
                        prim_path=entry_prim_path(entry),
                        filter_prim_paths_expr=exprs,
                        history_length=1,
                    )
        return fields_cfg

    # ---------------------------------------------------------------- #
    # bind (after construction)
    # ---------------------------------------------------------------- #
    def bind_env(self, env) -> None:
        """Resolve bodies + grab sensor handles; fail-fast on mismatches.

        Unresolvable sensor declarations refuse with a
        ``ContactCandidateError`` carrying the robot's discovered prim
        inventory (``usd_robot_inventory``: rigid-body prims + collision
        coverage, nested scopes included) -- foreign assets with unfamiliar
        naming become self-describing at the failure site.
        """
        if spec_body_frame_unsupported(self.specs):
            name = next(s.name for s in self.specs if s.frame == "body")
            raise NotImplementedError(
                f"contact group {name!r}: isaaclab frame='body' transform "
                f"is not implemented in v1 (plan §6 allowed list); use "
                f"frame: world"
            )
        robot_bodies = [str(n) for n in env.robot.data.body_names]

        def inventory():
            return usd_robot_inventory(env.robot)

        for spec in self.specs:
            if spec.mode == "general":
                fields = general_field_names(spec.name, list(spec.sensor_bodies))
                buckets = _general_buckets(list(spec.sensor_bodies))
                pieces, perms = [], []
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
                self._sensors[spec.name] = {"mode": "general", "sensors": pieces}
                self._perm[spec.name] = perms
            else:
                pieces, perms = [], []
                resolved: list[str] = []
                for j, entry in enumerate(spec.sensor_bodies):
                    sensor = env.scene[f"contact_{spec.name}_{j}"]
                    name = resolve_entry_name(entry, robot_bodies, spec, j, inventory_fn=inventory)
                    idx = sensor.body_names.index(name)
                    pieces.append(sensor)
                    perms.append([idx])
                    resolved.append(name)
                self._sensors[spec.name] = {"mode": "pairwise", "sensors": pieces}
                self._perm[spec.name] = perms
        self._bound = True

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
        return len(perms), len(spec.filter_bodies)

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
        self._bound = False


def _fullmatch(names: list[str], patterns: list[str]) -> list[str]:
    """Names (input order preserved) fullmatching ANY pattern."""
    compiled = [re.compile(p) for p in patterns]
    return [n for n in names if any(c.fullmatch(n) for c in compiled)]


def spec_body_frame_unsupported(specs: list[ContactGroupSpec]) -> bool:
    """True when any isaaclab spec asks for the v1-unsupported body frame."""
    return any(s.frame == "body" for s in specs)
