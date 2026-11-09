"""contacts package contract: key space, ContactGroupSpec, output shapes.

This module is **numpy/torch-only** (no isaaclab, no mujoco) so the spec
parsing, the validation and the format/component/aggregate unified layer are
unit-testable offline; both backend modules import lazily (mirroring the
``cameras/`` package convention: isaaclab imports live exclusively in
``backends/isaaclab2/contacts.py``, mujoco access in ``backends/mujoco/contacts.py``).

Key space (plan UniGym-contacts contact-manager plan)
=======================================================================
Every contact *group* named ``<name>`` serves exactly one record key::

    contact/<name>   -> shape table below (torch (N, ...) on env.device)

Shapes (plan §2; B = resolved sensor bodies, F = resolved filter bodies,
N = num_envs; aggregate may remove the F axis on pairwise groups):

============  ==============  ==============
format        general         pairwise
============  ==============  ==============
raw           (N, B, 3)       (N, B, F, 3)
norm          (N, B)          (N, B, F)
contact       (N, B) bool     (N, B, F) bool
============  ==============  ==============

Rules baked in here:

* ``frame`` is **required with no default** (world|body) — no silent
  world-frame semantics;
* ``component`` defaults to ``mag``; ``x|y|z`` are SIGNED projections
  (so ``format: norm, component: z`` generalises the scalar
  ``foot_contact_force_z`` channel); ``raw`` ignores ``component``;
* per-group isolation over sharing (one sensor cfg per group / pairwise
  entry), same isolation philosophy as cameras/;
* baseline protocol: a config without a ``contacts:`` section produces
  zero code paths (``build_contact_manager(None) -> None``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import ClassVar, Optional, Protocol, Sequence, cast

import numpy as np
import torch

FORMATS = ("raw", "norm", "contact")
COMPONENTS = ("mag", "x", "y", "z")
FRAMES = ("world", "body")
MODES = ("general", "pairwise")
# aggregate vocabulary by format (plan §2: contact -> any|all; value -> sum|mean|max)
AGGREGATE_CONTACT = ("any", "all", "none")
AGGREGATE_VALUE = ("sum", "mean", "max", "none")

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
ENV_PREFIX_SENTINEL = "{ENV_REGEX_NS}"


# --------------------------------------------------------------------------- #
# key space helpers
# --------------------------------------------------------------------------- #
def contact_key(name: str) -> str:
    """Record/obs key of contact group *name* (``contact/<name>``)."""
    return f"contact/{name}"


# --------------------------------------------------------------------------- #
# group spec (yaml-declarable, backend-agnostic)
# --------------------------------------------------------------------------- #
@dataclass
class ContactGroupSpec:
    """One declared contact group (yaml ``contacts:`` list entry).

    Attributes:
        name: group name (unique across the run; part of the
            ``contact/<name>`` key; restricted to ``[A-Za-z0-9_-]+`` because
            it also becomes an isaaclab scene-cfg field suffix).
        sensor_bodies: body-name regex list matched (fullmatch) against the
            robot's body names; the resolved set defines axis B (backend
            asset order).
        frame: ``world`` or ``body`` — REQUIRED, no default.
            ``body`` forces the per-sensor-body frame (isaaclab: v1
            NotImplementedError; mujoco: rotates by the sensor body
            orientation).
        filter_bodies: pairwise filter body-name regex list; a bare entry is
            resolved under the robot prefix, an entry starting with
            ``/`` or ``{ENV_REGEX_NS}`` is used as an explicit prim path
            (props ``Prop_*``).  Presence of a
            non-empty list makes the group pairwise.
        mode: ``general`` (no filter — net force on the sensor bodies) or
            ``pairwise`` (per-filter force matrix).  Default derived from
            ``filter_bodies``; an explicit value must be consistent.
        format: ``raw`` (force vectors), ``norm`` (component magnitude /
            signed projection) or ``contact`` (boolean vs ``threshold``).
        component: ``mag`` (default) or ``x|y|z`` (signed projection);
            ignored by ``raw``.
        threshold: contact threshold in N; REQUIRED iff ``format: contact``.
        aggregate: optional F-axis reduction.  ``contact``: any|all|none;
            ``norm``/``raw``: sum|mean|max|none.  ``general`` groups have no
            F axis, so aggregate must stay ``none``.
        history: only 1 supported in v1 (>1 raises NotImplementedError,
            plan §6 allowed list).
    """

    name: str
    sensor_bodies: Sequence[str]
    frame: str  # REQUIRED, no default
    filter_bodies: Optional[Sequence[str]] = None
    mode: Optional[str] = None
    format: str = "raw"
    component: str = "mag"
    threshold: Optional[float] = None
    aggregate: str = "none"
    history: int = 1

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name
                or self.name.strip() != self.name or "/" in self.name
                or not re.fullmatch(r"[A-Za-z0-9_-]+", self.name)):
            raise ValueError(
                f"contact group name {self.name!r} must be non-empty, "
                f"trimmed, free of '/' and match [A-Za-z0-9_-]+ (it is part "
                f"of the contact/<name> key and a scene field suffix)")
        self.sensor_bodies = _str_list(self.sensor_bodies, "sensor_bodies",
                                       self.name)
        if self.filter_bodies is not None:
            self.filter_bodies = _str_list(self.filter_bodies,
                                           "filter_bodies", self.name)
        if self.mode is None:
            self.mode = "pairwise" if self.filter_bodies else "general"
        if self.mode not in MODES:
            raise ValueError(f"contact {self.name!r}: mode must be one of "
                             f"{MODES}, got {self.mode!r}")
        if (self.mode == "pairwise") != (self.filter_bodies is not None):
            raise ValueError(
                f"contact {self.name!r}: mode/filter_bodies inconsistent "
                f"(mode={self.mode!r}, filter_bodies={self.filter_bodies})")
        if self.format not in FORMATS:
            raise ValueError(f"contact {self.name!r}: format must be one of "
                             f"{FORMATS}, got {self.format!r}")
        if self.frame not in FRAMES:
            raise ValueError(
                f"contact {self.name!r}: frame is REQUIRED with no default; "
                f"must be one of {FRAMES}, got {self.frame!r}")
        if self.component not in COMPONENTS:
            raise ValueError(f"contact {self.name!r}: component must be one "
                             f"of {COMPONENTS}, got {self.component!r}")
        if self.format == "contact":
            if self.threshold is None:
                raise ValueError(
                    f"contact {self.name!r}: format 'contact' requires "
                    f"threshold (N)")
            self.threshold = float(self.threshold)
        elif self.threshold is not None:
            raise ValueError(
                f"contact {self.name!r}: threshold is only valid with "
                f"format 'contact' (got {self.threshold!r} with "
                f"format {self.format!r})")
        allowed_agg = (AGGREGATE_CONTACT if self.format == "contact"
                       else AGGREGATE_VALUE)
        if self.aggregate not in allowed_agg:
            raise ValueError(
                f"contact {self.name!r}: aggregate {self.aggregate!r} not "
                f"valid for format {self.format!r} (allowed: {allowed_agg})")
        if self.mode == "general" and self.aggregate != "none":
            raise ValueError(
                f"contact {self.name!r}: general groups have no filter axis "
                f"to aggregate; aggregate must be 'none'")
        self.history = int(self.history)
        if self.history < 1:
            raise ValueError(f"contact {self.name!r}: history must be >= 1")
        if self.history > 1:
            raise NotImplementedError(
                f"contact {self.name!r}: history > 1 is not implemented in "
                f"v1 (plan §6 allowed list); got history={self.history}")

    # ---- parsing -------------------------------------------------------- #
    @classmethod
    def from_yaml(cls, entry: dict) -> ContactGroupSpec:
        """Build a spec from one yaml ``contacts:`` list entry.

        Unknown keys raise (fail-fast against typos); ``name``,
        ``sensor_bodies`` and ``frame`` are required (frame has NO default).
        """
        if not isinstance(entry, dict):
            raise ValueError(f"contact group entry must be a mapping, got "
                             f"{type(entry).__name__}")
        allowed = {f.name for f in fields(cls)}
        unknown = set(entry) - allowed
        if unknown:
            raise ValueError(f"contact spec unknown keys {sorted(unknown)}; "
                             f"allowed: {sorted(allowed)}")
        missing = [k for k in ("name", "sensor_bodies", "frame")
                   if entry.get(k) is None]
        if missing:
            raise ValueError(f"contact spec missing required keys {missing} "
                             f"(frame is mandatory)")
        for arg in ("sensor_bodies", "filter_bodies"):
            if isinstance(entry.get(arg), dict):
                raise ValueError(
                    f"contact {entry['name']!r}: {arg} is still in the "
                    f"reference form {entry[arg]!r} -- resolve reference "
                    f"forms ({{body_parts: ...}} / {{prop: ...}}) via "
                    f"build_contact_manager/resolve_contact_entries first")
        kwargs = dict(entry)
        kwargs["sensor_bodies"] = list(kwargs["sensor_bodies"])
        if kwargs.get("filter_bodies") is not None:
            kwargs["filter_bodies"] = list(kwargs["filter_bodies"])
        return cls(**kwargs)

    # ---- resolution helpers --------------------------------------------- #
    def resolved_mode(self) -> str:
        return "pairwise" if self.filter_bodies else "general"

    # ---- shape table (plan §2, single source of truth) ------------------- #
    def raw_shape(self, num_envs: int, B: int, F: int = 0) -> tuple:
        """Raw backend-array shape BEFORE format/aggregate."""
        if self.mode == "general":
            return (num_envs, B, 3)
        return (num_envs, B, int(F), 3)

    def output_shape(self, num_envs: int, B: int, F: int = 0) -> tuple:
        """Final ``contact/<name>`` shape AFTER format + aggregate."""
        if self.mode == "general":
            base = {"raw": 3, "norm": 1, "contact": 1}[self.format]
            shape = (num_envs, B, base) if self.format == "raw" \
                else (num_envs, B)
            return shape
        shape = {"raw": (num_envs, B, int(F), 3),
                 "norm": (num_envs, B, int(F)),
                 "contact": (num_envs, B, int(F))}[self.format]
        if self.aggregate != "none":  # drop the F axis (axis=-2), never xyz
            shape = shape[:2] + shape[3:]
        return shape

    def keys(self) -> list[str]:
        """Record/obs keys served by this spec (canonical single key)."""
        return [contact_key(self.name)]


def _str_list(value, arg: str, name: str) -> list[str]:
    if (not isinstance(value, (list, tuple)) or len(value) == 0
            or not all(isinstance(v, str) and v for v in value)):
        raise ValueError(f"contact {name!r}: {arg} must be a non-empty "
                         f"list of non-empty strings, got {value!r}")
    return list(value)


# --------------------------------------------------------------------------- #
# declarative reference resolution : {body_parts: ...} / {prop: ...}
# --------------------------------------------------------------------------- #
BACKENDS = ("mujoco", "isaaclab")


def _body_spec_list(value, where: str) -> list[str]:
    """Validate a resolved sensor/filter list (non-empty, non-empty strs)."""
    if (not isinstance(value, (list, tuple)) or len(value) == 0
            or not all(isinstance(v, str) and v for v in value)):
        raise ValueError(f"{where} must be a non-empty list of non-empty "
                         f"strings, got {value!r}")
    return [str(v) for v in value]


class _ContactBackendFace(Protocol):
    """The backend-native reference-resolution face (registry boundary)."""

    backend_key: ClassVar[str]

    def __init__(self, specs) -> None: ...

    @classmethod
    def group_bodies(cls, value, path: str) -> list[str]: ...

    @classmethod
    def prop_filter_paths(cls, name: str,
                          articulated: bool) -> list[str]: ...


def _contact_face(backend: str) -> type[_ContactBackendFace]:
    """Resolve the backend contact face lazily through the registry.

    The face class owns BOTH the raw-reading backend object and the
    backend-native name/path resolution (body-parts map keys, prop body
    addresses), so the neutral package carries no backend-name branch.
    """
    from ..backends.registry import BackendSelectionError, backend_info
    try:
        info = backend_info(backend)
    except BackendSelectionError as err:
        raise NotImplementedError(
            f"contacts are not implemented for backend {backend!r} "
            f"(available: {', '.join(BACKENDS)})") from err
    factory = info.contact_backend
    if factory is None:
        raise NotImplementedError(
            f"contacts are not implemented for backend {backend!r} "
            f"(available: {', '.join(BACKENDS)})")
    return cast("type[_ContactBackendFace]", factory())


def build_contact_backend(backend: str, specs) -> _ContactBackendFace:
    """Construct the backend contact object for *specs* (registry face)."""
    return _contact_face(backend)(specs)


def resolve_body_part_group(body_parts, path: str, backend: str) -> list[str]:
    """Resolve a ``{body_parts: <path>}`` reference to concrete body specs.

    ``path`` addresses the robot profile's ``body_parts`` dict
    (``unigym/robots/*_constants.py``; collect_cli passes
    ``profile.body_parts``):

    * ``"<key>"`` -- the key's value must be a flat body list shared by
      both backends (e.g. ``feet``);
    * ``"<key>.<group>"`` -- the key's value is a NAMED-GROUP dict; the
      group's value is either a flat shared list or an explicit per-backend
      mapping ``{mujoco: [...], isaaclab: [...]}`` -- the entry for
      *backend* is returned (the engines name robot bodies differently;
      G1's ``grasp.left`` / ``grasp.right`` use the per-backend form).

    Name-addressing only: no backend knowledge is hardcoded here -- the
    resolution fails fast listing the available keys/groups/backends.
    """
    face = _contact_face(backend)
    if not isinstance(path, str) or not path:
        raise ValueError(f"body_parts path must be a non-empty string, "
                         f"got {path!r}")
    parts = path.split(".")
    if len(parts) > 2:
        raise ValueError(f"body_parts path {path!r} must be '<key>' or "
                         f"'<key>.<group>' (at most one '.')")
    if not body_parts or parts[0] not in body_parts:
        raise ValueError(
            f"body_parts path {path!r}: key {parts[0]!r} not in body_parts "
            f"(available: {sorted(body_parts or {})})")
    value = body_parts[parts[0]]
    if len(parts) == 1:
        if isinstance(value, dict):
            raise ValueError(
                f"body_parts path {path!r}: key {parts[0]!r} holds named "
                f"groups {sorted(value)} -- address one explicitly "
                f"({parts[0]}.<group>)")
        return _body_spec_list(value, f"body_parts[{path!r}]")
    group = parts[1]
    if not isinstance(value, dict) or group not in value:
        available = sorted(value) if isinstance(value, dict) else \
            "<flat list -- no named groups>"
        raise ValueError(
            f"body_parts path {path!r}: group {group!r} not in "
            f"{parts[0]!r} (available: {available})")
    gval = value[group]
    selected = face.group_bodies(gval, path)
    where = (f"body_parts[{path!r}][{face.backend_key!r}]"
             if isinstance(gval, dict) else f"body_parts[{path!r}]")
    return _body_spec_list(selected, where)


def resolve_prop_filter(name, backend: str, prop_names,
                        articulated_props=frozenset()) -> list[str]:
    """Resolve a ``{prop: <name>}`` filter reference to the prop body.

    The address is the backend-native prop body name: MJ body
    ``Prop_<name>`` (mujoco) vs the cloned prim path
    ``{ENV_REGEX_NS}/Prop_<name>`` (isaaclab).  Articulation props
     own a whole SUBTREE (``Prop_<name>_<link>``), so their
    filter is the subtree pattern.  Unknown names fail fast listing the
    declared props (the declaration is the single source of truth --
    zero hardcoded addresses).
    """
    face = _contact_face(backend)
    names = list(prop_names or [])
    if not isinstance(name, str) or not name:
        raise ValueError(f"prop reference must be a non-empty string, "
                         f"got {name!r}")
    if name not in names:
        raise ValueError(
            f"prop filter reference {name!r}: no such declared prop "
            f"(available: {sorted(names)}); the contacts section may only "
            f"reference props declared in the same run config")
    return face.prop_filter_paths(name, name in articulated_props)


def _resolve_ref(value, backend: str, arg: str, group: str,
                 body_parts, prop_names,
                 articulated_props=frozenset()) -> list[str]:
    """One sensor_bodies/filter_bodies field: reference form OR list."""
    if isinstance(value, dict):
        keys = set(value)
        if arg == "sensor_bodies":
            if keys != {"body_parts"}:
                raise ValueError(
                    f"contact {group!r}: {arg} reference form is "
                    f"{{body_parts: <key>[.<group>]}}, got keys {sorted(keys)}")
            return resolve_body_part_group(body_parts, value["body_parts"],
                                           backend)
        if keys != {"prop"}:
            raise ValueError(
                f"contact {group!r}: {arg} reference form is "
                f"{{prop: <name>}}, got keys {sorted(keys)}")
        return resolve_prop_filter(value["prop"], backend, prop_names,
                                   articulated_props)
    return _body_spec_list(value, f"contact {group!r}: {arg}")


def resolve_contact_entry(entry: dict, backend: str,
                          body_parts=None, prop_names=None,
                          articulated_props=frozenset()) -> dict:
    """Resolve ONE yaml ``contacts:`` entry's reference forms to lists.

    ``sensor_bodies`` is either an explicit list of body specs or the
    reference form ``{body_parts: "<key>[.<group>]"}``; ``filter_bodies``
    either an explicit list or ``{prop: <name>}``.  Everything else passes
    through untouched for :class:`ContactGroupSpec` validation.  Returns a
    NEW dict (the input is never mutated).
    """
    if not isinstance(entry, dict):
        raise ValueError(f"contact group entry must be a mapping, got "
                         f"{type(entry).__name__}")
    out = dict(entry)
    if "sensor_bodies" in out:
        out["sensor_bodies"] = _resolve_ref(out["sensor_bodies"], backend,
                                            "sensor_bodies",
                                            str(out.get("name", "?")),
                                            body_parts, prop_names,
                                            articulated_props)
    if out.get("filter_bodies") is not None:
        out["filter_bodies"] = _resolve_ref(out["filter_bodies"], backend,
                                            "filter_bodies",
                                            str(out.get("name", "?")),
                                            body_parts, prop_names,
                                            articulated_props)
    return out


def resolve_contact_entries(entries, backend: str,
                            body_parts=None, prop_names=None,
                            articulated_props=frozenset()) -> list[dict]:
    """Resolve every entry of a ``contacts:`` list (see
    :func:`resolve_contact_entry`); the list itself must be a list."""
    if entries is None:
        return []
    if not isinstance(entries, (list, tuple)):
        raise ValueError(f"yaml 'contacts' section must be a list of group "
                         f"entries, got {type(entries).__name__}")
    return [resolve_contact_entry(e, backend, body_parts, prop_names,
                                  articulated_props)
            for e in entries]


# --------------------------------------------------------------------------- #
# unified format/component/aggregate layer (numpy + torch; both backends
# produce raw arrays only — the semantics live HERE, never twice)
# --------------------------------------------------------------------------- #
def _is_torch(x) -> bool:
    return torch.is_tensor(x)


def _sum_last(x, axis: int = -1):
    return x.sum(dim=axis) if _is_torch(x) else x.sum(axis=axis)


def _mean_last(x, axis: int = -1):
    return x.mean(dim=axis) if _is_torch(x) else x.mean(axis=axis)


def _amax_last(x, axis: int = -1):
    return x.amax(dim=axis) if _is_torch(x) else x.max(axis=axis)


def _any_last(x, axis: int = -1):
    return x.any(dim=axis) if _is_torch(x) else x.any(axis=axis)


def _all_last(x, axis: int = -1):
    return x.all(dim=axis) if _is_torch(x) else x.all(axis=axis)


def apply_format(raw, spec: ContactGroupSpec):
    """Unified layer: raw array -> ``contact/<name>`` value (numpy or torch).

    ``raw`` is the backend-produced array with shape ``spec.raw_shape``:
    (N, B, 3) for general groups, (N, B, F, 3) for pairwise groups, forces
    already expressed in ``spec.frame``.  Steps:

    1. ``raw``   -> passthrough (``component`` ignored);
    2. ``norm``  -> component projection: ``mag`` = |F| (2-norm over xyz),
       ``x|y|z`` = SIGNED projection on that axis;
    3. ``contact``-> (component quantity > threshold) boolean;
    4. aggregate -> pairwise F-axis reduction (``sum|mean|max`` for
       raw/norm, ``any|all`` for contact); ``none`` keeps the F axis.

    Returns a NEW array (never mutates *raw*); dtype float32 for
    raw/norm, bool for contact.
    """
    if spec.format == "raw":
        out = raw
        agg = spec.aggregate
        if agg != "none":  # pairwise only (spec validation enforces); F axis
            out = {"sum": _sum_last, "mean": _mean_last,       # is -2 here
                   "max": _amax_last}[agg](out, axis=-2)
        return _cast(out, np.float32 if not _is_torch(out) else torch.float32)

    if spec.component == "mag":
        quantity = _norm_last(raw)
    else:
        quantity = raw[..., _AXIS_INDEX[spec.component]]

    if spec.format == "norm":
        agg = spec.aggregate
        if agg != "none":  # xyz is squeezed -> the F axis is LAST here
            quantity = {"sum": _sum_last, "mean": _mean_last,
                        "max": _amax_last}[agg](quantity, axis=-1)
        return _cast(quantity, np.float32 if not _is_torch(quantity)
                     else torch.float32)

    # format == contact
    mask = quantity > spec.threshold
    agg = spec.aggregate
    if agg == "any":
        mask = _any_last(mask, axis=-1)
    elif agg == "all":
        mask = _all_last(mask, axis=-1)
    return mask


def _norm_last(raw):
    return _sqrt_last(_sum_last(raw * raw, axis=-1))


def _sqrt_last(x):
    return x.sqrt() if _is_torch(x) else np.sqrt(x)


def _cast(x, dtype):
    return x.to(dtype) if _is_torch(x) else x.astype(dtype)
