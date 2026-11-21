"""contacts package contract: key space, ContactGroupSpec, output shapes.

This module is **numpy/torch-only** (no isaaclab, no mujoco) so the spec
parsing, the validation and the format/component/aggregate unified layer are
unit-testable offline; both backend modules import lazily (mirroring the
``cameras/`` package convention: isaaclab imports live exclusively in
``backends/isaaclab2/contacts.py``, mujoco access in
``backends/mujoco/contacts.py``).

Key space (plan UniGym-contacts contact-manager plan)
=======================================================================
Every contact *group* named ``<name>`` serves exactly one record key::

    contact/<name>   -> shape table below (torch (N, ...) on env.device)

Shapes (B = resolved sensor bodies, F = resolved filter bodies,
N = num_envs; aggregate may remove the F axis on pairwise groups):

============  ==============  ==============
format        general         pairwise
============  ==============  ==============
raw           (N, B, 3)       (N, B, F, 3)
norm          (N, B)          (N, B, F)
contact       (N, B) bool     (N, B, F) bool
============  ==============  ==============

Rules baked in here:

* a group names its sensor and filter bodies with explicit typed
  :class:`~unigym.contacts.refs.ContactBodyRef` records only (Scheme A,
  breaking): no string body lists, no ``auto`` owner, no
  ``{prop: ...}`` / ``{contact_group: ...}`` references and no implicit
  global regex union;
* ``mode`` is derived: non-empty ``filter_refs`` -> pairwise, otherwise
  general.  ``owner`` is derived metadata from the refs, never a resolver;
* ``frame`` is **required with no default** (world|body) — no silent
  world-frame semantics;
* ``component`` defaults to ``mag``; ``x|y|z`` are SIGNED projections
  (``raw`` ignores ``component``);
* per-group isolation over sharing (one sensor cfg per group / pairwise
  entry), same isolation philosophy as cameras/;
* baseline protocol: a config without a ``contacts:`` section produces
  zero code paths (``build_contact_manager(None) -> None``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import ClassVar, Mapping, Optional, Protocol, Sequence, cast

import numpy as np
import torch

from .refs import ContactBodyRef

FORMATS = ("raw", "norm", "contact")
COMPONENTS = ("mag", "x", "y", "z")
FRAMES = ("world", "body")
MODES = ("general", "pairwise")
AGGREGATIONS = ("net", "sum_magnitudes")
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
_REMOVED_KEYS = {
    "sensor_bodies": (
        "key 'sensor_bodies' was removed by the breaking contact schema; "
        "declare sensor_refs: [{owner: robot|object|support|fixture|environment, ...}] "
        "(explicit typed refs -- no string body lists, no auto owner, no global regex union)"
    ),
    "filter_bodies": (
        "key 'filter_bodies' was removed by the breaking contact schema; "
        "declare filter_refs: [{owner: ..., ...}] (an empty/omitted list keeps "
        "the group general; a non-empty list makes it pairwise)"
    ),
    "articulation": (
        "a group-level 'articulation' is no longer a resolver; put "
        "articulation: <name> inside each robot ref "
        "({owner: robot, articulation: <name>, body|body_set: ...})"
    ),
    "mode": (
        "'mode' is derived: a non-empty filter_refs list makes the group "
        "pairwise, otherwise it is general"
    ),
    "owner": (
        "'owner' is derived metadata from sensor_refs/filter_refs; remove the key "
        "and declare the owner on each ref"
    ),
}


@dataclass
class ContactGroupSpec:
    """One declared contact group (yaml ``contacts:`` list entry).

    Attributes:
        name: group name (unique across the run; part of the
            ``contact/<name>`` key; restricted to ``[A-Za-z0-9_-]+`` because
            it also becomes an isaaclab scene-cfg field suffix).
        sensor_refs: explicit :class:`ContactBodyRef` sequence defining the B
            axis in declaration order (backend-native order within each ref).
        frame: ``world`` or ``body`` — REQUIRED, no default.
            ``body`` forces the per-sensor-body frame (isaaclab: v1
            NotImplementedError; mujoco: rotates by the sensor body
            orientation).
        filter_refs: explicit filter refs; non-empty -> pairwise.  F follows
            declaration order, backend-native body order within each ref.
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
        aggregation: force-fact aggregation for ``contact/<name>/force``.
            ``net`` (default) uses the group's net sensor force vector
            magnitude; ``sum_magnitudes`` sums per-contact force magnitudes
            BEFORE the configured aggregate.
        manifold: opt in to a backend runtime-query contact manifold for this
            group (``q.contacts.manifold(name)``).  Default ``False`` keeps the
            force/obs behavior unchanged.  ``True`` requires a pairwise group
            (non-empty ``filter_refs``): a general group has no filtered
            contact source to report.
    """

    name: str
    sensor_refs: Sequence[ContactBodyRef]
    frame: str  # REQUIRED, no default
    filter_refs: Sequence[ContactBodyRef] = ()
    format: str = "raw"
    component: str = "mag"
    threshold: Optional[float] = None
    aggregate: str = "none"
    history: int = 1
    aggregation: str = "net"
    manifold: bool = False

    def __post_init__(self):
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name.strip() != self.name
            or "/" in self.name
            or not re.fullmatch(r"[A-Za-z0-9_-]+", self.name)
        ):
            raise ValueError(
                f"contact group name {self.name!r} must be non-empty, "
                f"trimmed, free of '/' and match [A-Za-z0-9_-]+ (it is part "
                f"of the contact/<name> key and a scene field suffix)"
            )
        self.sensor_refs = _ref_list(self.sensor_refs, "sensor_refs", self.name, required=True)
        self.filter_refs = _ref_list(self.filter_refs, "filter_refs", self.name, required=False)
        if self.format not in FORMATS:
            raise ValueError(
                f"contact {self.name!r}: format must be one of " f"{FORMATS}, got {self.format!r}"
            )
        if self.frame not in FRAMES:
            raise ValueError(
                f"contact {self.name!r}: frame is REQUIRED with no default; "
                f"must be one of {FRAMES}, got {self.frame!r}"
            )
        if self.component not in COMPONENTS:
            raise ValueError(
                f"contact {self.name!r}: component must be one "
                f"of {COMPONENTS}, got {self.component!r}"
            )
        if self.format == "contact":
            if self.threshold is None:
                raise ValueError(
                    f"contact {self.name!r}: format 'contact' requires " f"threshold (N)"
                )
            self.threshold = float(self.threshold)
        elif self.threshold is not None:
            raise ValueError(
                f"contact {self.name!r}: threshold is only valid with "
                f"format 'contact' (got {self.threshold!r} with "
                f"format {self.format!r})"
            )
        allowed_agg = AGGREGATE_CONTACT if self.format == "contact" else AGGREGATE_VALUE
        if self.aggregate not in allowed_agg:
            raise ValueError(
                f"contact {self.name!r}: aggregate {self.aggregate!r} not "
                f"valid for format {self.format!r} (allowed: {allowed_agg})"
            )
        if self.mode == "general" and self.aggregate != "none":
            raise ValueError(
                f"contact {self.name!r}: general groups have no filter axis "
                f"to aggregate; aggregate must be 'none'"
            )
        self.history = int(self.history)
        if self.history < 1:
            raise ValueError(f"contact {self.name!r}: history must be >= 1")
        if self.history > 1:
            raise NotImplementedError(
                f"contact {self.name!r}: history > 1 is not implemented in "
                f"v1 (plan §6 allowed list); got history={self.history}"
            )
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"contact {self.name!r}: aggregation must be one of "
                f"{AGGREGATIONS}, got {self.aggregation!r}"
            )
        if not isinstance(self.manifold, bool):
            raise ValueError(
                f"contact {self.name!r}: manifold must be a bool, " f"got {self.manifold!r}"
            )
        if self.manifold and self.mode != "pairwise":
            raise ValueError(
                f"contact {self.name!r}: manifold: true requires a pairwise "
                f"group (non-empty filter_refs); a general group has no "
                f"filtered contact source to report"
            )

    # ---- derived metadata ------------------------------------------------ #
    @property
    def mode(self) -> str:
        """Derived mode: non-empty filter_refs -> pairwise."""
        return "pairwise" if self.filter_refs else "general"

    @property
    def owner(self) -> str:
        """Derived group owner: one sensor owner, else ``mixed``."""
        owners = self.owners()
        return owners[0] if len(owners) == 1 else "mixed"

    def owners(self) -> list[str]:
        """Distinct sensor owners in declaration order."""
        seen: list[str] = []
        for ref in self.sensor_refs:
            if ref.owner not in seen:
                seen.append(ref.owner)
        return seen

    def articulation_names(self) -> list[str]:
        """Explicit robot articulations referenced by any ref (declaration order)."""
        out: list[str] = []
        for ref in (*self.sensor_refs, *self.filter_refs):
            if ref.owner == "robot" and ref.articulation and ref.articulation not in out:
                out.append(ref.articulation)
        return out

    def prop_names(self, owners: Sequence[str] = ("object", "fixture", "support")) -> list[str]:
        """Declared prop/support scopes referenced by any ref."""
        allowed = set(owners)
        out: list[str] = []
        for ref in (*self.sensor_refs, *self.filter_refs):
            if ref.owner in allowed and ref.name and ref.name not in out:
                out.append(ref.name)
        return out

    # ---- parsing -------------------------------------------------------- #
    @classmethod
    def from_yaml(cls, entry: dict) -> ContactGroupSpec:
        """Build a spec from one yaml ``contacts:`` list entry (Scheme A only).

        Unknown keys raise (fail-fast against typos).  Removed legacy keys
        raise a teaching error naming the explicit ref replacement.
        ``name``, ``sensor_refs`` and ``frame`` are required.
        """
        if not isinstance(entry, dict):
            raise ValueError(
                f"contact group entry must be a mapping, got " f"{type(entry).__name__}"
            )
        removed = sorted(set(entry) & set(_REMOVED_KEYS))
        if removed:
            details = "; ".join(f"{key}: {_REMOVED_KEYS[key]}" for key in removed)
            name = entry.get("name", "?")
            raise ValueError(f"contact spec {name!r} uses removed keys: {details}")
        allowed = {f.name for f in fields(cls)}
        unknown = set(entry) - allowed
        if unknown:
            raise ValueError(
                f"contact spec unknown keys {sorted(unknown)}; " f"allowed: {sorted(allowed)}"
            )
        missing = [k for k in ("name", "sensor_refs", "frame") if entry.get(k) is None]
        if missing:
            raise ValueError(
                f"contact spec missing required keys {missing} "
                f"(sensor_refs and frame are mandatory; declare explicit "
                f"{{owner: ...}} refs)"
            )
        for arg in ("sensor_refs", "filter_refs"):
            value = entry.get(arg)
            if isinstance(value, Mapping):
                raise ValueError(
                    f"contact {entry['name']!r}: {arg} is a mapping "
                    f"{value!r}; the legacy reference forms "
                    f"({{contact_group: ...}} / {{prop: ...}}) are removed -- "
                    f"declare a list of explicit {{owner: ...}} refs"
                )
        kwargs = dict(entry)
        return cls(**kwargs)

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
            shape = (num_envs, B, base) if self.format == "raw" else (num_envs, B)
            return shape
        shape = {
            "raw": (num_envs, B, int(F), 3),
            "norm": (num_envs, B, int(F)),
            "contact": (num_envs, B, int(F)),
        }[self.format]
        if self.aggregate != "none":  # drop the F axis (axis=-2), never xyz
            shape = shape[:2] + shape[3:]
        return shape

    def keys(self) -> list[str]:
        """Record/obs keys served by this spec (canonical single key)."""
        return [contact_key(self.name)]

    def force_key(self) -> str:
        """Scalar force-fact key (``contact/<name>/force``)."""
        return f"{contact_key(self.name)}/force"

    def active_bodies_key(self) -> str:
        """Active-body-count fact key (``contact/<name>/active_bodies``)."""
        return f"{contact_key(self.name)}/active_bodies"

    def metadata(self) -> dict:
        """Static per-group metadata for ``meta["contacts"]`` (JSON-safe)."""
        return {
            "owner": self.owner,
            "owners": self.owners(),
            "aggregation": self.aggregation,
            "frame": self.frame,
            "mode": self.mode,
            "unit": "N",
            "sensor_refs": [ref.to_dict() for ref in self.sensor_refs],
            "filter_refs": [ref.to_dict() for ref in self.filter_refs],
        }


def _ref_list(value, arg: str, name: str, *, required: bool) -> list[ContactBodyRef]:
    if value is None:
        if required:
            raise ValueError(f"contact {name!r}: {arg} is required")
        return []
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple)):
        raise ValueError(
            f"contact {name!r}: {arg} must be a list of explicit "
            f"{{owner: ...}} refs, got {value!r}"
        )
    refs = [ContactBodyRef.from_mapping(entry) for entry in value]
    if required and not refs:
        raise ValueError(f"contact {name!r}: {arg} must not be empty")
    return refs


# --------------------------------------------------------------------------- #
# backend construction registry face
# --------------------------------------------------------------------------- #
#: Canonical env backends that expose a contact face (no ``isaaclab`` /
#: ``isaaclab3`` aliases: those family spellings are not selectable).
BACKENDS = ("mujoco", "mjwarp", "isaaclab2", "isaaclab3_physx", "isaaclab3_newton")


class _ContactBackendFace(Protocol):
    """The backend contact face (registry boundary)."""

    backend_key: ClassVar[str]

    def __init__(
        self,
        specs,
        *,
        articulation_names: Sequence[str] | None = None,
        articulation_body_sets: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
        prop_names: Sequence[str] | None = None,
        articulated_props: Sequence[str] | None = None,
        contact_assets: Mapping[str, str] | None = None,
    ) -> None: ...


def _contact_face(backend: str) -> type[_ContactBackendFace]:
    """Resolve the backend contact face lazily through the registry."""
    from ..backends.registry import BackendSelectionError, backend_info

    try:
        info = backend_info(backend)
    except BackendSelectionError as err:
        raise NotImplementedError(
            f"contacts are not implemented for backend {backend!r} "
            f"(available: {', '.join(BACKENDS)})"
        ) from err
    factory = info.contact_backend
    if factory is None:
        raise NotImplementedError(
            f"contacts are not implemented for backend {backend!r} "
            f"(available: {', '.join(BACKENDS)})"
        )
    return cast("type[_ContactBackendFace]", factory())


def build_contact_backend(
    backend: str,
    specs,
    *,
    articulation_names: Sequence[str] | None = None,
    articulation_body_sets: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    prop_names: Sequence[str] | None = None,
    articulated_props: Sequence[str] | None = None,
    contact_assets: Mapping[str, str] | None = None,
) -> _ContactBackendFace:
    """Construct the backend contact object for *specs* (registry face)."""
    return _contact_face(backend)(
        specs,
        articulation_names=articulation_names,
        articulation_body_sets=articulation_body_sets,
        prop_names=prop_names,
        articulated_props=articulated_props,
        contact_assets=contact_assets,
    )


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
            out = {
                "sum": _sum_last,
                "mean": _mean_last,  # is -2 here
                "max": _amax_last,
            }[agg](out, axis=-2)
        return _cast(out, np.float32 if not _is_torch(out) else torch.float32)

    if spec.component == "mag":
        quantity = _norm_last(raw)
    else:
        quantity = raw[..., _AXIS_INDEX[spec.component]]

    if spec.format == "norm":
        agg = spec.aggregate
        if agg != "none":  # xyz is squeezed -> the F axis is LAST here
            quantity = {"sum": _sum_last, "mean": _mean_last, "max": _amax_last}[agg](
                quantity, axis=-1
            )
        return _cast(quantity, np.float32 if not _is_torch(quantity) else torch.float32)

    # format == contact
    mask = quantity > spec.threshold
    agg = spec.aggregate
    if agg == "any":
        mask = _any_last(mask, axis=-1)
    elif agg == "all":
        mask = _all_last(mask, axis=-1)
    return mask


def contact_force_facts(raw, spec: ContactGroupSpec):
    """Derive ``(force, active_bodies)`` from ONE raw backend read.

    ``force`` is ``(N,)`` in newtons: the magnitude of the group's net force
    vector (``aggregation: net``) or the sum of per-contact force magnitudes
    (``aggregation: sum_magnitudes``).  ``active_bodies`` is ``(N,)`` int:
    how many sensor bodies carry a contact/force above the group threshold
    (``threshold`` for ``format: contact`` groups, otherwise > 0 N).

    The raw array is never mutated; numpy and torch inputs are both
    supported so the two backends can never drift.
    """
    magnitudes = _norm_last(raw)  # (N, B) or (N, B, F)
    threshold = float(spec.threshold) if spec.format == "contact" else 0.0
    pairwise = magnitudes.ndim > 2
    if spec.aggregation == "net":
        # sum the force VECTORS over the sensor axis (and the filter axis for
        # pairwise groups), then take the magnitude of the net vector.
        net_vector = _sum_axes(raw, (-3, -2) if pairwise else (-2,))
        force = _norm_last(net_vector)
    else:
        # sum_magnitudes: first reduce the per-contact magnitudes over the
        # FILTER axis with the group's configured aggregate, then sum over
        # the sensor bodies.  ``none`` means "keep every filter contact" and
        # therefore sums the filter axis for the scalar fact.
        reduced = magnitudes
        if pairwise:
            if spec.aggregate == "max":
                reduced = _amax_last(magnitudes)
            elif spec.aggregate == "mean":
                reduced = _mean_last(magnitudes)
            else:
                reduced = _sum_last(magnitudes)
        force = _sum_last(reduced)
    above = magnitudes > threshold
    if pairwise:
        above = _any_last(above)  # (N, B): this body has any active contact
    active = _sum_axes(_as_int(above), (-1,))
    return force, active


def _sum_axes(x, axes: tuple[int, ...]):
    axes = tuple(int(a) for a in axes)
    if _is_torch(x):
        for axis in sorted(axes, reverse=True):
            x = x.sum(dim=axis)
        return x
    return x.sum(axis=axes)


def _as_int(x):
    return x.to(torch.int64) if _is_torch(x) else x.astype(np.int64)


def _norm_last(raw):
    return _sqrt_last(_sum_last(raw * raw, axis=-1))


def _sqrt_last(x):
    return x.sqrt() if _is_torch(x) else np.sqrt(x)


def _cast(x, dtype):
    return x.to(dtype) if _is_torch(x) else x.astype(dtype)
