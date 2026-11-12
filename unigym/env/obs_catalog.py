"""ObsCatalog: the observation catalog + request validation .

Symmetry principle with the NamedAction contract: ACTIONS are
produced by the adapter WITH names and consumed by the backend by name;
OBSERVATIONS are produced by the env WITH names (this catalog is the single
published source) and consumed by the adapter, which gets its requests
CHECKED against the catalog at env construction.  The joint
keys publish the FULL articulation state and requests select a named
SUBSET of it — the exact mirror of the action side (publish full, address
by name), so the catalog mediates the per-step data flow for the joint
keys only (name gather into the request order); ``obs_cache`` stays a
plain tensor dict otherwise.

What the catalog locks
======================
* **Fixed keys** — the requestable obs-cache keys every backend serves
  (``gravity_b`` / ``joint_pos_rel`` / ``joint_vel`` / ``command`` /
  ``prev_action`` / ``base_yaw`` / ``base_pos_w`` / ``base_quat_w`` /
  ``base_ang_vel_b`` / ``foot_contact_force``) plus the record-only extra
  keys that document the recorded channel (``base_lin_vel_b`` /
  ``foot_contact`` / ``foot_contact_force_z`` / ``foot_pos_z``).  Joint
  order semantics are EXPLICIT here: ``joint_pos_rel`` / ``joint_vel``
  publish the run's FULL articulation joint set with ``joint_names``
  naming every element (isaaclab = the asset's own
  ``robot.joint_names`` order, mujoco = the MJ hinge order — name-explicit,
  hence backend-agnostic; 43 joints for the G1 with-hand asset, fingers
  included).  The former ``joint_pos_rel_hand`` special case (mujoco-only)
  is subsumed and DELETED — both backends publish the same
  full set symmetrically.
* **Camera keys** — ``cam/<name>/<rgb|depth>`` + the depth aliases, from
  the wired CameraManager (shape/dtype are the manager's frame contract).
* **Prop keys** — ``<name>_pos_b`` / ``_vel_b`` obs keys from the wired
  PropManager's ``state_requests``; without props the zero-fill
  table stays requestable (isaaclab backend keeps the parity behavior).
* **Object keys** — ``object/<name>/{pos,quat,held[_<group>]}`` record keys
  (record-only, never obs).
* **Contact keys** — ``contact/<name>`` from the wired ContactManager
  (requestable; shapes resolved from the bound backend).
* **Articulation namespace (A4 rule)** — a run with exactly ONE
  policy controller keeps every historical bare key above (byte-identical).
  When the catalog is built for more than one controller, or for a
  non-policy (named/passive) articulation, that articulation's keys use
  ``articulation/<name>/<key>`` (:func:`articulation_key`); the policy
  articulation keeps the bare keys.  A4 ships the helper + rule; A5 wires
  the multi-controller catalog.

Request validation (subset semantics)
=====================================
``ObsCatalog.validate_request`` runs at env construction over every
``adapter.get_obs_requests()`` entry:

* unknown key → ``ValueError`` listing the available keys;
* a joint-vector key (``joint_pos_rel`` / ``joint_vel``) requested without
  ``joint_names`` → ``ValueError`` (no-name joint requests are still
  illegal — the explicitness twin of the NamedAction action contract);
* ``joint_names`` must be a NON-EMPTY, DUPLICATE-FREE SUBSET of the
  catalog names, in ANY order (a relaxation of the earlier
  "exactly the catalog order" rule): the request order IS the served
  order — the env gathers the full-articulation state by name into an
  ``(N, len(joint_names))`` cache entry.  Unknown names fail listing the
  published set;
* ``dim`` on a joint key must equal ``len(joint_names)``; other
  ``dim``/``shape``/dtype checks are unchanged.  ``command`` /
  ``prev_action`` are caller-fed passthroughs whose width the env cannot
  know statically — their specs carry ``shape=None`` (documented
  "runtime-determined") and the per-step last-dim check stays
  authoritative for them.

This module is torch-only (importable without isaaclab/mujoco).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..adapters.base import ImageObsRequest, RawObsRequest
from ..props.object_state import object_record_keys
from ..scenes.articulation import ARTICULATION_PREFIX, articulation_key

__all__ = [
    "ARTICULATION_PREFIX",
    "CATALOG_VERSION",
    "ObsCatalog",
    "ObsSpec",
    "PER_ARTICULATION_KEYS",
    "articulation_key",
    "build_obs_catalog",
    "canonicalize_obs_requests",
    "is_per_articulation_key",
    "resolve_obs_request_key",
    "split_articulation_key",
]

# catalog schema version (bump on semantic changes to the snapshot format)
# v2 : joint keys publish the full articulation set (43 for the
# G1 with-hand asset) instead of the 29-joint policy order
# v3 : every per-articulation robot-state key is published under
# ``articulation/<name>/...`` (single and multi alike; no bare projection)
CATALOG_VERSION = 3

# Every fixed-table key that belongs to ONE articulation (robot state).  The
# catalog publishes these under ``articulation/<name>/...`` only; a bare
# request is resolved to the adapter's unique controller at the env boundary.
PER_ARTICULATION_KEYS = frozenset(
    {
        "gravity_b",
        "joint_pos_rel",
        "joint_vel",
        "applied_torque",
        "prev_action",
        "command",
        "base_yaw",
        "base_pos_w",
        "base_quat_w",
        "base_ang_vel_b",
        "foot_contact_force",
        "base_lin_vel_b",
        "foot_contact",
        "foot_contact_force_z",
        "foot_pos_z",
    }
)


def is_per_articulation_key(key: str) -> bool:
    """True when *key* is a bare per-articulation robot-state key."""
    return key in PER_ARTICULATION_KEYS


def split_articulation_key(key: str) -> tuple[str | None, str]:
    """``(articulation name, base key)`` for a namespaced key, else ``(None, key)``.

    ``articulation/<name>/<base>`` -> ``(name, base)`` when ``<base>`` is a
    known per-articulation key; every other key is returned verbatim.
    """
    prefix = f"{ARTICULATION_PREFIX}/"
    if not key.startswith(prefix):
        return None, key
    rest = key[len(prefix) :]
    name, _, base = rest.partition("/")
    if name and base in PER_ARTICULATION_KEYS:
        return name, base
    return None, key


def resolve_obs_request_key(req) -> str:
    """Canonical observation-cache key for one adapter request.

    Every robot-state request MUST carry its own explicit
    ``req.articulation`` (v104.3): there is no env-side controller
    inference and no "first" fallback.  Global keys (``cam/``, ``contact/``,
    props, ``object/``, ``site/``) pass through untouched.
    """
    key = str(req.key)
    if not is_per_articulation_key(key):
        return key
    name = getattr(req, "articulation", None)
    if not name:
        raise ValueError(
            f"obs request '{key}' is articulation-scoped but carries no "
            f"articulation; set RawObsRequest.articulation explicitly"
        )
    return articulation_key(str(name), key)


def canonicalize_obs_requests(requests):
    """Rewrite bare per-articulation request keys to their namespaced form.

    Every request keeps its other fields; global and already-namespaced keys
    pass through unchanged.
    """
    out = []
    for req in requests:
        key = resolve_obs_request_key(req)
        if key != req.key:
            req = req._replace(key=key)
        out.append(req)
    return list(out)


# torch dtype -> snapshot name (and back)
_DTYPE_TO_NAME = {torch.float32: "f32", torch.uint8: "u8", torch.bool: "bool"}

# zero-fill prop keys: when NO PropManager is wired the isaaclab
# backend still serves these (zero-filled every step) so old adapters keep
# running without a props section.  Dims mirror props/{door,ball,generic}.py
# state_requests; the table moves here (single source) with the catalog.
_ZERO_FILL_PROP_KEYS = {
    "door_pos": 3,
    "door_normal": 3,
    "door_angle": 1,
    "left_or_right": 2,
    "ball_pos_b": 3,
    "ball_vel_b": 3,
    "prop_pos_b": 3,
    "prop_vel_b": 3,
}

# keys whose width is decided by the caller at run time (passthrough values:
# sampler command / adapter raw action).  shape=None in the spec; the env's
# per-step last-dim check stays the authority.
_PASSTHROUGH_KEYS = ("command", "prev_action")


# --------------------------------------------------------------------------- #
# ObsSpec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ObsSpec:
    """Contract of one cataloged observation key.

    Attributes:
        key: obs-cache / record key (e.g. ``"gravity_b"``,
            ``"joint_pos_rel"``, ``"cam/ego/rgb"``,
            ``"contact/feet_ground"``).
        shape: per-env shape WITHOUT the batch axis; ``None`` marks a
            caller-fed passthrough key (``command`` / ``prev_action``)
            whose width the env validates per step against the request.
        dtype: snapshot dtype name — ``"f32" | "u8" | "bool"``.
        joint_names: the exact joint name of every element (joint-vector
            keys only); ``None`` for non-joint keys.  This is the published
            order contract (== ``profile.joint_order`` for the joint keys).
        unit: SI unit hint (``"rad"``, ``"m"``, ``"N"``, ...; None = n/a).
        frame: reference frame hint (``"base"``, ``"world"``, ...; None = n/a).
        source: producer family — ``"robot" | "computed" | "adapter" |
            "sampler" | "camera" | "prop" | "contact" | "object" |
            "contact_sensor"`` (documentation for humans and consumers).
    """

    key: str
    shape: tuple | None
    dtype: str
    joint_names: tuple | None = None
    unit: str | None = None
    frame: str | None = None
    source: str = "robot"


def _dtype_name(dtype) -> str:
    name = _DTYPE_TO_NAME.get(dtype)
    if name is None:
        raise ValueError(
            f"unsupported frame dtype {dtype!r} for the "
            f"obs catalog (allowed: {sorted(_DTYPE_TO_NAME, key=str)})"
        )
    return name


# --------------------------------------------------------------------------- #
# the catalog
# --------------------------------------------------------------------------- #
class ObsCatalog:
    """The run's observation catalog: specs per key + request validation."""

    def __init__(self, entries: dict, requestable=None, articulation: str | None = None):
        self.entries: dict[str, ObsSpec] = dict(entries)
        self.requestable: set[str] = set(self.entries) if requestable is None else set(requestable)
        # the articulation this per-controller catalog belongs to; bare
        # requests validate against its namespaced keys (None for the merged
        # multi catalog -- every request there is already namespaced)
        self.articulation = articulation
        unknown = self.requestable - set(self.entries)
        if unknown:
            raise ValueError(
                f"requestable keys {sorted(unknown)} have no "
                f"catalog entry (internal wiring bug)"
            )

    # ---- container surface ---------------------------------------------- #
    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def __getitem__(self, key: str) -> ObsSpec:
        return self.entries[key]

    def __iter__(self):
        return iter(sorted(self.entries))

    def __len__(self) -> int:
        return len(self.entries)

    def keys(self) -> list[str]:
        """All cataloged keys, sorted (obs + record)."""
        return sorted(self.entries)

    def requestable_keys(self) -> set:
        """Keys an adapter may request into ``obs_cache``."""
        return set(self.requestable)

    # ---- request validation (env construction time) ---------------------- #
    def validate_request(self, req) -> None:
        """Check one adapter obs request against the catalog.

        Accepts :class:`~unigym.adapters.base.RawObsRequest` and
        :class:`~unigym.adapters.base.ImageObsRequest`.  A bare
        per-articulation key is resolved with :func:`resolve_obs_request_key`
        from the request's own explicit ``articulation`` (no inference).
        Raises ``ValueError`` on: unknown key, record-only key,
        missing/mismatched ``joint_names`` on a joint-vector key, or
        dim/shape/dtype diff.
        """
        key = resolve_obs_request_key(req)
        spec = self.entries.get(key)
        if spec is None:
            raise ValueError(
                f"adapter requested unsupported obs key '{key}'; "
                f"available obs keys: {sorted(self.requestable)}"
            )
        if key not in self.requestable:
            raise ValueError(
                f"obs key '{key}' is record-only "
                f"(source={spec.source}); it never enters obs_cache; "
                f"requestable obs keys: {sorted(self.requestable)}"
            )

        if isinstance(req, ImageObsRequest):
            self._validate_image(req, spec)
        else:
            self._validate_raw(req, spec)

    def _validate_raw(self, req: RawObsRequest, spec: ObsSpec) -> None:
        self._validate_joint_names(req, spec)
        if spec.shape is None:  # passthrough key: runtime check stays authoritative
            return
        if spec.joint_names is not None:
            # joint key: the served width IS the request's name count
            # (subset semantics; the names were validated above)
            want = len(tuple(getattr(req, "joint_names", None) or ()))
            if req.dim != want:
                raise ValueError(
                    f"obs request '{req.key}' dim {req.dim} != " f"len(joint_names) {want}"
                )
            return
        if req.dim != spec.shape[-1]:
            raise ValueError(
                f"obs request '{req.key}' dim {req.dim} != catalog "
                f"{list(spec.shape)} (last dim {spec.shape[-1]})"
            )

    def _validate_image(self, req: ImageObsRequest, spec: ObsSpec) -> None:
        if tuple(req.shape) != tuple(spec.shape or ()):
            raise ValueError(
                f"obs request '{req.key}' shape {tuple(req.shape)} != "
                f"catalog {list(spec.shape or ())}"
            )
        want = _DTYPE_TO_NAME.get(req.dtype)
        if want != spec.dtype:
            raise ValueError(
                f"obs request '{req.key}' dtype {req.dtype} != catalog " f"'{spec.dtype}'"
            )

    def _validate_joint_names(self, req: RawObsRequest, spec: ObsSpec) -> None:
        """Subset semantics: explicit, non-empty, duplicate-free,
        every name published by the catalog; ANY order (request order ==
        served order)."""
        req_names = getattr(req, "joint_names", None)
        if spec.joint_names is None:
            if req_names is not None:
                raise ValueError(
                    f"obs request '{req.key}' declares joint_names "
                    f"{list(req_names)} but the catalog key is not a "
                    f"joint-vector key"
                )
            return
        if req_names is None:
            raise ValueError(
                f"obs request '{req.key}' addresses joints but carries "
                f"no joint_names; joint-vector requests "
                f"must be explicit (the catalog publishes "
                f"{len(spec.joint_names)} joints: "
                f"{list(spec.joint_names)})"
            )
        names = tuple(req_names)
        if not names:
            raise ValueError(
                f"obs request '{req.key}' carries an empty joint_names "
                f"tuple (the catalog publishes {len(spec.joint_names)} "
                f"joints: {list(spec.joint_names)})"
            )
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"obs request '{req.key}' has duplicate joint names "
                f"{dupes} (a duplicate would make the column mapping "
                f"ambiguous)"
            )
        known = set(spec.joint_names)
        unknown = [n for n in names if n not in known]
        if unknown:
            raise ValueError(
                f"obs request '{req.key}' names joints the catalog does "
                f"not publish: {unknown}; the catalog publishes "
                f"{len(spec.joint_names)} joints: {list(spec.joint_names)}"
            )
        # names ⊆ catalog names, in the requester's order — accepted.

    # ---- meta snapshot ---------------------------------------------------- #
    def to_meta(self) -> dict:
        """JSON-safe snapshot (meta.json ``obs_catalog`` key, additive)."""
        return {
            "version": CATALOG_VERSION,
            "keys": {
                spec.key: {
                    "shape": (None if spec.shape is None else [int(d) for d in spec.shape]),
                    "dtype": spec.dtype,
                    "joint_names": (None if spec.joint_names is None else list(spec.joint_names)),
                    "unit": spec.unit,
                    "frame": spec.frame,
                    "source": spec.source,
                }
                for spec in (self.entries[k] for k in sorted(self.entries))
            },
            "requestable": sorted(self.requestable),
        }


def _fixed_entries(profile, joint_names=None) -> tuple[dict, set]:
    """The fixed key table (obs-requestable + record-only) from the profile.

    ``joint_names`` is the run's FULL articulation joint order (every
    actuated DOF incl. non-policy fingers); ``None`` falls back to
    ``profile.full_joint_names`` then ``profile.joint_order``.
    """
    joints = tuple(joint_names or profile.full_joint_names or profile.joint_order or ())
    feet = tuple(profile.foot_bodies or ())
    f = len(feet)
    entries = {
        # ---- requestable obs-cache keys (fixed, every backend) ----
        "gravity_b": ObsSpec("gravity_b", (3,), "f32", unit="g", frame="base", source="computed"),
        "joint_pos_rel": ObsSpec(
            "joint_pos_rel", (len(joints),), "f32", joint_names=joints, unit="rad", source="robot"
        ),
        "joint_vel": ObsSpec(
            "joint_vel", (len(joints),), "f32", joint_names=joints, unit="rad/s", source="robot"
        ),
        "applied_torque": ObsSpec(
            "applied_torque", (len(joints),), "f32", joint_names=joints, unit="N·m", source="robot"
        ),
        "prev_action": ObsSpec("prev_action", None, "f32", source="adapter"),
        "command": ObsSpec("command", None, "f32", source="sampler"),
        "base_yaw": ObsSpec("base_yaw", (1,), "f32", unit="rad", frame="world", source="robot"),
        "base_pos_w": ObsSpec("base_pos_w", (3,), "f32", unit="m", frame="world", source="robot"),
        "base_quat_w": ObsSpec(
            "base_quat_w", (4,), "f32", frame="world", source="robot"
        ),  # (w,x,y,z)
        "base_ang_vel_b": ObsSpec(
            "base_ang_vel_b", (3,), "f32", unit="rad/s", frame="base", source="robot"
        ),
        "foot_contact_force": ObsSpec(
            "foot_contact_force", (f,), "f32", unit="N", source="contact_sensor"
        ),
        # ---- record-only extras (documented; never obs_cache) ----
        "base_lin_vel_b": ObsSpec(
            "base_lin_vel_b", (3,), "f32", unit="m/s", frame="base", source="robot"
        ),
        "foot_contact": ObsSpec("foot_contact", (f,), "bool", source="contact_sensor"),
        "foot_contact_force_z": ObsSpec(
            "foot_contact_force_z", (f,), "f32", unit="N", frame="world", source="contact_sensor"
        ),
        "foot_pos_z": ObsSpec("foot_pos_z", (f,), "f32", unit="m", frame="world", source="robot"),
    }
    requestable = set(entries) - {
        "base_lin_vel_b",
        "foot_contact",
        "foot_contact_force_z",
        "foot_pos_z",
    }
    return entries, requestable


def build_obs_catalog(
    profile,
    camera_manager=None,
    prop_manager=None,
    contact_manager=None,
    zero_fill_props: bool = False,
    articulation_joint_names=None,
    record_sites=None,
    *,
    articulation: str,
) -> ObsCatalog:
    """Build the run's :class:`ObsCatalog` at env-construction time.

    Args:
        profile: the run's :class:`~unigym.robots.base.RobotProfile`
            (joint/foot orders feed the fixed key table).
        camera_manager: optional wired
            :class:`~unigym.cameras.CameraManager` — its ``cam/*`` keys and
            depth aliases join the catalog (shape/dtype from the
            manager's frame requests).
        prop_manager: optional wired :class:`~unigym.props.PropManager` —
            ``state_requests`` obs keys join as requestable and
            ``object/<name>/*`` record keys join as record-only.
        contact_manager: optional wired
            :class:`~unigym.contacts.ContactManager` — ``contact/<name>``
            keys join as requestable with backend-resolved shapes.
        zero_fill_props: include the zero-fill prop table (the
            isaaclab backend serves those keys zero-filled when no
            PropManager is wired; the mujoco backend does not).
        articulation_joint_names: the backend's resolved FULL articulation
            joint order for the joint keys (isaaclab passes the
            asset's ``robot.joint_names``, mujoco its MJ hinge order —
            fingers included, prop freejoints excluded; both backends
            publish the same name SET symmetrically).  ``None`` falls back
            to ``profile.full_joint_names`` then ``profile.joint_order``.
        articulation: the articulation name every per-articulation robot
            state key is namespaced under (required; v103.5).
        record_sites: optional tuple of MJ site names the backend records
            (T3.2 EEF site family: ``site/<name>/pos`` world-frame
            record keys, e.g. the GR1 grip sites the pack criteria's far
            predicates consume).  Record-only (never obs_cache); empty /
            ``None`` (the default) joins nothing -- zero behavior diff.

    Returns the catalog; the caller validates every adapter obs request
    against it (construction-time fail-fast) and snapshots ``to_meta()``
    into the run meta.
    """
    if not articulation:
        raise ValueError(
            "build_obs_catalog requires an articulation name (v103.5: every "
            "per-articulation robot-state key is namespaced "
            "articulation/<name>/...; there is no bare projection)"
        )
    entries, requestable = _fixed_entries(profile, articulation_joint_names)
    # v103.5: the fixed table is robot state owned by ONE articulation --
    # publish it namespaced (single and multi use the same rule).
    entries = {
        articulation_key(articulation, key): ObsSpec(
            articulation_key(articulation, spec.key),
            spec.shape,
            spec.dtype,
            joint_names=spec.joint_names,
            unit=spec.unit,
            frame=spec.frame,
            source=spec.source,
        )
        for key, spec in entries.items()
    }
    requestable = {articulation_key(articulation, key) for key in requestable}

    # ---- camera keys (cam/* + depth aliases; manager-owned shapes)
    if camera_manager is not None:
        for key in sorted(camera_manager.obs_keys()):
            frame = camera_manager.frame_request(key)
            entries[key] = ObsSpec(
                key,
                tuple(int(d) for d in frame.shape),
                _dtype_name(frame.dtype),
                frame="sensor",
                source="camera",
            )
        requestable |= set(camera_manager.obs_keys())

    # ---- prop obs keys + object record keys
    if prop_manager is not None:
        for req in prop_manager.state_requests:
            entries[req.key] = ObsSpec(req.key, (int(req.dim),), "f32", frame="base", source="prop")
            requestable.add(req.key)
        for key in object_record_keys(prop_manager):
            if key.endswith("/pos"):
                spec = ObsSpec(key, (3,), "f32", unit="m", frame="world", source="object")
            elif "/joints/" in key:
                spec = ObsSpec(
                    key, (1,), "f32", unit="rad", frame="world", source="object"
                )  # joint state
            else:  # /quat: (w,x,y,z)
                spec = ObsSpec(key, (4,), "f32", frame="world", source="object")
            entries[key] = spec
    elif zero_fill_props:
        for key, dim in sorted(_ZERO_FILL_PROP_KEYS.items()):
            entries[key] = ObsSpec(
                key, (dim,), "f32", frame="base", source="prop"
            )  # zero-filled, no props
            requestable.add(key)

    # ---- contact keys (shapes resolved by the bound backend)
    if contact_manager is not None:
        for spec in contact_manager.specs:
            key = f"contact/{spec.name}"
            shape = contact_manager.group_shape(spec.name)
            entries[key] = ObsSpec(
                key,
                shape,
                "bool" if spec.format == "contact" else "f32",
                frame=spec.frame,
                source="contact",
            )
            requestable.add(key)
            # v057 ops detail face: per-contact records (list of dicts,
            # variable length) -- passthrough shape, runtime authoritative
            ops_key = f"contact_ops/{spec.name}"
            entries[ops_key] = ObsSpec(ops_key, None, "json", frame=spec.frame, source="contact")
            requestable.add(ops_key)

    # ---- declared record sites (T3.2 EEF site family): the
    # backend emits ``site/<name>/pos`` per step; record-only keys
    # (never obs_cache), consumed by pack criteria (FormulaTask keys).
    for name in tuple(record_sites or ()):
        key = f"site/{name}/pos"
        entries[key] = ObsSpec(key, (3,), "f32", unit="m", frame="world", source="site")

    return ObsCatalog(entries, requestable, articulation=articulation)
