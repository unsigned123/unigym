"""Reset initializers: the declarative per-episode reset face (scene packs).

A scene pack declares HOW the world initializes each episode through an
``initializer:`` block (``type:`` + ``params:``); the engine dispatches it
through ``INITIALIZER_REGISTRY`` and knows nothing else -- no key names,
no protocol names, no robot names (engine contract).  An initializer owns one
slice of
reset knowledge: geometry resolution, sampling/rejection math and the
prop placement writes; everything it writes goes through
:class:`ResetPrimitives` (never raw engine handles).

Three phases (the implementer contract):

``validate()``    load-time schema fail-fast (teaching errors).
``resolve()``     bind-time: declared geom/region names resolved against
                  the compiled model (one call, read-only on the model).
``apply()``       per-episode reset: sample -> write through the
                  primitives -> return the sampled record for the
                  engine's name-neutrality check (sampled names must be a
                  subset of the declared props).

The three phases are implemented here as a thin adapter: a mounted pack
may supply the sampling mechanics as a ``policy_class`` (a pack-private
object with ``normalize`` / ``bind`` / ``place`` / ``commit``), and the
phases delegate in order -- ``validate`` -> normalize, ``resolve`` ->
bind, ``apply`` -> place then commit.  The core never imports the pack;
the type's registered class carries its own policy.

The single RNG stream lives here (``params.seed``, default 0): ``apply``
and the optional :meth:`InitializerBase.apply_robot_stages` hook consume
it in that fixed order (placement draws first, then stage noise) --
consumption order is a determinism red line (same seed -> bit-identical
resets).

External initializers mount through the ``module`` key (the watchers
callable precedent, resolved in the run process):
``{"module": "pkg.mod:factory", "params": {...}}`` -- the factory is
called as ``factory(params, profile, primitives)`` and returns an
:class:`InitializerBase` instance.
"""

from __future__ import annotations

import math
from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Optional

import numpy as np

from unigym.provenance import OriginTable, ProviderOrigin, ProviderRef, RegistrySnapshot

__all__ = [
    "CollisionPairSpec",
    "InitializerBase",
    "ProbeCapabilityError",
    "ProbePose",
    "ProbeRequest",
    "ProbeResult",
    "ResetOutcome",
    "ResetPrimitives",
    "INITIALIZER_REGISTRY",
    "register",
    "attribute_initializer",
    "initializer_origin",
    "initializer_providers",
    "snapshot_initializers",
    "restore_initializers",
    "build_initializer",
    "normalize_initializer_spec",
    "validate_initializer_shape",
    "wrap_initializer_result",
]


# --------------------------------------------------------------------------- #
# neutral probe request/result types (engine-free)
# --------------------------------------------------------------------------- #
def _probe_vector(value, *, field: str, size: int) -> tuple[float, ...]:
    """Validate one finite ``(size,)`` probe vector and canonicalize it."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a numeric vector") from exc
    if array.shape != (size,):
        raise ValueError(f"{field} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    return tuple(float(component) for component in array)


def _probe_name(value, *, field: str) -> str:
    """Validate and canonicalize one non-empty owner/body/geom name."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string, got {value!r}")
    return value.strip()


class ProbeCapabilityError(RuntimeError):
    """The backend cannot probe the requested free-root pose.

    Raised by ``ResetPrimitives.probe`` for an unsupported capability (an
    articulated internal body pose, a fixed/anchored root, a candidate
    without collision geoms), an unknown declared owner/geom/support
    reference, or an unavailable backend query.  It is the neutral,
    pack-catchable capability failure; malformed shapes/types remain value
    errors.
    """


@dataclass(frozen=True)
class ProbePose:
    """One candidate free-root pose in world frame.

    ``owner`` is the declared prop or articulation name; ``body`` is
    ``None`` for a free root.  V1 probes free roots only: a non-``None``
    body is rejected by the backend with a capability error.
    ``pos_w`` is world XYZ in metres and ``quat_wxyz`` is a world-frame
    ``(w, x, y, z)`` quaternion (normalized by the backend).
    """

    owner: str
    body: str | None
    pos_w: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner", _probe_name(self.owner, field="ProbePose.owner"))
        if self.body is not None:
            object.__setattr__(self, "body", _probe_name(self.body, field="ProbePose.body"))
        object.__setattr__(
            self, "pos_w", _probe_vector(self.pos_w, field="ProbePose.pos_w", size=3)
        )
        quat = _probe_vector(self.quat_wxyz, field="ProbePose.quat_wxyz", size=4)
        if float(np.linalg.norm(quat)) <= 1e-12:
            raise ValueError("ProbePose.quat_wxyz must be a non-zero quaternion")
        object.__setattr__(self, "quat_wxyz", quat)


@dataclass(frozen=True)
class CollisionPairSpec:
    """One explicitly checked candidate geom / other geom pair.

    ``allow_support_contact`` permits a near-zero contact (within
    ``penetration_tolerance_m``) to satisfy the requested clearance; it
    never excuses actual overlap.  This is the region-placement support
    exception expressed neutrally.
    """

    candidate_geom: str
    other_geom: str
    allow_support_contact: bool = False
    penetration_tolerance_m: float = 1e-7

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_geom",
            _probe_name(self.candidate_geom, field="CollisionPairSpec.candidate_geom"),
        )
        object.__setattr__(
            self, "other_geom", _probe_name(self.other_geom, field="CollisionPairSpec.other_geom")
        )
        if not isinstance(self.allow_support_contact, bool):
            raise TypeError(
                "CollisionPairSpec.allow_support_contact must be bool, got "
                f"{type(self.allow_support_contact).__name__}"
            )
        try:
            tolerance = float(self.penetration_tolerance_m)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "CollisionPairSpec.penetration_tolerance_m must be a finite " "non-negative number"
            ) from exc
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                "CollisionPairSpec.penetration_tolerance_m must be a finite " "non-negative number"
            )
        object.__setattr__(self, "penetration_tolerance_m", tolerance)


@dataclass(frozen=True)
class ProbeRequest:
    """One backend-scratch collision/clearance probe request.

    ``overrides`` is a non-empty sequence of candidate free-root poses.
    ``pairs=None`` asks the backend to treat every override owner's full
    body subtree (root plus descendants) as candidates, skip same-owner
    internal pairs, cross-pair different overridden owners, and pair each
    candidate against the collision geoms outside all overridden subtrees.
    A declared ``support_body`` is never removed from that world set: its
    geoms stay in the query, and only pairs against that exact body allow
    a near-zero contact.  Explicit ``pairs`` are evaluated exactly as
    given; the request-level ``support_body`` does not alter them.
    ``min_clearance_m`` is surface clearance in metres; it is not an
    owner-origin distance.
    """

    overrides: tuple[ProbePose, ...]
    pairs: tuple[CollisionPairSpec, ...] | None = None
    min_clearance_m: float = 0.0
    support_body: str | None = None

    def __post_init__(self) -> None:
        try:
            overrides = tuple(self.overrides)
        except TypeError as exc:
            raise TypeError("ProbeRequest.overrides must be a sequence of ProbePose") from exc
        if not overrides:
            raise ValueError("ProbeRequest.overrides must be a non-empty sequence of ProbePose")
        seen: set[tuple[str, str | None]] = set()
        for index, pose in enumerate(overrides):
            if not isinstance(pose, ProbePose):
                raise TypeError(
                    f"ProbeRequest.overrides[{index}] must be a ProbePose, got "
                    f"{type(pose).__name__}"
                )
            key = (pose.owner, pose.body)
            if key in seen:
                raise ValueError(f"ProbeRequest.overrides[{index}] duplicates owner/body {key!r}")
            seen.add(key)
        object.__setattr__(self, "overrides", overrides)

        if self.pairs is not None:
            try:
                pairs = tuple(self.pairs)
            except TypeError as exc:
                raise TypeError(
                    "ProbeRequest.pairs must be None or a sequence of CollisionPairSpec"
                ) from exc
            for index, pair in enumerate(pairs):
                if not isinstance(pair, CollisionPairSpec):
                    raise TypeError(
                        f"ProbeRequest.pairs[{index}] must be a CollisionPairSpec, got "
                        f"{type(pair).__name__}"
                    )
            object.__setattr__(self, "pairs", pairs)

        try:
            clearance = float(self.min_clearance_m)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "ProbeRequest.min_clearance_m must be a finite non-negative number"
            ) from exc
        if not math.isfinite(clearance) or clearance < 0:
            raise ValueError("ProbeRequest.min_clearance_m must be a finite non-negative number")
        object.__setattr__(self, "min_clearance_m", clearance)

        if self.support_body is not None:
            object.__setattr__(
                self,
                "support_body",
                _probe_name(self.support_body, field="ProbeRequest.support_body"),
            )


@dataclass(frozen=True)
class ProbeResult:
    """Probe verdict and measured signed distances, in metres.

    ``reason`` is a stable query outcome (``penetration`` /
    ``insufficient_clearance``) or ``None`` when accepted.  Distances are
    signed and may be capped by the backend's query horizon (MuJoCo caps at
    ``max(1.0, min_clearance_m + penetration_tolerance_m)``).  Capability
    failures raise instead of returning a result.
    """

    accepted: bool
    reason: str | None
    distances_m: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError(f"ProbeResult.accepted must be bool, got {self.accepted!r}")
        reason: str | None = None
        if self.reason is not None:
            if not isinstance(self.reason, str):
                raise TypeError(f"ProbeResult.reason must be str or None, got {self.reason!r}")
            reason = self.reason.strip()
            if not reason:
                raise ValueError("ProbeResult.reason must be non-empty when set")
        if self.accepted and reason is not None:
            raise ValueError("ProbeResult.reason must be None when accepted")
        if not self.accepted and reason is None:
            raise ValueError("ProbeResult.reason must be set when rejected")
        object.__setattr__(self, "reason", reason)
        distances = tuple(float(distance) for distance in self.distances_m)
        if not all(math.isfinite(distance) for distance in distances):
            raise ValueError("ProbeResult.distances_m must be finite")
        object.__setattr__(self, "distances_m", distances)


# --------------------------------------------------------------------------- #
# ResetPrimitives: the only write face an initializer gets
# --------------------------------------------------------------------------- #
@dataclass
class ResetPrimitives:
    """The engine-provided write/physics face (initializer-side contract).

    An initializer NEVER touches engine handles directly: sampling reads
    ``model`` (read-only) and every mutation goes through a primitive.
    ``write_freejoint`` is the prop freejoint write:
    ``(prop_name, pos(3,), yaw=None, *, quat_wxyz=None, lin_vel_w=None,
    ang_vel_w=None)``.  Exactly one of ``yaw`` (pure +Z, the placement
    sampler's call shape) or ``quat_wxyz`` (world-frame wxyz, normalized
    by the primitive) is required; velocities are world-frame and
    optional (absent writes zero).  The placement sampler still passes
    ``yaw`` today; a pack needing arbitrary orientation constructs the
    quaternion and calls the write directly.
    ``set_root_state`` is the robot free-root write:
    ``(name, pos_w(3,), quat_wxyz(4,), lin_vel_w=None, ang_vel_w=None)``
    -- world pose/quaternion (wxyz, normalized downstream) and optional
    world velocities at the root body origin (absent writes zero).
    ``name`` addresses a free-root articulation or a free rigid prop;
    a fixed/anchored root fails with the backend's state-write error.
    ``set_joint_state`` is the robot joint write:
    ``(articulation, joint_names, position, velocity=None)`` -- absolute
    1-DOF positions for an explicit, duplicate-free name sequence;
    ``velocity=None`` leaves the selected joints' velocity unchanged.
    Both write ONLY the current reset env and never settle implicitly
    (the reset chain's settle/forward tail stays authoritative); calling
    them outside ``initializer.apply()`` is an error.
    ``settle_fn(substeps)`` runs the backend's zero-action settle loop;
    ``forward()`` refreshes derived quantities; ``probe(request)`` evaluates
    a :class:`ProbeRequest` on a private scratch data copy (never the live
    reset data); ``prop_extents(name)`` yields the ``(half_xy, height)``
    z-stack inputs and ``prop_names`` is the declared prop-name set (the
    neutrality check's reference).
    The former ``joint_group`` / ``set_targets`` stubs were removed: they
    had no core caller, and an external pack that declared them must
    switch to ``set_root_state`` / ``set_joint_state``.
    """

    model: Any
    """The compiled backend model (read-only for initializers)."""
    settle_fn: Callable[[int], None]
    """Zero-action settle loop (substeps)."""
    write_freejoint: Callable[..., None]
    """``(prop_name, pos(3,), yaw=None, *, quat_wxyz=None, lin_vel_w=None,
    ang_vel_w=None)`` -- world pose (+ optional world velocity) write."""
    set_root_state: Callable[..., None]
    """``(name, pos_w(3,), quat_wxyz(4,), lin_vel_w=None, ang_vel_w=None)``
    -- free-root world pose/quaternion (+ optional world velocity) write
    for the CURRENT reset env (free-root articulation or free rigid prop)."""
    set_joint_state: Callable[..., None]
    """``(articulation, joint_names, position, velocity=None)`` -- absolute
    1-DOF joint write for the CURRENT reset env (``velocity=None`` leaves
    velocity unchanged)."""
    forward: Callable[[], None]
    """``mj_forward`` equivalent (derived-quantity refresh)."""
    probe: Callable[[ProbeRequest], ProbeResult]
    """Evaluate candidate free-root poses on a private scratch data copy.

    Runs ``mj_copyData`` -> apply the request overrides -> ``mj_forward``
    -> ``mj_geomDistance`` and returns a :class:`ProbeResult`.  Live reset
    data, qpos/qvel and the model are never mutated.  V1 supports
    freejoint/free roots only; capability failures raise
    :class:`ProbeCapabilityError`."""
    prop_extents: Callable[[str], tuple] = None  # type: ignore[assignment]
    """``(prop_name) -> (half_xy, height)`` (z-stack sampler inputs)."""
    prop_names: frozenset = field(default_factory=frozenset)
    """Declared prop names (the name-neutrality reference set)."""
    resolve_articulation: Callable[[str], str] = None  # type: ignore[assignment]
    """``(profile_name) -> articulation_name`` for the CURRENT run.

    The env articulation name may differ from the robot profile name
    (``articulations[].params.profile``); robot-stage writers must resolve
    it here instead of assuming ``profile.name``."""
    resolve_region: Callable[[str], Any] = None  # type: ignore[assignment]
    """Resolve a named region or raise PlacementCapabilityError if unsupported."""
    object_collision_bounds: Callable[[str], Any] = None  # type: ignore[assignment]
    """Return conservative collision bounds or raise PlacementCapabilityError."""
    probe_candidate: Callable[..., Any] = None  # type: ignore[assignment]
    """Deprecated region candidate probe (behavior-preserving alias).

    Emits ``DeprecationWarning``; migrate to ``probe(ProbeRequest(...))``.
    Retained until zero-caller proof: the only remaining callers are this
    definition/wiring, the pack API docs and the backend regression test.
    Removal requires a separate owner-approved change."""


@dataclass(frozen=True)
class ResetOutcome:
    """Structured result of one ``initializer.apply()`` reset.

    Attributes:
        valid: whether the sampled reset is legal (``None`` when the
            initializer does not judge legality).
        reason: human-readable reason for an invalid reset.
        metrics: flat finite-float reset diagnostics.
        sample: selected pre-policy-step observation arrays captured by the
            reset chain (``None`` when the initializer does not expose one).
        placements: sampled prop placement records (the legacy return face).
    """

    valid: bool | None = None
    reason: str | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    sample: Mapping[str, np.ndarray] | None = None
    placements: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.valid is not None and not isinstance(self.valid, bool):
            raise TypeError(f"ResetOutcome.valid must be bool or None, got {self.valid!r}")
        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError(f"ResetOutcome.reason must be str or None, got {self.reason!r}")
        if not isinstance(self.placements, Mapping):
            raise TypeError(
                f"ResetOutcome.placements must be a mapping, got "
                f"{type(self.placements).__name__}"
            )
        metrics: dict[str, float] = {}
        for key, value in dict(self.metrics).items():
            if not isinstance(key, str) or not key:
                raise ValueError(
                    f"ResetOutcome metric names must be non-empty strings, got {key!r}"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"ResetOutcome metric {key!r} is not finite ({number!r})")
            metrics[key] = number
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "placements", dict(self.placements))
        if self.sample is not None:
            if not isinstance(self.sample, Mapping):
                raise TypeError(
                    f"ResetOutcome.sample must be a mapping or None, got "
                    f"{type(self.sample).__name__}"
                )
            sample: dict[str, np.ndarray] = {}
            for key, value in dict(self.sample).items():
                if not isinstance(key, str) or not key:
                    raise ValueError("ResetOutcome sample keys must be non-empty strings")
                sample[key] = np.asarray(value)
            object.__setattr__(self, "sample", sample)


def wrap_initializer_result(result: object) -> ResetOutcome:
    """Normalize a legacy placements dict (or a ResetOutcome) to ResetOutcome.

    Backward compatibility: an initializer that still returns its
    ``{prop_name: record}`` mapping is wrapped with ``valid=None`` /
    ``reason=None`` / ``sample=None``; the placements are preserved verbatim.
    """
    if isinstance(result, ResetOutcome):
        return result
    if isinstance(result, Mapping):
        return ResetOutcome(valid=None, reason=None, sample=None, placements=dict(result))
    raise TypeError(
        "initializer.apply() must return a ResetOutcome or a placements "
        f"mapping; got {type(result).__name__}"
    )


# --------------------------------------------------------------------------- #
# InitializerBase
# --------------------------------------------------------------------------- #
class InitializerBase(ABC):
    """One initializer type: params vocabulary + the three phases.

    Class attribute ``name`` is the registry key (yaml ``type:``).
    ``params`` is the initializer's own block (``initializer.params``);
    the base consumes exactly one generic key, ``seed`` (the numpy
    stream seed, default 0 -- matching the legacy reset face's
    protocol-local default).  ``profile`` is the robot profile (pure
    constants; unused by placement-only initializers) and
    ``primitives`` the engine-provided write face.
    """

    name: ClassVar[str]
    policy_class: ClassVar[Any] = None
    """The type's placement policy class (pack-supplied; ``None`` = none).

    The policy owns the sampling mechanics; the phases below delegate to
    its ``normalize`` / ``bind`` / ``place`` / ``commit`` methods.  The
    core only knows the call order, never the pack implementation.
    """

    def __init__(self, params: Optional[dict], profile, primitives):
        self.params: dict = dict(params or {})
        self.profile = profile
        self.primitives = primitives
        # the one RNG stream (apply is the only in-interface consumer;
        # the engine may share it for its residual reset stages -- the
        # consumption order is the determinism red line)
        self.rng = np.random.default_rng(int(self.params.get("seed", 0)))
        self.normalized: Optional[dict] = None
        self._policy: Any = None
        self._bound: Any = None

    def _resolve_policy(self):
        """The type's placement policy instance (created once per initializer)."""
        if self._policy is None:
            policy_cls = self.policy_class
            if policy_cls is None:
                raise RuntimeError(
                    f"{type(self).__name__} declares no placement policy: "
                    "set 'policy_class' or override the three phases"
                )
            self._policy = policy_cls()
        return self._policy

    def snapshot_state(self) -> dict:
        """Capture state needed to continue a restored initializer timeline."""
        return {"rng": dict(self.rng.bit_generator.state)}

    def restore_state(self, state: dict) -> None:
        """Restore state captured by :meth:`snapshot_state`."""
        self.rng.bit_generator.state = state["rng"].copy()

    @classmethod
    def default_params(cls) -> dict:
        """The type's default layout table (data file or inline)."""
        return {}

    def validate(self) -> None:
        """Load-time schema fail-fast (the policy's ``normalize`` step)."""
        self.normalized = self._resolve_policy().normalize(self.params, self.default_params())

    def resolve(self) -> None:
        """Bind-time model resolution (the policy's ``bind`` step)."""
        self._bound = self._resolve_policy().bind(self.primitives)

    def apply(self) -> Dict[str, dict] | ResetOutcome:
        """One episode's reset: sample -> write -> return the sample.

        Returning a ``{prop_name: record}`` mapping keeps the historical
        contract (the core wrapper turns it into a :class:`ResetOutcome`).
        New initializers may return a ``ResetOutcome`` directly to record
        legality/reason/metrics/sample as well.

        The returned placements feed the engine's name-neutrality check
        (every name must be a declared prop).
        """
        if self._bound is None:
            raise RuntimeError(f"{self.name}.resolve() must run before apply()")
        policy = self._resolve_policy()
        placements = policy.place(self._bound, self.rng)
        return policy.commit(self.primitives, placements)

    def apply_outcome(self) -> ResetOutcome:
        """The structured reset result consumed by the core reset chain.

        Wraps a legacy placements dict as ``valid=None`` / ``reason=None`` /
        ``sample=None``; a ``ResetOutcome`` return passes through unchanged.
        Never infers legality from recorded frame 0.
        """
        return wrap_initializer_result(self.apply())

    @property
    def places_props(self) -> bool:
        """True when this initializer writes prop placements.

        The engine's props-wired prerequisite (a placement initializer
        without injected prop bodies is a config error).
        """
        return False

    def apply_robot_stages(self) -> None:
        """Optional robot-stage hook; default no-op.

        Runs once after :meth:`apply` in the backend reset chain (before
        the reset tail forward), so placement draws precede stage noise
        on the shared ``self.rng`` stream.  An initializer type that owns
        robot-body reset stages writes them here through ``self.primitives``;
        the core never imports a robot module for it.
        """


# --------------------------------------------------------------------------- #
# registry + mounting
# --------------------------------------------------------------------------- #
INITIALIZER_REGISTRY: Dict[str, type] = {}
"""Initializer type name -> class (yaml ``initializer.type`` keys)."""

_INITIALIZER_ORIGINS = OriginTable()
"""Private provenance side table beside :data:`INITIALIZER_REGISTRY`."""

InitializerRegistrySnapshot = RegistrySnapshot[type]
"""The origin-aware initializer registry snapshot type."""


def register(cls: type, *, origin: ProviderOrigin | None = None) -> type:
    """Registry decorator (key = the class's ``name`` ClassVar)."""
    from unigym.registration import PackFace, require_registration

    name = getattr(cls, "name", None)
    require_registration(PackFace.INITIALIZER, name=name if isinstance(name, str) else None)
    if not name:
        raise ValueError(
            f"{cls.__name__}: initializer classes must declare a "
            f"'name' ClassVar (the yaml initializer.type key)"
        )
    if name in INITIALIZER_REGISTRY:
        raise ValueError(
            f"initializer type {name!r} already registered "
            f"({INITIALIZER_REGISTRY[name].__module__}."
            f"{INITIALIZER_REGISTRY[name].__name__})"
        )
    INITIALIZER_REGISTRY[name] = cls
    if origin is None:
        _INITIALIZER_ORIGINS.remove(name)
    else:
        if not isinstance(origin, ProviderOrigin):
            raise TypeError(
                f"initializer {name!r}: origin must be a ProviderOrigin, got "
                f"{type(origin).__name__}"
            )
        _INITIALIZER_ORIGINS.set(name, origin)
    return cls


def attribute_initializer(name: str, origin: ProviderOrigin) -> None:
    """Label one already-registered initializer with *origin*."""
    if name in INITIALIZER_REGISTRY:
        _INITIALIZER_ORIGINS.set(name, origin)


def initializer_origin(name: str) -> ProviderOrigin:
    """Return the origin of one registered initializer (never ``None``)."""
    return _INITIALIZER_ORIGINS.origin(name)


def initializer_providers(kind: str | None = None) -> tuple[ProviderRef, ...]:
    """Return the registered initializers as live :class:`ProviderRef` records.

    ``kind`` selects the provider kind; the initializer registry owns only
    ``"initializer"``, so any other kind returns an empty tuple (``None``
    returns every record).  Records are ordered by name.
    """
    if kind is not None and kind != "initializer":
        return ()
    return tuple(
        ProviderRef(
            name=name,
            kind="initializer",
            origin=_INITIALIZER_ORIGINS.origin(name),
        )
        for name in sorted(INITIALIZER_REGISTRY)
    )


def snapshot_initializers() -> InitializerRegistrySnapshot:
    """Capture the initializer registry and origins together."""
    return RegistrySnapshot.capture(INITIALIZER_REGISTRY, _INITIALIZER_ORIGINS)


def restore_initializers(snapshot: InitializerRegistrySnapshot) -> None:
    """Restore a snapshot captured by :func:`snapshot_initializers`."""
    snapshot.restore(INITIALIZER_REGISTRY, _INITIALIZER_ORIGINS)


def _resolve_factory(spec: str):
    """``"pkg.mod:attr"`` -> the attribute (run-process resolution)."""
    mod_name, sep, attr = str(spec).partition(":")
    if not sep or not mod_name or not attr:
        raise ValueError(
            f"initializer module {spec!r} must spell 'pkg.mod:factory' "
            "(the watchers callable precedent; resolved in the run "
            "process)"
        )
    import importlib

    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def validate_initializer_shape(cfg: object) -> None:
    """Validate the closed initializer shape without resolving the registry.

    ``cfg`` must be a mapping declaring exactly one of ``type`` (a registered
    builtin) or ``module`` (``pkg.mod:factory``) plus an optional mapping
    ``params``; the removed ``name`` key fails even when ``type`` / ``module``
    is also present.  The preset loader calls this at document validation time
    so a legacy block fails before any backend is constructed.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"initializer block must be a mapping, got {type(cfg).__name__}")
    if "name" in cfg:
        raise ValueError(
            "initializer block declares the removed key 'name'; declare 'type' "
            "(builtin registry) or 'module' (pkg.mod:factory mount) -- there "
            "is no name alias"
        )
    has_type = "type" in cfg
    has_module = "module" in cfg
    if has_type == has_module:
        raise ValueError(
            "initializer block must declare exactly one of 'type' "
            f"(builtin registry) or 'module' (pkg.mod:factory mount), "
            f"got keys {sorted(cfg)}"
        )
    params = cfg.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"initializer.params must be a mapping, got {type(params).__name__}")


def build_initializer(cfg: dict, profile, primitives) -> InitializerBase:
    """Registry/factory dispatch + construction + ``validate()``.

    ``cfg`` is the normalized ``initializer`` block: either
    ``{"type": <registry name>}`` or the external mount
    ``{"module": "pkg.mod:factory"}``, each with an optional ``params``
    mapping.  Fail-fast on unknown types and on schema errors (the
    load-time validate phase).
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"initializer block must be a mapping, got " f"{type(cfg).__name__}")
    has_type = "type" in cfg
    has_module = "module" in cfg
    if has_type == has_module:
        raise ValueError(
            "initializer block must declare exactly one of 'type' "
            f"(builtin registry) or 'module' (pkg.mod:factory mount), "
            f"got keys {sorted(cfg)}"
        )
    params = cfg.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"initializer.params must be a mapping, got " f"{type(params).__name__}")
    if has_type:
        itype = str(cfg["type"])
        cls = INITIALIZER_REGISTRY.get(itype)
        if cls is None:
            raise ValueError(
                f"initializer.type {itype!r} unknown | registered: "
                f"{sorted(INITIALIZER_REGISTRY)}"
            )
        inst = cls(params, profile, primitives)
    else:
        factory = _resolve_factory(cfg["module"])
        inst = factory(params, profile, primitives)
        if not isinstance(inst, InitializerBase):
            raise ValueError(
                f"initializer factory {cfg['module']!r} returned "
                f"{type(inst).__name__}, not an InitializerBase"
            )
    inst.validate()
    return inst


def normalize_initializer_spec(params: dict) -> Optional[dict]:
    """Segment-read step: the declared initializer block + legacy sugar.

    Returns the normalized ``{"type" | "module", "params": {}}`` block,
    or ``None`` when the config declares no initializer (the legacy
    no-initializer reset, bit-identical).  Two sources, one output:

    * ``params["initializer"]`` -- the scene-preset face (wins);
    * ``params["reset"]`` -- the legacy T1.4 block (protocol + optional
      placement), transposed to an ANONYMOUS INLINE initializer of the
      ``splita_tabletop`` builtin: placement fields hoist to the top
      level, the robot-stage keys and the protocol ride along.  Core
      validates only the mapping shape; the selected initializer (and
      its pack) owns protocol validation.  Existing non-scene configs
      keep working unchanged (zero-migration guarantee).

    Declaring both is an ambiguity error.
    """
    init = params.get("initializer")
    legacy = params.get("reset")
    if init is not None and legacy is not None:
        raise ValueError(
            "env.params declares both 'initializer' and the legacy "
            "'reset' block -- declare one (the initializer face wins "
            "nowhere; the ambiguity is a config error)"
        )
    if init is not None:
        if not isinstance(init, dict) or not ("type" in init or "module" in init):
            raise ValueError(
                f"env.params.initializer must be a mapping with a "
                f"'type' (or 'module') key, got {init!r}"
            )
        return init
    if legacy is None:
        return None
    if not isinstance(legacy, dict):
        raise ValueError(f"env.params.reset must be a mapping, got " f"{type(legacy).__name__}")
    protocol = legacy.get("protocol")
    placement = legacy.get("placement") or {}
    rest = {k: v for k, v in legacy.items() if k not in ("protocol", "placement")}
    params = {**placement, **rest}
    if protocol is not None:
        params["protocol"] = protocol
    return {"type": "splita_tabletop", "params": params}
