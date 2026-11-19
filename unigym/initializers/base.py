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

__all__ = [
    "InitializerBase",
    "ResetOutcome",
    "ResetPrimitives",
    "INITIALIZER_REGISTRY",
    "register",
    "build_initializer",
    "normalize_initializer_spec",
    "validate_initializer_shape",
    "wrap_initializer_result",
]


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
    ``forward()`` refreshes derived quantities; ``prop_extents(name)``
    yields the ``(half_xy, height)`` z-stack inputs and ``prop_names``
    is the declared prop-name set (the neutrality check's reference).
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
    """Probe a candidate without mutating live simulation state."""


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


def register(cls: type) -> type:
    """Registry decorator (key = the class's ``name`` ClassVar)."""
    name = getattr(cls, "name", None)
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
    return cls


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
