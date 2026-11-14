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
is the only consumer inside the interface, and the engine may share the
stream (``.rng``) with its own residual reset stages while those still
exist -- consumption order is a determinism red line (same seed ->
bit-identical resets).

External initializers mount through the ``module`` key (the watchers
callable precedent, resolved in the run process):
``{"module": "pkg.mod:factory", "params": {...}}`` -- the factory is
called as ``factory(params, profile, primitives)`` and returns an
:class:`InitializerBase` instance.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Optional

import numpy as np

__all__ = [
    "InitializerBase",
    "ResetPrimitives",
    "INITIALIZER_REGISTRY",
    "register",
    "build_initializer",
    "normalize_initializer_spec",
    "validate_initializer_shape",
]


# --------------------------------------------------------------------------- #
# ResetPrimitives: the only write face an initializer gets
# --------------------------------------------------------------------------- #
@dataclass
class ResetPrimitives:
    """The engine-provided write/physics face (initializer-side contract).

    An initializer NEVER touches engine handles directly: sampling reads
    ``model`` (read-only) and every mutation goes through a primitive.
    ``write_freejoint(name, pos, yaw)`` is the prop freejoint write
    (posed + zeroed velocity, the backend's own address table);
    ``settle_fn(substeps)`` runs the backend's zero-action settle loop;
    ``forward()`` refreshes derived quantities; ``prop_extents(name)``
    yields the ``(half_xy, height)`` z-stack inputs and ``prop_names``
    is the declared prop-name set (the neutrality check's reference).
    ``joint_group``/``set_targets`` are the robot-side write seams for
    future initializer types -- interface-complete stubs until the
    robot-body reset migrates (planned).
    """

    model: Any
    """The compiled backend model (read-only for initializers)."""
    settle_fn: Callable[[int], None]
    """Zero-action settle loop (substeps)."""
    write_freejoint: Callable[[str, Any, float], None]
    """``(prop_name, pos(3,), yaw_rad)`` -- pose + zero-velocity write."""
    joint_group: Callable[[str], Any]
    """Named joint group -> indices (robot-side seam; stub)."""
    set_targets: Callable[[Any, Any], None]
    """Joint-group target write (robot-side seam; stub)."""
    forward: Callable[[], None]
    """``mj_forward`` equivalent (derived-quantity refresh)."""
    prop_extents: Callable[[str], tuple] = None  # type: ignore[assignment]
    """``(prop_name) -> (half_xy, height)`` (z-stack sampler inputs)."""
    prop_names: frozenset = field(default_factory=frozenset)
    """Declared prop names (the name-neutrality reference set)."""
    resolve_region: Callable[[str], Any] = None  # type: ignore[assignment]
    """Resolve a named region or raise PlacementCapabilityError if unsupported."""
    object_collision_bounds: Callable[[str], Any] = None  # type: ignore[assignment]
    """Return conservative collision bounds or raise PlacementCapabilityError."""
    probe_candidate: Callable[..., Any] = None  # type: ignore[assignment]
    """Probe a candidate without mutating live simulation state."""


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

    def apply(self) -> Dict[str, dict]:
        """One episode's reset: sample -> write -> return the sample.

        The returned ``{prop_name: record}`` mapping feeds the engine's
        name-neutrality check (every name must be a declared prop).
        """
        if self._bound is None:
            raise RuntimeError(f"{self.name}.resolve() must run before apply()")
        policy = self._resolve_policy()
        placements = policy.place(self._bound, self.rng)
        return policy.commit(self.primitives, placements)

    @property
    def places_props(self) -> bool:
        """True when this initializer writes prop placements.

        The engine's props-wired prerequisite (a placement initializer
        without injected prop bodies is a config error).
        """
        return False

    def stage_spec(self) -> Optional[dict]:
        """Optional residual robot-stage spec (transition seam).

        The robot-body reset stages still run engine-side (their move to
        the robots layer is the v068 route); an initializer type that
        owns that knowledge returns its stage spec dict here (consumed
        by the backend's legacy stage face).  Default: none.
        """
        return None


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
      level, the robot-stage keys ride along.  Existing non-scene
      configs keep working unchanged (zero-migration guarantee).

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
    from unigym.robots.gr1_reset import GR1_RESET_PROTOCOL

    protocol = legacy.get("protocol")
    if protocol != GR1_RESET_PROTOCOL:
        raise ValueError(
            f"env.params.reset.protocol {protocol!r} unknown | the "
            f"legacy block transposes to the splita_tabletop "
            f"initializer whose only protocol is "
            f"{GR1_RESET_PROTOCOL!r}"
        )
    placement = legacy.get("placement") or {}
    rest = {k: v for k, v in legacy.items() if k not in ("protocol", "placement")}
    return {"type": "splita_tabletop", "params": {**placement, **rest, "protocol": protocol}}
