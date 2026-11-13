"""Trait registry: the capability contract between run configs and backends.

Single source of truth for "what can a backend do".  The enum table is the
mechanized touch matrix (docs/designs/backend-traits.md §1.1 → §3.1): every
optional feature a yaml/adapter/director-op can request maps to a trait, and
a backend missing that trait fails fast at build time (``{error, hint}``) —
never a silent degradation.

Backends are *equal citizens*: ``mujoco`` / ``isaaclab2``
/ ``isaaclab3``.  ``isaaclab2`` (IsaacLab 2.3 series) and ``isaaclab3``
(3.0+ series) are two different backends, selected by import probing at
construction time; a config ``backend:`` key overrides the probe.

This module is **stdlib-only at import time**: engine presence is probed
lazily (importlib.metadata first — no side effects — then a guarded
``import``), so unit tests can import it on any host.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from collections.abc import Callable, Mapping
from typing import Any
from dataclasses import dataclass, field, replace
from enum import Enum

from unigym.assets.asset_ref import ASSET_FORMATS

__all__ = [
    "CANONICAL_BACKENDS",
    "BACKEND_ALIASES",
    "MJWARP_BACKEND",
    "BackendInfo",
    "BackendSelectionError",
    "build_name",
    "TRAIT_RULES",
    "Trait",
    "TraitRule",
    "TraitScope",
    "backend_info",
    "isaaclab_version",
    "known_backends",
    "mujoco_version",
    "mujoco_warp_version",
    "select_backend",
    "validate_backend_options",
    "validate_trait_rules",
]


class Trait(Enum):
    """One addressable backend capability.

    Env-core traits (the first three) are mandatory: a backend that cannot
    deliver them is not registrable.  All other traits are optional; a
    request for a missing one fails fast with the trait name, the request
    source and the backend's trait set.
    """

    # --- env core (registration prerequisite) ---
    ENV_STEP_RESET = "ENV_STEP_RESET"  # step/reset_envs/reset_all + extra 17-key contract
    ACTION_FACES_NAMED = "ACTION_FACES_NAMED"  # consumes dict[str, NamedAction]
    JOINT_PERMUTATION = "JOINT_PERMUTATION"  # policy/il orders + inverse perm
    # --- env shape (optional: single-env backends omit it) ---
    MULTI_ENV = "MULTI_ENV"  # native N>1 parallel-env runs
    # --- env layout (exactly one ENV_LAYOUT_* per registered backend) ---
    ENV_LAYOUT_TILED = "ENV_LAYOUT_TILED"  # env_spacing; one view spans envs
    ENV_LAYOUT_COINCIDENT = "ENV_LAYOUT_COINCIDENT"  # same-coordinate worlds
    # --- observation / recording ---
    CAMERAS_RGB = "CAMERAS_RGB"  # rig-level RGB frames
    CAMERAS_GLOBAL = "CAMERAS_GLOBAL"  # scene-wide single camera/image
    CAMERAS_PER_ENV = "CAMERAS_PER_ENV"  # one camera instance per env (N>=1)
    CAMERAS_PER_ENV_STATE = "CAMERAS_PER_ENV_STATE"  # (camera, env) own pose state
    CAMERAS_BATCH_RENDER = "CAMERAS_BATCH_RENDER"  # one render covers all instances
    CAMERAS_PER_ENV_RESOLUTION = "CAMERAS_PER_ENV_RESOLUTION"  # reserved
    CAMERAS_PER_ENV_INTRINSICS = "CAMERAS_PER_ENV_INTRINSICS"  # reserved
    CAMERAS_DEPTH_RIG = "CAMERAS_DEPTH_RIG"  # rig-level metric z-depth
    DEPTH_OBS_ALIASES = "DEPTH_OBS_ALIASES"  # depth obs alias keys (isaaclab)
    RECORDER_VIDEO = "RECORDER_VIDEO"  # recorder.enabled mp4
    # --- contacts ---
    CONTACTS_GROUPS = "CONTACTS_GROUPS"  # declarative groups + read_raw
    CONTACTS_OVERRIDE = "CONTACTS_OVERRIDE"  # director contact op
    # --- props ---
    PROPS_RIGID = "PROPS_RIGID"  # sphere/box-family/usd tiers
    PROPS_ARTICULATION_URDF = "PROPS_ARTICULATION_URDF"
    PROPS_ARTICULATION_USD = "PROPS_ARTICULATION_USD"
    PROPS_LATCH = "PROPS_LATCH"  # runtime drive swap
    PROPS_ACTUATED_ACTIONS = "PROPS_ACTUATED_ACTIONS"  # actuated joints -> NamedAction
    # --- director ---
    DIRECTOR_TELEPORT = "DIRECTOR_TELEPORT"
    DIRECTOR_SETVEL = "DIRECTOR_SETVEL"
    DIRECTOR_ATTACH_WELD = "DIRECTOR_ATTACH_WELD"  # frozen relative pose
    DIRECTOR_ATTACH_SLAVE = "DIRECTOR_ATTACH_SLAVE"  # per-tick kinematic follow
    DIRECTOR_GRIPPER = "DIRECTOR_GRIPPER"
    # --- scene / randomization ---
    SCENE_COMPOSE = "SCENE_COMPOSE"  # neutral Scene -> composed world face
    TERRAIN = "TERRAIN"
    EVENTS_DR = "EVENTS_DR"
    INIT_POSE = "INIT_POSE"
    SPAWN_XY_YAW = "SPAWN_XY_YAW"  # both backends (parity)
    MOCAP = "MOCAP"  # mocap body carrier (mujoco)
    # --- rewind (Round B B1) ---
    SNAPSHOT = "SNAPSHOT"  # mjSTATE-exact snapshot()/restore()
    # --- model introspection (no second capability enum) ---
    MODEL_ADDRESSES = "MODEL_ADDRESSES"  # qpos/dof/geom/actuator ids
    MODEL_MASS_INERTIA = "MODEL_MASS_INERTIA"  # compiled mass/inertia
    MODEL_MOCAP = "MODEL_MOCAP"  # mocap body carriers
    MODEL_EQ_SLOTS = "MODEL_EQ_SLOTS"  # equality slots
    MODEL_USD_PRIMS = "MODEL_USD_PRIMS"  # USD prim hierarchy


# --------------------------------------------------------------------------- #
# TraitScope: per-backend availability predicate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TraitScope:
    """One (backend, trait) availability predicate; PRESENCE stays the enum.

    A scope only *narrows* a declared trait (``has()`` stays presence-only):
    the backend has the trait, but a run with more than ``max_envs``
    environments cannot use it.  ``max_envs=None`` means "any num_envs".
    """

    max_envs: int | None = None
    note: str = ""


# --------------------------------------------------------------------------- #
# BackendInfo
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BackendInfo:
    """Identity + capability set of one registered backend."""

    name: str  # canonical: "mujoco" | "isaaclab2" | "isaaclab3"
    version: tuple[int, int]  # engine version (probe; (0, 0) when absent)
    traits: frozenset[Trait]  # capability set (registration declares it)
    options_schema: Mapping[str, object] = field(default_factory=dict)
    """Backend-private ``env.backend_options.<backend>`` schema.

    Nested mappings declare option groups; a non-mapping leaf is the default
    and its type is validated (bool leaves require a bool).  An empty schema
    means the backend declares no options and any supplied block fails fast.
    """
    scene_engine: Callable[[], object] | None = None
    """Optional zero-argument factory for the backend's scene engine.

    Set only by backends that consume the neutral scene IR; the factory is
    called lazily so this module stays stdlib-only at import time.
    """
    scopes: Mapping[Trait, TraitScope] = field(default_factory=dict)
    """Per-trait availability narrowing; keys must be declared traits."""
    env_class: Callable[[], type] | None = None
    """Optional zero-argument factory for the backend env class.

    The factory performs the engine import lazily, so importing this module
    (or ``unigym.env.backends``) never loads an engine.
    """
    live_provider: Callable[[], type] | None = None
    """Optional zero-argument factory for the backend live-model binder.

    Returns the binder class used by :class:`unigym.model.live.LiveModel`;
    the engine import stays inside the returned class, so the neutral model
    layer imports no engine.  ``mujoco`` and ``mjwarp`` share the same
    ``MjModel``/``MjData`` binder.
    """
    # Optional construction faces (default None; each backend fills them in).
    model_provider: Callable[[], Callable[[], Any]] | None = None
    model_handle: Callable[[], Callable[[object], object | None]] | None = None
    """Optional factory for the backend's compiled-model handle resolver.

    Returns a callable ``handle(env) -> model-or-None``.  The neutral
    ``describe --compiled`` face asks the registry for this instead of
    sniffing env attributes; ``None`` means the backend
    exposes no compiled model handle.
    """
    prop_backend: Callable[[], object] | None = None
    camera_backend: Callable[[], object] | None = None
    """Optional zero-argument factory for the backend camera rig builder.

    Returns the callable ``build(spec, render_groups=None)`` used by
    :func:`unigym.cameras.rigs.build_rig`; the backend module import stays
    inside the factory.
    """
    director_ops: Callable[[], object] | None = None
    contact_backend: Callable[[], object] | None = None
    gripper_backend: Callable[[], object] | None = None
    """Optional zero-argument factory for the backend director gripper.

    Returns the runtime gripper class (``MujocoGripper`` /
    ``IsaacLabGripper``); the backend module import stays inside the
    factory so the neutral director layer imports no engine.
    """
    terrain_presets: Callable[[], object] | None = None
    # --- neutral construction faces (lazy factories) ---
    # The neutral run/camera/scene layers ask these faces instead of
    # comparing ``backend`` strings; every engine import stays inside the
    # backend package.
    env_kwargs: Callable[[], object] | None = None
    """Factory -> ``env_kwargs(env_cfg, artifacts_dir, world_base, spec)``.

    Returns the backend-specific builder kwargs for the env constructor.
    """
    build_terrain: Callable[[], object] | None = None
    """Factory -> ``build_terrain(cfg, num_envs, env_spacing, spec_terrain)``.

    Returns the terrain object (or None for the native-IR pass-through);
    ``None`` on the info means the backend declares no terrain face.
    """
    build_events: Callable[[], object] | None = None
    """Factory -> ``build_events(section)`` (DR events; None = unsupported)."""
    content_router: Callable[[], Callable[..., object]] | None = None
    """Factory -> ``route_content(spec, env_cfg)`` for the ``objects:`` fill."""
    scene_ops: Callable[[], object] | None = None
    """Factory -> the backend's ``SceneOps`` class (typed scene query)."""
    kit_launcher: Callable[[], Callable[..., tuple[Any, Any]]] | None = None
    """Factory -> ``kit_launcher(parser, warn=None)`` (kit launch before build)."""
    robot_realizer: Callable[[], object] | None = None
    """Factory -> ``realize(profile, *, spec, params, asset_plan)``.

    The realizer turns a neutral :class:`~unigym.robots.base.RobotProfile`
    plus its :class:`~unigym.scenes.articulation.ArticulationSpec` into the
    backend's robot realization (e.g. an Isaac Lab ``ArticulationCfg``).
    ``None`` means the backend consumes the neutral profile directly and
    builds no backend robot object.
    """
    asset_formats: tuple[str, ...] = ()
    """Supported robot asset formats in backend preference order.

    The run layer selects the first declared format present on the effective
    ``AssetRef``; this tuple is also the native-name key set used by
    ``BodyBinding`` and contact resolution.  An empty tuple is invalid.
    """
    default_lighting: Mapping[str, float] | None = None
    """Default lights for a textured usd prop when the config declares none."""
    camera_enable_required: bool = False
    """Cameras require the kit ``--enable_cameras`` pipeline."""
    bind_all_declared_cameras: bool = False
    """Bind every declared camera rig (vs request-driven instantiation)."""
    world_base_required: bool = False
    """A run-level ``world_base: {mjcf: ...}`` declaration is mandatory."""
    mj_scene_consumer: bool = False
    """Consumes the ``env.params.mj_scene`` / scene-IR fill."""
    inline_static_scene: bool = False
    """Consumes the inline world scene as ``env.params.mj_scene`` (mujoco)."""
    probe_in_host: bool = False
    """``unigym probe`` runs in the host venv (vs the container lane)."""
    probe_env: tuple[tuple[str, str], ...] = ()
    """Environment defaults injected into a host-lane probe subprocess."""

    def __post_init__(self) -> None:
        """Validate and freeze the backend asset-format preference list."""
        formats = tuple(str(value) for value in self.asset_formats)
        unknown = sorted(set(formats) - set(ASSET_FORMATS))
        if not formats:
            raise ValueError(
                f"backend {self.name!r} must declare at least one asset format; "
                f"allowed formats are {list(ASSET_FORMATS)}"
            )
        if unknown:
            raise ValueError(
                f"backend {self.name!r} declares unknown asset format(s) {unknown}; "
                f"allowed formats are {list(ASSET_FORMATS)}"
            )
        if len(set(formats)) != len(formats):
            raise ValueError(f"backend {self.name!r} repeats an asset format: {list(formats)}")
        object.__setattr__(self, "asset_formats", formats)

    def has(self, *traits: Trait) -> bool:
        """Every listed trait present?"""
        return set(traits) <= set(self.traits)

    def has_any(self, *traits: Trait) -> bool:
        """At least one listed trait present?"""
        return bool(set(traits) & set(self.traits))

    def missing(self, *traits: Trait) -> tuple[Trait, ...]:
        """The listed traits this backend lacks (stable declaration order)."""
        want = [t for t in Trait if t in set(traits)]
        return tuple(t for t in want if t not in self.traits)

    def available(self, trait: Trait, num_envs: int | None = None) -> bool:
        """Is *trait* present AND usable at *num_envs* (``None`` = any)?"""
        if trait not in self.traits:
            return False
        if num_envs is None:
            return True
        scope = self.scopes.get(trait)
        return scope is None or scope.max_envs is None or num_envs <= scope.max_envs

    def blocked(self, traits, num_envs: int | None = None):
        """Present-but-scope-blocked ``(trait, scope)`` pairs (enum order).

        Missing traits are not reported here (use :meth:`missing`); a
        ``num_envs`` of ``None`` means "no scope check" and always yields
        an empty tuple.
        """
        if num_envs is None:
            return ()
        want = [t for t in Trait if t in set(traits)]
        blocked = []
        for trait in want:
            if trait not in self.traits:
                continue
            scope = self.scopes.get(trait)
            if scope is None or scope.max_envs is None:
                continue
            if num_envs > scope.max_envs:
                blocked.append((trait, scope))
        return tuple(blocked)


class BackendSelectionError(RuntimeError):
    """A backend was requested that the probe cannot satisfy.

    Carries the ``{error, hint}`` pair the CLI/daemon surfaces verbatim
    (fail-fast: no silent fallback to another backend).
    """

    def __init__(self, error: str, hint: str = ""):
        super().__init__(error)
        self.error = error
        self.hint = hint or "(none)"


# --------------------------------------------------------------------------- #
# backend-private options
# --------------------------------------------------------------------------- #
def _dotted_options(schema: Mapping[str, object], prefix: str = "") -> list[str]:
    """Every supported option path of one nested schema (stable order)."""
    out = []
    for key, value in schema.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.extend(_dotted_options(value, f"{path}."))
        else:
            out.append(path)
    return out


def _validate_options_block(
    schema: Mapping[str, object], options: Mapping[str, object], path: str
) -> dict[str, object]:
    """Validate one nested option block against *schema* (fail-fast)."""
    unknown = sorted(set(options) - set(schema))
    if unknown:
        supported = _dotted_options(schema, path)
        raise ValueError(
            f"backend_options: unknown option keys {unknown} at "
            f"{path.rstrip('.')!r}; supported options: {supported}"
        )
    out = {}
    for key, spec in schema.items():
        full = f"{path}{key}"
        value = options.get(key, spec if not isinstance(spec, Mapping) else {})
        if isinstance(spec, Mapping):
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"backend_options: {full} must be a mapping; supported "
                    f"options: {_dotted_options(schema, path)}"
                )
            out[key] = _validate_options_block(spec, value, f"{full}.")
            continue
        if isinstance(spec, bool) and not isinstance(value, bool):
            raise ValueError(f"backend_options: {full} must be a bool, got {value!r}")
        out[key] = value
    return out


_OPTION_NAMESPACES = {
    "mujoco": ("mujoco",),
    "mjwarp": ("mjwarp",),
    "isaaclab2": ("isaaclab2", "isaaclab"),
    "isaaclab3": ("isaaclab3", "isaaclab"),
}


def validate_backend_options(
    backend: str, options: Mapping[str, object] | None
) -> dict[str, object]:
    """Validate ``env.backend_options`` and return only *backend*'s options.

    Args:
        backend: The requested backend spelling (canonical or alias).
        options: The raw ``env.backend_options`` mapping.

    Returns:
        The normalized option tree for this backend only (defaults applied).

    Raises:
        ValueError: A malformed block, an unknown backend namespace, an
            unknown option key or an option the backend does not declare.
    """
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise ValueError(f"env.backend_options must be a mapping, got " f"{type(options).__name__}")
    canonical = _resolve_name(backend)
    accepted = _OPTION_NAMESPACES.get(canonical, (canonical,))
    foreign = sorted(set(options) - set(accepted))
    if foreign:
        raise ValueError(
            f"env.backend_options declares {foreign}, but the selected "
            f"backend {canonical!r} only accepts its own namespace "
            f"{list(accepted)}"
        )
    supplied = {}
    for key in accepted:
        if key in options:
            supplied = options[key]
            break
    info = _STATIC_INFO[canonical]
    if not info.options_schema:
        if supplied:
            keys = sorted(supplied) if isinstance(supplied, Mapping) else []
            raise ValueError(
                f"backend {canonical!r} declares no backend_options (unknown "
                f"keys {keys}); supported options: []"
            )
        return {}
    if not isinstance(supplied, Mapping):
        raise ValueError(
            f"env.backend_options.{canonical} must be a mapping, got " f"{type(supplied).__name__}"
        )
    return _validate_options_block(info.options_schema, supplied, "")


# --------------------------------------------------------------------------- #
# trait rules (the single source of truth for cross-trait relations)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TraitRule:
    """Declarative relations of one trait to the rest of the set."""

    requires: tuple[Trait, ...] = ()
    exclusive_with: tuple[Trait, ...] = ()


TRAIT_RULES: dict[Trait, TraitRule] = {
    Trait.ENV_LAYOUT_TILED: TraitRule(exclusive_with=(Trait.ENV_LAYOUT_COINCIDENT,)),
    Trait.ENV_LAYOUT_COINCIDENT: TraitRule(exclusive_with=(Trait.ENV_LAYOUT_TILED,)),
    Trait.CAMERAS_GLOBAL: TraitRule(requires=(Trait.ENV_LAYOUT_TILED,)),
    Trait.CAMERAS_PER_ENV_STATE: TraitRule(requires=(Trait.CAMERAS_PER_ENV,)),
    Trait.CAMERAS_BATCH_RENDER: TraitRule(requires=(Trait.CAMERAS_PER_ENV, Trait.MULTI_ENV)),
    Trait.CAMERAS_PER_ENV_RESOLUTION: TraitRule(requires=(Trait.CAMERAS_PER_ENV,)),
    Trait.CAMERAS_PER_ENV_INTRINSICS: TraitRule(requires=(Trait.CAMERAS_PER_ENV_STATE,)),
}

_LAYOUT_TRAITS = (Trait.ENV_LAYOUT_TILED, Trait.ENV_LAYOUT_COINCIDENT)


def validate_trait_rules(backends=None) -> None:
    """Validate every registered backend's trait set against ``TRAIT_RULES``.

    Called at import (so a bad declaration table fails before any run) and
    re-run independently by ``tests/meta/traits/test_trait_rules.py``.
    Empty trait sets (the isaaclab3 skeleton) are skipped: they declare no
    layout because they implement nothing yet.
    """
    infos = _STATIC_INFO if backends is None else backends
    for name, info in infos.items():
        traits = set(info.traits)
        # A scope may only narrow a DECLARED trait, never add one.
        for trait, scope in info.scopes.items():
            if trait not in traits:
                raise RuntimeError(
                    f"backend {name!r}: scope declared for undeclared trait "
                    f"{trait.name} (scopes only narrow presence)"
                )
            if scope.max_envs is not None and scope.max_envs < 1:
                raise RuntimeError(
                    f"backend {name!r}: trait {trait.name} scope max_envs "
                    f"must be >= 1, got {scope.max_envs}"
                )
        if not traits:
            continue
        layouts = [t for t in _LAYOUT_TRAITS if t in traits]
        if len(layouts) != 1:
            raise RuntimeError(
                f"backend {name!r}: exactly one ENV_LAYOUT_* trait is "
                f"required, got {[t.name for t in layouts]}"
            )
        for trait in sorted(traits, key=lambda t: t.value):
            rule = TRAIT_RULES.get(trait)
            if rule is None:
                continue
            for other in rule.exclusive_with:
                if other in traits:
                    raise RuntimeError(
                        f"backend {name!r}: trait {trait.name} is "
                        f"exclusive with {other.name} but both are declared"
                    )
            for required in rule.requires:
                if required not in traits:
                    raise RuntimeError(
                        f"backend {name!r}: trait {trait.name} requires "
                        f"{required.name}, which is not declared"
                    )


# --------------------------------------------------------------------------- #
# per-backend trait tables (declaration = registration)
# --------------------------------------------------------------------------- #
_CORE = frozenset(
    {
        Trait.ENV_STEP_RESET,
        Trait.ACTION_FACES_NAMED,
        Trait.JOINT_PERMUTATION,
    }
)

# MuJoCo-family compose options (backend-private).
_MJ_COMPOSE_OPTIONS = {"compose": {"auto_prefix": False}}

MUJOCO_TRAITS = _CORE | frozenset(
    {
        Trait.MULTI_ENV,  # native num_envs=N independent MjData
        Trait.ENV_LAYOUT_COINCIDENT,  # one MjModel / one world per run
        Trait.SCENE_COMPOSE,
        Trait.CAMERAS_RGB,
        Trait.CAMERAS_DEPTH_RIG,
        Trait.RECORDER_VIDEO,
        Trait.CAMERAS_PER_ENV,  # N=1 degenerate instance
        Trait.CAMERAS_PER_ENV_STATE,
        Trait.CONTACTS_GROUPS,
        Trait.CONTACTS_OVERRIDE,
        Trait.PROPS_RIGID,
        Trait.PROPS_ARTICULATION_URDF,
        Trait.PROPS_ACTUATED_ACTIONS,
        Trait.DIRECTOR_TELEPORT,
        Trait.DIRECTOR_SETVEL,
        Trait.DIRECTOR_ATTACH_WELD,
        Trait.DIRECTOR_GRIPPER,
        Trait.SPAWN_XY_YAW,
        Trait.MOCAP,
        Trait.SNAPSHOT,
        # Model introspection: the compiled MjModel carries them all
        Trait.MODEL_ADDRESSES,
        Trait.MODEL_MASS_INERTIA,
        Trait.MODEL_MOCAP,
        Trait.MODEL_EQ_SLOTS,
    }
)

# The independent MuJoCo Warp backend.  N>1 is ONE batched warp
# Data (nworld=N) with N CPU mirrors, plus scene compose through the
# shared MujocoSceneEngine, box-family props, contact groups, RGB rigs,
# planar spawn, and the mirror-faithful snapshot/restore face.
MJWARP_TRAITS = _CORE | frozenset(
    {
        Trait.MULTI_ENV,
        Trait.ENV_LAYOUT_COINCIDENT,  # N same-coordinate worlds
        Trait.SCENE_COMPOSE,
        Trait.PROPS_RIGID,
        Trait.PROPS_ARTICULATION_URDF,  # free-joint articulation N>1
        Trait.PROPS_ACTUATED_ACTIONS,  # prop-name NamedAction -> servo
        Trait.CONTACTS_GROUPS,
        Trait.CONTACTS_OVERRIDE,  # manager-level record override
        Trait.CAMERAS_RGB,
        Trait.CAMERAS_DEPTH_RIG,  # metric z from rc.depth_data
        Trait.CAMERAS_PER_ENV,  # CPU mirror per world
        Trait.CAMERAS_PER_ENV_STATE,
        Trait.CAMERAS_BATCH_RENDER,  # native warp renderer
        Trait.RECORDER_VIDEO,  # manager cache -> mp4
        Trait.DIRECTOR_TELEPORT,
        Trait.DIRECTOR_SETVEL,  # N=1 mirror + push
        Trait.DIRECTOR_ATTACH_WELD,
        Trait.DIRECTOR_GRIPPER,
        Trait.MOCAP,
        Trait.SPAWN_XY_YAW,
        Trait.SNAPSHOT,
        # Same compiled MjModel as mujoco
        Trait.MODEL_ADDRESSES,
        Trait.MODEL_MASS_INERTIA,
        Trait.MODEL_MOCAP,
        Trait.MODEL_EQ_SLOTS,
    }
)

ISAACLAB2_TRAITS = _CORE | frozenset(
    {
        Trait.MULTI_ENV,  # native num_envs=N scene cloning
        Trait.ENV_LAYOUT_TILED,  # env_spacing clone grid
        Trait.CAMERAS_RGB,
        Trait.CAMERAS_DEPTH_RIG,
        Trait.DEPTH_OBS_ALIASES,
        # CAMERAS_GLOBAL is the 1-prim mechanism (num_instances = parent prim
        # matches = 1); upstream docs do not name it -- container-verify.
        Trait.CAMERAS_GLOBAL,
        Trait.CAMERAS_PER_ENV,
        Trait.CAMERAS_PER_ENV_STATE,
        Trait.CAMERAS_BATCH_RENDER,
        Trait.RECORDER_VIDEO,
        Trait.CONTACTS_GROUPS,
        Trait.CONTACTS_OVERRIDE,
        Trait.PROPS_RIGID,
        Trait.PROPS_ARTICULATION_URDF,
        Trait.PROPS_ARTICULATION_USD,
        Trait.PROPS_LATCH,
        Trait.PROPS_ACTUATED_ACTIONS,
        Trait.DIRECTOR_TELEPORT,
        Trait.DIRECTOR_SETVEL,
        Trait.DIRECTOR_ATTACH_SLAVE,
        Trait.DIRECTOR_GRIPPER,
        Trait.TERRAIN,
        Trait.EVENTS_DR,
        Trait.INIT_POSE,
        Trait.SPAWN_XY_YAW,
        # The USD prim hierarchy is the isaaclab model face
        Trait.MODEL_USD_PRIMS,
    }
)

# EA skeleton: registered so selection/probing can name it, implements
# nothing yet (backends/isaaclab3/README.md declares the gap; build -> stub
# NotImplementedError).  Traits fill in as capabilities land.
ISAACLAB3_TRAITS = frozenset()

# mujoco's model-global writes are one-MjModel only (no per-env
# face); they stay declared traits but are unavailable at N>1.  mjwarp
# declares the same N=1 scope (director writes the CPU mirror + pushes the
# device; one MjModel / one mocap namespace).  isaaclab supports these at
# N>1 (no scope).
_MODEL_GLOBAL = "model-global writes (one MjModel): no per-env face"
_MJWARP_MODEL_GLOBAL = "model-global writes (one MjModel); mirror-only " "at N>1"
MUJOCO_SCOPES: dict[Trait, TraitScope] = {
    Trait.DIRECTOR_TELEPORT: TraitScope(max_envs=1, note=_MODEL_GLOBAL),
    Trait.DIRECTOR_SETVEL: TraitScope(max_envs=1, note=_MODEL_GLOBAL),
    Trait.DIRECTOR_ATTACH_WELD: TraitScope(max_envs=1, note=_MODEL_GLOBAL),
    Trait.DIRECTOR_GRIPPER: TraitScope(max_envs=1, note=_MODEL_GLOBAL),
    Trait.MOCAP: TraitScope(max_envs=1, note=_MODEL_GLOBAL),
}
MJWARP_SCOPES: dict[Trait, TraitScope] = {
    Trait.DIRECTOR_TELEPORT: TraitScope(max_envs=1, note=_MJWARP_MODEL_GLOBAL),
    Trait.DIRECTOR_SETVEL: TraitScope(max_envs=1, note=_MJWARP_MODEL_GLOBAL),
    Trait.DIRECTOR_ATTACH_WELD: TraitScope(max_envs=1, note=_MJWARP_MODEL_GLOBAL),
    Trait.DIRECTOR_GRIPPER: TraitScope(max_envs=1, note=_MJWARP_MODEL_GLOBAL),
    Trait.MOCAP: TraitScope(max_envs=1, note=_MJWARP_MODEL_GLOBAL),
}


def _mujoco_scene_engine() -> object:
    """Construct the MuJoCo scene engine (lazy: no engine import here)."""
    from unigym.backends.shared.mjfamily.scene.scene_engine import (
        MujocoSceneEngine,
    )

    return MujocoSceneEngine()


def _mujoco_env() -> type:
    """Load the MuJoCo env class lazily (engine import stays inside)."""
    from unigym.backends.mujoco.env import MujocoEnv

    return MujocoEnv


def _mjwarp_env() -> type:
    """Load the MuJoCo Warp env class lazily."""
    from unigym.backends.mjwarp.env import MjwarpEnv

    return MjwarpEnv


def _isaaclab2_env() -> type:
    """Load the IsaacLab 2.x env class lazily."""
    from unigym.backends.isaaclab2.env import IsaaclabEnv

    return IsaaclabEnv


def _isaaclab3_env() -> type:
    """Load the IsaacLab 3.x env class lazily (EA skeleton)."""
    from unigym.backends.isaaclab3 import Isaaclab3Env

    return Isaaclab3Env


def _mujoco_live() -> type:
    """Load the MuJoCo-family live binder lazily (shared with mjwarp)."""
    from unigym.backends.mujoco.live import MujocoLiveBackend

    return MujocoLiveBackend


def _mujoco_model_provider() -> Callable[[], Any]:
    """Load the MuJoCo model provider class lazily."""
    from unigym.backends.shared.mjfamily.model.provider import MujocoModelProvider

    return MujocoModelProvider


def _mjwarp_model_provider() -> Callable[[], Any]:
    """Load the mjwarp-labelled model provider class lazily."""
    from unigym.backends.mjwarp.model import MjwarpModelProvider

    return MjwarpModelProvider


def _isaaclab2_model_provider() -> Callable[[], Any]:
    """Load the IsaacLab model provider class lazily."""
    from unigym.backends.isaaclab2.model import IsaacLabModelProvider

    return IsaacLabModelProvider


def _mujoco_model_handle() -> Callable[[object], object | None]:
    """Load the MuJoCo compiled-model handle resolver lazily."""
    from unigym.backends.mujoco.model_handle import model_handle

    return model_handle


def _mjwarp_model_handle() -> Callable[[object], object | None]:
    """Load the mjwarp compiled-model handle resolver lazily."""
    from unigym.backends.mjwarp.model_handle import model_handle

    return model_handle


def _mujoco_rig_factory() -> object:
    """Load the MuJoCo camera rig builder lazily."""
    from unigym.backends.mujoco.cameras import build_mj_rig

    return build_mj_rig


def _mjwarp_rig_factory() -> object:
    """Load the MuJoCo-Warp camera rig builder lazily."""
    from unigym.backends.mjwarp.cameras import build_mjwarp_rig

    return build_mjwarp_rig


def _isaaclab2_rig_factory() -> object:
    """Load the IsaacLab camera rig builder lazily."""
    from unigym.backends.isaaclab2.cameras.rigs import build_isaaclab_rig

    return build_isaaclab_rig


def _mujoco_prop_backend() -> object:
    """Load the MuJoCo prop backend face lazily."""
    from unigym.backends.shared.mjfamily.props.props import build_mujoco_prop_backend

    return build_mujoco_prop_backend()


def _isaaclab2_prop_backend() -> object:
    """Load the IsaacLab prop backend face lazily."""
    from unigym.backends.isaaclab2.props import build_isaaclab_prop_backend

    return build_isaaclab_prop_backend()


def _mujoco_director_ops() -> object:
    """Load the MuJoCo director ops class lazily."""
    from unigym.backends.shared.mjfamily.director.ops import MujocoDirectorOps

    return MujocoDirectorOps


def _mjwarp_director_ops() -> object:
    """Load the MuJoCo-Warp director ops class lazily."""
    from unigym.backends.mjwarp.director_ops import MjwarpDirectorOps

    return MjwarpDirectorOps


def _isaaclab2_director_ops() -> object:
    """Load the IsaacLab director ops class lazily."""
    from unigym.backends.isaaclab2.director_ops import IsaacLabDirectorOps

    return IsaacLabDirectorOps


def _mujoco_contact_backend() -> object:
    """Load the MuJoCo contact backend class lazily."""
    from unigym.backends.mujoco.contacts import MujocoContactBackend

    return MujocoContactBackend


def _isaaclab2_contact_backend() -> object:
    """Load the IsaacLab contact backend class lazily."""
    from unigym.backends.isaaclab2.contacts import IsaacLabContactBackend

    return IsaacLabContactBackend


def _mujoco_gripper_backend() -> object:
    """Load the MuJoCo gripper runtime class lazily."""
    from unigym.backends.mujoco.director.gripper import MujocoGripper

    return MujocoGripper


def _isaaclab2_gripper_backend() -> object:
    """Load the IsaacLab gripper runtime class lazily."""
    from unigym.backends.isaaclab2.director.gripper import IsaacLabGripper

    return IsaacLabGripper


# --------------------------------------------------------------------------- #
# construction faces (lazy: engine imports stay in the packages)
# --------------------------------------------------------------------------- #
def _mujoco_env_kwargs() -> object:
    from unigym.backends.mujoco.build import env_kwargs

    return env_kwargs


def _mjwarp_env_kwargs() -> object:
    from unigym.backends.mjwarp.build import env_kwargs

    return env_kwargs


def _mujoco_content_router() -> Callable[..., object]:
    from unigym.backends.mujoco.build import route_content

    return route_content


def _isaaclab2_env_kwargs() -> object:
    from unigym.backends.isaaclab2.build import env_kwargs

    return env_kwargs


def _isaaclab2_build_terrain() -> object:
    from unigym.backends.isaaclab2.build import build_terrain

    return build_terrain


def _isaaclab2_build_events() -> object:
    from unigym.backends.isaaclab2.build import build_events

    return build_events


def _mujoco_scene_ops() -> object:
    from unigym.backends.mujoco.scene import MujocoSceneOps

    return MujocoSceneOps


def _isaaclab2_scene_ops() -> object:
    from unigym.backends.isaaclab2.scene import IsaacLabSceneOps

    return IsaacLabSceneOps


def _isaaclab2_kit_launcher() -> Callable[..., tuple[Any, Any]]:
    from unigym.backends.isaaclab2.launcher import kit_launcher

    return kit_launcher


def _isaaclab2_robot_realizer() -> object:
    from unigym.backends.isaaclab2.robots import build_robot_realization

    return build_robot_realization


_STATIC_INFO: dict[str, BackendInfo] = {
    "mujoco": BackendInfo(
        "mujoco",
        (0, 0),
        MUJOCO_TRAITS,
        asset_formats=("mjcf",),
        options_schema=_MJ_COMPOSE_OPTIONS,
        scene_engine=_mujoco_scene_engine,
        scopes=MUJOCO_SCOPES,
        env_class=_mujoco_env,
        live_provider=_mujoco_live,
        model_provider=_mujoco_model_provider,
        model_handle=_mujoco_model_handle,
        prop_backend=_mujoco_prop_backend,
        camera_backend=_mujoco_rig_factory,
        director_ops=_mujoco_director_ops,
        contact_backend=_mujoco_contact_backend,
        gripper_backend=_mujoco_gripper_backend,
        env_kwargs=_mujoco_env_kwargs,
        content_router=_mujoco_content_router,
        scene_ops=_mujoco_scene_ops,
        bind_all_declared_cameras=True,
        world_base_required=True,
        mj_scene_consumer=True,
        inline_static_scene=True,
        probe_in_host=True,
        probe_env=(("MUJOCO_GL", "egl"),),
    ),
    "mjwarp": BackendInfo(
        "mjwarp",
        (0, 0),
        MJWARP_TRAITS,
        asset_formats=("mjcf",),
        options_schema=_MJ_COMPOSE_OPTIONS,
        scene_engine=_mujoco_scene_engine,
        scopes=MJWARP_SCOPES,
        env_class=_mjwarp_env,
        live_provider=_mujoco_live,
        model_provider=_mjwarp_model_provider,
        model_handle=_mjwarp_model_handle,
        prop_backend=_mujoco_prop_backend,
        camera_backend=_mjwarp_rig_factory,
        director_ops=_mjwarp_director_ops,
        contact_backend=_mujoco_contact_backend,
        gripper_backend=_mujoco_gripper_backend,
        env_kwargs=_mjwarp_env_kwargs,
        scene_ops=_mujoco_scene_ops,
        bind_all_declared_cameras=True,
        world_base_required=True,
        mj_scene_consumer=True,
        probe_in_host=True,
        probe_env=(("MUJOCO_GL", "egl"),),
    ),
    "isaaclab2": BackendInfo(
        "isaaclab2",
        (2, 3),
        ISAACLAB2_TRAITS,
        asset_formats=("usd", "urdf"),
        env_class=_isaaclab2_env,
        model_provider=_isaaclab2_model_provider,
        prop_backend=_isaaclab2_prop_backend,
        camera_backend=_isaaclab2_rig_factory,
        director_ops=_isaaclab2_director_ops,
        contact_backend=_isaaclab2_contact_backend,
        gripper_backend=_isaaclab2_gripper_backend,
        env_kwargs=_isaaclab2_env_kwargs,
        build_terrain=_isaaclab2_build_terrain,
        build_events=_isaaclab2_build_events,
        scene_ops=_isaaclab2_scene_ops,
        kit_launcher=_isaaclab2_kit_launcher,
        robot_realizer=_isaaclab2_robot_realizer,
        default_lighting={"dome_intensity": 600.0},
        camera_enable_required=True,
    ),
    "isaaclab3": BackendInfo(
        "isaaclab3",
        (3, 0),
        ISAACLAB3_TRAITS,
        asset_formats=("usd", "urdf"),
        env_class=_isaaclab3_env,
        camera_backend=_isaaclab2_rig_factory,
        director_ops=_isaaclab2_director_ops,
        contact_backend=_isaaclab2_contact_backend,
        gripper_backend=_isaaclab2_gripper_backend,
        env_kwargs=_isaaclab2_env_kwargs,
        build_terrain=_isaaclab2_build_terrain,
        build_events=_isaaclab2_build_events,
        scene_ops=_isaaclab2_scene_ops,
        kit_launcher=_isaaclab2_kit_launcher,
        robot_realizer=_isaaclab2_robot_realizer,
        default_lighting={"dome_intensity": 600.0},
        camera_enable_required=True,
    ),
}

CANONICAL_BACKENDS: tuple[str, ...] = tuple(_STATIC_INFO)

#: canonical name of the MuJoCo-Warp backend (the single truth callers key
#: config/CLI binding tables on; never hand-write the literal elsewhere).
MJWARP_BACKEND: str = "mjwarp"

# import-time self-check: a bad trait table must never reach a run
validate_trait_rules()

# config-facing aliases -> canonical family.  "isaaclab" is the historical
# default name; it resolves to isaaclab2/isaaclab3 by version probe.
BACKEND_ALIASES: dict[str, str] = {"isaaclab": "isaaclab:auto"}

# build-name aliases: the single truth for the env factory table.
# "isaaclab" is the historical build spelling of the 2.3-series module.
_BUILD_ALIASES: dict[str, str] = {"isaaclab": "isaaclab2"}


# --------------------------------------------------------------------------- #
# version probing (lazy, side-effect free)
# --------------------------------------------------------------------------- #
def _dist_version(dist: str) -> tuple[int, int] | None:
    """(major, minor) of an installed distribution, else None."""
    try:
        raw = importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None
    nums = re.findall(r"\d+", raw)
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]))
    if len(nums) == 1:
        return (int(nums[0]), 0)
    return None


def _import_version(module: str) -> tuple[int, int] | None:
    """Fallback probe: guarded import + ``__version__``.

    Post-AppLauncher
    the metadata lookup may already have sufficed; this catches exotic
    installs where isaaclab is importable but not pip-registered.
    """
    try:
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001 — any failure means "not usable here"
        return None
    raw = getattr(mod, "__version__", None)
    if not raw:
        return None
    nums = re.findall(r"\d+", str(raw))
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]))
    return None


def isaaclab_version() -> tuple[int, int] | None:
    """Return the IsaacLab (major, minor) version probe.

    Metadata first, guarded import second.
    """
    return _dist_version("isaaclab") or _import_version("isaaclab")


def mujoco_version() -> tuple[int, int] | None:
    """Return the MuJoCo (major, minor) version probe."""
    return _dist_version("mujoco") or _import_version("mujoco")


def mujoco_warp_version() -> tuple[int, int] | None:
    """Return the ``mujoco-warp`` (major, minor) version probe."""
    return _dist_version("mujoco-warp") or _import_version("mujoco_warp")


def known_backends() -> list[str]:
    """Canonical backend names (registry order, no probing)."""
    return list(CANONICAL_BACKENDS)


def build_name(name: str | None = None) -> str:
    """Resolve a build spelling to a canonical backend name (no probing).

    The historical ``"isaaclab"`` build name pins ``isaaclab2`` (the 2.3
    module family); canonical names pass through unchanged.
    """
    raw = "isaaclab" if name is None else str(name).strip().lower()
    raw = _BUILD_ALIASES.get(raw, raw)
    if raw in _STATIC_INFO:
        return raw
    raise ValueError(f"unknown env backend {raw!r}; " f"available: {', '.join(CANONICAL_BACKENDS)}")


def backend_info(name: str) -> BackendInfo:
    """Registry info for *name* (canonical, alias or ``isaaclab``).

    The returned ``version`` is the live probe for the requested family —
    ``(0, 0)`` when the stack is absent (callers gate on it via
    :func:`select_backend`, which is the fail-fast entry point).
    """
    canonical = _resolve_name(name)
    info = _STATIC_INFO[canonical]
    if canonical == "mujoco":
        ver = mujoco_version()
    elif canonical == "mjwarp":
        ver = mujoco_warp_version()
    else:
        ver = isaaclab_version()
        if canonical == "isaaclab3" and ver is not None and ver[0] < 3:
            ver = None  # declared il3, stack is 2.x
        if canonical == "isaaclab2" and ver is not None and ver[0] >= 3:
            ver = None  # declared il2, stack is 3.x
    if ver is not None:
        info = replace(info, version=ver)
    return info


def _resolve_name(name: str | None) -> str:
    """Normalize a requested backend name to a canonical one.

    ``None`` / ``"isaaclab"`` -> the isaaclab family by version probe
    (2.x -> isaaclab2, 3.x+ -> isaaclab3, absent -> isaaclab2 so the
    fail-fast message names the *default* stack).
    """
    if name is None:
        name = "isaaclab"
    name = str(name).strip().lower()
    if name in ("isaaclab", "isaaclab:auto"):
        ver = isaaclab_version()
        if ver is not None and ver[0] >= 3:
            return "isaaclab3"
        return "isaaclab2"
    if name in _STATIC_INFO:
        return name
    raise BackendSelectionError(
        f"unknown env backend {name!r}",
        f"available backends: {', '.join(CANONICAL_BACKENDS)} "
        f"(alias 'isaaclab' auto-selects isaaclab2/isaaclab3 by version)",
    )


def select_backend(name: str | None = None) -> str:
    """Resolve a config ``backend:`` value to a canonical, usable backend.

    Import-probe auto-selection for ``None``/``"isaaclab"``; explicit
    ``isaaclab2``/``isaaclab3``/``mujoco`` pins are verified against the
    probe (stack absent or version mismatch -> :class:`BackendSelectionError`
    with an actionable hint — never a silent fallback).
    """
    canonical = _resolve_name(name)
    if canonical == "mujoco":
        ver = mujoco_version()
        if ver is None:
            raise BackendSelectionError(
                "backend 'mujoco' requested but the mujoco stack is not "
                "importable in this environment",
                "run on the host .venv_sim (pip install mujoco) or pick a " "different env.backend",
            )
        return canonical
    if canonical == "mjwarp":
        ver = mujoco_warp_version()
        if ver is None:
            raise BackendSelectionError(
                "backend 'mjwarp' requested but the mujoco-warp stack is not "
                "importable in this environment",
                "install the warp extra (pip install 'unigym[warp]'; "
                "mujoco>=3.13 + mujoco-warp + warp-lang) or pick a "
                "different env.backend",
            )
        return canonical
    ver = isaaclab_version()
    if ver is None:
        raise BackendSelectionError(
            f"backend '{canonical}' requested but the isaaclab stack is not "
            f"importable in this environment",
            "isaaclab runs need the isaac-lab container (AppLauncher must "
            "start first); check env.backend in the config",
        )
    if canonical == "isaaclab2" and ver[0] >= 3:
        raise BackendSelectionError(
            f"backend 'isaaclab2' requested but installed isaaclab is " f"{ver[0]}.{ver[1]}",
            "set env.backend: isaaclab3 (or install the isaaclab 2.3 stack)",
        )
    if canonical == "isaaclab3" and ver[0] < 3:
        raise BackendSelectionError(
            f"backend 'isaaclab3' requested but installed isaaclab is " f"{ver[0]}.{ver[1]}",
            "set env.backend: isaaclab2 (or install isaaclab 3.x)",
        )
    return canonical
