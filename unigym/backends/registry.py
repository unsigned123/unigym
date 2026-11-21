"""Trait registry: the capability contract between run configs and backends.

Single source of truth for "what can a backend do".  The enum table is the
mechanized touch matrix (docs/designs/backend-traits.md §1.1 → §3.1): every
optional feature a yaml/adapter/director-op can request maps to a trait, and
a backend missing that trait fails fast at build time (``{error, hint}``) —
never a silent degradation.

Backends are *equal citizens*: ``mujoco`` / ``mjwarp`` / ``isaaclab2`` /
``isaaclab3_physx`` / ``isaaclab3_newton``.  The two IsaacLab 3 identities share
one implementation package but differ in engine, runtime stack and capability
set.  Selection is canonical-only: there are no ``isaaclab`` / ``isaaclab3``
aliases and no version-based family auto-selection.  ``backend`` is required;
the explicitly chosen stack is verified against a live probe and never falls
back silently.

This module is **stdlib-only at import time**: engine presence is probed
lazily (importlib.metadata first — no side effects — then a guarded
``import``), so unit tests can import it on any host.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from collections.abc import Callable, Mapping
from typing import Any, TYPE_CHECKING
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType

from unigym.assets.asset_ref import ASSET_FORMATS
from unigym.registration import (
    ACTUATOR_IMPLEMENTATION_WINDOWS,
    BackendRegistrationPolicy,
    PackFace,
)

if TYPE_CHECKING:
    from unigym.query.facade import QueryBackend

__all__ = [
    "BACKEND_REGISTRATION_POLICIES",
    "CANONICAL_BACKENDS",
    "MJWARP_BACKEND",
    "BackendInfo",
    "BackendSelectionError",
    "ModelEditField",
    "ModelEditProvider",
    "StackProbe",
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
    "newton_version",
    "select_backend",
    "selection_error",
    "validate_backend_options",
    "validate_trait_rules",
    "warp_version",
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
    ACTION_FACES_NAMED = "ACTION_FACES_NAMED"  # consumes Sequence[NamedAction]
    JOINT_PERMUTATION = "JOINT_PERMUTATION"  # policy/il orders + inverse perm
    # --- env shape (optional: single-env backends omit it) ---
    MULTI_ENV = "MULTI_ENV"  # native N>1 parallel-env runs
    # --- state writes (optional: collection-only backends omit it) ---
    STATE_WRITE = "STATE_WRITE"  # explicit world-frame simulator state writes
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
    # --- model edits (global; no per-env face) ---
    MODEL_EDIT = "MODEL_EDIT"  # run.model_edit() transaction
    # --- on-demand runtime query (not an observation channel) ---
    RUNTIME_QUERY = "RUNTIME_QUERY"  # run.query -> Query facade
    # --- native escape hatch (external packs only; unstable) ---
    NATIVE_HANDLE = "NATIVE_HANDLE"  # env.native(backend) -> NativeHandle


class ModelEditField(Enum):
    """One addressable model parameter inside a ``MODEL_EDIT`` transaction.

    The enum is the single source of truth for the field capability matrix;
    backends declare the subset they can write through
    :attr:`BackendInfo.model_edit_fields`.
    """

    BODY_MASS = "body.mass"
    BODY_COM = "body.com"
    BODY_INERTIA = "body.inertia"
    BODY_INERTIA_QUAT = "body.inertia_quat"
    GEOM_FRICTION = "geom.friction"
    GEOM_RESTITUTION = "geom.restitution"


BODY_MODEL_EDIT_FIELDS = frozenset(
    {
        ModelEditField.BODY_MASS,
        ModelEditField.BODY_COM,
        ModelEditField.BODY_INERTIA,
        ModelEditField.BODY_INERTIA_QUAT,
    }
)
GEOM_MODEL_EDIT_FIELDS = frozenset(
    {
        ModelEditField.GEOM_FRICTION,
        ModelEditField.GEOM_RESTITUTION,
    }
)
MUJOCO_MODEL_EDIT_FIELDS = BODY_MODEL_EDIT_FIELDS | GEOM_MODEL_EDIT_FIELDS
ISAACLAB2_MODEL_EDIT_FIELDS = BODY_MODEL_EDIT_FIELDS


@dataclass(frozen=True)
class ModelEditProvider:
    """A backend's model-edit writer declaration (engine import stays lazy).

    Attributes:
        fields: The exact field set this provider can write.  It must equal
            the owning backend's :attr:`BackendInfo.model_edit_fields`.
        loader: Zero-argument factory returning the provider class/object;
            called only when a model-edit transaction opens.
    """

    fields: frozenset[ModelEditField]
    loader: Callable[[], object] = field(repr=False)


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

    name: str  # canonical: "mujoco" | "mjwarp" | "isaaclab2" | "isaaclab3_*"
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
    model_edit_provider: ModelEditProvider | None = None
    """Optional model-edit writer declaration (``Trait.MODEL_EDIT``)."""
    model_edit_fields: frozenset[ModelEditField] = frozenset()
    """The model fields the backend's writer can edit (must equal provider)."""
    query_backend: Callable[[], QueryBackend] | None = None
    """Optional zero-argument factory for the backend's runtime-query class.

    The returned class is constructed with the live env (``cls(env)``) and
    implements :class:`unigym.query.facade.Query`.  Set exactly when the
    backend declares ``Trait.RUNTIME_QUERY``; the engine import stays inside
    the factory so this module remains stdlib-only at import time.
    """
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
    """Bind every declared camera rig (vs config-selected instantiation)."""
    world_base_consumer: bool = False
    """Consumes the run-level ``world_base: {mjcf|urdf|usd: ...}`` asset."""
    scene_provider_consumer: bool = False
    """Consumes ``env.params.scene_provider`` and composes its neutral scene."""
    backend_param_keys: frozenset[str] = frozenset()
    """Keys a backend accepts under ``backend_params.<backend>``."""
    probe_in_host: bool = False
    """``unigym probe`` runs in the host venv (vs the container lane)."""
    probe_env: tuple[tuple[str, str], ...] = ()
    """Environment defaults injected into a host-lane probe subprocess."""
    registration_policy: BackendRegistrationPolicy | None = None
    """Optional narrowing of the pack registration windows."""

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
        keys = frozenset(str(key) for key in self.backend_param_keys)
        if any(not key.replace("_", "").isalnum() or key != key.lower() for key in keys):
            raise ValueError(
                f"backend {self.name!r} backend_param_keys must be snake_case: {sorted(keys)}"
            )
        object.__setattr__(self, "backend_param_keys", keys)
        edit_fields = frozenset(self.model_edit_fields)
        if not edit_fields <= set(ModelEditField):
            raise ValueError(
                f"backend {self.name!r} declares unknown model-edit fields "
                f"{sorted(field.value for field in edit_fields - set(ModelEditField))}"
            )
        object.__setattr__(self, "model_edit_fields", edit_fields)

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
    "isaaclab2": ("isaaclab2",),
    "isaaclab3_physx": ("isaaclab3_physx",),
    "isaaclab3_newton": ("isaaclab3_newton",),
}


def validate_backend_options(
    backend: str, options: Mapping[str, object] | None
) -> dict[str, object]:
    """Validate ``env.backend_options`` and return only *backend*'s options.

    Args:
        backend: The requested canonical backend name.
        options: The raw ``env.backend_options`` mapping.

    Returns:
        The normalized option tree for this backend only (defaults applied).

    Raises:
        ValueError: A malformed block, an unknown backend namespace, an
            unknown option key or an option the backend does not declare.
        BackendSelectionError: *backend* is not a canonical name.
    """
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise ValueError(f"env.backend_options must be a mapping, got " f"{type(options).__name__}")
    canonical = _canonical_or_error(backend)
    accepted = _OPTION_NAMESPACES[canonical]
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


def _validate_model_edit_declaration(
    name: str,
    info: BackendInfo,
    traits: set[Trait],
) -> None:
    """Enforce provider/field/trait consistency for ``MODEL_EDIT``."""
    provider = info.model_edit_provider
    fields = set(info.model_edit_fields)
    if Trait.MODEL_EDIT in traits:
        if provider is None:
            raise RuntimeError(
                f"backend {name!r}: MODEL_EDIT is declared but " f"model_edit_provider is None"
            )
        if not fields:
            raise RuntimeError(
                f"backend {name!r}: MODEL_EDIT is declared but " f"model_edit_fields is empty"
            )
        if set(provider.fields) != fields:
            raise RuntimeError(
                f"backend {name!r}: model_edit_provider fields "
                f"{sorted(field.value for field in provider.fields)} != "
                f"model_edit_fields {sorted(field.value for field in fields)}"
            )
        return
    if provider is not None or fields:
        raise RuntimeError(
            f"backend {name!r}: model-edit provider/fields declared without "
            f"the MODEL_EDIT trait"
        )


def _validate_query_declaration(
    name: str,
    info: BackendInfo,
    traits: set[Trait],
) -> None:
    """Enforce ``RUNTIME_QUERY`` <-> ``query_backend`` consistency."""
    declared = Trait.RUNTIME_QUERY in traits
    if declared and info.query_backend is None:
        raise RuntimeError(
            f"backend {name!r}: RUNTIME_QUERY is declared but " f"query_backend is None"
        )
    if not declared and info.query_backend is not None:
        raise RuntimeError(
            f"backend {name!r}: query_backend is declared without the " f"RUNTIME_QUERY trait"
        )


def validate_trait_rules(backends=None) -> None:
    """Validate every registered backend's trait set against ``TRAIT_RULES``.

    Called at import (so a bad declaration table fails before any run) and
    re-run independently by ``tests/meta/traits/test_trait_rules.py``.
    Empty trait sets (the IsaacLab 3 P1 skeletons) are skipped: they declare no
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
        _validate_model_edit_declaration(name, info, traits)
        _validate_query_declaration(name, info, traits)
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
        Trait.STATE_WRITE,  # per-env set_root_state/set_joint_state/settle
        # Model introspection: the compiled MjModel carries them all
        Trait.MODEL_ADDRESSES,
        Trait.MODEL_MASS_INERTIA,
        Trait.MODEL_MOCAP,
        Trait.MODEL_EQ_SLOTS,
        Trait.MODEL_EDIT,  # global body/geom parameter transaction
        Trait.RUNTIME_QUERY,  # on-demand query face (not an obs channel)
        Trait.NATIVE_HANDLE,  # external-pack-only identity handle
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
        Trait.STATE_WRITE,  # CPU mirror write + per-world device push
        # Same compiled MjModel as mujoco
        Trait.MODEL_ADDRESSES,
        Trait.MODEL_MASS_INERTIA,
        Trait.MODEL_MOCAP,
        Trait.MODEL_EQ_SLOTS,
        Trait.MODEL_EDIT,  # global body/geom parameter transaction
        Trait.NATIVE_HANDLE,  # external-pack-only identity handle
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
        Trait.SCENE_COMPOSE,
        Trait.TERRAIN,
        Trait.EVENTS_DR,
        Trait.INIT_POSE,
        Trait.SPAWN_XY_YAW,
        Trait.STATE_WRITE,  # per-env set_root_state/set_joint_state/settle
        # The USD prim hierarchy is the isaaclab model face
        Trait.MODEL_USD_PRIMS,
        Trait.MODEL_EDIT,  # body mass/COM/inertia across every env row
        Trait.NATIVE_HANDLE,  # external-pack-only identity handle
    }
)

# IsaacLab 3 identities: PhysX and Newton share one implementation package
# but own separate runtime stacks and capability sets.  C2-1 lights up the
# verified shared core conservatively; prop traits stay undeclared until
# their backend tests land (P2).  The rigid-prop path is lane-tested, but the
# shared props bit projects all three PROPS_* traits, so its declaration is
# deferred until the capability model can express it.  The PhysX director,
# camera and recorder paths are lane-proven, so those traits are declared;
# weld stays absent until an engine-hosted weld test exists.
_IL3_CORE = _CORE | frozenset(
    {
        Trait.MULTI_ENV,  # native num_envs=N scene cloning
        Trait.ENV_LAYOUT_TILED,  # env_spacing clone grid
        Trait.SCENE_COMPOSE,  # neutral Scene -> composed asset scene
        Trait.TERRAIN,
        Trait.INIT_POSE,
        Trait.STATE_WRITE,  # per-env set_root_state/set_joint_state/settle
        Trait.MODEL_USD_PRIMS,  # USD prim hierarchy is the isaaclab model face
        Trait.NATIVE_HANDLE,  # external-pack-only identity handle
    }
)

ISAACLAB3_PHYSX_TRAITS = _IL3_CORE | frozenset(
    {
        Trait.EVENTS_DR,  # PhysX profile owns the CoM DR write
        Trait.CONTACTS_GROUPS,  # contacts: all world-frame groups
        Trait.DIRECTOR_TELEPORT,  # root-pose writes on a rigid prop
        Trait.DIRECTOR_SETVEL,  # root-velocity writes on a rigid prop
        Trait.DIRECTOR_ATTACH_SLAVE,  # per-tick kinematic follow
        Trait.DIRECTOR_GRIPPER,  # kinematic gripper pose face
        Trait.CAMERAS_RGB,  # RTX CameraCfg rigs
        Trait.CAMERAS_DEPTH_RIG,  # distance_to_camera annotator
        Trait.CAMERAS_GLOBAL,  # global request-axis world rig
        Trait.CAMERAS_PER_ENV,  # cloned sensor per env
        Trait.CAMERAS_PER_ENV_STATE,  # per-(camera, env) pose state
        Trait.RECORDER_VIDEO,  # camera_ref capture -> MP4
    }
)

ISAACLAB3_NEWTON_TRAITS = _IL3_CORE

# mujoco's model-global writes are one-MjModel only (no per-env
# face); they stay declared traits but are unavailable at N>1.  mjwarp
# declares the same N=1 scope (director writes the CPU mirror + pushes the
# device; one MjModel / one mocap namespace).  isaaclab2 supports these at
# N>1 (no scope); the isaaclab3 neutral Director writes env 0 only.
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


def _isaaclab3_physx_env() -> type:
    """Load the IsaacLab 3.x PhysX env class lazily."""
    from unigym.backends.isaaclab3_physx.env import Isaaclab3PhysxEnv

    return Isaaclab3PhysxEnv


def _isaaclab3_newton_env() -> type:
    """Load the IsaacLab 3.x Newton env class lazily."""
    from unigym.backends.isaaclab3_newton.env import Isaaclab3NewtonEnv

    return Isaaclab3NewtonEnv


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


def _mujoco_model_edit_provider() -> object:
    """Load the MuJoCo model-edit writer lazily."""
    from unigym.backends.mujoco.model_edit import MujocoModelEditProvider

    return MujocoModelEditProvider


def _mjwarp_model_edit_provider() -> object:
    """Load the mjwarp model-edit writer lazily."""
    from unigym.backends.mjwarp.model_edit import MjwarpModelEditProvider

    return MjwarpModelEditProvider


def _isaaclab2_model_edit_provider() -> object:
    """Load the IsaacLab2 model-edit writer lazily."""
    from unigym.backends.isaaclab2.model_edit import IsaacLab2ModelEditProvider

    return IsaacLab2ModelEditProvider


def _mujoco_model_handle() -> Callable[[object], object | None]:
    """Load the MuJoCo compiled-model handle resolver lazily."""
    from unigym.backends.mujoco.model_handle import model_handle

    return model_handle


def _mujoco_query_backend() -> QueryBackend:
    """Load the MuJoCo runtime-query class lazily."""
    from unigym.backends.mujoco.query import MujocoQuery

    return MujocoQuery


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


def _isaaclab2_scene_engine() -> object:
    """Construct the IsaacLab asset-scene engine (lazy: no engine import here)."""
    from unigym.backends.isaaclab2.asset_scene.engine import AssetSceneEngine

    return AssetSceneEngine()


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


def _isaaclab3_faces(backend: str) -> object:
    """Import one IsaacLab 3 backend's lazy registry-face module.

    Called only from inside a face factory, so the engine-owned module (and
    any engine import it triggers) never loads when this stdlib-only module
    is imported.
    """
    return importlib.import_module(f"unigym.backends.{backend}.registry_faces")


def _isaaclab3_face(backend: str, face: str) -> Callable[[], object]:
    """Return a registry factory that resolves one face from *backend*.

    Args:
        backend: the canonical IsaacLab 3 identity
            (``isaaclab3_physx`` / ``isaaclab3_newton``).
        face: the zero-argument factory name in that backend's
            ``registry_faces`` module.
    """

    def factory() -> object:
        return getattr(_isaaclab3_faces(backend), face)()

    return factory


def _isaaclab3_seam(backend: str, name: str) -> Callable[[], object]:
    """Return a registry factory for an exported builder/launcher object.

    ``env_kwargs`` / ``build_terrain`` / ``build_events`` / ``kit_launcher``
    are the callables themselves (not zero-argument factories), so the
    registry factory resolves the module attribute without calling it.
    """

    def factory() -> object:
        return getattr(_isaaclab3_faces(backend), name)

    return factory


BACKEND_REGISTRATION_POLICIES: Mapping[str, BackendRegistrationPolicy] = MappingProxyType(
    {
        backend: BackendRegistrationPolicy(
            backend=backend,
            windows={PackFace.ACTUATOR_IMPLEMENTATION: window},
        )
        for backend, window in ACTUATOR_IMPLEMENTATION_WINDOWS.items()
    }
)
"""Canonical backend -> registration policy (only implementations narrow)."""


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
        model_edit_provider=ModelEditProvider(
            MUJOCO_MODEL_EDIT_FIELDS, _mujoco_model_edit_provider
        ),
        model_edit_fields=MUJOCO_MODEL_EDIT_FIELDS,
        model_handle=_mujoco_model_handle,
        query_backend=_mujoco_query_backend,
        prop_backend=_mujoco_prop_backend,
        camera_backend=_mujoco_rig_factory,
        director_ops=_mujoco_director_ops,
        contact_backend=_mujoco_contact_backend,
        gripper_backend=_mujoco_gripper_backend,
        env_kwargs=_mujoco_env_kwargs,
        content_router=_mujoco_content_router,
        scene_ops=_mujoco_scene_ops,
        bind_all_declared_cameras=True,
        world_base_consumer=True,
        scene_provider_consumer=True,
        backend_param_keys=frozenset({"base", "initializer"}),
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
        model_edit_provider=ModelEditProvider(
            MUJOCO_MODEL_EDIT_FIELDS, _mjwarp_model_edit_provider
        ),
        model_edit_fields=MUJOCO_MODEL_EDIT_FIELDS,
        model_handle=_mjwarp_model_handle,
        prop_backend=_mujoco_prop_backend,
        camera_backend=_mjwarp_rig_factory,
        director_ops=_mjwarp_director_ops,
        contact_backend=_mujoco_contact_backend,
        gripper_backend=_mujoco_gripper_backend,
        env_kwargs=_mjwarp_env_kwargs,
        content_router=_mujoco_content_router,
        scene_ops=_mujoco_scene_ops,
        bind_all_declared_cameras=True,
        world_base_consumer=True,
        scene_provider_consumer=True,
        backend_param_keys=frozenset({"base", "initializer"}),
        probe_in_host=True,
        probe_env=(("MUJOCO_GL", "egl"),),
    ),
    "isaaclab2": BackendInfo(
        "isaaclab2",
        (2, 3),
        ISAACLAB2_TRAITS,
        asset_formats=("usd", "urdf"),
        scene_engine=_isaaclab2_scene_engine,
        env_class=_isaaclab2_env,
        model_provider=_isaaclab2_model_provider,
        model_edit_provider=ModelEditProvider(
            ISAACLAB2_MODEL_EDIT_FIELDS, _isaaclab2_model_edit_provider
        ),
        model_edit_fields=ISAACLAB2_MODEL_EDIT_FIELDS,
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
        scene_provider_consumer=True,
    ),
    "isaaclab3_physx": BackendInfo(
        "isaaclab3_physx",
        (3, 0),
        ISAACLAB3_PHYSX_TRAITS,
        asset_formats=("usd", "urdf"),
        scene_engine=_isaaclab3_face("isaaclab3_physx", "scene_engine"),
        env_class=_isaaclab3_physx_env,
        model_provider=_isaaclab3_face("isaaclab3_physx", "model_provider"),
        prop_backend=_isaaclab3_face("isaaclab3_physx", "prop_backend"),
        camera_backend=_isaaclab3_face("isaaclab3_physx", "camera_backend"),
        director_ops=_isaaclab3_face("isaaclab3_physx", "director_ops"),
        contact_backend=_isaaclab3_face("isaaclab3_physx", "contact_backend"),
        gripper_backend=_isaaclab3_face("isaaclab3_physx", "gripper_backend"),
        env_kwargs=_isaaclab3_seam("isaaclab3_physx", "env_kwargs"),
        build_terrain=_isaaclab3_seam("isaaclab3_physx", "build_terrain"),
        build_events=_isaaclab3_seam("isaaclab3_physx", "build_events"),
        scene_ops=_isaaclab3_face("isaaclab3_physx", "scene_ops"),
        kit_launcher=_isaaclab3_seam("isaaclab3_physx", "kit_launcher"),
        robot_realizer=_isaaclab3_face("isaaclab3_physx", "robot_realizer"),
        default_lighting={"dome_intensity": 600.0},
        camera_enable_required=True,
        scene_provider_consumer=True,
    ),
    "isaaclab3_newton": BackendInfo(
        "isaaclab3_newton",
        (3, 0),
        ISAACLAB3_NEWTON_TRAITS,
        asset_formats=("usd", "urdf"),
        scene_engine=_isaaclab3_face("isaaclab3_newton", "scene_engine"),
        env_class=_isaaclab3_newton_env,
        model_provider=_isaaclab3_face("isaaclab3_newton", "model_provider"),
        prop_backend=_isaaclab3_face("isaaclab3_newton", "prop_backend"),
        camera_backend=_isaaclab3_face("isaaclab3_newton", "camera_backend"),
        director_ops=_isaaclab3_face("isaaclab3_newton", "director_ops"),
        contact_backend=_isaaclab3_face("isaaclab3_newton", "contact_backend"),
        gripper_backend=_isaaclab3_face("isaaclab3_newton", "gripper_backend"),
        env_kwargs=_isaaclab3_seam("isaaclab3_newton", "env_kwargs"),
        build_terrain=_isaaclab3_seam("isaaclab3_newton", "build_terrain"),
        build_events=_isaaclab3_seam("isaaclab3_newton", "build_events"),
        scene_ops=_isaaclab3_face("isaaclab3_newton", "scene_ops"),
        kit_launcher=_isaaclab3_seam("isaaclab3_newton", "kit_launcher"),
        robot_realizer=_isaaclab3_face("isaaclab3_newton", "robot_realizer"),
        default_lighting={"dome_intensity": 600.0},
        camera_enable_required=False,
        scene_provider_consumer=True,
    ),
}

_STATIC_INFO = {
    name: replace(info, registration_policy=BACKEND_REGISTRATION_POLICIES.get(name))
    for name, info in _STATIC_INFO.items()
}

CANONICAL_BACKENDS: tuple[str, ...] = tuple(_STATIC_INFO)

#: canonical name of the MuJoCo-Warp backend (the single truth callers key
#: config/CLI binding tables on; never hand-write the literal elsewhere).
MJWARP_BACKEND: str = "mjwarp"

# import-time self-check: a bad trait table must never reach a run
validate_trait_rules()


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


def newton_version() -> tuple[int, int] | None:
    """Return the ``newton`` (major, minor) version probe."""
    return _dist_version("newton") or _import_version("newton")


def warp_version() -> tuple[int, int] | None:
    """Return the ``warp-lang`` (major, minor) version probe.

    The distribution name is ``warp-lang``; the import/module name is ``warp``.
    """
    return _dist_version("warp-lang") or _import_version("warp")


def known_backends() -> list[str]:
    """Canonical backend names (registry order, no probing)."""
    return list(CANONICAL_BACKENDS)


def _canonical_hint() -> str:
    """The teaching hint shared by every rejection path."""
    return (
        f"available backends: {', '.join(CANONICAL_BACKENDS)} "
        f"(canonical names only; 'isaaclab'/'isaaclab3' aliases were removed)"
    )


def _canonical_or_error(name: str | None) -> str:
    """Return *name* if it is canonical, else raise ``BackendSelectionError``.

    No aliases and no version probing: ``None``/``""``/``isaaclab``/
    ``isaaclab3`` are rejected just like any unknown spelling.
    """
    raw = "" if name is None else str(name).strip().lower()
    if raw in _STATIC_INFO:
        return raw
    raise BackendSelectionError(f"unknown env backend {name!r}", _canonical_hint())


def build_name(name: str) -> str:
    """Validate a canonical build spelling (no probing, no aliases).

    Only :data:`CANONICAL_BACKENDS` is accepted.  ``None``, ``""``,
    ``"isaaclab"``, ``"isaaclab3"`` and unknown spellings raise a teaching
    :class:`ValueError` listing the five canonical names.
    """
    raw = "" if name is None else str(name).strip().lower()
    if raw in _STATIC_INFO:
        return raw
    raise ValueError(
        f"unknown env backend {name!r}; available backends: "
        f"{', '.join(CANONICAL_BACKENDS)} "
        f"(canonical names only; 'isaaclab'/'isaaclab3' aliases were removed)"
    )


def _stack_version(canonical: str) -> tuple[int, int] | None:
    """Probed version of *canonical*'s stack, or None if absent/mismatched."""
    if canonical == "mujoco":
        return mujoco_version()
    if canonical == "mjwarp":
        mjwarp = mujoco_warp_version()
        if mjwarp is None or warp_version() is None:
            return None
        return mjwarp
    ver = isaaclab_version()
    if canonical == "isaaclab2":
        return None if ver is None or ver[0] >= 3 else ver
    if canonical == "isaaclab3_physx":
        return None if ver is None or ver[0] < 3 else ver
    # isaaclab3_newton: needs IsaacLab 3.x and the whole Newton stack.
    if ver is None or ver[0] < 3:
        return None
    if newton_version() is None or mujoco_warp_version() is None or warp_version() is None:
        return None
    return ver


def backend_info(name: str) -> BackendInfo:
    """Registry info for the canonical backend *name*.

    The returned ``version`` is the live probe for that backend's stack;
    ``(0, 0)`` when the stack is absent or its major version does not match
    the backend identity (callers gate on it via :func:`select_backend`, the
    fail-fast entry point).

    Raises:
        BackendSelectionError: *name* is not a canonical name (no aliases,
            no ``None``).
    """
    canonical = _canonical_or_error(name)
    info = _STATIC_INFO[canonical]
    ver = _stack_version(canonical)
    return replace(info, version=(0, 0) if ver is None else ver)


@dataclass(frozen=True)
class StackProbe:
    """A snapshot of the engine stacks a backend selection may require.

    The pure decision function :func:`selection_error` consumes this value
    object, so unit tests feed fake probes instead of monkeypatching the live
    version probes.  Each field is ``(major, minor)`` or ``None`` (absent).
    """

    isaaclab: tuple[int, int] | None = None
    newton: tuple[int, int] | None = None
    mujoco: tuple[int, int] | None = None
    mujoco_warp: tuple[int, int] | None = None
    warp: tuple[int, int] | None = None


def selection_error(canonical: str, probe: StackProbe) -> BackendSelectionError | None:
    """Decide whether *canonical* is usable against *probe* (pure).

    Never calls a version probe: the caller owns the :class:`StackProbe`.
    Returns the :class:`BackendSelectionError` for an unusable selection, or
    ``None`` when the explicitly chosen stack is satisfied.
    """
    if canonical not in _STATIC_INFO:
        return BackendSelectionError(
            f"unknown env backend {canonical!r}",
            _canonical_hint(),
        )

    if canonical == "mujoco":
        if probe.mujoco is None:
            return BackendSelectionError(
                "backend 'mujoco' requested but the mujoco stack is not "
                "importable in this environment",
                "run on the host .venv_sim (pip install mujoco) or pick a " "different env.backend",
            )
        return None

    if canonical == "mjwarp":
        missing = [
            dist
            for dist, ver in (("mujoco-warp", probe.mujoco_warp), ("warp-lang", probe.warp))
            if ver is None
        ]
        if missing:
            return BackendSelectionError(
                "backend 'mjwarp' requested but the "
                + " + ".join(missing)
                + " stack is not importable in this environment",
                "install the warp extra (pip install 'unigym[warp]'; "
                "mujoco>=3.13 + mujoco-warp + warp-lang) or pick a "
                "different env.backend",
            )
        return None

    ver = probe.isaaclab
    if canonical == "isaaclab2":
        if ver is None:
            return BackendSelectionError(
                "backend 'isaaclab2' requested but the isaaclab stack is not "
                "importable in this environment",
                "isaaclab runs need the isaac-lab container (AppLauncher must "
                "start first); check env.backend in the config",
            )
        if ver[0] >= 3:
            return BackendSelectionError(
                f"backend 'isaaclab2' requested but installed isaaclab is {ver[0]}.{ver[1]}",
                "set env.backend: isaaclab3_physx or isaaclab3_newton "
                "(or install the isaaclab 2.3 stack)",
            )
        return None

    # The two IsaacLab 3 identities both require IsaacLab 3.x.
    if ver is None:
        return BackendSelectionError(
            f"backend {canonical!r} requested but the isaaclab stack is not "
            f"importable in this environment",
            "isaaclab runs need the isaac-lab container (AppLauncher must "
            "start first); check env.backend in the config",
        )
    if ver[0] < 3:
        return BackendSelectionError(
            f"backend {canonical!r} requested but installed isaaclab is {ver[0]}.{ver[1]}",
            "set env.backend: isaaclab2 (or install isaaclab 3.x)",
        )
    if canonical == "isaaclab3_physx":
        return None

    # isaaclab3_newton: IsaacLab 3.x plus newton / mujoco-warp / warp-lang.
    missing = [
        dist
        for dist, dist_ver in (
            ("newton", probe.newton),
            ("mujoco-warp", probe.mujoco_warp),
            ("warp-lang", probe.warp),
        )
        if dist_ver is None
    ]
    if missing:
        return BackendSelectionError(
            "backend 'isaaclab3_newton' requested but the "
            + " + ".join(missing)
            + " stack is not importable in this environment",
            "install newton + mujoco-warp + warp-lang (the IsaacLab 3 Newton "
            "extra) or set env.backend: isaaclab3_physx",
        )
    return None


def _live_probe() -> StackProbe:
    """Snapshot the live stacks for :func:`select_backend` (probes once)."""
    return StackProbe(
        isaaclab=isaaclab_version(),
        newton=newton_version(),
        mujoco=mujoco_version(),
        mujoco_warp=mujoco_warp_version(),
        warp=warp_version(),
    )


def select_backend(name: str) -> str:
    """Verify the explicitly chosen canonical backend against the live stack.

    Canonical-only: no aliases and no automatic family selection.  ``None``,
    ``""``, ``"isaaclab"``, ``"isaaclab3"`` and unknown names are rejected.
    The chosen backend's stack is verified with :func:`selection_error`; a
    missing or mismatched stack raises :class:`BackendSelectionError` with an
    actionable hint — never a silent fallback.
    """
    canonical = _canonical_or_error(name)
    error = selection_error(canonical, _live_probe())
    if error is not None:
        raise error
    return canonical
