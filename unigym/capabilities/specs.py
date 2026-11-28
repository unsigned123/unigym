"""Pure backend metadata specs (``BackendSpec`` and the canonical spec table).

A :class:`BackendSpec` describes *what* a backend is and what it can do.  It
contains no callables and imports no concrete backend module.  Construction
faces live in :mod:`unigym.backends.bindings`; the two are combined at the
composition boundary (:mod:`unigym.backends.compose`).

The removed ``terrain_presets`` and ``registration_policy`` fields are
intentionally absent: neither had a production reader, and the registration
policy is owned by :mod:`unigym.registration`.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from unigym.assets.asset_ref import ASSET_FORMATS
from unigym.capabilities.selection import canonical_or_error
from unigym.capabilities.traits import (
    CANONICAL_BACKENDS,
    ISAACLAB2_MODEL_EDIT_FIELDS,
    ISAACLAB2_TRAITS,
    ISAACLAB3_NEWTON_TRAITS,
    ISAACLAB3_PHYSX_TRAITS,
    MJWARP_SCOPES,
    MJWARP_TRAITS,
    MUJOCO_MODEL_EDIT_FIELDS,
    MUJOCO_SCOPES,
    MUJOCO_TRAITS,
    ModelEditField,
    Trait,
    TraitScope,
    validate_trait_relations,
    validate_trait_scopes,
)

__all__ = [
    "BACKEND_SPECS",
    "BackendSpec",
    "backend_spec",
    "blocked_traits",
    "has_traits",
    "missing_traits",
    "normalize_asset_formats",
    "normalize_backend_param_keys",
    "normalize_model_edit_fields",
    "validate_backend_options",
    "validate_trait_specs",
]


def normalize_asset_formats(name: str, formats: Iterable[str]) -> tuple[str, ...]:
    """Validate and freeze one backend's asset-format preference list."""
    ordered = tuple(str(value) for value in formats)
    unknown = sorted(set(ordered) - set(ASSET_FORMATS))
    if not ordered:
        raise ValueError(
            f"backend {name!r} must declare at least one asset format; "
            f"allowed formats are {list(ASSET_FORMATS)}"
        )
    if unknown:
        raise ValueError(
            f"backend {name!r} declares unknown asset format(s) {unknown}; "
            f"allowed formats are {list(ASSET_FORMATS)}"
        )
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"backend {name!r} repeats an asset format: {list(ordered)}")
    return ordered


def normalize_backend_param_keys(name: str, keys: Iterable[str]) -> frozenset[str]:
    """Validate and freeze the ``backend_params.<backend>`` key vocabulary."""
    normalized = frozenset(str(key) for key in keys)
    if any(not key.replace("_", "").isalnum() or key != key.lower() for key in normalized):
        raise ValueError(
            f"backend {name!r} backend_param_keys must be snake_case: {sorted(normalized)}"
        )
    return normalized


def normalize_model_edit_fields(
    name: str, fields: Iterable[ModelEditField]
) -> frozenset[ModelEditField]:
    """Validate and freeze one backend's model-edit field set."""
    normalized = frozenset(fields)
    if not normalized <= set(ModelEditField):
        raise ValueError(
            f"backend {name!r} declares unknown model-edit fields "
            f"{sorted(field.value for field in normalized - set(ModelEditField))}"
        )
    return normalized


def has_traits(declared: Collection[Trait], *traits: Trait) -> bool:
    """True when every requested trait is declared."""
    return set(traits) <= set(declared)


def missing_traits(declared: Collection[Trait], *traits: Trait) -> tuple[Trait, ...]:
    """The requested traits not declared (stable ``Trait`` order)."""
    wanted = [trait for trait in Trait if trait in set(traits)]
    return tuple(trait for trait in wanted if trait not in set(declared))


def blocked_traits(
    declared: Collection[Trait],
    scopes: Mapping[Trait, TraitScope],
    traits: Iterable[Trait],
    num_envs: int | None = None,
) -> tuple[tuple[Trait, TraitScope], ...]:
    """Present-but-scope-blocked ``(trait, scope)`` pairs (enum order).

    Missing traits are not reported here (use :func:`missing_traits`); a
    ``num_envs`` of ``None`` means "no scope check" and yields an empty tuple.
    """
    if num_envs is None:
        return ()
    declared_set = set(declared)
    wanted = [trait for trait in Trait if trait in set(traits)]
    blocked: list[tuple[Trait, TraitScope]] = []
    for trait in wanted:
        if trait not in declared_set:
            continue
        scope = scopes.get(trait)
        if scope is None or scope.max_envs is None:
            continue
        if num_envs > scope.max_envs:
            blocked.append((trait, scope))
    return tuple(blocked)


@dataclass(frozen=True)
class BackendSpec:
    """Pure metadata of one registered backend (no construction callables)."""

    name: str
    """Canonical backend name (``mujoco`` / ``mjwarp`` / ``isaaclab2`` / …)."""
    traits: frozenset[Trait]
    """Capability set (registration declares it)."""
    scopes: Mapping[Trait, TraitScope] = field(default_factory=dict)
    """Per-trait availability narrowing; keys must be declared traits."""
    options_schema: Mapping[str, object] = field(default_factory=dict)
    """Backend-private ``env.backend_options.<backend>`` schema."""
    model_edit_fields: frozenset[ModelEditField] = frozenset()
    """The model fields the backend's writer can edit."""
    asset_formats: tuple[str, ...] = ()
    """Supported robot asset formats in backend preference order."""
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
    content_fill: bool = False
    """Supports the neutral ``objects:`` content fill (has a content router)."""
    backend_param_keys: frozenset[str] = frozenset()
    """Keys a backend accepts under ``backend_params.<backend>``."""
    probe_in_host: bool = False
    """``unigym probe`` runs in the host venv (vs the container lane)."""
    probe_env: tuple[tuple[str, str], ...] = ()
    """Environment defaults injected into a host-lane probe subprocess."""

    def __post_init__(self) -> None:
        """Validate and freeze the pure metadata fields."""
        object.__setattr__(
            self, "asset_formats", normalize_asset_formats(self.name, self.asset_formats)
        )
        object.__setattr__(
            self,
            "backend_param_keys",
            normalize_backend_param_keys(self.name, self.backend_param_keys),
        )
        object.__setattr__(
            self,
            "model_edit_fields",
            normalize_model_edit_fields(self.name, self.model_edit_fields),
        )

    def has(self, *traits: Trait) -> bool:
        """Every listed trait present?"""
        return has_traits(self.traits, *traits)

    def has_any(self, *traits: Trait) -> bool:
        """At least one listed trait present?"""
        return bool(set(traits) & set(self.traits))

    def missing(self, *traits: Trait) -> tuple[Trait, ...]:
        """The listed traits this backend lacks (stable declaration order)."""
        return missing_traits(self.traits, *traits)

    def available(self, trait: Trait, num_envs: int | None = None) -> bool:
        """Is *trait* present AND usable at *num_envs* (``None`` = any)?"""
        if trait not in self.traits:
            return False
        if num_envs is None:
            return True
        scope = self.scopes.get(trait)
        return scope is None or scope.max_envs is None or num_envs <= scope.max_envs

    def blocked(
        self, traits: Iterable[Trait], num_envs: int | None = None
    ) -> tuple[tuple[Trait, TraitScope], ...]:
        """Present-but-scope-blocked ``(trait, scope)`` pairs (enum order)."""
        return blocked_traits(self.traits, self.scopes, traits, num_envs)


# MuJoCo-family compose options (backend-private).
_MJ_COMPOSE_OPTIONS: Mapping[str, object] = {"compose": {"auto_prefix": False}}

BACKEND_SPECS: Mapping[str, BackendSpec] = MappingProxyType(
    {
        "mujoco": BackendSpec(
            name="mujoco",
            traits=MUJOCO_TRAITS,
            scopes=MUJOCO_SCOPES,
            options_schema=_MJ_COMPOSE_OPTIONS,
            model_edit_fields=MUJOCO_MODEL_EDIT_FIELDS,
            asset_formats=("mjcf",),
            bind_all_declared_cameras=True,
            world_base_consumer=True,
            scene_provider_consumer=True,
            content_fill=True,
            backend_param_keys=frozenset({"base", "initializer"}),
            probe_in_host=True,
            probe_env=(("MUJOCO_GL", "egl"),),
        ),
        "mjwarp": BackendSpec(
            name="mjwarp",
            traits=MJWARP_TRAITS,
            scopes=MJWARP_SCOPES,
            options_schema=_MJ_COMPOSE_OPTIONS,
            model_edit_fields=MUJOCO_MODEL_EDIT_FIELDS,
            asset_formats=("mjcf",),
            bind_all_declared_cameras=True,
            world_base_consumer=True,
            scene_provider_consumer=True,
            content_fill=True,
            backend_param_keys=frozenset({"base", "initializer"}),
            probe_in_host=True,
            probe_env=(("MUJOCO_GL", "egl"),),
        ),
        "isaaclab2": BackendSpec(
            name="isaaclab2",
            traits=ISAACLAB2_TRAITS,
            model_edit_fields=ISAACLAB2_MODEL_EDIT_FIELDS,
            asset_formats=("usd", "urdf"),
            default_lighting={"dome_intensity": 600.0},
            camera_enable_required=True,
            scene_provider_consumer=True,
        ),
        "isaaclab3_physx": BackendSpec(
            name="isaaclab3_physx",
            traits=ISAACLAB3_PHYSX_TRAITS,
            asset_formats=("usd", "urdf"),
            default_lighting={"dome_intensity": 600.0},
            camera_enable_required=True,
            scene_provider_consumer=True,
        ),
        "isaaclab3_newton": BackendSpec(
            name="isaaclab3_newton",
            traits=ISAACLAB3_NEWTON_TRAITS,
            asset_formats=("usd", "urdf"),
            default_lighting={"dome_intensity": 600.0},
            scene_provider_consumer=True,
        ),
    }
)

if tuple(BACKEND_SPECS) != CANONICAL_BACKENDS:
    raise RuntimeError(
        "backend spec table order does not match "
        "unigym.capabilities.CANONICAL_BACKENDS: "
        f"{tuple(BACKEND_SPECS)} != {CANONICAL_BACKENDS}"
    )


def backend_spec(name: str) -> BackendSpec:
    """Return the pure metadata spec for canonical backend *name*.

    Raises:
        BackendSelectionError: *name* is not a canonical backend name.
    """
    return BACKEND_SPECS[canonical_or_error(name)]


def validate_trait_specs(backends: Mapping[str, BackendSpec] | None = None) -> None:
    """Validate every backend spec's trait/scope declaration.

    Pure: no construction faces are consulted.  Empty trait sets (the
    IsaacLab 3 identity skeletons) are skipped because they declare no layout
    yet.
    """
    specs = BACKEND_SPECS if backends is None else backends
    for name, spec in specs.items():
        traits = set(spec.traits)
        validate_trait_scopes(name, traits, spec.scopes)
        if not traits:
            continue
        validate_trait_relations(name, traits)


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
        raise ValueError(f"env.backend_options must be a mapping, got {type(options).__name__}")
    canonical = canonical_or_error(backend)
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
    spec = BACKEND_SPECS[canonical]
    if not spec.options_schema:
        if supplied:
            keys = sorted(supplied) if isinstance(supplied, Mapping) else []
            raise ValueError(
                f"backend {canonical!r} declares no backend_options (unknown "
                f"keys {keys}); supported options: []"
            )
        return {}
    if not isinstance(supplied, Mapping):
        raise ValueError(
            f"env.backend_options.{canonical} must be a mapping, got {type(supplied).__name__}"
        )
    return _validate_options_block(spec.options_schema, supplied, "")
