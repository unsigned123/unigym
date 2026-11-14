"""Scene preset resolution over the pack runtime ``scene_presets`` face.

A scene preset is a reusable world declaration: an optional provider (a
registered :class:`~unigym.scenes.providers.SceneProvider`) plus its params, a
format-neutral world asset (``world: {mjcf|urdf|usd: <path>}``), run content
(``objects:``) and backend-only fields (``backend_params.<backend>``).  This
module owns the preset document, the three-layer merge and the routing into
the existing run-config consumption points.  It is backend-free: the run layer
passes a :class:`BackendBinding` describing the selected backend.

There is one clean cutover: the old ``scene:`` / ``scenes`` vocabulary and the
``backend.mujoco`` shape are gone, with teaching errors naming the replacement.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeGuard, cast

import yaml

from unigym._typing import JsonValue
from unigym.assets.asset_ref import AssetRef, AssetRefError, select_asset_format
from unigym.scenes.provider_params import (
    SceneProviderParamError,
    fill_and_validate,
    normalize_manifest_params,
)
from unigym.scenes.recipe import normalize_content_keys

if TYPE_CHECKING:
    from unigym.packs.packset import ScenePresetRef
    from unigym.packs.runtime import PackRuntime

__all__ = [
    "SCHEMA_ID",
    "BackendBinding",
    "ScenePlan",
    "ScenePreset",
    "ScenePresetError",
    "apply_scene_preset",
    "load_scene_preset",
    "reject_legacy_scene_keys",
    "resolve_scene_plan",
    "serialize_scene_preset",
]

SCHEMA_ID = "unigym.scene_preset/1"
"""The one preset document schema this parser accepts."""

_PRESET_KEYS = frozenset(
    {"schema", "name", "provider", "params", "world", "objects", "backend_params"}
)
_OBJECT_KEYS = frozenset({"types", "enabled", "sets", "demands", "policy", "filler", "seed"})
_CONTENT_FILL_KEYS = frozenset({"sets", "demands", "policy", "filler", "seed"})
_PROVIDER_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_PRESET_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

_RUN_MAPPING_KEYS = frozenset({"id", "provider", "params", "world", "objects", "backend_params"})
_INTERNAL_ENV_KEYS = ("scene_provider", "scene_provider_params", "scene_provider_preset")
_REMOVED_RUN_KEYS = ("scene_pack", "scene_bundle", "scenes")


class ScenePresetError(ValueError):
    """A scene preset document or resolution failed validation."""


@dataclass(frozen=True)
class BackendBinding:
    """The selected backend's data a backend-free preset resolver needs."""

    name: str
    asset_formats: tuple[str, ...]
    backend_param_keys: frozenset[str]
    known_backends: tuple[str, ...]


@dataclass(frozen=True)
class ScenePlan:
    """The fully merged, backend-selected preset resolution result."""

    preset_id: str | None
    provider: str | None
    provider_params: Mapping[str, JsonValue]
    world: AssetRef | None
    world_format: str | None
    world_path: str | None
    props: Mapping[str, JsonValue] | None
    content: Mapping[str, JsonValue] | None
    backend_params: Mapping[str, JsonValue]


@dataclass(frozen=True)
class ScenePreset:
    """One parsed ``unigym.scene_preset/1`` document."""

    schema: str
    name: str
    provider: str | None
    params: Mapping[str, JsonValue]
    world: AssetRef | None
    objects: Mapping[str, JsonValue] | None
    backend_params: Mapping[str, Mapping[str, JsonValue]]
    catalog_id: str | None = None

    def __post_init__(self) -> None:
        """Deep-freeze every mapping field so a preset cannot mutate."""
        object.__setattr__(self, "params", _freeze_mapping(self.params))
        object.__setattr__(self, "objects", _freeze_optional(self.objects))
        object.__setattr__(
            self,
            "backend_params",
            MappingProxyType(
                {str(name): _freeze_mapping(block) for name, block in self.backend_params.items()}
            ),
        )


def _is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    """Narrow an opaque value to a mapping with opaque keys and values."""
    return isinstance(value, Mapping)


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    """Narrow an opaque value to a non-string sequence."""
    return isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes))


def _thaw(value: object) -> JsonValue:
    """A mutable, validated deep copy of a JSON value (mapping proxies included)."""
    if _is_mapping(value):
        mapping: dict[str, JsonValue] = {}
        for key, item in value.items():
            mapping[str(key)] = _thaw(item)
        return mapping
    if _is_sequence(value):
        return [_thaw(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ScenePresetError(f"value {value!r} is not JSON data (type {type(value).__name__})")


def _freeze_json(value: object) -> JsonValue:
    """Freeze one validated JSON value, recursively for nested mappings."""
    if _is_mapping(value):
        proxy = MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
        # MappingProxyType is a read-only mapping; JsonValue's alias spells a
        # mutable dict, so this is the one frozen-data boundary cast.
        return cast(JsonValue, proxy)
    if _is_sequence(value):
        return [_freeze_json(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ScenePresetError(f"value {value!r} is not JSON data (type {type(value).__name__})")


def _freeze_mapping(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    """Freeze one JSON mapping, including its nested mappings."""
    return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})


def _freeze_optional(
    value: Mapping[str, JsonValue] | None,
) -> Mapping[str, JsonValue] | None:
    """Freeze an optional mapping."""
    return None if value is None else _freeze_mapping(value)


def _mapping_or_none(value: object, what: str) -> Mapping[str, JsonValue] | None:
    """Validate an optional mapping and return a plain dict copy."""
    if value is None:
        return None
    return _require_mapping(value, what)


def _require_mapping(value: object, what: str) -> dict[str, JsonValue]:
    """Validate a required mapping and return a plain dict copy."""
    if not _is_mapping(value):
        raise ScenePresetError(f"{what} must be a mapping, got {type(value).__name__}")
    thawed = _thaw(value)
    if not isinstance(thawed, dict):
        raise ScenePresetError(f"{what} must be a mapping, got {type(value).__name__}")
    return thawed


def _validate_initializer(block: object, where: str) -> None:
    """Validate the runtime initializer shape without resolving the registry."""
    from unigym.initializers.base import validate_initializer_shape

    try:
        validate_initializer_shape(block)
    except ValueError as err:
        raise ScenePresetError(f"{where}: {err}") from err


def _validate_base(block: object, where: str) -> None:
    """Validate ``base``: only key ``joint`` and (today) only ``joint: fixed``."""
    base = _require_mapping(block, f"{where} ({{joint: fixed}})")
    unknown = sorted(key for key in base if key != "joint")
    if unknown:
        raise ScenePresetError(
            f"{where} declares unsupported key(s) {unknown}; the only supported key is 'joint' "
            "(currently only 'joint: fixed')"
        )
    joint = base.get("joint", "fixed")
    if joint != "fixed":
        raise ScenePresetError(
            f"{where}.joint must be 'fixed', got {joint!r}; free-base spawning is not implemented"
        )


def _validate_objects(block: object, where: str, *, provider: str | None) -> dict[str, JsonValue]:
    """Validate the closed ``objects`` block and the content-only rule."""
    objects = _require_mapping(block, where)
    unknown = sorted(str(key) for key in objects if key not in _OBJECT_KEYS)
    if "provider" in objects:
        raise ScenePresetError(
            f"{where}: 'objects.provider' was renamed to 'objects.filler' -- "
            "declare the content filler with the 'filler' key"
        )
    if unknown:
        raise ScenePresetError(
            f"{where} declares unknown key(s) {unknown}; known keys: {sorted(_OBJECT_KEYS)}"
        )
    if provider is None:
        fill_keys = sorted(set(objects) & _CONTENT_FILL_KEYS)
        if fill_keys:
            raise ScenePresetError(
                f"{where}: content fill key(s) {fill_keys} need a composed scene; "
                "declare 'provider:' at the preset top level or drop the fill keys"
            )
    return objects


def _validate_backend_params(block: object, where: str) -> dict[str, Mapping[str, JsonValue]]:
    """Validate the ``backend_params`` table and generic backend fields."""
    table = _require_mapping(block, where)
    parsed: dict[str, Mapping[str, JsonValue]] = {}
    for backend, raw_block in table.items():
        name = str(backend)
        if not name:
            raise ScenePresetError(f"{where} declares an empty backend name")
        if not isinstance(raw_block, Mapping):
            raise ScenePresetError(
                f"{where}.{name} must be a mapping, got {type(raw_block).__name__}"
            )
        backend_block = {str(key): copy.deepcopy(item) for key, item in raw_block.items()}
        for legacy in ("mj_scene", "mj_scene_params", "mj_model"):
            if legacy in backend_block:
                raise ScenePresetError(
                    f"{where}.{name}.{legacy} was removed; declare the provider through "
                    "'provider:'/'params:' and the world asset through top-level 'world:'"
                )
        if "initializer" in backend_block:
            _validate_initializer(backend_block["initializer"], f"{where}.{name}.initializer")
        if "base" in backend_block:
            _validate_base(backend_block["base"], f"{where}.{name}.base")
        parsed[name] = backend_block
    return parsed


def _parse_world(value: object, where: str) -> AssetRef | None:
    """Parse an optional format-neutral ``world`` AssetRef mapping."""
    if value is None:
        return None
    mapping = _require_mapping(value, f"{where} ({{mjcf|urdf|usd: <path>}})")
    try:
        ref = AssetRef.from_mapping(mapping, where)
    except AssetRefError as err:
        raise ScenePresetError(f"{where}: {err}") from err
    if not any(getattr(ref, key) for key in ("mjcf", "urdf", "usd")):
        raise ScenePresetError(
            f"{where} must declare at least one non-empty format path (mjcf/urdf/usd)"
        )
    return ref


def _parse_document(
    doc: Mapping[object, object], *, source: str, catalog_id: str | None
) -> ScenePreset:
    """Validate a raw preset mapping into the frozen :class:`ScenePreset`."""
    raw = _require_mapping(doc, source)

    if "backend" in raw:
        raise ScenePresetError(
            f"{source}: top-level 'backend:' was removed; declare backend-only fields under "
            "'backend_params.<backend>' and the world asset under top-level 'world:'"
        )
    if "initializer" in raw:
        raise ScenePresetError(
            f"{source}: top-level 'initializer:' was removed; declare "
            "'backend_params.<backend>.initializer' with 'type:' or 'module:' and 'params:'"
        )
    if "scene" in raw:
        raise ScenePresetError(
            f"{source}: top-level 'scene:' was removed; use 'provider:'/'params:' for behavior "
            "and 'world:' for the world asset"
        )
    unknown = sorted(key for key in raw if key not in _PRESET_KEYS)
    if unknown:
        raise ScenePresetError(
            f"{source}: unknown key(s) {unknown}; known keys: {sorted(_PRESET_KEYS)}"
        )

    schema = raw.get("schema")
    if schema != SCHEMA_ID:
        raise ScenePresetError(f"{source}: schema {schema!r} must be {SCHEMA_ID!r}")

    name = raw.get("name")
    if not isinstance(name, str) or _PRESET_NAME_RE.fullmatch(name) is None:
        raise ScenePresetError(
            f"{source}: name {name!r} must be lowercase snake_case, e.g. 'tabletop'"
        )

    provider = raw.get("provider")
    if provider is not None:
        if not isinstance(provider, str) or _PROVIDER_NAME_RE.fullmatch(provider) is None:
            raise ScenePresetError(
                f"{source}: provider {provider!r} must use the adapter/robot grammar "
                "(lowercase snake_case, optionally dot-namespaced)"
            )

    params = _mapping_or_none(raw.get("params"), f"{source}: params") or {}
    if params and provider is None:
        raise ScenePresetError(
            f"{source}: 'params:' is only legal with 'provider:'; declare provider or drop params"
        )

    world = _parse_world(raw.get("world"), f"{source}: world")

    objects_raw = raw.get("objects")
    objects = (
        None
        if objects_raw is None
        else _validate_objects(objects_raw, f"{source}: objects", provider=provider)
    )

    backend_raw = raw.get("backend_params")
    backend_params = (
        {}
        if backend_raw is None
        else _validate_backend_params(backend_raw, f"{source}: backend_params")
    )

    return ScenePreset(
        schema=SCHEMA_ID,
        name=name,
        provider=provider,
        params=params,
        world=world,
        objects=objects,
        backend_params=backend_params,
        catalog_id=catalog_id,
    )


def _validate_provider_reference(
    preset: ScenePreset,
    *,
    ref: str,
    pack: str,
    runtime: PackRuntime,
) -> None:
    """Enforce same-pack ownership and manifest-schema validation at load.

    A catalog preset may name only a ``kind: scene`` provider declared by its
    own pack.  A dynamically registered pack provider has no declarable owner
    and is not referenceable from a catalog preset.  When the manifest item
    carries a ``params`` schema, the preset's declared values are validated
    (unknown keys, types and choices) here; a required parameter may still be
    supplied by a run-level override before dispatch.
    """
    provider = preset.provider
    if provider is None:
        return
    manifest = runtime.catalog.get(pack)
    match = next(
        (
            item
            for item in manifest.provides.get("providers", ())
            if item.kind == "scene" and item.id == provider
        ),
        None,
    )
    if match is None:
        owners = sorted(
            {
                entry.pack
                for entry in runtime.catalog.entries("providers")
                if entry.kind == "scene" and entry.id == provider
            }
        )
        if owners:
            raise ScenePresetError(
                f"scene preset {ref!r} declares provider {provider!r} owned by pack "
                f"{owners[0]!r}; a catalog preset may reference only a kind: scene "
                f"provider declared by its own pack {pack!r}"
            )
        raise ScenePresetError(
            f"scene preset {ref!r} declares provider {provider!r}, which pack {pack!r} "
            "does not declare as a kind: scene provider; catalog presets may reference "
            "only a manifest-declared same-pack scene provider"
        )
    if match.params is None:
        return
    try:
        schema = normalize_manifest_params(match.params)
    except SceneProviderParamError as err:
        raise ScenePresetError(
            f"scene preset {ref!r}: provider {provider!r} manifest params are invalid: {err}"
        ) from err
    try:
        fill_and_validate(
            schema,
            preset.params,
            provider=provider,
            preset=ref,
            require_all=False,
        )
    except SceneProviderParamError as err:
        raise ScenePresetError(f"scene preset {ref!r}: {err}") from err


def _scene_preset_identity(ref: object) -> tuple[str, str | None]:
    """Normalize a string catalog id or preset handle to ``(id, declared_pack)``."""
    from unigym.packs.packset import ScenePresetRef

    if isinstance(ref, ScenePresetRef):
        return ref.id, ref.pack
    if not isinstance(ref, str) or not ref:
        raise ScenePresetError(
            "scene preset ref must be a non-empty pack catalog id string or a "
            f"ScenePresetRef, got {ref!r}"
        )
    return ref, None


def load_scene_preset(ref: str | ScenePresetRef, runtime: PackRuntime) -> ScenePreset:
    """Resolve one exact ``scene_presets`` catalog id or handle to its preset."""
    ref_id, declared_pack = _scene_preset_identity(ref)
    if runtime is None:
        raise ScenePresetError(
            f"scene preset ref {ref_id!r} requires a pack runtime: pass runtime=... | "
            "hint: scene preset refs are exact PackRuntime 'scene_presets' catalog ids"
        )
    entry = runtime.catalog.entry("scene_presets", ref_id)
    if declared_pack is not None and entry.pack != declared_pack:
        raise ScenePresetError(
            f"scene preset id {ref_id!r} is declared by pack {entry.pack!r}, "
            f"not {declared_pack!r}"
        )
    path: Path = runtime.resource_for("scene_presets", ref_id)
    try:
        raw_doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as err:
        raise ScenePresetError(f"scene preset {ref_id!r} at {path} is not readable: {err}") from err
    except yaml.YAMLError as err:
        raise ScenePresetError(
            f"scene preset {ref_id!r} at {path} is not valid YAML: {err}"
        ) from err
    doc: object = {} if raw_doc is None else raw_doc
    if not _is_mapping(doc):
        raise ScenePresetError(
            f"scene preset {ref_id!r} at {path} must be a mapping, got {type(doc).__name__}"
        )
    preset = _parse_document(doc, source=f"scene preset {ref_id!r}", catalog_id=ref_id)
    _validate_provider_reference(preset, ref=ref_id, pack=entry.pack, runtime=runtime)
    return preset


def _merge(base: object, override: object) -> JsonValue:
    """Three-layer merge: override wins; mappings merge key-wise."""
    if override is None:
        return _thaw(base) if base is not None else None
    if _is_mapping(base) and _is_mapping(override):
        base_json = _thaw(base)
        override_json = _thaw(override)
        if isinstance(base_json, dict) and isinstance(override_json, dict):
            merged = dict(base_json)
            for key, value in override_json.items():
                merged[key] = _merge(base_json.get(key), value)
            return merged
    return _thaw(override)


def _normalize_objects_block(
    block: Mapping[str, JsonValue] | None, *, source: str
) -> tuple[Mapping[str, JsonValue] | None, Mapping[str, JsonValue] | None]:
    """Split one raw ``objects`` block into ``props`` and ``content``."""
    if block is None:
        return None, None
    wrapper: dict[str, JsonValue] = {"objects": _thaw(block)}
    normalized = normalize_content_keys(wrapper, source=source)
    props = normalized.get("props")
    content = normalized.get("content")
    props_map = props if isinstance(props, Mapping) and props else None
    content_map = content if isinstance(content, Mapping) and content else None
    return props_map, content_map


def _scene_preset_value(
    cfg: Mapping[str, JsonValue],
    *,
    runtime: PackRuntime | None,
    source: str,
) -> tuple[ScenePreset | None, Mapping[str, JsonValue], str | None]:
    """Resolve the run's ``scene_preset`` value to a preset and its overrides."""
    from unigym.errors import ConfigError

    raw = cfg.get("scene_preset")
    if raw is None:
        return None, {}, None
    if isinstance(raw, str):
        if not raw:
            raise ConfigError(f"{source}: scene_preset must be a non-empty catalog id string")
        if runtime is None:
            raise ConfigError(
                f"{source}: scene preset {raw!r} requires a pack runtime: pass runtime=... | "
                "hint: catalog presets resolve through the 'scene_presets' face"
            )
        return load_scene_preset(raw, runtime), {}, raw
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"{source}: scene_preset must be a catalog id string or a mapping, "
            f"got {type(raw).__name__}"
        )
    unknown = sorted(str(key) for key in raw if key not in _RUN_MAPPING_KEYS)
    if unknown:
        raise ConfigError(
            f"{source}: scene_preset mapping declares unknown key(s) {unknown}; "
            f"known keys: {sorted(_RUN_MAPPING_KEYS)}"
        )
    has_id = raw.get("id") is not None
    has_provider = raw.get("provider") is not None
    if has_id == has_provider:
        raise ConfigError(
            f"{source}: scene_preset mapping requires exactly one of 'id' or 'provider' "
            "(both or neither is ambiguous)"
        )
    overrides = {key: value for key, value in raw.items() if key not in ("id", "provider")}
    if has_id:
        preset_id = raw.get("id")
        if not isinstance(preset_id, str) or not preset_id:
            raise ConfigError(f"{source}: scene_preset.id must be a non-empty catalog id string")
        if runtime is None:
            raise ConfigError(
                f"{source}: scene preset {preset_id!r} requires a pack runtime: pass runtime=... | "
                "hint: catalog presets resolve through the 'scene_presets' face"
            )
        return load_scene_preset(preset_id, runtime), overrides, preset_id

    provider = raw.get("provider")
    if not isinstance(provider, str) or not provider:
        raise ConfigError(f"{source}: scene_preset.provider must be a non-empty provider name")
    objects = None
    if overrides.get("objects") is not None:
        objects = _validate_objects(
            overrides["objects"], f"{source}: scene_preset.objects", provider=provider
        )
    backend_raw = overrides.get("backend_params")
    backend_params = (
        {}
        if backend_raw is None
        else _validate_backend_params(backend_raw, f"{source}: scene_preset.backend_params")
    )
    params = _mapping_or_none(overrides.get("params"), f"{source}: scene_preset.params") or {}
    world = _parse_world(overrides.get("world"), f"{source}: scene_preset.world")
    preset = ScenePreset(
        schema=SCHEMA_ID,
        name="inline",
        provider=provider,
        params=params,
        world=world,
        objects=objects,
        backend_params=backend_params,
        catalog_id=None,
    )
    # The inline form's fields are already consumed; only shared fields remain.
    overrides = {
        key: value
        for key, value in overrides.items()
        if key not in ("params", "world", "objects", "backend_params")
    }
    return preset, overrides, None


def resolve_scene_plan(
    cfg: dict[str, JsonValue],
    *,
    runtime: PackRuntime | None,
    binding: BackendBinding,
    source: str = "",
) -> ScenePlan | None:
    """Resolve ``cfg['scene_preset']`` into a backend-selected :class:`ScenePlan`."""
    from unigym.errors import ConfigError

    if cfg.get("scene_preset") is None:
        return None
    label = source or "<run config>"
    # A direct caller may still carry the raw ``objects`` surface block; the
    # collector's single load point already normalized its own copy.
    work: dict[str, JsonValue] = (
        normalize_content_keys(copy.deepcopy(cfg), source=label) if "objects" in cfg else cfg
    )
    preset, overrides, preset_id = _scene_preset_value(work, runtime=runtime, source=label)
    assert preset is not None  # a present scene_preset always yields a preset

    provider = preset.provider
    params_merged = _merge(preset.params, overrides.get("params"))
    provider_params = params_merged if isinstance(params_merged, Mapping) else {}

    # world: preset -> scene_preset mapping -> run top-level world_base.
    world_merged: JsonValue = _thaw(preset.world.to_dict()) if preset.world is not None else {}
    if overrides.get("world") is not None:
        override_world = _parse_world(overrides["world"], f"{label}: scene_preset.world")
        assert override_world is not None
        world_merged = _merge(world_merged, override_world.to_dict())
    if work.get("world_base") is not None:
        world_merged = _merge(world_merged, work["world_base"])
    world_ref: AssetRef | None = None
    if isinstance(world_merged, Mapping) and world_merged:
        try:
            world_ref = AssetRef.from_mapping(world_merged, "world")
        except AssetRefError as err:
            raise ConfigError(f"{label}: {err}") from err

    # objects: preset -> scene_preset mapping -> run top-level props/content.
    preset_props, preset_content = _normalize_objects_block(
        preset.objects, source=f"{label}: preset objects"
    )
    override_objects = overrides.get("objects")
    run_props: Mapping[str, JsonValue] | None = None
    run_content: Mapping[str, JsonValue] | None = None
    if override_objects is not None:
        run_props, run_content = _normalize_objects_block(
            _validate_objects(
                override_objects, f"{label}: scene_preset.objects", provider=provider
            ),
            source=f"{label}: scene_preset objects",
        )
    props_merged = _merge(preset_props, run_props)
    props_merged = _merge(props_merged, work.get("props"))
    content_merged = _merge(preset_content, run_content)
    content_merged = _merge(content_merged, work.get("content"))
    props = props_merged if isinstance(props_merged, Mapping) and props_merged else None
    content = content_merged if isinstance(content_merged, Mapping) and content_merged else None

    # backend_params: preset -> scene_preset mapping -> run env.params overrides.
    backend_merged = _merge(dict(preset.backend_params), overrides.get("backend_params"))
    backend_table = backend_merged if isinstance(backend_merged, Mapping) else {}
    for name in backend_table:
        if str(name) not in binding.known_backends:
            raise ConfigError(
                f"{label}: backend_params declares unknown backend {name!r}; "
                f"known backends: {list(binding.known_backends)}"
            )
    selected = backend_table.get(binding.name)
    selected_block = (
        {str(key): copy.deepcopy(item) for key, item in selected.items()}
        if isinstance(selected, Mapping)
        else {}
    )
    env = work.get("env")
    params_obj = env.get("params") if isinstance(env, Mapping) else None
    if isinstance(params_obj, Mapping):
        # The run's env.params block merges LAST over the preset defaults and
        # the scene_preset mapping overrides; a partial ``initializer.params``
        # override keeps the preset's ``type`` (and every other param).
        for key in ("initializer", "base"):
            if key in params_obj:
                selected_block[key] = _merge(selected_block.get(key), params_obj[key])
    unknown_params = sorted(key for key in selected_block if key not in binding.backend_param_keys)
    if unknown_params:
        raise ConfigError(
            f"{label}: backend_params.{binding.name} declares unsupported key(s) "
            f"{unknown_params}; allowed keys: {sorted(binding.backend_param_keys)}"
        )
    if "initializer" in selected_block:
        _validate_initializer(selected_block["initializer"], f"{label}: initializer")
    if "base" in selected_block:
        _validate_base(selected_block["base"], f"{label}: base")

    if params_merged and provider is None:
        raise ConfigError(
            f"{label}: 'params:' requires 'provider:'; a content-only preset has no provider"
        )
    if content is not None:
        if provider is None:
            raise ConfigError(
                f"{label}: content fill needs a composed scene; declare 'provider:' or drop "
                "the objects fill keys"
            )
        filler = content.get("filler")
        if not isinstance(filler, str) or not filler:
            raise ConfigError(
                f"{label}: content fill requires 'objects.filler' -- declare the content filler"
            )

    world_format: str | None = None
    world_path: str | None = None
    if world_ref is not None:
        try:
            world_format = select_asset_format(world_ref, binding.asset_formats)
        except AssetRefError as err:
            raise ConfigError(
                f"{label}: no compatible world format for backend {binding.name!r}: {err}"
            ) from err
        world_path = getattr(world_ref, world_format)

    return ScenePlan(
        preset_id=preset_id,
        provider=provider,
        provider_params=_freeze_mapping(dict(provider_params)),
        world=world_ref,
        world_format=world_format,
        world_path=world_path,
        props=_freeze_optional(dict(props) if props is not None else None),
        content=_freeze_optional(dict(content) if content is not None else None),
        backend_params=_freeze_mapping(selected_block),
    )


def apply_scene_preset(
    cfg: dict[str, JsonValue],
    *,
    runtime: PackRuntime | None = None,
    binding: BackendBinding | None = None,
    source: str = "",
) -> dict[str, JsonValue]:
    """Route ``cfg['scene_preset']`` into the run-config consumption points."""
    from unigym.errors import ConfigError

    if cfg.get("scene_preset") is None:
        return cfg
    label = source or "<run config>"
    reject_legacy_scene_keys(cfg, source=label)
    if binding is None:
        raise ConfigError(
            f"{label}: scene_preset needs a BackendBinding (the run layer selects the backend "
            "before preset resolution)"
        )
    work = normalize_content_keys(copy.deepcopy(cfg), source=label) if "objects" in cfg else cfg
    plan = resolve_scene_plan(work, runtime=runtime, binding=binding, source=label)
    assert plan is not None  # scene_preset is present
    out = copy.deepcopy(work)
    out.pop("scene_preset", None)
    if plan.world_format is not None and plan.world_path is not None:
        out["world_base"] = {plan.world_format: plan.world_path}
    if plan.props is not None:
        out["props"] = _thaw(plan.props)
    if plan.content is not None:
        out["content"] = _thaw(plan.content)
    env = out.setdefault("env", {})
    if not isinstance(env, dict):
        raise ConfigError(f"{label}: env must be a mapping")
    params = env.setdefault("params", {})
    if not isinstance(params, dict):
        raise ConfigError(f"{label}: env.params must be a mapping")
    if plan.provider is not None:
        params["scene_provider"] = plan.provider
        params["scene_provider_params"] = _thaw(plan.provider_params)
        params["scene_provider_preset"] = plan.preset_id
    for key, value in plan.backend_params.items():
        params[key] = _thaw(value)
    return out


def serialize_scene_preset(
    cfg: Mapping[str, JsonValue], declaration: JsonValue
) -> dict[str, JsonValue]:
    """Write an effective config's scene declaration on the authoring surface.

    ``RunSpec.cfg`` is the *effective* config:
    :func:`apply_scene_preset` routed the preset into the internal
    ``env.params.scene_provider*`` keys, and those keys are refused as
    authoring keys by :func:`reject_legacy_scene_keys`.  This is the one
    place that reverses that routing for a serialized document: the internal
    keys are dropped and the run's original ``scene_preset:`` *declaration*
    is re-emitted, so the document reloads through the same authoring gate
    it was constructed from.

    Args:
        cfg: The effective run config (``RunSpec.cfg``); not mutated.
        declaration: The declaration the run was constructed from -- a
            catalog id string or the inline provider mapping, or ``None``
            when the run declared no preset.

    Returns:
        A deep copy of *cfg* without the internal routing keys and carrying
        ``scene_preset`` when a declaration was recorded.
    """
    thawed = _thaw(cfg)
    assert isinstance(thawed, dict)  # _thaw maps any Mapping to a fresh dict
    out = thawed
    env = out.get("env")
    params = env.get("params") if isinstance(env, dict) else None
    if isinstance(params, dict):
        for key in _INTERNAL_ENV_KEYS:
            params.pop(key, None)
    if declaration is not None:
        out["scene_preset"] = copy.deepcopy(declaration)
    return out


def reject_legacy_scene_keys(cfg: Mapping[str, JsonValue], *, source: str = "") -> None:
    """Fail fast on the removed run-config scene vocabulary (no aliases)."""
    from unigym.errors import ConfigError

    label = source or "<run config>"
    if "scene" in cfg:
        raise ConfigError(
            f"{label}: top-level 'scene:' was removed; declare "
            "'scene_preset: <pack>.<preset>' or the inline form "
            "'scene_preset: {provider: ..., params: {...}}'; there is no alias"
        )
    for key in _REMOVED_RUN_KEYS:
        if key in cfg:
            raise ConfigError(
                f"{label}: top-level {key!r} was removed with the scene vocabulary; "
                "declare 'scene_preset:' instead -- there is no alias"
            )
    env = cfg.get("env")
    params = env.get("params") if isinstance(env, Mapping) else None
    if not isinstance(params, Mapping):
        return
    for key in ("mj_scene", "mj_scene_params"):
        if key in params:
            raise ConfigError(
                f"{label}: env.params.{key} was removed; declare 'scene_preset:' (catalog or "
                "inline provider) and provider 'params:' instead -- there is no alias"
            )
    if "mj_model" in params:
        raise ConfigError(
            f"{label}: env.params.mj_model was removed; declare the run-level "
            "'world_base: {mjcf: <path>}' or a scene preset 'world:' instead -- "
            "there is no alias and no fallback"
        )
    for key in _INTERNAL_ENV_KEYS:
        if key in params:
            raise ConfigError(
                f"{label}: env.params.{key} is an internal routing key; declare 'scene_preset:' "
                "instead -- direct declaration is rejected"
            )
