"""Pack-owned actuator law descriptors and late-bound implementations.

``providers`` items with ``kind: actuator_law`` are loaded during ``ACTIVATE``
and become pure :class:`~unigym.actuator_law.ActuatorLawDescriptor` records.
``kind: actuator_implementation`` items stay inert until the run layer
materializes the selected backend: MuJoCo/mjWarp at ``BACKEND_BUILD``,
IsaacLab at ``ASSEMBLE``.  No implementation module is imported during
``ACTIVATE``/``SPEC``.

The manifest schema stays ``unigym.pack/1``; these are additive provider
kinds.  A law resource uses the ``unigym.actuator_law/1`` document schema.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from unigym.actuator_law import (
    ACTUATOR_LAW_REGISTRY,
    ActuatorLawDescriptor,
    attribute_actuator_law,
    register_actuator_law,
)
from unigym.packs import PackError
from unigym.packs.loader import (
    PackLoader,
    _add_to_sys_path,
    _import_boundary,
    _resolve_entrypoint,
)
from unigym.packs.schema import Face, Pack, PackItem
from unigym.provenance import ProviderOrigin
from unigym.registration import PackFace, require_registration

__all__ = [
    "LAW_SCHEMA_ID",
    "activate_actuator_law_providers",
    "actuator_implementation_items",
    "load_actuator_law_resource",
    "materialize_actuator_implementations",
]

LAW_SCHEMA_ID = "unigym.actuator_law/1"

_LAW_KEYS = frozenset(
    {
        "schema",
        "name",
        "config_keys",
        "gains",
        "parts",
        "drives",
        "backends",
        "compiled_plant",
        "backend_owned_limits",
        "command",
        "implementation",
        "implementation_id",
    }
)


def _string_list(value: object, what: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise PackError(f"{what} must be a list of strings, got {value!r}")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise PackError(f"{what} must contain non-empty strings, got {item!r}")
        out.append(item)
    return out


def _bool_word(value: object, what: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("allowed", "required", "true", "yes"):
            return True
        if lowered in ("forbidden", "false", "no", "none"):
            return False
    raise PackError(f"{what} must be a boolean or one of allowed/forbidden, got {value!r}")


def load_actuator_law_resource(path: str | Path, *, name: str) -> ActuatorLawDescriptor:
    """Load one ``unigym.actuator_law/1`` YAML document into a descriptor."""
    path = Path(path)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError) as err:
        raise PackError(f"unreadable actuator law resource {path}: {err}") from err
    if not isinstance(document, Mapping):
        raise PackError(f"actuator law resource {path} must be a mapping")
    unknown = sorted(str(key) for key in document if key not in _LAW_KEYS)
    if unknown:
        raise PackError(
            f"actuator law resource {path} declares unknown keys {unknown}; known keys: "
            f"{sorted(_LAW_KEYS)}"
        )
    schema = document.get("schema")
    if schema != LAW_SCHEMA_ID:
        raise PackError(f"actuator law resource {path} schema {schema!r} must be {LAW_SCHEMA_ID!r}")
    declared = document.get("name")
    if declared != name:
        raise PackError(
            f"actuator law resource {path} declares name {declared!r}; a manifest "
            f"actuator_law item id must equal the descriptor name {name!r}"
        )
    gains_raw = document.get("gains") or {}
    if not isinstance(gains_raw, Mapping):
        raise PackError(f"actuator law {name!r} gains must be a mapping")
    allowed_gains = _string_list(gains_raw.get("allowed"), f"actuator law {name!r} gains.allowed")
    required_gains = _string_list(
        gains_raw.get("required"), f"actuator law {name!r} gains.required"
    )
    compiled_raw = document.get("compiled_plant") or {}
    if not isinstance(compiled_raw, Mapping):
        raise PackError(f"actuator law {name!r} compiled_plant must be a mapping")
    compiled = {}
    for backend, plant in compiled_raw.items():
        if not isinstance(backend, str) or not backend.strip():
            raise PackError(f"actuator law {name!r} compiled_plant key must be a backend name")
        if not isinstance(plant, str) or not plant.strip():
            raise PackError(f"actuator law {name!r} compiled_plant[{backend!r}] must be a string")
        compiled[backend] = plant
    return ActuatorLawDescriptor(
        name=name,
        config_keys=frozenset(
            _string_list(document.get("config_keys"), f"law {name!r} config_keys")
        ),
        gain_keys=frozenset(allowed_gains),
        required_gains=tuple(required_gains),
        allows_parts=_bool_word(document.get("parts", "forbidden"), f"law {name!r} parts"),
        drives=frozenset(_string_list(document.get("drives"), f"law {name!r} drives")),
        backends=(
            None
            if document.get("backends") is None
            else frozenset(_string_list(document.get("backends"), f"law {name!r} backends"))
        ),
        compiled_plant=compiled,
        backend_owned_limits=frozenset(
            _string_list(document.get("backend_owned_limits"), f"law {name!r} backend_owned_limits")
        ),
        allows_command=_bool_word(document.get("command", "forbidden"), f"law {name!r} command"),
        requires_implementation=_bool_word(
            document.get("implementation", "forbidden"), f"law {name!r} implementation"
        ),
        implementation_id=(
            None
            if document.get("implementation_id") is None
            else str(document.get("implementation_id"))
        ),
    )


def _manifest_origin(runtime: PackLoader, manifest: Pack, item: PackItem) -> ProviderOrigin:
    return ProviderOrigin(
        kind="manifest",
        pack=manifest.name,
        pack_version=manifest.version,
        root=str(manifest.root),
        discovered_by=runtime.origins.get(manifest.name, "unknown"),
        face="providers",
        item_id=item.id,
        resource=item.resource,
        entrypoint=item.entrypoint,
        activation_entrypoint=manifest.entrypoint.raw if manifest.entrypoint is not None else None,
    )


def _law_items(manifest: Pack) -> tuple[PackItem, ...]:
    return manifest.items_for(face=Face.PROVIDERS, kind="actuator_law")


def activate_actuator_law_providers(manifest: Pack, runtime: PackLoader) -> tuple[str, ...]:
    """Load and register every ``kind: actuator_law`` item for one pack."""
    items = _law_items(manifest)
    if not items:
        return ()
    require_registration(PackFace.ACTUATOR_LAW, name=items[0].id)
    entrypoint_items = [item for item in items if item.entrypoint is not None]
    if entrypoint_items:
        python_dir = _import_boundary(manifest, entrypoint_items[0].entrypoint)
        _add_to_sys_path(python_dir)
    for item in items:
        where = f"pack {manifest.name!r} actuator law id {item.id!r}"
        origin = _manifest_origin(runtime, manifest, item)
        if item.resource is not None:
            descriptor = load_actuator_law_resource(
                runtime.resource(manifest.name, item.resource), name=item.id
            )
            register_actuator_law(descriptor, origin=origin)
            continue
        if item.entrypoint is None:  # pragma: no cover - schema enforces
            raise PackError(f"{where} must declare resource or entrypoint")
        try:
            target = _resolve_entrypoint(item.entrypoint)
        except Exception as err:
            raise PackError(
                f"{where} entrypoint {item.entrypoint!r} failed to import: {err}"
            ) from err
        if not callable(target):
            raise PackError(f"{where} entrypoint {item.entrypoint!r} target must be callable")
        result = target()
        if result is None:
            if item.id not in ACTUATOR_LAW_REGISTRY:
                raise PackError(
                    f"{where} entrypoint {item.entrypoint!r} returned None but did not "
                    "register its descriptor; return the descriptor or call "
                    "pack_api.register_actuator_law"
                )
            attribute_actuator_law(item.id, origin)
            continue
        if not isinstance(result, ActuatorLawDescriptor):
            raise PackError(
                f"{where} entrypoint {item.entrypoint!r} must return an "
                f"ActuatorLawDescriptor, got {type(result).__name__}"
            )
        if result.name != item.id:
            raise PackError(
                f"{where} entrypoint {item.entrypoint!r} returned descriptor "
                f"{result.name!r}; the descriptor name must equal the item id"
            )
        register_actuator_law(result, origin=origin)
    return tuple(item.id for item in items)


def actuator_implementation_items(manifest: Pack) -> tuple[PackItem, ...]:
    """Every ``kind: actuator_implementation`` item declared by *manifest*."""
    return manifest.items_for(face=Face.PROVIDERS, kind="actuator_implementation")


def _implementation_selector(item: PackItem) -> tuple[str, str, str | None]:
    params = item.params
    if not isinstance(params, Mapping):
        raise PackError(f"actuator implementation {item.id!r} requires flat law/backend params")
    law = params.get("law")
    backend = params.get("backend")
    kind = params.get("kind")
    if not isinstance(law, str) or not isinstance(backend, str):
        raise PackError(
            f"actuator implementation {item.id!r} params must declare string law/backend"
        )
    if kind is not None and not isinstance(kind, str):
        raise PackError(f"actuator implementation {item.id!r} params kind must be a string")
    return law, backend, kind


def materialize_actuator_implementations(
    runtime: PackLoader | None, backend: str
) -> tuple[str, ...]:
    """Import and register the implementation items selected for *backend*.

    Called by the run layer at the backend's legal phase.  Implementation
    modules are imported here and nowhere earlier.
    """
    if runtime is None:
        return ()
    from unigym.actuators.implementations import (
        default_implementation_kind,
        ensure_core_implementations,
        register_actuator_implementation,
    )
    from unigym.capabilities.traits import CANONICAL_BACKENDS

    if backend not in CANONICAL_BACKENDS:
        raise PackError(
            f"cannot materialize actuator implementations for unknown backend "
            f"{backend!r}; canonical backends are {list(CANONICAL_BACKENDS)}"
        )
    ensure_core_implementations(backend)
    materialized: list[str] = []
    for pack in runtime.pack_set.packs():
        manifest = pack
        items = actuator_implementation_items(manifest)
        if not items:
            continue
        owned_laws = {item.id for item in _law_items(manifest)}
        python_dir: Path | None = None
        for item in items:
            law, declared_backend, kind = _implementation_selector(item)
            where = f"pack {manifest.name!r} actuator implementation id {item.id!r}"
            if declared_backend not in CANONICAL_BACKENDS:
                raise PackError(
                    f"{where} declares backend {declared_backend!r} which is not "
                    f"canonical; choose exactly one of {list(CANONICAL_BACKENDS)}"
                )
            if declared_backend != backend:
                continue
            if law not in owned_laws:
                raise PackError(
                    f"{where} declares law {law!r} which this pack does not own; an "
                    "actuator_implementation must name a kind: actuator_law item "
                    "declared by the same pack"
                )
            if item.entrypoint is None:  # pragma: no cover - schema enforces
                raise PackError(f"{where} requires an entrypoint")
            if python_dir is None:
                python_dir = _import_boundary(manifest, item.entrypoint)
                _add_to_sys_path(python_dir)
            try:
                target = _resolve_entrypoint(item.entrypoint)
            except Exception as err:
                raise PackError(
                    f"{where} entrypoint {item.entrypoint!r} failed to import: {err}"
                ) from err
            if not callable(target):
                raise PackError(f"{where} entrypoint {item.entrypoint!r} target must be callable")
            try:
                register_actuator_implementation(
                    law,
                    backend,
                    target,
                    kind=kind or default_implementation_kind(backend),
                    entrypoint=item.entrypoint,
                    origin=_manifest_origin(runtime, manifest, item),
                )
            except Exception as err:
                raise PackError(
                    f"{where} failed to register implementation for ({law!r}, "
                    f"{backend!r}): {err}"
                ) from err
            materialized.append(item.id)
    return tuple(materialized)
