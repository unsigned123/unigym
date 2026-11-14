"""Public ``unigym.pack/1`` manifest schema and loader.

This module owns the Phase 1 pack document contract: it validates the
``unigym.pack.yaml`` mapping, its declared provider faces and the
resource/entrypoint rules.  Loading is pure: it reads YAML and checks
filesystem paths but never imports pack Python, so a manifest can be
enumerated without activating a pack.

The schema is the only manifest contract: discovery, catalog lookup and
activation in :mod:`unigym.packs.runtime` all read it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml

from unigym._typing import JsonValue

from . import PackError

MANIFEST_NAME = "unigym.pack.yaml"
"""The single manifest file every pack root carries."""

SCHEMA_ID = "unigym.pack/1"
"""The one document schema this parser accepts."""

FACE_NAMES = (
    "runs",
    "scene_presets",
    "tasks",
    "assets",
    "benchmarks",
    "props",
    "terminations",
    "providers",
    "selection",
)
"""Provider faces a ``unigym.pack/1`` manifest may declare."""

TRUST_LEVELS = ("content", "code")
"""Supported trust levels; only ``code`` may declare entrypoints."""

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "name",
        "version",
        "api",
        "trust",
        "entrypoint",
        "requires",
        "provides",
    }
)
_ITEM_KEYS = frozenset({"id", "resource", "entrypoint", "kind", "params"})
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION_RE = re.compile(r"^[0-9]+(\.[0-9]+)*$")
_ENTRYPOINT_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"
)

#: provider ids use the adapter/robot grammar: dotted lowercase snake_case.
PROVIDER_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")


def _freeze_json(value: object) -> JsonValue:
    """Freeze one validated JSON value into read-only data, recursively.

    Nested mappings become ``MappingProxyType`` proxies and nested sequences
    become tuples, so no caller can mutate a declared provider parameter
    schema after the manifest (or catalog) is validated.  ``JsonValue`` spells
    mutable containers, so the frozen containers cross that boundary cast.
    """
    if isinstance(value, Mapping):
        proxy = MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
        return cast(JsonValue, proxy)
    if isinstance(value, (list, tuple)):
        frozen = tuple(_freeze_json(item) for item in value)
        return cast(JsonValue, frozen)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise PackError(f"provider params value {value!r} is not JSON data")


def freeze_provider_params(
    params: Mapping[str, Mapping[str, JsonValue]],
) -> Mapping[str, Mapping[str, JsonValue]]:
    """Deep-freeze one provider parameter-schema declaration mapping.

    This is the one freeze boundary shared by :class:`ProvidedItem` and
    :class:`~unigym.packs.catalog.CatalogEntry`; both expose the schema
    read-only (mappings are proxies, sequences are tuples).
    """
    frozen: dict[str, Mapping[str, JsonValue]] = {}
    for name, declaration in params.items():
        value = _freeze_json(declaration)
        if not isinstance(value, Mapping):
            raise PackError(f"provider params[{name!r}] must be a mapping")
        frozen[str(name)] = value
    return MappingProxyType(frozen)


def _scene_provider_params(
    raw: object, *, where: str, item_id: str
) -> Mapping[str, Mapping[str, JsonValue]] | None:
    """Validate one manifest ``params:`` declaration and deep-freeze it."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise PackError(
            f"{where} (id {item_id!r}) params must be a mapping of "
            f"parameter declarations, got {type(raw).__name__}"
        )
    # The normalizer is the one parameter-schema truth.  It validates the
    # declaration shape while this loader owns the manifest-level wrapping.
    from unigym.scenes.provider_params import SceneProviderParamError, normalize_manifest_params

    declarations = cast(Mapping[str, Mapping[str, JsonValue]], raw)
    try:
        normalize_manifest_params(declarations)
    except SceneProviderParamError as err:
        raise PackError(f"{where} (id {item_id!r}) params are invalid: {err}") from err
    return freeze_provider_params(declarations)


@dataclass(frozen=True)
class ProvidedItem:
    """One item a pack provides under a declared face.

    Exactly one of ``resource`` and ``entrypoint`` is set; ``kind`` optionally
    names a provider kind (for example ``asset_generator``, ``adapter``,
    ``robot`` or ``scene``).  ``params`` is the inline parameter schema of a
    ``providers`` item with ``kind: scene``; it is deep-frozen and ``None`` for
    every other item.

    A robot provider is resource-backed or entrypoint-backed::

        - id: acme.arm
          kind: robot
          resource: robots/acme_arm.yaml

        - id: acme.factory
          kind: robot
          entrypoint: acme_robot:make_profile

    A scene provider is entrypoint-backed and may declare an inline parameter
    schema.  That schema is deep-frozen recursively: mappings are read-only
    proxies and sequences are tuples, so neither the outer item nor a nested
    ``choices``/``default`` sequence can be mutated after validation::

        - id: acme.table_top
          kind: scene
          entrypoint: acme_scenes:table_top
          params:
            style: {type: int, default: 1}
    """

    id: str
    resource: str | None
    entrypoint: str | None
    kind: str | None
    params: Mapping[str, Mapping[str, JsonValue]] | None = None

    def __post_init__(self) -> None:
        """Deep-freeze the optional provider parameter schema."""
        if self.params is not None:
            object.__setattr__(self, "params", freeze_provider_params(self.params))


@dataclass(frozen=True)
class PackManifestV1:
    """A validated ``unigym.pack/1`` document rooted at ``root``."""

    schema: str
    name: str
    version: str
    api: int
    trust: str
    entrypoint: str | None
    requires: Mapping[str, str]
    provides: Mapping[str, tuple[ProvidedItem, ...]]
    root: Path

    def __post_init__(self) -> None:
        """Freeze the caller-facing mappings so the manifest cannot mutate."""
        object.__setattr__(self, "requires", MappingProxyType(dict(self.requires)))
        provides = {face: tuple(items) for face, items in self.provides.items()}
        object.__setattr__(self, "provides", MappingProxyType(provides))


def _read_document(root: Path) -> dict[str, object]:
    """Read the raw manifest mapping or raise :class:`PackError`."""
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise PackError(f"missing pack manifest: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError) as err:
        raise PackError(f"unreadable pack manifest: {path}: {err}") from err
    if not isinstance(document, Mapping):
        raise PackError(f"pack manifest must be a mapping: {path}")
    return dict(document)


def _check_resource(root: Path, resource: str, where: str) -> None:
    """Require *resource* to exist inside the manifest root."""
    rel = Path(resource)
    if not rel.parts:
        raise PackError(f"{where} resource {resource!r} must be a non-empty relative path")
    if rel.is_absolute():
        raise PackError(f"{where} resource {resource!r} must be relative to the manifest root")
    if any(part == ".." for part in rel.parts):
        raise PackError(f"{where} resource {resource!r} must not contain '..'")
    base = root.resolve()
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise PackError(f"{where} resource {resource!r} escapes the manifest root {base}") from None
    if not candidate.exists():
        raise PackError(f"{where} resource {resource!r} does not exist under {base}")


def _provided_item(
    root: Path,
    face: str,
    index: int,
    raw: object,
    trust: str,
    path: Path,
) -> ProvidedItem:
    """Validate one ``provides`` item and return its typed form."""
    where = f"pack face {face!r} item {index} in {path}"
    if not isinstance(raw, Mapping):
        raise PackError(f"{where} must be a mapping")
    unknown = sorted(str(key) for key in raw if key not in _ITEM_KEYS)
    if unknown:
        raise PackError(
            f"unknown provided-item key(s) {unknown} in {where}; "
            f"known keys: {sorted(_ITEM_KEYS)}"
        )

    item_id = raw.get("id")
    if not isinstance(item_id, str) or not item_id.strip():
        raise PackError(f"{where} requires a non-empty string id")
    if face == "providers" and PROVIDER_ID_RE.fullmatch(item_id) is None:
        raise PackError(
            f"{where} provider id {item_id!r} must use the adapter/robot grammar "
            "[a-z][a-z0-9_]*(?:\\.[a-z][a-z0-9_]*)*"
        )

    raw_resource = raw.get("resource")
    raw_entrypoint = raw.get("entrypoint")
    raw_kind = raw.get("kind")
    raw_params = raw.get("params")

    resource: str | None = None
    if raw_resource is not None:
        if not isinstance(raw_resource, str) or not raw_resource.strip():
            raise PackError(f"{where} (id {item_id!r}) resource must be a non-empty string")
        _check_resource(root, raw_resource, f"{where} (id {item_id!r})")
        resource = raw_resource

    entrypoint: str | None = None
    if raw_entrypoint is not None:
        if not isinstance(raw_entrypoint, str) or not _ENTRYPOINT_RE.fullmatch(raw_entrypoint):
            raise PackError(
                f"{where} (id {item_id!r}) entrypoint {raw_entrypoint!r} "
                f"must be 'module:attribute'"
            )
        if trust != "code":
            raise PackError(
                f"{where} (id {item_id!r}) declares entrypoint {raw_entrypoint!r} "
                f"but the pack trust is {trust!r}; entrypoints require trust: code"
            )
        entrypoint = raw_entrypoint

    kind: str | None = None
    if raw_kind is not None:
        if not isinstance(raw_kind, str) or not raw_kind.strip():
            raise PackError(f"{where} (id {item_id!r}) kind must be a non-empty string")
        kind = raw_kind

    params: Mapping[str, Mapping[str, JsonValue]] | None = None
    if face == "scene_presets":
        if entrypoint is not None:
            raise PackError(
                f"{where} (id {item_id!r}) is a scene preset item and must be "
                "resource-backed only; declare 'resource: scene_presets/<file>.yaml' "
                "(scene providers are declared on the 'providers' face with kind: scene)"
            )
        if kind is not None:
            raise PackError(
                f"{where} (id {item_id!r}) is a scene preset item and must be "
                "resource-backed only; drop 'kind:' (scene providers are declared "
                "on the 'providers' face with kind: scene)"
            )
        if raw_params is not None:
            raise PackError(
                f"{where} (id {item_id!r}) is a scene preset item and cannot declare "
                "'params:'; provider parameter schemas belong to a 'providers' item "
                "with kind: scene"
            )
        if resource is None:
            raise PackError(
                f"{where} (id {item_id!r}) is a scene preset item and must be "
                "resource-backed; declare 'resource: scene_presets/<file>.yaml'"
            )
    elif face == "providers" and kind == "scene":
        if resource is not None:
            raise PackError(
                f"{where} (id {item_id!r}) is a scene provider item and must be "
                "entrypoint-backed only; declare 'entrypoint: module:factory' and "
                "drop 'resource:'"
            )
        if entrypoint is None:
            raise PackError(
                f"{where} (id {item_id!r}) is a scene provider item and must be "
                "entrypoint-backed; declare 'entrypoint: module:factory'"
            )
        params = _scene_provider_params(raw_params, where=where, item_id=item_id)
    elif raw_params is not None:
        raise PackError(
            f"unknown provided-item key(s) ['params'] in {where} (id {item_id!r}); "
            "'params' is legal only on a 'providers' item with kind: scene"
        )

    if (resource is None) == (entrypoint is None):
        raise PackError(
            f"{where} (id {item_id!r}) must declare exactly one of " f"'resource' or 'entrypoint'"
        )

    return ProvidedItem(
        id=item_id,
        resource=resource,
        entrypoint=entrypoint,
        kind=kind,
        params=params,
    )


def _provides(
    root: Path,
    raw: object,
    trust: str,
    path: Path,
) -> dict[str, tuple[ProvidedItem, ...]]:
    """Validate the whole ``provides`` table."""
    if not isinstance(raw, Mapping):
        raise PackError(f"pack provides must be a mapping ({path})")
    provides: dict[str, tuple[ProvidedItem, ...]] = {}
    for face, raw_items in raw.items():
        if face not in FACE_NAMES:
            if face == "scenes":
                raise PackError(
                    f"unknown pack face 'scenes' in {path}; scene presets are declared "
                    "under 'scene_presets'; scene providers are declared under "
                    "'providers' with kind: scene"
                )
            raise PackError(
                f"unknown pack face {face!r} in {path}; known faces: {list(FACE_NAMES)}"
            )
        if not isinstance(raw_items, list):
            raise PackError(f"pack face {face!r} must be a list of items ({path})")
        items: list[ProvidedItem] = []
        seen: set[str] = set()
        for index, raw_item in enumerate(raw_items):
            item = _provided_item(root, face, index, raw_item, trust, path)
            if item.id in seen:
                raise PackError(f"duplicate id {item.id!r} on pack face {face!r} in {path}")
            seen.add(item.id)
            items.append(item)
        provides[face] = tuple(items)
    return provides


def load_manifest_v1(root: str | Path) -> PackManifestV1:
    """Load and validate ``unigym.pack.yaml`` rooted at *root*.

    The loader never imports pack code; entrypoints are validated as strings
    only.
    """
    root = Path(root).resolve()
    path = root / MANIFEST_NAME
    document = _read_document(root)

    unknown = [str(key) for key in document if key not in _TOP_LEVEL_KEYS]
    if unknown:
        raise PackError(
            f"unknown pack manifest key(s) {sorted(unknown)} in {path}; "
            f"known keys: {sorted(_TOP_LEVEL_KEYS)}"
        )

    schema = document.get("schema")
    if schema != SCHEMA_ID:
        raise PackError(f"pack manifest schema {schema!r} must be {SCHEMA_ID!r} ({path})")

    name = document.get("name")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise PackError(f"pack name {name!r} must match [a-z][a-z0-9_]* ({path})")

    version = document.get("version")
    if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
        raise PackError(f"pack version {version!r} must be dotted digits, e.g. 1.2.0 ({path})")

    api = document.get("api")
    if isinstance(api, bool) or not isinstance(api, int) or api <= 0:
        raise PackError(f"pack api {api!r} must be a positive integer ({path})")

    trust = document.get("trust")
    if trust not in TRUST_LEVELS:
        raise PackError(
            f"pack {name!r} trust {trust!r} must be one of {list(TRUST_LEVELS)} ({path})"
        )

    raw_entrypoint = document.get("entrypoint")
    entrypoint: str | None = None
    if raw_entrypoint is not None:
        if not isinstance(raw_entrypoint, str) or not _ENTRYPOINT_RE.fullmatch(raw_entrypoint):
            raise PackError(
                f"pack {name!r} entrypoint {raw_entrypoint!r} must be "
                f"'module:attribute' ({path})"
            )
        if trust != "code":
            raise PackError(
                f"pack {name!r} declares entrypoint {raw_entrypoint!r} with trust "
                f"{trust!r}; entrypoints require trust: code ({path})"
            )
        entrypoint = raw_entrypoint

    raw_requires = document.get("requires")
    if raw_requires is None:
        requires: dict[str, str] = {}
    elif not isinstance(raw_requires, Mapping):
        raise PackError(f"pack {name!r} requires must be a mapping ({path})")
    else:
        requires = {}
        for key, value in raw_requires.items():
            if not isinstance(key, str) or not key.strip():
                raise PackError(f"pack {name!r} requires keys must be non-empty strings ({path})")
            if not isinstance(value, str) or not value.strip():
                raise PackError(
                    f"pack {name!r} requires value for {key!r} must be a "
                    f"non-empty string ({path})"
                )
            requires[key] = value

    raw_provides = document.get("provides")
    if raw_provides is None:
        raise PackError(f"pack {name!r} must declare a provides mapping ({path})")
    provides = _provides(root, raw_provides, trust, path)

    return PackManifestV1(
        schema=SCHEMA_ID,
        name=name,
        version=version,
        api=api,
        trust=trust,
        entrypoint=entrypoint,
        requires=requires,
        provides=provides,
        root=root,
    )


__all__ = [
    "FACE_NAMES",
    "MANIFEST_NAME",
    "PROVIDER_ID_RE",
    "SCHEMA_ID",
    "TRUST_LEVELS",
    "PackManifestV1",
    "ProvidedItem",
    "freeze_provider_params",
    "load_manifest_v1",
]
