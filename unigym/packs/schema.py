"""Public ``unigym.pack/1`` manifest schema and stored pack model.

This module owns the Phase 1 pack document contract: it validates the
``unigym.pack.yaml`` mapping, its declared faces and the resource/entrypoint
rules, then returns one immutable :class:`Pack` with a single ordered
:class:`PackItem` tuple.  Loading is pure: it reads YAML and checks filesystem
paths but never imports pack Python, so a manifest can be enumerated without
activating a pack.

The schema is the only manifest contract: discovery, pack-set lookup and
activation in :mod:`unigym.packs.loader` all read it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import cast

import yaml

from unigym._typing import JsonValue
from unigym.provenance import ProviderOrigin

from . import PackError

MANIFEST_NAME = "unigym.pack.yaml"
"""The single manifest file every pack root carries."""

SCHEMA_ID = "unigym.pack/1"
"""The one document schema this parser accepts."""


class Face(StrEnum):
    """The faces a ``unigym.pack/1`` manifest may declare."""

    RUNS = "runs"
    SCENE_PRESETS = "scene_presets"
    ASSETS = "assets"
    PROPS = "props"
    TERMINATIONS = "terminations"
    PROVIDERS = "providers"
    SELECTION = "selection"


MANIFEST_FACES: tuple[Face, ...] = tuple(Face)
"""The manifest face vocabulary, in declaration and render order."""

FACE_NAMES = tuple(face.value for face in MANIFEST_FACES)
"""String spelling of :data:`MANIFEST_FACES` for existing string callers."""

TRUST_LEVELS = ("content", "code")
"""Supported trust levels; only ``code`` may declare entrypoints."""


class ItemStatus(StrEnum):
    """Snapshot state of one stored pack item.

    ``FAILED`` is diagnostic-only: a failed activation raises and caches no
    snapshot, so a stored :class:`PackItem` never carries it.
    """

    DECLARED = "declared"
    REGISTERED = "registered"
    ADOPTED = "adopted"
    INERT = "inert"
    FAILED = "failed"


@dataclass(frozen=True)
class ResourceLocator:
    """A pack-relative resource path validated by the manifest loader."""

    path: str


@dataclass(frozen=True)
class EntrypointLocator:
    """A parsed ``module:attribute`` entrypoint with its exact spelling."""

    module: str
    attribute: str
    raw: str


Locator = ResourceLocator | EntrypointLocator
"""Exactly one locator is carried by every stored item."""

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


def coerce_face(value: Face | str) -> Face:
    """Return *value* as a :class:`Face` or fail with the known vocabulary."""
    if isinstance(value, Face):
        return value
    try:
        return Face(value)
    except ValueError:
        hint = ""
        if value == "scenes":
            hint = (
                "; scene presets are declared under 'scene_presets'; scene providers "
                "use providers with kind: scene"
            )
        raise PackError(
            f"unknown pack face {value!r}{hint}; known faces: {list(FACE_NAMES)}"
        ) from None


def _freeze_json(value: object) -> JsonValue:
    """Freeze one validated JSON value into read-only data, recursively.

    Nested mappings become ``MappingProxyType`` proxies and nested sequences
    become tuples, so no caller can mutate a declared provider parameter
    schema after the manifest is validated.  ``JsonValue`` spells mutable
    containers, so the frozen containers cross that boundary cast.
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

    This is the one freeze boundary used by :class:`PackItem`; the schema is
    exposed read-only (mappings are proxies, sequences are tuples).
    """
    frozen: dict[str, Mapping[str, JsonValue]] = {}
    for name, declaration in params.items():
        value = _freeze_json(declaration)
        if not isinstance(value, Mapping):
            raise PackError(f"provider params[{name!r}] must be a mapping")
        frozen[str(name)] = value
    return MappingProxyType(frozen)


def provider_param_declarations(
    params: Mapping[str, JsonValue],
) -> Mapping[str, Mapping[str, JsonValue]]:
    """Narrow deep-frozen params to named mapping declarations.

    The manifest validator already requires each parameter declaration to be a
    mapping; this check keeps hand-built :class:`PackItem` values honest at the
    one normalization boundary.
    """
    declarations: dict[str, Mapping[str, JsonValue]] = {}
    for name, declaration in params.items():
        if not isinstance(declaration, Mapping):
            raise PackError(f"provider params[{name!r}] must be a mapping")
        declarations[str(name)] = declaration
    return MappingProxyType(declarations)


def _scene_provider_params(
    raw: object, *, where: str, item_id: str
) -> Mapping[str, JsonValue] | None:
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
    # normalize_manifest_params validated every value as a declaration mapping;
    # the runtime JsonValue alias cannot express that recursive invariant.
    return cast(Mapping[str, JsonValue], freeze_provider_params(declarations))


def _parse_entrypoint(raw: str) -> EntrypointLocator:
    """Split one validated ``module:attribute`` spelling into a locator."""
    module, _, attribute = raw.partition(":")
    return EntrypointLocator(module=module, attribute=attribute, raw=raw)


@dataclass(frozen=True)
class PackItem:
    """One stored item a pack provides under a declared face.

    Every stored item carries exactly one :class:`Locator`.  ``kind`` names a
    provider kind for ``providers`` items and is optional elsewhere.  ``params``
    is deep-frozen JSON and ``None`` for items without parameters.

    ``origin`` is manifest provenance built once at load time.  ``status`` is
    snapshot metadata: it never mutates in place and a failed activation never
    stores ``FAILED``.

    ``resource``/``entrypoint``/``id`` are read-only projections over
    ``locator``/``name``; the stored field remains ``locator``.
    """

    pack: str
    face: Face
    kind: str | None
    name: str
    locator: Locator
    params: Mapping[str, JsonValue] | None
    origin: ProviderOrigin
    status: ItemStatus = ItemStatus.DECLARED

    def __post_init__(self) -> None:
        """Validate identity, locator ownership and status; deep-freeze params."""
        if not isinstance(self.pack, str) or not self.pack.strip():
            raise PackError(f"pack item pack must be a non-empty string, got {self.pack!r}")
        if not isinstance(self.name, str) or not self.name.strip():
            raise PackError(f"pack item name must be a non-empty string, got {self.name!r}")
        object.__setattr__(self, "face", coerce_face(self.face))
        if not isinstance(self.locator, (ResourceLocator, EntrypointLocator)):
            raise PackError(
                f"pack item {self.name!r} locator must be a ResourceLocator or "
                f"EntrypointLocator, got {type(self.locator).__name__}"
            )
        if self.face is Face.PROVIDERS and (
            not isinstance(self.kind, str) or not self.kind.strip()
        ):
            raise PackError(
                f"pack item {self.name!r} on the 'providers' face requires a "
                f"non-empty kind, got {self.kind!r}"
            )
        if not isinstance(self.status, ItemStatus):
            raise PackError(
                f"pack item {self.name!r} status must be an ItemStatus, got "
                f"{type(self.status).__name__}"
            )
        if self.status is ItemStatus.FAILED:
            raise PackError(
                f"pack item {self.name!r} cannot store ItemStatus.FAILED; "
                "failed activation raises and caches no snapshot"
            )
        if not isinstance(self.origin, ProviderOrigin):
            raise PackError(
                f"pack item {self.name!r} origin must be a ProviderOrigin, got "
                f"{type(self.origin).__name__}"
            )
        if self.params is not None:
            frozen = _freeze_json(self.params)
            if not isinstance(frozen, Mapping):
                raise PackError(f"pack item {self.name!r} params must be a mapping")
            object.__setattr__(self, "params", frozen)

    @property
    def id(self) -> str:
        """The manifest item id spelling (``name`` is the stored field)."""
        return self.name

    @property
    def resource(self) -> str | None:
        """The resource path when this item is resource-backed, else ``None``."""
        return self.locator.path if isinstance(self.locator, ResourceLocator) else None

    @property
    def entrypoint(self) -> str | None:
        """The exact entrypoint spelling when this item is code-backed, else ``None``."""
        return self.locator.raw if isinstance(self.locator, EntrypointLocator) else None


def pack_origin(pack: Pack, discovered_by: str) -> ProviderOrigin:
    """Build the pack-level origin for one discovery label."""
    return ProviderOrigin(
        kind="pack",
        pack=pack.name,
        pack_version=pack.version,
        root=str(pack.root),
        discovered_by=discovered_by,
        location_imported=discovered_by == "entry-point",
    )


def with_pack_origin(pack: Pack, origin: ProviderOrigin) -> Pack:
    """Return *pack* relabeled with *origin*, rebuilding item origins once."""
    if not isinstance(origin, ProviderOrigin):
        raise PackError(f"pack origin must be a ProviderOrigin, got {type(origin).__name__}")
    items = tuple(
        replace(
            item,
            origin=replace(
                item.origin,
                kind="manifest",
                pack=pack.name,
                pack_version=pack.version,
                root=str(pack.root),
                discovered_by=origin.discovered_by,
                location_imported=origin.location_imported,
            ),
        )
        for item in pack.items
    )
    return replace(pack, origin=origin, items=items)


@dataclass(frozen=True)
class Pack:
    """A validated ``unigym.pack/1`` document and its ordered stored items."""

    schema: str
    name: str
    version: str
    api: int
    trust: str
    root: Path
    entrypoint: EntrypointLocator | None
    requires: Mapping[str, str]
    items: tuple[PackItem, ...]
    origin: ProviderOrigin

    def __post_init__(self) -> None:
        """Freeze caller-facing mappings and the ordered item tuple."""
        object.__setattr__(self, "requires", MappingProxyType(dict(self.requires)))
        items = tuple(self.items)
        object.__setattr__(self, "items", items)
        for item in items:
            if not isinstance(item, PackItem):
                raise PackError(
                    f"pack {self.name!r} items must be PackItem values, got "
                    f"{type(item).__name__}"
                )
            if item.pack != self.name:
                raise PackError(
                    f"pack {self.name!r} contains item {item.name!r} owned by "
                    f"pack {item.pack!r}"
                )

    def items_for(
        self,
        *,
        face: Face | None = None,
        kind: str | None = None,
        status: ItemStatus | None = None,
    ) -> tuple[PackItem, ...]:
        """Filter this pack's items by face, kind and/or status in stored order."""
        return tuple(
            item
            for item in self.items
            if (face is None or item.face == coerce_face(face))
            and (kind is None or item.kind == kind)
            and (status is None or item.status == status)
        )

    def item(self, face: Face | str, name: str) -> PackItem:
        """Return the exact ``(face, name)`` item declared by this pack."""
        face_value = coerce_face(face)
        for item in self.items:
            if item.face == face_value and item.name == name:
                return item
        known = sorted(item.name for item in self.items if item.face == face_value)
        raise PackError(
            f"pack {self.name!r} declares no {face_value.value!r} id {name!r}; "
            f"known ids: {known}"
        )

    def provider(self, kind: str, name: str) -> PackItem:
        """Return this pack's ``providers`` item for an exact ``(kind, name)`` pair."""
        for item in self.items:
            if item.face is Face.PROVIDERS and item.kind == kind and item.name == name:
                return item
        known = sorted(
            item.name for item in self.items if item.face is Face.PROVIDERS and item.kind == kind
        )
        raise PackError(
            f"pack {self.name!r} has no {kind} id {name!r}; known {kind} ids "
            f"on {self.name!r}: {known}"
        )

    @property
    def face_counts(self) -> Mapping[Face, int]:
        """Declared item counts per face in manifest face order."""
        counts: dict[Face, int] = {}
        for item in self.items:
            counts[item.face] = counts.get(item.face, 0) + 1
        return MappingProxyType({face: counts[face] for face in MANIFEST_FACES if face in counts})


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


def _actuator_implementation_params(
    raw: object, *, where: str, item_id: str
) -> Mapping[str, JsonValue]:
    """Validate the flat ``law/backend/kind`` implementation selector."""
    if not isinstance(raw, Mapping):
        raise PackError(
            f"{where} (id {item_id!r}) actuator implementation requires a mapping "
            "'params' with law/backend (and optional kind)"
        )
    unknown = sorted(str(key) for key in raw if key not in {"law", "backend", "kind"})
    if unknown:
        raise PackError(
            f"{where} (id {item_id!r}) actuator implementation params declare unknown "
            f"keys {unknown}; known keys: ['backend', 'kind', 'law']"
        )
    for key in ("law", "backend"):
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise PackError(
                f"{where} (id {item_id!r}) actuator implementation params require a "
                f"non-empty string {key!r}"
            )
    backend = raw["backend"]
    from unigym.backends.registry import CANONICAL_BACKENDS

    if backend not in CANONICAL_BACKENDS:
        raise PackError(
            f"{where} (id {item_id!r}) actuator implementation backend {backend!r} is "
            f"not canonical; choose exactly one of {list(CANONICAL_BACKENDS)}"
        )
    kind = raw.get("kind")
    if kind is not None and (not isinstance(kind, str) or not kind.strip()):
        raise PackError(
            f"{where} (id {item_id!r}) actuator implementation params 'kind' must be "
            "a non-empty string"
        )
    frozen = _freeze_json(raw)
    assert isinstance(frozen, Mapping)
    return frozen


def _provided_item(
    root: Path,
    face: str,
    index: int,
    raw: object,
    trust: str,
    path: Path,
    *,
    pack_name: str,
    pack_version: str,
    pack_entrypoint: str | None,
) -> PackItem:
    """Validate one ``provides`` item and return its typed stored form."""
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

    locator: Locator | None = None
    resource: str | None = None
    if raw_resource is not None:
        if not isinstance(raw_resource, str) or not raw_resource.strip():
            raise PackError(f"{where} (id {item_id!r}) resource must be a non-empty string")
        _check_resource(root, raw_resource, f"{where} (id {item_id!r})")
        resource = raw_resource
        locator = ResourceLocator(path=raw_resource)

    entrypoint_text: str | None = None
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
        entrypoint_text = raw_entrypoint
        locator = _parse_entrypoint(raw_entrypoint)

    kind: str | None = None
    if raw_kind is not None:
        if not isinstance(raw_kind, str) or not raw_kind.strip():
            raise PackError(f"{where} (id {item_id!r}) kind must be a non-empty string")
        kind = raw_kind

    params: Mapping[str, JsonValue] | None = None
    if face == "scene_presets":
        if entrypoint_text is not None:
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
    elif face == "providers" and kind == "actuator_law":
        if raw_params is not None:
            raise PackError(
                f"{where} (id {item_id!r}) is an actuator law item and cannot "
                "declare 'params:'; declare the law fields in the law resource "
                "or the entrypoint's descriptor"
            )
    elif face == "providers" and kind == "actuator_implementation":
        if resource is not None:
            raise PackError(
                f"{where} (id {item_id!r}) is an actuator implementation item and "
                "must be entrypoint-backed only; drop 'resource:'"
            )
        if entrypoint_text is None:
            raise PackError(
                f"{where} (id {item_id!r}) is an actuator implementation item and "
                "requires 'entrypoint: module:factory'"
            )
        params = _actuator_implementation_params(raw_params, where=where, item_id=item_id)
    elif face == "providers" and kind == "scene":
        if resource is not None:
            raise PackError(
                f"{where} (id {item_id!r}) is a scene provider item and must be "
                "entrypoint-backed only; declare 'entrypoint: module:factory' and "
                "drop 'resource:'"
            )
        if entrypoint_text is None:
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

    if (resource is None) == (entrypoint_text is None):
        raise PackError(
            f"{where} (id {item_id!r}) must declare exactly one of " f"'resource' or 'entrypoint'"
        )
    assert locator is not None

    origin = ProviderOrigin(
        kind="manifest",
        pack=pack_name,
        pack_version=pack_version,
        root=str(root),
        discovered_by="unknown",
        face=face,
        item_id=item_id,
        resource=resource,
        entrypoint=entrypoint_text,
        activation_entrypoint=pack_entrypoint,
        location_imported=False,
    )
    return PackItem(
        pack=pack_name,
        face=Face(face),
        kind=kind,
        name=item_id,
        locator=locator,
        params=params,
        origin=origin,
    )


def _provides(
    root: Path,
    raw: object,
    trust: str,
    path: Path,
    *,
    pack_name: str,
    pack_version: str,
    pack_entrypoint: str | None,
) -> tuple[PackItem, ...]:
    """Validate the whole ``provides`` table into one ordered item tuple."""
    if not isinstance(raw, Mapping):
        raise PackError(f"pack provides must be a mapping ({path})")
    items: list[PackItem] = []
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
        seen: set[str] = set()
        for index, raw_item in enumerate(raw_items):
            item = _provided_item(
                root,
                face,
                index,
                raw_item,
                trust,
                path,
                pack_name=pack_name,
                pack_version=pack_version,
                pack_entrypoint=pack_entrypoint,
            )
            if item.name in seen:
                raise PackError(f"duplicate id {item.name!r} on pack face {face!r} in {path}")
            seen.add(item.name)
            items.append(item)
    return tuple(items)


def load_manifest_v1(root: str | Path) -> Pack:
    """Load and validate ``unigym.pack.yaml`` rooted at *root*.

    The loader never imports pack code; entrypoints are validated as strings
    only.  The returned :class:`Pack` carries every item as ``DECLARED`` and
    an unattributed pack origin; discovery relabels it with the real source.
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
    entrypoint: EntrypointLocator | None = None
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
        entrypoint = _parse_entrypoint(raw_entrypoint)

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
    items = _provides(
        root,
        raw_provides,
        trust,
        path,
        pack_name=name,
        pack_version=version,
        pack_entrypoint=None if entrypoint is None else entrypoint.raw,
    )

    pack = Pack(
        schema=SCHEMA_ID,
        name=name,
        version=version,
        api=api,
        trust=trust,
        root=root,
        entrypoint=entrypoint,
        requires=requires,
        items=items,
        origin=ProviderOrigin(
            kind="pack",
            pack=name,
            pack_version=version,
            root=str(root),
            discovered_by="unknown",
        ),
    )
    return pack


__all__ = [
    "FACE_NAMES",
    "MANIFEST_FACES",
    "MANIFEST_NAME",
    "PROVIDER_ID_RE",
    "SCHEMA_ID",
    "TRUST_LEVELS",
    "EntrypointLocator",
    "Face",
    "ItemStatus",
    "Locator",
    "Pack",
    "PackItem",
    "ResourceLocator",
    "coerce_face",
    "freeze_provider_params",
    "load_manifest_v1",
]
