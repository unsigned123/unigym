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

import yaml

from . import PackError

MANIFEST_NAME = "unigym.pack.yaml"
"""The single manifest file every pack root carries."""

SCHEMA_ID = "unigym.pack/1"
"""The one document schema this parser accepts."""

FACE_NAMES = (
    "runs",
    "scenes",
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
_ITEM_KEYS = frozenset({"id", "resource", "entrypoint", "kind"})
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION_RE = re.compile(r"^[0-9]+(\.[0-9]+)*$")
_ENTRYPOINT_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"
)


@dataclass(frozen=True)
class ProvidedItem:
    """One item a pack provides under a declared face.

    Exactly one of ``resource`` and ``entrypoint`` is set; ``kind`` optionally
    names a provider kind (for example ``asset_generator``).
    """

    id: str
    resource: str | None
    entrypoint: str | None
    kind: str | None


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

    raw_resource = raw.get("resource")
    raw_entrypoint = raw.get("entrypoint")
    raw_kind = raw.get("kind")

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

    if (resource is None) == (entrypoint is None):
        raise PackError(
            f"{where} (id {item_id!r}) must declare exactly one of " f"'resource' or 'entrypoint'"
        )

    kind: str | None = None
    if raw_kind is not None:
        if not isinstance(raw_kind, str) or not raw_kind.strip():
            raise PackError(f"{where} (id {item_id!r}) kind must be a non-empty string")
        kind = raw_kind

    return ProvidedItem(id=item_id, resource=resource, entrypoint=entrypoint, kind=kind)


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
    "SCHEMA_ID",
    "TRUST_LEVELS",
    "PackManifestV1",
    "ProvidedItem",
    "load_manifest_v1",
]
