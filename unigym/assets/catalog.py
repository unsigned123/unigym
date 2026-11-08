"""Asset catalog: the rebuildable fact index over asset manifests (v086 §5).

The catalog answers "what exists" -- it is derived from ``unigym.asset/1``
manifests and generator metadata and can be rebuilt at any time.  It never
records a selection: that is the decision layer (v086 §6), which pins a
catalog digest and resolves to explicit ids.

This module is L0.  It imports the standard library plus
:mod:`unigym.assets.manifest` only, and it takes manifests as already-loaded
data.  Filesystem discovery and pack mounting belong to
``tools/build_asset_catalog.py``, which sits above the library.

Determinism is part of the contract: keys are sorted, the three index lists
(``tags``/``uses``/``variants``) are sorted and deduplicated, and no
absolute path, mtime or timestamp may enter a document -- so two builds of
the same input produce byte-identical JSON and the same digest.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from .manifest import KINDS, ManifestError, split_asset_id

SCHEMA = "unigym.catalog/1"
"""The catalog document schema id."""

CARRIERS = frozenset({"file", "generator"})
"""The two asset carriers (v086 §2.1): fixed payload vs computed payload."""

CATALOG_NAME = "catalog.json"
"""The catalog document file name inside an output directory."""

DIGEST_NAME = "catalog.digest"
"""The digest sidecar file name inside an output directory."""

_ABSOLUTE_PATH_RE = re.compile(r"^(?:[/\\]|[A-Za-z]:[/\\])")
"""An absolute POSIX path, an absolute Windows path, or a rooted path."""

_VOLATILE_KEYS = frozenset({
    "mtime", "mtime_ns", "ctime", "ctime_ns", "timestamp", "built_at",
    "created_at", "generated_at", "build_time",
})
"""Keys that would make the catalog build-dependent; they fail closed."""

_INDEX_KEYS = frozenset({
    "id", "kind", "carrier", "pack", "tags", "uses", "variants", "files",
    "capabilities",
})
"""The catalog-owned keys; every other manifest key is passed through."""


class CatalogError(ValueError):
    """A catalog entry or document is malformed; the message names the fix."""


def _error(where: str, detail: str) -> CatalogError:
    """Build a fail-closed teaching error naming the entry and the problem."""
    return CatalogError(f"{where}: {detail}" if where else detail)


def _label(entry: object, index: int) -> str:
    """Name one input entry for an error message."""
    if isinstance(entry, Mapping):
        asset_id = entry.get("id")
        if isinstance(asset_id, str) and asset_id:
            return f"entry #{index} ({asset_id})"
    return f"entry #{index}"


def _string_list(where: str, key: str, value: object) -> list[str]:
    """Normalize one index list: sorted, deduplicated, strings only."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise _error(where, f"{key} must be a list of strings, got "
                            f"{type(value).__name__}")
    items = list(value)
    for item in items:
        if not isinstance(item, str) or not item:
            raise _error(where, f"{key} entries must be non-empty strings, "
                                f"got {item!r}")
    return sorted(set(items))


def _scan_volatile(value: object, where: str, path: str) -> None:
    """Reject volatile keys and absolute paths anywhere in a document (R3)."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in _VOLATILE_KEYS:
                raise _error(
                    where,
                    f"{child_path!r} is a volatile key; the catalog is "
                    "rebuilt and digested, so mtime/timestamp data cannot "
                    "enter it -- remove the key")
            _scan_volatile(child, where, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_volatile(child, where, f"{path}[{index}]")
    elif isinstance(value, str) and _ABSOLUTE_PATH_RE.match(value):
        raise _error(
            where,
            f"{path!r} holds the absolute path {value!r}; the catalog "
            "carries relative facts only -- make it relative to the pack")


def _files_count(files: object, where: str) -> int:
    """Count the payload paths a manifest's ``files`` mapping ships."""
    if isinstance(files, bool) or not isinstance(files, int) or files < 0:
        raise _error(where, f"files must be a non-negative int count, got "
                            f"{files!r}; the collector counts the manifest's "
                            "file payloads")
    return files


def _normalize_entry(raw: object, index: int) -> dict:
    """Normalize one manifest-like entry into the catalog entry shape."""
    where = _label(raw, index)
    if not isinstance(raw, Mapping):
        raise _error(
            where,
            f"a catalog entry must be a mapping, got {type(raw).__name__}; "
            "pass the manifest dicts the collector produced")
    asset_id = raw.get("id")
    if not isinstance(asset_id, str) or not asset_id:
        raise _error(
            where,
            "is missing a non-empty 'id'; every entry names its asset "
            "(e.g. 'pack:acme/assets/generators/objects/x/y')")
    kind = raw.get("kind")
    if not isinstance(kind, str) or kind not in KINDS:
        raise _error(
            where,
            f"is missing a legal 'kind', got {kind!r}; legal kinds are "
            f"{sorted(KINDS)} (the manifest kind set stays closed)")
    carrier = raw.get("carrier", "file")
    if carrier not in CARRIERS:
        raise _error(
            where,
            f"carrier must be one of {sorted(CARRIERS)}, got {carrier!r} "
            "(v086 §2.1: file = fixed payload, generator = computed payload)")
    try:
        namespace, _ = split_asset_id(asset_id)
    except ManifestError as exc:
        raise _error(where, str(exc)) from exc
    declared_pack = raw.get("pack")
    if declared_pack is not None and declared_pack != namespace:
        raise _error(
            where,
            f"declares pack {declared_pack!r} but its id namespace is "
            f"{namespace!r}; the id is the single source of the pack name")

    tags = _string_list(where, "tags", raw.get("tags"))
    uses = _string_list(where, "uses", raw.get("uses"))
    variants = _string_list(where, "variants", raw.get("variants"))
    if carrier == "file" and variants:
        raise _error(
            where,
            "declares 'variants' on a file carrier; variants index the "
            "computed payload of a generator (carrier: generator)")

    extras = {
        key: value
        for key, value in raw.items()
        if key not in _INDEX_KEYS and key != "schema"
    }
    entry: dict = {
        "id": asset_id,
        "kind": kind,
        "carrier": carrier,
        "pack": namespace,
        "tags": tags,
        "uses": uses,
        "files": _files_count(raw.get("files", 0), where),
        "capabilities": raw.get("capabilities", {}),
    }
    if carrier == "generator":
        entry["variants"] = variants
    entry.update(extras)
    _scan_volatile(entry, where, "")
    return entry


def build_catalog(entries: Iterable[dict]) -> dict:
    """Normalize manifest-like entries into the catalog document.

    Args:
        entries: Manifest dicts (or generator entries built from registry
            metadata).  Every entry needs an ``id`` and a manifest ``kind``;
            ``carrier`` defaults to ``file``.

    Returns:
        The catalog document ``{"schema": SCHEMA, "entries": [...]}`` with
        entries sorted by id and every index list sorted.

    Raises:
        CatalogError: An entry is malformed (and the message names it), or
            two entries share one asset id (ids are unique, v086 R0).
    """
    normalized: list[dict] = []
    seen: dict[str, int] = {}
    for index, raw in enumerate(entries):
        entry = _normalize_entry(raw, index)
        asset_id = entry["id"]
        first = seen.get(asset_id)
        if first is not None:
            raise _error(
                _label(raw, index),
                f"duplicate asset id {asset_id!r}, also declared by entry "
                f"#{first}; asset ids are unique across packs (v086 R0) -- "
                "fix the two manifests or their directory paths")
        seen[asset_id] = index
        normalized.append(entry)
    normalized.sort(key=lambda item: item["id"])
    return {"schema": SCHEMA, "entries": normalized}


def canonical_json(catalog: dict) -> str:
    """Render a catalog document as canonical JSON text (sorted, trailing NL).

    Sorted keys, UTF-8 text, two-space indent and a trailing newline; the
    digest and the on-disk file are both derived from exactly these bytes.
    """
    return json.dumps(
        catalog,
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"


def catalog_digest(catalog: dict) -> str:
    """Return ``sha256:<hex>`` over the catalog's canonical JSON bytes."""
    payload = canonical_json(catalog).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def find(catalog: dict, kind: str | None = None, tags: Iterable[str] | None = None,
         pack: str | None = None, uses: Iterable[str] | None = None) -> list[str]:
    """Return the sorted asset ids matching every given filter (AND).

    Args:
        catalog: A catalog document from :func:`build_catalog`.
        kind: Require this manifest kind (``model``, ``texture_set``, ...).
        tags: Require every listed tag (subset test, not equality).
        pack: Require this pack namespace.
        uses: Require every listed asset id in the entry's ``uses`` (subset
            test).

    Returns:
        The matching asset ids, sorted.
    """
    wanted_tags = _filter_set("tags", tags)
    wanted_uses = _filter_set("uses", uses)
    matches: list[str] = []
    for entry in catalog.get("entries", ()):
        if kind is not None and entry.get("kind") != kind:
            continue
        if pack is not None and entry.get("pack") != pack:
            continue
        if wanted_tags and not wanted_tags <= set(entry.get("tags", ())):
            continue
        if wanted_uses and not wanted_uses <= set(entry.get("uses", ())):
            continue
        matches.append(entry["id"])
    return sorted(matches)


def _filter_set(key: str, value: Iterable[str] | None) -> frozenset[str]:
    """Normalize one ``find`` filter into a subset-test set."""
    if value is None:
        return frozenset()
    if isinstance(value, (str, bytes)):
        raise _error("", f"{key} filter must be a list of strings, not a bare "
                        f"string; pass [{value!r}] to require one {key} entry")
    return frozenset(value)


def write_catalog(catalog: dict, out_dir: str | Path) -> Path:
    """Write ``catalog.json`` and ``catalog.digest`` into ``out_dir``.

    Returns:
        The path of the written ``catalog.json``.  The directory is created
        when absent; nothing else on disk is touched.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CATALOG_NAME
    path.write_text(canonical_json(catalog), encoding="utf-8")
    (directory / DIGEST_NAME).write_text(
        catalog_digest(catalog) + "\n", encoding="utf-8")
    return path


def load_catalog(out_dir: str | Path) -> dict:
    """Read ``catalog.json`` from ``out_dir`` and return the document.

    Raises:
        CatalogError: The file is absent, unreadable, not JSON, or does not
            carry the ``unigym.catalog/1`` shape.
    """
    path = Path(out_dir) / CATALOG_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(
            f"{path}: cannot read the catalog ({exc}); run "
            "`python tools/build_asset_catalog.py --out <dir>` first") from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CatalogError(
            f"{path}: the catalog is not valid JSON ({exc}); rebuild it with "
            "`python tools/build_asset_catalog.py --out <dir>`") from exc
    if not isinstance(document, Mapping):
        raise CatalogError(f"{path}: the catalog must be a JSON object")
    if document.get("schema") != SCHEMA:
        raise CatalogError(
            f"{path}: schema must be {SCHEMA!r}, got {document.get('schema')!r};"
            " rebuild the catalog instead of editing it")
    if not isinstance(document.get("entries"), list):
        raise CatalogError(
            f"{path}: 'entries' must be a list; rebuild the catalog instead "
            "of editing it")
    return document
