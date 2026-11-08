"""Asset manifests: the closed ``unigym.asset/1`` schema (v086 §4).

The manifest is the L0 fact sheet of ONE asset: its stable id, its
carrier kind, the files it ships, its capability vocabulary, its license
and its provenance.  It is deliberately scene-blind -- ``scene``,
``layout``, ``style`` and ``engine`` are forbidden at ANY depth (R3),
because scene knowledge belongs to L1.

This module validates SHAPE only.  It never touches the filesystem and
never resolves a pack: file existence, id-vs-directory agreement and
``uses`` resolvability belong to the catalog and the guards (v086 §9).
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import yaml

#: the manifest schema id every document must carry verbatim.
SCHEMA = "unigym.asset/1"

#: the closed carrier-kind set (v086 §4); anything else is an error.
KINDS = frozenset({"model", "texture_set", "mesh", "material"})

#: keys that would smuggle L1 scene knowledge into an L0 manifest (R3).
FORBIDDEN_KEYS = frozenset({"scene", "layout", "style", "engine"})

#: keys every manifest must carry (v086 R2/R4, narrowed by the v091 card).
REQUIRED_KEYS = ("schema", "id", "kind", "files", "capabilities", "license",
                 "provenance")

#: a namespace is lowercase snake/dash; the path below it keeps its
#: on-disk spelling (case included), so only the namespace is checked here.
_NAMESPACE_RE = re.compile(r"[a-z][a-z0-9_-]*")

#: file-map role names are snake_case (the closed-registry charset rule).
_ROLE_RE = re.compile(r"[a-z][a-z0-9_]*")

#: the tag name after ``pack:<ns>/tag=``; dots are legal (file stems).
_TAG_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*")

_TAG_MARKER = "/tag="


class ManifestError(ValueError):
    """A manifest is malformed; the message names the file, key and fix."""


def _error(where: str, key: str | None, detail: str) -> ManifestError:
    """Build a fail-closed teaching error naming file, key and problem."""
    location = ": ".join(part for part in (str(where), f"key {key!r}" if key
                                           else "") if part)
    return ManifestError(f"{location}: {detail}" if location else detail)


def _asset_id_problem(value: object) -> str | None:
    """Return a teaching message when ``value`` is not a legal asset id."""
    if not isinstance(value, str):
        return (f"must be an asset id string, got {type(value).__name__} "
                "(e.g. 'pack:acme/objects/lettuce/Lettuce004')")
    if not value.startswith("pack:"):
        return ("must start with 'pack:' -- an asset id is "
                "'pack:<ns>/<path>' or 'pack:<ns>/generator/<name>', "
                "e.g. 'pack:acme/objects/lettuce/Lettuce004'")
    rest = value[len("pack:"):]
    namespace, sep, remainder = rest.partition("/")
    if not sep:
        return ("must carry a namespace AND a path ('pack:<ns>/<path>'), "
                "e.g. 'pack:acme/objects/lettuce/Lettuce004'")
    if not namespace:
        return ("has an empty namespace; write 'pack:<ns>/<path>' with a "
                "lowercase snake/dash namespace, e.g. 'pack:acme/...'")
    if _NAMESPACE_RE.fullmatch(namespace) is None:
        return (f"has a non-conforming namespace {namespace!r}; namespaces "
                "are lowercase snake/dash only (uppercase is rejected), "
                "e.g. 'pack:acme/...'")
    if "\\" in value:
        return ("contains a backslash; asset ids are '/'-separated "
                "('pack:<ns>/<path>'), never Windows paths")
    if remainder.startswith("/") or remainder.endswith("/") or "//" in remainder:
        return ("has an empty path segment (leading, trailing or doubled "
                "'/'); write a plain relative path, e.g. "
                "'pack:acme/objects/lettuce'")
    if any(segment in ("", ".", "..") for segment in remainder.split("/")):
        return ("contains a '.' or '..' segment; an asset id is a relative "
                "path that cannot escape its pack ('pack:<ns>/<path>')")
    return None


def _require_asset_id(value: object, where: str, key: str, rule: str) -> str:
    """Validate ``value`` as an asset id and return it."""
    problem = _asset_id_problem(value)
    if problem is not None:
        raise _error(where, key, f"{value!r} {problem} ({rule})")
    return value


def _path_problem(value: object) -> str | None:
    """Return a teaching message when ``value`` is not a safe relative path."""
    if not isinstance(value, str) or not value:
        return (f"must be a non-empty relative path string, got {value!r}; "
                "paths are relative to the asset directory")
    if "\\" in value:
        return (f"{value!r} contains a backslash; manifest paths use '/' "
                "separators")
    if value.startswith("/"):
        return (f"{value!r} is absolute; manifest paths are relative to the "
                "asset directory (no leading '/')")
    if any(segment in ("", ".", "..") for segment in value.split("/")):
        return (f"{value!r} contains an empty, '.' or '..' segment; a "
                "manifest cannot escape its asset directory")
    return None


def _tag_problem(value: object) -> str | None:
    """Return a teaching message when ``value`` is not a namespaced tag (R7)."""
    if not isinstance(value, str) or not value:
        return (f"must be a non-empty string, got {value!r}; tags are "
                "namespaced 'pack:<ns>/tag=<name>', e.g. "
                "'pack:acme/tag=wood'")
    if not value.startswith("pack:"):
        return (f"{value!r} is a bare tag; R7 requires every tag to be "
                "namespaced -- write 'pack:<ns>/tag=<name>', e.g. "
                "'pack:acme/tag=wood'")
    namespace, sep, name = value[len("pack:"):].partition(_TAG_MARKER)
    if (not sep or _NAMESPACE_RE.fullmatch(namespace) is None
            or _TAG_NAME_RE.fullmatch(name) is None):
        return (f"{value!r} is not a legal namespaced tag; R7 requires "
                "'pack:<ns>/tag=<name>' with a lowercase namespace, e.g. "
                "'pack:acme/tag=wood'")
    return None


def _scan_forbidden(value: object, where: str, path: str) -> None:
    """Recursively reject L1 keys anywhere in the document (R3)."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in FORBIDDEN_KEYS:
                raise _error(
                    where, str(key),
                    f"is forbidden at any depth (R3): found at {child_path!r}. "
                    "An asset manifest carries L0 facts only; scene/layout/"
                    "style/engine knowledge belongs to L1 -- remove the key "
                    "and keep it in the scene or content layer")
            _scan_forbidden(child, where, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_forbidden(child, where, f"{path}[{index}]")


def _validate_files(files: object, where: str) -> None:
    """Validate the ``files`` role -> path(s) mapping (shape only)."""
    if not isinstance(files, Mapping) or not files:
        raise _error(
            where, "files",
            "must be a non-empty mapping role -> relative path(s), e.g. "
            "{model: model.xml, visual: [visual/base.png]}")
    for role, payload in files.items():
        if not isinstance(role, str) or _ROLE_RE.fullmatch(role) is None:
            raise _error(
                where, "files",
                f"role {role!r} must be lowercase snake_case, e.g. "
                "'model' or 'texture_base'")
        if isinstance(payload, str):
            paths: list[object] = [payload]
        elif isinstance(payload, (list, tuple)):
            if not payload:
                raise _error(where, "files",
                             f"role {role!r} lists no path; remove the role or "
                             "give it at least one relative path")
            paths = list(payload)
        else:
            raise _error(
                where, "files",
                f"role {role!r} must map to a relative path string or a list "
                f"of them, got {type(payload).__name__}")
        for relative in paths:
            problem = _path_problem(relative)
            if problem is not None:
                raise _error(where, "files", f"role {role!r}: {problem}")


def _validate_aliases(aliases: object, asset_id: str, where: str) -> None:
    """Validate the optional alias list: ids, no self-reference, no dupes."""
    if not isinstance(aliases, (list, tuple)):
        raise _error(
            where, "aliases",
            f"must be a list of asset ids, got {type(aliases).__name__}; "
            "write aliases: [pack:<ns>/<old-path>]")
    seen: set[str] = set()
    for alias in aliases:
        _require_asset_id(alias, where, "aliases", "R8 aliases are asset ids")
        if alias == asset_id:
            raise _error(
                where, "aliases",
                f"{alias!r} is the manifest's own id; an alias must be an "
                "OLD id (a rename tombstone, R8), never the current one")
        if alias in seen:
            raise _error(where, "aliases",
                         f"{alias!r} is listed twice; aliases are unique (R8)")
        seen.add(alias)


def _validate_uses(uses: object, where: str) -> None:
    """Validate the optional ``uses`` list: every entry is an asset id."""
    if not isinstance(uses, (list, tuple)):
        raise _error(
            where, "uses",
            f"must be a list of asset ids, got {type(uses).__name__}; "
            "write uses: [pack:<ns>/<path>]")
    for target in uses:
        _require_asset_id(target, where, "uses",
                          "R4 a cross-asset reference is an asset id, never a "
                          "relative path")


def _validate_tags(tags: object, where: str) -> None:
    """Validate the optional tag list (R7 namespace rule)."""
    if not isinstance(tags, (list, tuple)):
        raise _error(where, "tags",
                     f"must be a list of namespaced tags, got "
                     f"{type(tags).__name__}")
    for tag in tags:
        problem = _tag_problem(tag)
        if problem is not None:
            raise _error(where, "tags", problem)


def _validate(doc: object, where: str = "") -> dict:
    """Run every schema gate on ``doc`` and return it unchanged."""
    if not isinstance(doc, Mapping):
        raise _error(
            where, None,
            f"an asset manifest must be a mapping, got {type(doc).__name__}; "
            f"expected the keys {list(REQUIRED_KEYS)}")
    _scan_forbidden(doc, where, "")
    for key in REQUIRED_KEYS:
        if key not in doc:
            raise _error(
                where, key,
                "is required by schema 'unigym.asset/1'; add it to the "
                f"manifest (required keys: {', '.join(REQUIRED_KEYS)})")
    if doc["schema"] != SCHEMA:
        raise _error(
            where, "schema",
            f"must be {SCHEMA!r} exactly, got {doc['schema']!r}; bump the "
            "asset schema deliberately, never by typo")
    asset_id = _require_asset_id(
        doc["id"], where, "id",
        "R0 an asset id is 'pack:<ns>/<path>' or 'pack:<ns>/generator/<name>'")
    kind = doc["kind"]
    if not isinstance(kind, str) or kind not in KINDS:
        raise _error(
            where, "kind",
            f"{kind!r} is not a known kind; legal kinds are "
            f"{sorted(KINDS)} (v086 §4)")
    _validate_files(doc["files"], where)
    if not isinstance(doc["capabilities"], Mapping):
        raise _error(
            where, "capabilities",
            f"must be a mapping (capability vocabulary), got "
            f"{type(doc['capabilities']).__name__}; the pack owns the "
            "sub-keys, this gate only requires a mapping")
    if not isinstance(doc["license"], str) or not doc["license"]:
        raise _error(where, "license",
                     f"must be a non-empty string, got {doc['license']!r}; "
                     "state the license (e.g. 'CC-BY-4.0')")
    provenance = doc["provenance"]
    if not isinstance(provenance, Mapping):
        raise _error(
            where, "provenance",
            f"must be a mapping carrying 'origin', got "
            f"{type(provenance).__name__}")
    origin = provenance.get("origin")
    if not isinstance(origin, str) or not origin:
        raise _error(
            where, "provenance",
            "must carry a non-empty 'origin' string (registry / grade / "
            "source, v086 §4); add e.g. origin: 'registry lightwheel'")
    if doc.get("tags") is not None:
        _validate_tags(doc["tags"], where)
    if doc.get("aliases") is not None:
        _validate_aliases(doc["aliases"], asset_id, where)
    if doc.get("uses") is not None:
        _validate_uses(doc["uses"], where)
    return doc


def validate_manifest(doc: dict) -> dict:
    """Validate one ``unigym.asset/1`` manifest; return the document as-is.

    Shape only: no filesystem touch, no pack resolution.  Raises
    :class:`ManifestError` on the first violated rule, naming the key and
    the fix (v086 §9 fail-closed/teach style).
    """
    return _validate(doc)


def load_manifest(path: str | Path) -> dict:
    """Read one ``asset.yaml`` file, validate it and return the document.

    The only file read in this module.  I/O and yaml errors become
    :class:`ManifestError` naming the file and the fix, so a broken
    manifest fails closed at the loader with a teaching message.
    """
    location = str(path)
    try:
        with open(path, encoding="utf-8") as handle:
            doc = yaml.safe_load(handle)
    except OSError as exc:
        raise ManifestError(
            f"{location}: cannot read the asset manifest ({exc}); every "
            "asset directory carries one 'asset.yaml' -- check the path and "
            "the pack mount") from exc
    except yaml.YAMLError as exc:
        raise ManifestError(
            f"{location}: asset manifest is not valid yaml ({exc}); fix the "
            "file -- one asset gets one 'asset.yaml' mapping") from exc
    if not isinstance(doc, Mapping):
        raise ManifestError(
            f"{location}: an asset manifest must be a yaml mapping with keys "
            f"{list(REQUIRED_KEYS)}, got {type(doc).__name__}; check the "
            "file content, not just its name")
    return _validate(doc, where=location)


def is_asset_id(value: str) -> bool:
    """Return whether ``value`` is an asset id by spelling alone (no I/O)."""
    return _asset_id_problem(value) is None


def split_asset_id(value: str) -> tuple[str, str]:
    """Split a legal asset id into ``(namespace, path-or-generator-name)``.

    Raises :class:`ManifestError` when the spelling is illegal, using the
    same teaching messages as :func:`validate_manifest`.
    """
    problem = _asset_id_problem(value)
    if problem is not None:
        raise ManifestError(f"invalid asset id {value!r}: {problem}")
    assert isinstance(value, str)
    namespace, _, remainder = value[len("pack:"):].partition("/")
    return namespace, remainder
