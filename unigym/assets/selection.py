"""Selection specs and locks: the decision layer over the asset catalog (v086 §6).

The catalog answers "what exists" and is rebuildable from the manifests.  A
selection answers "which of those did we choose, and by what rule" -- a
decision that cannot be derived from the manifests, so it is recorded in git
as a spec (the intent) plus a lock (the resolved ids pinned to one catalog
digest).

Resolution is deterministic by construction:

* candidates are filtered by ``kind``, by ``tags`` (every listed tag must be
  present -- the AND form) and ``any_tags`` (at least one must be present --
  the OR form; an object pool that unions several categories needs it),
  present) and by ``pack``;
* candidates are ordered by ``(pack, id)`` -- never by directory walk order;
* without ``seed`` the first ``choose`` ids of that order win;
* with ``seed`` the choice is ``random.Random(seed).sample(ids, choose)`` over
  that stable order, sorted, so the same seed always yields the same ids.

An empty or too-small result fails closed, naming the selector and the catalog
size: a selection that matches nothing is a broken decision, not an empty one.

This module is the L0 decision layer: the standard library, :mod:`yaml` and
:mod:`unigym.assets.catalog` only.  It never scans the filesystem and never
revises the catalog -- it pins a catalog digest and records what was chosen.
"""
from __future__ import annotations

import random
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

import yaml

from .catalog import catalog_digest

SCHEMA = "unigym.selection/1"
"""The selection lock schema id."""

SPEC_KEYS = ("name", "selector")
"""The keys a selection spec carries (a closed set, v086 §6)."""

SELECTOR_KEYS = ("kind", "tags", "any_tags", "pack", "choose", "seed")
"""The keys a selector carries; ``choose`` is the only required one."""

LOCK_KEYS = ("schema", "catalog_digest", "resolved", "rule")
"""The keys a selection lock carries; nothing else enters a lock."""

_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")
"""A selection name is snake_case, like every registry name."""

_NAMESPACE_RE = re.compile(r"[a-z][a-z0-9_-]*")
"""A tag namespace / pack name is lowercase snake/dash."""

_TAG_MARKER = "/tag="
"""The marker separating a tag namespace from its name (R7)."""


class SelectionError(ValueError):
    """A selection spec, lock or resolution is invalid; the message names the fix."""


def _error(where: str, detail: str) -> SelectionError:
    """Build a fail-closed teaching error naming the subject and the problem."""
    return SelectionError(f"{where}: {detail}" if where else detail)


def _catalog(catalog: object) -> dict:
    """Validate the catalog document shape and return it as a plain dict.

    The catalog itself is built elsewhere; this gate only rejects a document
    that could not have come from :func:`unigym.assets.catalog.build_catalog`,
    so a resolution never fails later with a bare ``KeyError``.
    """
    if not isinstance(catalog, Mapping):
        raise _error(
            "",
            f"the catalog must be a mapping, got {type(catalog).__name__}; load "
            "it with unigym.assets.catalog.load_catalog")
    document = dict(catalog)
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise _error(
            "",
            "'entries' must be a list; rebuild the catalog with "
            "`python tools/build_asset_catalog.py --out <dir>`")
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str):
            raise _error(
                f"catalog entry #{index}",
                "must be a mapping with a string 'id'; rebuild the catalog "
                "instead of editing it")
    return document


def _tag_problem(value: object) -> str | None:
    """Return a teaching message when ``value`` is not a namespaced tag (R7)."""
    if not isinstance(value, str) or not value:
        return (f"must be a non-empty string, got {value!r}; tags are "
                "'pack:<ns>/tag=<name>', e.g. 'pack:acme/tag=tiles'")
    if not value.startswith("pack:"):
        return (f"{value!r} is a bare tag; R7 requires every tag to be "
                "namespaced -- write 'pack:<ns>/tag=<name>'")
    namespace, sep, name = value[len("pack:"):].partition(_TAG_MARKER)
    if not sep or _NAMESPACE_RE.fullmatch(namespace) is None or not name:
        return (f"{value!r} is not a legal namespaced tag; write "
                "'pack:<ns>/tag=<name>' (R7)")
    return None


def _validate_selector(selector: object) -> dict:
    """Validate one selector block; return it unchanged."""
    if not isinstance(selector, Mapping):
        raise _error(
            "selector",
            f"must be a mapping with keys {list(SELECTOR_KEYS)}, got "
            f"{type(selector).__name__}")
    unknown = sorted(set(selector) - set(SELECTOR_KEYS))
    if unknown:
        raise _error(
            "selector",
            f"unknown key(s) {unknown}; legal keys are {list(SELECTOR_KEYS)} "
            "-- a selector has exactly these filters and the choice size")
    if "choose" not in selector:
        raise _error(
            "selector",
            "'choose' is required: the number of ids the selection pins "
            "(e.g. choose: 1)")
    choose = selector["choose"]
    if isinstance(choose, bool) or not isinstance(choose, int) or choose < 1:
        raise _error(
            "selector",
            f"'choose' must be a positive int, got {choose!r}; it is the "
            "number of ids the selection pins")
    if "kind" in selector:
        kind = selector["kind"]
        if not isinstance(kind, str) or not kind:
            raise _error(
                "selector",
                f"'kind' must be a non-empty manifest kind, got {kind!r}; "
                "e.g. kind: texture_set")
    if "pack" in selector:
        pack = selector["pack"]
        if not isinstance(pack, str) or _NAMESPACE_RE.fullmatch(pack) is None:
            raise _error(
                "selector",
                f"'pack' must be a lowercase pack namespace, got {pack!r}; "
                "e.g. pack: acme")
    if "tags" in selector:
        tags = selector["tags"]
        if isinstance(tags, (str, bytes)) or not isinstance(tags, (list, tuple)):
            raise _error(
                "selector",
                f"'tags' must be a list of namespaced tags, got {tags!r}; "
                "pass a list even for one tag")
        for tag in tags:
            problem = _tag_problem(tag)
            if problem is not None:
                raise _error("selector", problem)
        if len(set(tags)) != len(tags):
            raise _error(
                "selector",
                f"'tags' lists a tag twice: {list(tags)!r}; tags are a set")
    if "any_tags" in selector:
        any_tags = selector["any_tags"]
        if isinstance(any_tags, (str, bytes)) or not isinstance(
                any_tags, (list, tuple)):
            raise _error(
                "selector",
                f"'any_tags' must be a list of namespaced tags (the OR "
                f"form of 'tags'), got {any_tags!r}")
        for tag in any_tags:
            problem = _tag_problem(tag)
            if problem is not None:
                raise _error("selector", f"'any_tags': {problem}")
        if len(set(any_tags)) != len(any_tags):
            raise _error(
                "selector",
                f"'any_tags' lists a tag twice: {list(any_tags)!r}; tags "
                "are a set")
    if "seed" in selector:
        seed = selector["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise _error(
                "selector",
                f"'seed' must be an int, got {seed!r}; the seed makes a "
                "random choice reproducible across runs")
    return selector


def validate_spec(doc: dict) -> dict:
    """Validate one selection spec; return the document unchanged.

    A spec carries exactly ``name`` and ``selector``; the selector carries the
    optional ``kind``/``tags``/``pack`` filters plus ``choose`` (required) and
    ``seed`` (optional).  Nothing is resolved here and no catalog is read.

    Raises:
        SelectionError: A key is missing, unknown or malformed; the message
            names the key and the legal form.
    """
    if not isinstance(doc, Mapping):
        raise _error(
            "",
            f"a selection spec must be a mapping with keys {list(SPEC_KEYS)}, "
            f"got {type(doc).__name__}")
    unknown = sorted(set(doc) - set(SPEC_KEYS))
    if unknown:
        raise _error(
            "",
            f"unknown spec key(s) {unknown}; legal keys are {list(SPEC_KEYS)} "
            "-- the spec is the selector's intent, not a scene fragment")
    for key in SPEC_KEYS:
        if key not in doc:
            raise _error(
                "",
                f"{key!r} is required; a spec is "
                "{name: <snake_case>, selector: {choose: <int>, ...}}")
    name = doc["name"]
    if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
        raise _error(
            "name",
            f"must be snake_case ('[a-z][a-z0-9_]*'), got {name!r}; e.g. "
            "name: my_selection")
    _validate_selector(doc["selector"])
    return doc


def _matches(entry: Mapping, selector: Mapping) -> bool:
    """Return whether one catalog entry satisfies every selector filter."""
    kind = selector.get("kind")
    if kind is not None and entry.get("kind") != kind:
        return False
    pack = selector.get("pack")
    if pack is not None and entry.get("pack") != pack:
        return False
    wanted = selector.get("tags")
    if wanted and not set(wanted) <= set(entry.get("tags", ())):
        return False
    any_wanted = selector.get("any_tags")
    if any_wanted and not (set(any_wanted) & set(entry.get("tags", ()))):
        return False
    return True


def _stable_order(entries: Iterable[Mapping], selector: Mapping) -> list[str]:
    """Filter the entries and order their ids by ``(pack, id)`` (R7 tie-break)."""
    chosen = sorted(
        (entry.get("pack"), entry["id"])
        for entry in entries
        if _matches(entry, selector))
    return [asset_id for _, asset_id in chosen]


def _rule(selector: Mapping) -> str:
    """Render the selector as the lock's human-readable one-liner."""
    parts: list[str] = []
    if selector.get("kind") is not None:
        parts.append(f"kind={selector['kind']}")
    if selector.get("pack") is not None:
        parts.append(f"pack={selector['pack']}")
    if selector.get("tags"):
        parts.append("tags=" + ",".join(sorted(selector["tags"])))
    if selector.get("any_tags"):
        parts.append("any_tags=" + ",".join(sorted(selector["any_tags"])))
    parts.append("order=(pack,id)")
    parts.append(f"choose={selector['choose']}")
    if selector.get("seed") is not None:
        parts.append(f"seed={selector['seed']}")
    return "; ".join(parts)


def resolve_spec(spec: dict, catalog: dict) -> list[str]:
    """Resolve one spec against a catalog into the pinned asset ids.

    Candidates are filtered by the selector, ordered by ``(pack, id)`` and
    then either taken in order (``choose`` first ids) or drawn with
    ``random.Random(seed).sample(candidates, choose)`` and sorted, so a seeded
    selection is a permutation over that stable order, not over walk order.

    Args:
        spec: A spec mapping (validated here).
        catalog: A catalog document from
            :func:`unigym.assets.catalog.build_catalog`.

    Returns:
        The chosen asset ids, sorted; never empty.

    Raises:
        SelectionError: The spec is malformed, or the selector matches fewer
            than ``choose`` entries (including zero); the message names the
            selector, its rule and the catalog size.
    """
    document = validate_spec(spec)
    selector = document["selector"]
    entries = _catalog(catalog)["entries"]
    name = document["name"]
    rule = _rule(selector)
    candidates = _stable_order(entries, selector)
    choose = selector["choose"]
    if not candidates:
        raise _error(
            f"selection {name!r}",
            f"matched 0 of {len(entries)} catalog entries for rule {rule!r}; "
            "widen the selector or rebuild the catalog -- an empty selection "
            "is a broken decision, not an empty one")
    if choose > len(candidates):
        raise _error(
            f"selection {name!r}",
            f"asks for {choose} ids but only {len(candidates)} of "
            f"{len(entries)} catalog entries match rule {rule!r}; lower "
            "choose or widen the selector")
    seed = selector.get("seed")
    if seed is None:
        return candidates[:choose]
    return sorted(random.Random(seed).sample(candidates, choose))


def _validate_lock(lock: object) -> dict:
    """Validate one lock document; return it unchanged."""
    if not isinstance(lock, Mapping):
        raise _error(
            "",
            f"a selection lock must be a mapping with keys {list(LOCK_KEYS)}, "
            f"got {type(lock).__name__}")
    missing = [key for key in LOCK_KEYS if key not in lock]
    unknown = sorted(set(lock) - set(LOCK_KEYS))
    if missing or unknown:
        raise _error(
            "",
            f"lock keys must be exactly {list(LOCK_KEYS)}; missing {missing}, "
            f"unknown {unknown}")
    if lock["schema"] != SCHEMA:
        raise _error(
            "schema",
            f"must be {SCHEMA!r} exactly, got {lock['schema']!r}; regenerate "
            "the lock instead of editing it")
    digest = lock["catalog_digest"]
    if not isinstance(digest, str) or not digest:
        raise _error(
            "catalog_digest",
            f"must be a 'sha256:<hex>' string, got {digest!r}; regenerate "
            "the lock instead of editing it")
    resolved = lock["resolved"]
    if not isinstance(resolved, (list, tuple)) or not resolved:
        raise _error(
            "resolved",
            f"must be a non-empty list of asset ids, got {resolved!r}; an "
            "empty lock is a broken decision")
    for asset_id in resolved:
        if not isinstance(asset_id, str) or not asset_id:
            raise _error(
                "resolved",
                f"entries must be non-empty asset id strings, got {asset_id!r}")
    if len(set(resolved)) != len(resolved):
        raise _error(
            "resolved", f"lists an id twice: {list(resolved)!r}")
    rule = lock["rule"]
    if not isinstance(rule, str) or not rule or "\n" in rule:
        raise _error(
            "rule", f"must be a single-line string, got {rule!r}")
    return lock


def lock_document(spec: dict, catalog: dict, resolved: list[str]) -> dict:
    """Build the lock document for one spec, catalog and resolved id list.

    The lock pins the catalog digest, the resolved ids and the human-readable
    rule; it carries nothing else, so a reader can tell exactly what decided
    the ids.  ``resolved`` is checked for shape and for membership in this
    catalog (a lock may not name an id the catalog does not ship).

    Raises:
        SelectionError: The spec, the ids or the catalog are malformed, or an
            id is absent from the catalog.
    """
    document = validate_spec(spec)
    current = _catalog(catalog)
    ids = list(resolved)
    lock = {
        "schema": SCHEMA,
        "catalog_digest": catalog_digest(current),
        "resolved": ids,
        "rule": _rule(document["selector"]),
    }
    _validate_lock(lock)
    known = {entry["id"] for entry in current["entries"]}
    absent = [asset_id for asset_id in ids if asset_id not in known]
    if absent:
        raise _error(
            "resolved",
            f"names ids absent from the catalog: {absent}; resolve the spec "
            "against this catalog first")
    return lock


def check_lock(lock: dict, catalog: dict, spec: dict | None = None) -> list[str]:
    """Verify a lock against a catalog (and, when given, its spec).

    The digest gate comes first: a lock resolved against another catalog is
    unreproducible by definition.  Every resolved id must still exist, and
    when ``spec`` is supplied, re-resolving it must reproduce ``resolved``
    exactly -- that is what catches a changed selector or resolution rule.

    Args:
        lock: A lock document (validated here).
        catalog: The catalog to check against.
        spec: The spec the lock belongs to; without it only the digest and
            id membership can be checked, with it the lock is re-derived.

    Returns:
        The verified resolved ids.

    Raises:
        SelectionError: On any mismatch; a stale digest names the
            re-resolve-and-rewrite path (the ``--update`` behaviour) rather
            than asking for a hand-edited digest.
    """
    document = _validate_lock(lock)
    current = _catalog(catalog)
    expected = catalog_digest(current)
    recorded = document["catalog_digest"]
    if recorded != expected:
        raise _error(
            f"selection lock ({document['rule']})",
            f"records catalog_digest {recorded} but the catalog digests to "
            f"{expected}; the assets behind this decision changed -- "
            "re-resolve and rewrite the lock with --update instead of editing "
            "the digest")
    resolved = list(document["resolved"])
    known = {entry["id"] for entry in current["entries"]}
    missing = [asset_id for asset_id in resolved if asset_id not in known]
    if missing:
        raise _error(
            "selection lock",
            f"names ids absent from the catalog: {missing}; re-resolve and "
            "rewrite the lock with --update")
    if spec is not None:
        spec_document = validate_spec(spec)
        fresh = resolve_spec(spec_document, current)
        if fresh != resolved:
            raise _error(
                f"selection {spec_document['name']!r}",
                f"the lock resolved {resolved} but re-resolution yields "
                f"{fresh}; the spec or the resolution rule changed -- rewrite "
                "the lock with --update")
    return resolved


def _dump(lock: dict) -> str:
    """Render a lock document as deterministic YAML text (one trailing newline)."""
    text = yaml.safe_dump(
        lock, sort_keys=True, default_flow_style=False, allow_unicode=True,
        width=4096)
    return text if text.endswith("\n") else text + "\n"


def write_lock(spec: dict, catalog: dict, path: str | Path) -> Path:
    """Resolve ``spec`` against ``catalog`` and write its lock to ``path``.

    Writing is the explicit ``--update`` path: it re-resolves and overwrites,
    so a stale digest is refreshed by re-deciding, never by editing the file.
    The rendering is canonical (sorted keys, block style, one trailing
    newline), so the same spec and catalog produce the same lock bytes.

    Returns:
        The path of the written lock.
    """
    target = Path(path)
    current = _catalog(catalog)
    resolved = resolve_spec(spec, current)
    lock = lock_document(spec, current, resolved)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_dump(lock), encoding="utf-8")
    return target


def load_spec(path: str | Path) -> dict:
    """Read one selection spec YAML file, validate it and return the document.

    The only file read in this module.  I/O and yaml errors become
    :class:`SelectionError` naming the file and the fix, so a broken spec
    fails closed at the loader.
    """
    location = str(path)
    try:
        with open(path, encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except OSError as exc:
        raise SelectionError(
            f"{location}: cannot read the selection spec ({exc}); a spec is a "
            "yaml mapping with 'name' and 'selector' (v086 §6)") from exc
    except yaml.YAMLError as exc:
        raise SelectionError(
            f"{location}: the selection spec is not valid yaml ({exc}); fix "
            "the file -- one spec is one mapping") from exc
    if not isinstance(document, Mapping):
        raise SelectionError(
            f"{location}: a selection spec must be a yaml mapping with keys "
            f"{list(SPEC_KEYS)}, got {type(document).__name__}; check the file "
            "content, not just its name")
    try:
        return validate_spec(document)
    except SelectionError as exc:
        raise SelectionError(f"{location}: {exc}") from exc
