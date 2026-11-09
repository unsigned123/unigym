"""Path-anchor placeholders for run configs (single expansion point).

Run configs spell host paths via ``${TOKEN}`` placeholders;
unigym/scripts/collect_cli.py expands them at the SINGLE config load
point, so both backends (mujoco host path; isaaclab container path,
which re-reads the same yaml through this same module) see concrete
paths.  Downstream needs zero changes: resolvers, raw paths (e.g.
``spawn.texture.file``) and the docker path translation all receive
expanded absolute paths.

Tokens:

======================  ==============================================
``${UNIGYM_ASSET_ROOT}``  shared asset root (audit 2026-10 P1-7): env
                          ``UNIGYM_ASSET_ROOT`` > yaml
                          ``unigym_asset_root`` (local overlay then
                          unigym.yaml) -- unset fails fast (v101 A1
                          removed the implicit machine default)
``${UNIGYM_REPO}``        this checkout's repo root -- COMPUTED from
                          ``unigym.__file__`` (follows the worktree /
                          checkout automatically)
``${UNIGYM_DEPS_ROOT}``   external dependency root: env
                          ``UNIGYM_DEPS_ROOT`` > ``unigym_deps_root``
                          key in unigym.local.yaml (machine overlay)
                          then unigym.yaml -- unset fails fast (v101 A1
                          removed the implicit machine default)
``${UNIGYM_ASSET_PACK}``  the unified content pack's ``assets`` face
                          (v091 B-0): env ``UNIGYM_ASSET_PACK`` > yaml
                          ``unigym_asset_pack_root`` (local overlay then
                          unigym.yaml) > the mounted pack's ``assets``
                          directory
======================  ==============================================

The asset-pack loader resolves generator leaves against that ONE face
(:func:`resolve_asset_path`): the retired standalone asset pack and the
upstream store are no longer rungs, so a leaf the face does not carry is
an error naming ``tools/pack_asset_closure.py`` rather than a silent
second lookup.

Semantics: deep expansion over the whole parsed yaml document
(dict/list/str tree -- non-strings pass through); a string whose
expansion still contains a placeholder is re-expanded (capped passes
-- cycle guard).  An UNKNOWN ``${TOKEN}`` is a hard error listing the
known tokens: a placeholder is a declaration, not a guess -- typos
must fail fast at load time.  Trust model unchanged: token values are
computed values / env / the repo's own yaml layers, never user input.

isaaclab note (unchanged): the backend resolves paths INSIDE the
container -- an alias/deps root must exist under a container-visible
prefix (unigym.yaml ``host_ro_prefix``) to be usable there.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, overload

import yaml

from unigym._typing import JsonValue
from unigym.errors import ConfigError

ENV_VAR = "UNIGYM_ASSET_ROOT"
PLACEHOLDER = "${" + ENV_VAR + "}"          # the original token spelling

DEPS_ENV_VAR = "UNIGYM_DEPS_ROOT"
DEPS_YAML_KEY = "unigym_deps_root"
ASSET_YAML_KEY = "unigym_asset_root"
# local overlay wins over the committed yaml (load_cfg layering)
_YAML_FILES = ("unigym.local.yaml", "unigym.yaml")
_DEPS_YAML_FILES = _YAML_FILES          # deps-root parity (legacy name)

PACK_ENV_VAR = "UNIGYM_ASSET_PACK"
PACK_YAML_KEY = "unigym_asset_pack_root"

_MAX_PASSES = 8                             # deep-expansion cycle guard
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

_IMAGE_SUFFIXES = frozenset({
    ".bmp", ".gif", ".hdr", ".exr", ".jpeg", ".jpg", ".png", ".tga", ".webp",
})
"""Suffixes that make a ``tex/`` leaf a texture image (the stem-dir rule)."""


class PlaceholderError(ValueError):
    """Unknown or non-terminating ``${TOKEN}`` in a run config."""


class AssetPathError(FileNotFoundError):
    """A generator asset is absent from the content pack's asset face."""


def asset_root(repo: str | None = None) -> str:
    """Asset root: env > yaml ``unigym_asset_root`` (no implicit default).

    ``repo`` is injectable for tests; defaults to :func:`repo_root`.

    Raises:
        ConfigError: Neither the env var nor a yaml key is set.  v101 A1
            removed the compiled-in machine default.
    """
    env = os.environ.get(ENV_VAR)
    if env:
        return env
    base = Path(repo) if repo else Path(repo_root())
    for name in _YAML_FILES:
        p = base / name
        if not p.is_file():
            continue
        try:
            doc = yaml.safe_load(p.read_text()) or {}
        except (OSError, ValueError):
            doc = {}
        v = doc.get(ASSET_YAML_KEY)
        if v:
            return str(v)
    raise ConfigError(
        f"no asset-root anchor: set {ENV_VAR}, or declare "
        f"`{ASSET_YAML_KEY}` in unigym.local.yaml / unigym.yaml -- "
        "v101 A1 removed the implicit machine default")


def repo_root() -> str:
    """The checkout's repo root (computed -- follows the worktree)."""
    import unigym
    return str(Path(unigym.__file__).resolve().parents[1])


def deps_root(repo: str | None = None) -> str:
    """External dependency root: env > yaml key (no implicit default).

    ``repo`` is injectable for tests; defaults to :func:`repo_root`.

    Raises:
        ConfigError: Neither the env var nor a yaml key is set.  v101 A1
            removed the compiled-in machine default.
    """
    env = os.environ.get(DEPS_ENV_VAR)
    if env:
        return env
    base = Path(repo) if repo else Path(repo_root())
    for name in _DEPS_YAML_FILES:
        p = base / name
        if not p.is_file():
            continue
        try:
            doc = yaml.safe_load(p.read_text()) or {}
        except (OSError, ValueError):
            doc = {}
        v = doc.get(DEPS_YAML_KEY)
        if v:
            return str(v)
    raise ConfigError(
        f"no dependency-root anchor: set {DEPS_ENV_VAR}, or declare "
        f"`{DEPS_YAML_KEY}` in unigym.local.yaml / unigym.yaml -- "
        "v101 A1 removed the implicit machine default")


def _yaml_key(base: Path, file_name: str, key: str):
    """One top-level yaml key from *file_name* under *base* (or None)."""
    path = base / file_name
    if not path.is_file():
        return None
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except (OSError, ValueError):
        return None
    return doc.get(key)


def asset_pack_root(repo: str | None = None) -> str:
    """The unified pack's asset face: env > yaml declaration.

    The face is the content pack's ``assets`` directory -- the single
    serving root for generator payload, textures and licenses since every
    referenced asset was folded in.  The checkout declares it in
    ``unigym.yaml`` (``unigym_asset_pack_root``) because core must not
    import the packs layer; the retired standalone asset pack is no longer
    a fallback rung, so a missing leaf is an error, not a second lookup.
    ``repo`` is injectable for tests; defaults to :func:`repo_root`.
    """
    env = os.environ.get(PACK_ENV_VAR)
    if env:
        return env
    base = Path(repo) if repo else Path(repo_root())
    for name in _YAML_FILES:
        value = _yaml_key(base, name, PACK_YAML_KEY)
        if value:
            return _expand_str(str(value), known_tokens())
    raise AssetPathError(
        "no asset-face anchor: set UNIGYM_ASSET_PACK, or declare "
        f"`{PACK_YAML_KEY}` (the content pack's assets directory, e.g. "
        "${UNIGYM_DEPS_ROOT}/unigym_packs/robocasa/assets) in unigym.yaml")


def texture_payload_rel(rel: Path) -> Path:
    """Map a declared ``tex/`` image leaf onto its own asset directory.

    Every texture image is its own self-contained asset (manifest beside
    payload), so the declared leaf ``tex/<a>/<stem>.<ext>`` names
    ``tex/<a>/<stem>/<stem>.<ext>``.  Deeper directories keep their
    prefix: ``tex/gentex/<d>/texNNN.png`` becomes
    ``tex/gentex/<d>/texNNN/texNNN.png``.  The mapping is computed from
    the declared spelling alone; the CALLER decides which of the two
    spellings the face actually carries (canonical wins, see
    :func:`canonical_asset_path`).

    Public because the closure/reporting tools must apply the SAME rule
    when they predict a texture's pack destination.
    """
    if rel.parts[:1] != ("tex",):
        return rel
    if rel.suffix.lower() not in _IMAGE_SUFFIXES:
        return rel
    return rel.parent / rel.stem / rel.name


#: the marker every asset-face path carries before its face-relative tail.
_GENERATORS_MARK = "/generators/"


def canonical_asset_path(value: str) -> str:
    """Canonicalize the texture tail of one asset-face path (pure, no I/O).

    Declared asset paths are INPUTS, never filesystem paths: a texture
    leaf names its own asset directory (``tex/<a>/<stem>/<stem>.<ext>``).
    Any string with a ``/generators/`` segment therefore maps through
    :func:`texture_payload_rel` -- including a retired anchor XML or a
    golden fragment written before the relocation.  Idempotent, so it is
    safe on an already-canonical path and safe to apply on both sides of
    a comparison.
    """
    head, sep, tail = str(value).partition(_GENERATORS_MARK)
    if not sep:
        return value
    return head + sep + texture_payload_rel(Path(tail)).as_posix()


_FILE_ATTR_RE = re.compile(r'(file=")([^"]*)(")')


@overload
def canonical_asset_values(obj: str) -> str: ...


@overload
def canonical_asset_values(obj: dict[Any, Any]) -> dict[Any, Any]: ...


@overload
def canonical_asset_values(obj: list[Any]) -> list[Any]: ...


@overload
def canonical_asset_values(obj: Any) -> Any: ...


def canonical_asset_values(obj):
    """Deep-walk a parsed document: canonicalize every asset-face string.

    The dict/list twin of :func:`canonical_asset_text` for template
    entries (``{"texture": {"file": "${UNIGYM_ASSET_PACK}/generators/...``).
    Non-strings and strings without the face marker pass through.
    """
    if isinstance(obj, str):
        return canonical_asset_path(obj)
    if isinstance(obj, dict):
        return {key: canonical_asset_values(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [canonical_asset_values(value) for value in obj]
    return obj


def canonical_asset_text(text: str) -> str:
    """Apply :func:`canonical_asset_path` to every ``file=`` of a document.

    MJCF documents spell asset leaves in ``file=`` attributes; a fragment
    or a retired anchor XML written before the texture relocation must
    still compile against the real payload, so the document door maps
    those attributes and nothing else (comments, names, values untouched).
    """
    return _FILE_ATTR_RE.sub(
        lambda match: match.group(1) + canonical_asset_path(match.group(2))
        + match.group(3), text)


def resolve_asset_path(value: str, repo: str | None = None) -> str:
    """Resolve one generator asset string to the unified pack's asset face.

    Accepted input spellings: the ``${UNIGYM_ASSET_PACK}/generators/<rel>``
    token, the unified face itself, the retired repo literal
    ``<repo>/assets/robocasa/generators/<rel>``, and a bare ``<rel>``.  The
    output is always ``<asset_pack_root>/generators/<rel>`` -- one root, no
    ladder.  A leaf the face does not carry fails closed and names the
    closure tool that measures the gap.

    Texture images get one extra, mandatory mapping: a leaf under ``tex/``
    that names an image resolves into its own directory (``rel`` ->
    ``tex/<a>/<stem>/<stem>.<ext>``, see :func:`texture_payload_rel`).  That
    is the ONLY mapping for texture leaves -- the canonical path is
    computed, never discovered, and an absent canonical file fails closed
    exactly like any other missing leaf (no flat-sibling fallback).

    A path that is absolute and outside every generator root is not a
    generator reference at all (a run config's own texture, say): it passes
    through unchanged, exactly as before.
    """
    expanded = expand_placeholders(value)
    base = Path(repo) if repo else Path(repo_root())
    face = Path(asset_pack_root(repo))
    rel = None
    for root in (face / "generators",
                 base / "assets" / "robocasa" / "generators"):
        try:
            rel = Path(expanded).relative_to(root)
            break
        except ValueError:
            continue
    if rel is None:
        if Path(expanded).is_absolute():
            return expanded
        rel = Path(expanded)
    candidate = face / "generators" / texture_payload_rel(rel)
    if not candidate.exists():
        raise AssetPathError(
            f"generator asset absent from the unified pack asset face: "
            f"{candidate} -- run `python tools/pack_asset_closure.py` to "
            f"measure the gap and fold it (v091 B-0), or fix the reference "
            f"{value!r}")
    return str(candidate)


def known_tokens() -> dict:
    """Token name -> resolver (also the known-token list for errors)."""
    return {"UNIGYM_ASSET_ROOT": asset_root,
            "UNIGYM_ASSET_PACK": asset_pack_root,
            "UNIGYM_REPO": repo_root,
            "UNIGYM_DEPS_ROOT": deps_root}


def _expand_str(s: str, tokens: dict) -> str:
    for _ in range(_MAX_PASSES):
        m = _PLACEHOLDER_RE.search(s)
        if m is None:
            return s
        name = m.group(1)
        if name not in tokens:
            raise PlaceholderError(
                f"unknown path placeholder ${{{name}}} in run config "
                f"(known tokens: {', '.join(sorted(tokens))})")
        s = s.replace("${" + name + "}", tokens[name]())
    raise PlaceholderError(
        f"path placeholder expansion did not settle within "
        f"{_MAX_PASSES} passes (cycle in placeholder values?)")


@overload
def expand_placeholders(obj: str) -> str: ...


@overload
def expand_placeholders(obj: dict[str, JsonValue]) -> dict[str, JsonValue]: ...


@overload
def expand_placeholders(obj: list[JsonValue]) -> list[JsonValue]: ...


@overload
def expand_placeholders(obj: JsonValue) -> JsonValue: ...


def expand_placeholders(obj: JsonValue) -> JsonValue:
    """Deep-expand every ``${TOKEN}`` in a parsed yaml document
    (dict/list/str tree).  Non-strings pass through; unknown tokens
    raise :class:`PlaceholderError`.

    The overloads keep the shape: a ``str`` input stays ``str`` (the
    generator-asset ladder in :func:`resolve_asset_path` depends on
    that), a mapping stays a mapping, a list stays a list.
    """
    if isinstance(obj, str):
        expanded = _expand_str(obj, known_tokens())
        if expanded.startswith("pack:"):
            # v081: external-pack references resolve at this same point.
            # Lazy import: unigym.packs sits above unigym.core (it reads
            # the deps/repo anchors from here), so a module-level import
            # would close the cycle.
            from unigym.packs.refs import resolve_ref
            return resolve_ref(expanded)
        return expanded
    if isinstance(obj, dict):
        return {k: expand_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_placeholders(v) for v in obj]
    return obj
