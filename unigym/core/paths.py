"""Path-anchor placeholders for run configs (single expansion point).

Run configs spell host paths via ``${TOKEN}`` placeholders; unigym expands
them at the SINGLE run/template/asset boundary, so both backends (mujoco
host path; isaaclab container path, which re-reads the same yaml through
this same module) see concrete paths.  Downstream needs zero changes:
resolvers, raw paths (e.g. ``spawn.texture.file``) and the docker path
translation all receive expanded absolute paths.

Tokens:

======================  ==============================================
``${UNIGYM_REPO}``        this checkout's repo root -- COMPUTED from
                          ``unigym.__file__`` (follows the worktree /
                          checkout automatically)
``${UNIGYM_DEPS_ROOT}``   external dependency root: env
                          ``UNIGYM_DEPS_ROOT`` > ``unigym_deps_root``
                          key in unigym.local.yaml (machine overlay)
                          then unigym.yaml -- unset fails fast (no
                          implicit machine default)
======================  ==============================================

``${UNIGYM_ASSET_ROOT}`` and ``${UNIGYM_ASSET_PACK}`` were removed from
the known-token vocabulary: an asset is addressed by a ``pack:`` reference
or an absolute path, resolved once at the composition boundary through an
:class:`unigym.packs.runtime.AssetContext`.  A ``pack:`` reference without
a context is a :class:`PlaceholderError`; there is no bare-relative or
legacy-token fallback.

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
from typing import TYPE_CHECKING, overload

import yaml

from unigym._typing import JsonValue
from unigym.errors import ConfigError

if TYPE_CHECKING:
    from unigym.packs.runtime import AssetContext

DEPS_ENV_VAR = "UNIGYM_DEPS_ROOT"
DEPS_YAML_KEY = "unigym_deps_root"
# local overlay wins over the committed yaml (load_cfg layering)
_DEPS_YAML_FILES = ("unigym.local.yaml", "unigym.yaml")

_MAX_PASSES = 8  # deep-expansion cycle guard
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class PlaceholderError(ValueError):
    """Unknown or non-terminating ``${TOKEN}`` in a run config."""


def repo_root() -> str:
    """The checkout's repo root (computed -- follows the worktree)."""
    import unigym

    return str(Path(unigym.__file__).resolve().parents[1])


def deps_root(repo: str | None = None) -> str:
    """External dependency root: env > yaml key (no implicit default).

    ``repo`` is injectable for tests; defaults to :func:`repo_root`.

    Raises:
        ConfigError: Neither the env var nor a yaml key is set (there is
            no compiled-in machine default).
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
        "v101 A1 removed the implicit machine default"
    )


def known_tokens() -> dict:
    """Token name -> resolver (also the known-token list for errors)."""
    return {
        "UNIGYM_REPO": repo_root,
        "UNIGYM_DEPS_ROOT": deps_root,
    }


#: Asset path anchors removed from the placeholder vocabulary by the asset
#: cutover.  They stay a teaching error instead of silently becoming an
#: unknown-token typo.
_REMOVED_ASSET_TOKENS = frozenset({"UNIGYM_ASSET_ROOT", "UNIGYM_ASSET_PACK"})


def _resolve_pack_ref(value: str, context: AssetContext | None) -> str:
    """Resolve one whole-string ``pack:`` reference through *context*."""
    if context is None:
        raise PlaceholderError(
            f"asset reference {value!r} requires an explicit asset context: "
            "build a pack runtime at the composition boundary and pass its "
            "AssetContext to expansion; there is no ambient or global runtime"
        )
    return context.resolve(value)


def _expand_str(s: str, tokens: dict, context: AssetContext | None) -> str:
    for _ in range(_MAX_PASSES):
        m = _PLACEHOLDER_RE.search(s)
        if m is None:
            return _resolve_pack_ref(s, context) if s.startswith("pack:") else s
        name = m.group(1)
        if name in _REMOVED_ASSET_TOKENS:
            raise PlaceholderError(
                f"path placeholder ${{{name}}} was removed by the asset "
                "cutover; address the asset with a 'pack:<pack>/<rel>' "
                "reference or an absolute filesystem path"
            )
        if name not in tokens:
            raise PlaceholderError(
                f"unknown path placeholder ${{{name}}} in run config "
                f"(known tokens: {', '.join(sorted(tokens))})"
            )
        s = s.replace("${" + name + "}", tokens[name]())
    raise PlaceholderError(
        f"path placeholder expansion did not settle within "
        f"{_MAX_PASSES} passes (cycle in placeholder values?)"
    )


@overload
def expand_placeholders(obj: str, context: AssetContext | None = None) -> str: ...


@overload
def expand_placeholders(
    obj: dict[str, JsonValue], context: AssetContext | None = None
) -> dict[str, JsonValue]: ...


@overload
def expand_placeholders(
    obj: list[JsonValue], context: AssetContext | None = None
) -> list[JsonValue]: ...


@overload
def expand_placeholders(obj: JsonValue, context: AssetContext | None = None) -> JsonValue: ...


def expand_placeholders(obj: JsonValue, context: AssetContext | None = None) -> JsonValue:
    """Deep-expand every ``${TOKEN}`` in a parsed yaml document
    (dict/list/str tree).  Non-strings pass through; unknown tokens
    raise :class:`PlaceholderError`.

    A whole-string ``pack:<pack>/<rel>`` reference resolves through
    *context*; without a context it is a teaching :class:`PlaceholderError`.
    The removed ``${UNIGYM_ASSET_ROOT}`` / ``${UNIGYM_ASSET_PACK}`` anchors
    are teaching errors too -- there is no legacy or bare-relative fallback.

    The overloads keep the shape: a ``str`` input stays ``str`` (a caller
    that spells a whole reference depends on that), a mapping stays a
    mapping, a list stays a list.
    """
    if isinstance(obj, str):
        return _expand_str(obj, known_tokens(), context)
    if isinstance(obj, dict):
        return {k: expand_placeholders(v, context) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_placeholders(v, context) for v in obj]
    return obj
