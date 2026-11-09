"""Content-key normalization for run configs and scene bundles (v091 D-3).

The yaml content block migrates from ``props:`` to ``objects:``; the old
spelling stays a long-lived alias.  :func:`normalize_content_keys` is the
ONE place the alias resolves: it maps ``objects:`` onto the internal
``props`` name the runtime consumes and warns -- never errors -- when the
legacy key is still spelled.  An ``objects:`` block may additionally carry
a content-fill spec (``sets`` / ``demands`` / ``policy`` / ``provider`` /
``seed``); those keys split into ``cfg["content"]`` while ``types:`` (the
static props) stays in ``cfg["props"]``.

The internal vocabulary does not move: ``PropManager``,
``build_prop_manager`` and the prop templates keep their names; only the
config key surface changes.

Callers: the collector's config-load point (a run yaml) and
:func:`unigym.scene_packs.apply_scene_bundle` (a bundle's own block), so a
run config and its scene bundle both migrate through this function.
"""
from __future__ import annotations

import warnings

from unigym._typing import JsonValue

__all__ = ["normalize_content_keys"]

_INTERNAL_KEY = "props"
"""The runtime's content key (``build_prop_manager(cfg.get("props"))``)."""
_SURFACE_KEY = "objects"
"""The yaml surface spelling every config migrates to."""
_CONTENT_KEY = "content"
"""The run's content-fill spec (split out of the ``objects:`` block)."""
_CONTENT_SPEC_KEYS = ("demands", "policy", "provider", "seed", "sets")
"""The keys an ``objects:`` content spec carries (``types`` stays props)."""

_MIGRATION = (
    "{source}: 'props:' is the legacy spelling of the content block; "
    "migrate it to 'objects:' (props: -> objects:).  The alias keeps "
    "working -- this warning never becomes an error"
)
_AMBIGUOUS = (
    "{source}: both 'objects:' and the legacy 'props:' are declared; "
    "'objects:' wins (props: -> objects:).  Drop the 'props:' block to "
    "silence this warning"
)


def _label(source: str) -> str:
    """The warning's file label (an anonymous mapping when unknown)."""
    return source or "<config mapping>"


def normalize_content_keys(
    cfg: dict[str, JsonValue], *, source: str = "",
) -> dict[str, JsonValue]:
    """Split the ``objects:`` surface block into props and content fill.

    The migration is one-way: ``objects`` is the only surface spelling and
    ``props`` stays the runtime vocabulary.  The block carries the static
    prop declarations under ``types:`` (plus ``enabled``), which become
    ``cfg["props"]``; the optional content spec keys
    (:data:`_CONTENT_SPEC_KEYS`) split into ``cfg["content"]``.  The input
    mapping is updated in place and returned, so a caller can splice the
    call into its load chain.  ``props:`` alone keeps working unchanged.
    When both spellings are present ``objects:`` wins deterministically
    (the run's new surface beats the legacy block).

    Args:
        cfg: A parsed run config or scene bundle mapping.
        source: The file the mapping came from, named in the warning (an
            anonymous label when empty).

    Returns:
        The same mapping, with ``props`` carrying the static declarations
        and ``content`` carrying the fill spec when one was declared.

    Warns:
        DeprecationWarning: when the legacy ``props:`` key is spelled, or
            when both spellings are present (``objects:`` then wins).  The
            alias is long-lived, so this is guidance, never an error.
    """
    if not isinstance(cfg, dict):
        return cfg
    legacy = _INTERNAL_KEY in cfg
    surface = _SURFACE_KEY in cfg
    if legacy and surface:
        warnings.warn(_AMBIGUOUS.format(source=_label(source)),
                      DeprecationWarning, stacklevel=2)
    elif legacy:
        warnings.warn(_MIGRATION.format(source=_label(source)),
                      DeprecationWarning, stacklevel=2)
    if surface:
        block = cfg.pop(_SURFACE_KEY)
        content = None
        if isinstance(block, dict):
            block = dict(block)
            content = {
                key: block.pop(key) for key in _CONTENT_SPEC_KEYS
                if key in block
            } or None
        cfg[_INTERNAL_KEY] = block
        if content is not None:
            cfg[_CONTENT_KEY] = content
    return cfg
