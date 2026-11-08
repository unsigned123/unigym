"""Pack-relative references inside run configs and scene bundles.

A string of the form ``pack:<name>/<rel>`` resolves through
:func:`unigym.packs.resolve`.  Both expansion points call
:func:`resolve_ref`: ``expand_placeholders`` (the single documented
config-expansion function) for every parsed document, and
:func:`expand_pack_refs` when a caller wants the recursive form directly.

This lets a bundle which lives in the repository point at content that
exists only in an external pack, without an absolute path and without a
repo-relative path that breaks once the content moves out.
"""
from __future__ import annotations

from typing import Any

from . import PackError, resolve

PREFIX = "pack:"


def resolve_ref(ref: str) -> str:
    """Resolve one ``pack:<name>/<rel>`` reference to an absolute path."""
    try:
        return str(resolve(ref))
    except PackError as exc:
        raise PackError(
            f"{exc} | pack references in configs are resolved by "
            f"unigym.packs.resolve at config load") from exc


def expand_pack_refs(obj: Any) -> Any:
    """Deep-expand every ``pack:<name>/<rel>`` string in a parsed document.

    Non-string leaves pass through; a reference that cannot be resolved
    raises :class:`PackError` with the resolver's candidate list.  The
    expansion is idempotent: an already-absolute path is untouched.
    """
    if isinstance(obj, str):
        return resolve_ref(obj) if obj.startswith(PREFIX) else obj
    if isinstance(obj, dict):
        return {key: expand_pack_refs(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [expand_pack_refs(value) for value in obj]
    return obj
