"""MJCF loading with path-token expansion (v080 P2).

Prop fragments are MJCF documents.  Runtime fragments (``mjcf_file``)
and the sealed test references spell asset leaves with
``${UNIGYM_ASSET_PACK}`` / other path tokens; MuJoCo itself has no
placeholder syntax, so the single load door here expands the document
text through :func:`unigym.core.asset_root.expand_placeholders` and
parses the result in memory.  Documents without a placeholder keep the
in-place :meth:`mujoco.MjSpec.from_file` path (relative includes and
file-relative resolution unchanged).

The expansion is document-wide: a fragment that spells a token in a
``<mesh file=...>`` / ``<texture file=...>`` attribute compiles with the
absolute resolved leaf.
"""
from __future__ import annotations

from pathlib import Path


def _document_text(path) -> str:
    return Path(path).read_text(encoding="utf-8")


def has_placeholders(path) -> bool:
    """True when the MJCF document carries any ``${TOKEN}``."""
    return "${" in _document_text(path)


def load_spec(path):
    """Load one MJCF document as a :class:`mujoco.MjSpec` (tokens expanded)."""
    import mujoco

    text = _document_text(path)
    if "${" not in text:
        return mujoco.MjSpec.from_file(str(path))
    from unigym.core.asset_root import canonical_asset_text, expand_placeholders

    # tokens resolve to absolute leaves, then the asset-face texture rule
    # makes a flat leaf written by an older fragment compile against the
    # payload that now owns its own directory
    return mujoco.MjSpec.from_string(
        canonical_asset_text(expand_placeholders(text)))


def load_model(path):
    """Compile one MJCF document to a :class:`mujoco.MjModel`."""
    return load_spec(path).compile()
