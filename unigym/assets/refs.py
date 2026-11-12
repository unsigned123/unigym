"""Asset-face path policy: texture-leaf mapping and canonical asset paths.

These helpers are pure path arithmetic over a declared asset spelling; they
perform no I/O and own no resolution context.
:class:`unigym.packs.runtime.PackRuntime` applies them while resolving a
``pack:`` reference, and the closure/reporting tools apply the same rule when
they predict a texture's pack destination.
"""

from __future__ import annotations

from pathlib import Path

_IMAGE_SUFFIXES = frozenset(
    {
        ".bmp",
        ".gif",
        ".hdr",
        ".exr",
        ".jpeg",
        ".jpg",
        ".png",
        ".tga",
        ".webp",
    }
)
"""Suffixes that make a ``tex/`` leaf a texture image (the stem-dir rule)."""


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
