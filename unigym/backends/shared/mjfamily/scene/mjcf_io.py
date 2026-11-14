"""MJCF loading for already-resolved documents.

Prop fragments and sealed test references are MJCF documents.  Asset leaves
inside them (``<mesh file=...>`` / ``<texture file=...>``) are resolved by
the composition boundary (:class:`unigym.assets.plan.AssetPlan`) before the
document reaches this loader; this module never expands placeholders and
never resolves a ``pack:`` reference or a legacy asset anchor.

Documents without an asset reference keep the in-place
:meth:`mujoco.MjSpec.from_file` path (relative includes and file-relative
resolution unchanged).  A document that still carries a ``${TOKEN}`` or a
``pack:`` reference fails fast with a teaching error: old content is not
migrated, and silently resolving it here would rebuild the legacy loader
inside the backend.

File-backed scene assets go through :func:`load_asset_spec`, which resolves a
document's placeholders and ``pack:`` leaves through the composition boundary
once.  A rewritten document is loaded with :meth:`mujoco.MjSpec.from_string`,
which has no document base, so every remaining ``file=`` leaf must be
absolute; there is no copying, rewriting or CWD-relative fallback.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unigym.assets.plan import AssetPlan

_FILE_ATTRIBUTE_RE = re.compile(r"""file\s*=\s*(['"])(.*?)\1""")
"""One MJCF ``file=`` attribute value (mesh/texture/include leaves)."""


def _document_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _unresolved_reference(text: str) -> str | None:
    """The first unresolved placeholder or ``pack:`` marker in *text*."""
    if "${" in text:
        start = text.find("${")
        return text[start : start + 40]
    if "pack:" in text:
        start = text.find("pack:")
        return text[start : start + 40]
    return None


def load_spec(path):
    """Load one resolved MJCF document as a :class:`mujoco.MjSpec`.

    Args:
        path: An absolute path to a resolved MJCF document.

    Raises:
        ValueError: The document still carries a ``${TOKEN}`` placeholder or
            a ``pack:`` reference.  Asset resolution belongs to the
            composition boundary, not to this loader.
    """
    import mujoco

    resolved = Path(path)
    text = _document_text(resolved)
    unresolved = _unresolved_reference(text)
    if unresolved is not None:
        raise ValueError(
            f"MJCF document {resolved} still carries an unresolved asset "
            f"reference {unresolved!r}; resolve it at the composition "
            "boundary (unigym.assets.plan.AssetPlan) -- old token and pack "
            "content is not migrated"
        )
    return mujoco.MjSpec.from_file(str(resolved))


def _require_absolute_file(path: str | Path, what: str) -> Path:
    """Validate that *path* is an absolute path to an existing file."""
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError(
            f"{what} {str(candidate)!r} must be an absolute path; asset "
            f"references are resolved by the composition boundary"
        )
    if not candidate.is_file():
        raise FileNotFoundError(f"{what} {str(candidate)!r} is not an existing file")
    return candidate


def load_asset_spec(path: str | Path, asset_plan: AssetPlan):
    """Load one file-backed MJCF asset with the MVP closure policy.

    A document with no placeholder and no ``pack:`` leaf loads in place
    through :meth:`mujoco.MjSpec.from_file`, so ordinary multi-file assets
    keep MuJoCo's file-relative include resolution.  Otherwise the document
    text resolves once through *asset_plan* and loads through
    :meth:`mujoco.MjSpec.from_string`; every remaining ``file=`` leaf must be
    absolute because ``from_string`` has no document base.

    Args:
        path: The absolute path of the asset document.
        asset_plan: The composition boundary that resolves placeholders and
            ``pack:`` leaves.

    Returns:
        The loaded :class:`mujoco.MjSpec`.

    Raises:
        ValueError: The path is relative, or the resolved document still
            carries a placeholder/``pack:`` reference, or a relative
            ``file=`` leaf remains.
        FileNotFoundError: The path does not name an existing file.
    """
    import mujoco

    resolved_path = _require_absolute_file(path, "asset")
    text = _document_text(resolved_path)
    if "${" not in text and "pack:" not in text:
        return mujoco.MjSpec.from_file(str(resolved_path))
    resolved = asset_plan.resolve_document(text)
    unresolved = _unresolved_reference(resolved)
    if unresolved is not None:
        raise ValueError(
            f"asset {str(resolved_path)!r} still carries an unresolved asset "
            f"reference {unresolved!r} after AssetPlan resolution; make the "
            f"document self-contained or use absolute/pack: leaves"
        )
    relative = [
        match.group(2)
        for match in _FILE_ATTRIBUTE_RE.finditer(resolved)
        if not Path(match.group(2)).is_absolute()
    ]
    if relative:
        raise ValueError(
            f"asset {str(resolved_path)!r} uses relative file leaves with pack: "
            f"references; make the document self-contained or use "
            f"absolute/pack: leaves (no copying/rewriting fallback) -- "
            f"relative leaves: {relative}"
        )
    return mujoco.MjSpec.from_string(resolved)


def load_model(path):
    """Compile one resolved MJCF document to a :class:`mujoco.MjModel`."""
    return load_spec(path).compile()
