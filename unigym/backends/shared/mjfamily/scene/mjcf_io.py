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
"""

from __future__ import annotations

from pathlib import Path


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


def load_model(path):
    """Compile one resolved MJCF document to a :class:`mujoco.MjModel`."""
    return load_spec(path).compile()
