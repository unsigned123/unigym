"""Pack manifest loading and validation.

The manifest is the only entry point of a pack.  It declares the pack
name, its own version, the kernel API it targets, the ``unigym`` version
range it supports, its trust level and the content faces it provides.
Validation is fail-closed: every malformed or unsupported declaration
raises :class:`~unigym.packs.PackError` with the offending path and the
legal form.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import yaml

from . import PACK_API, PackError, PackManifest

MANIFEST_NAME = "unigym.pack.yaml"
"""The single manifest file every pack root carries."""

FACE_NAMES = ("runs", "benchmarks", "scenes", "props", "terminations", "assets")
"""Content faces a manifest may declare."""

TRUST_LEVELS = ("content", "code")
"""Supported trust levels; ``code`` is the only level that imports Python."""

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_COMPARATOR_RE = re.compile(r"^(>=|<=|==|>|<)\s*([0-9][0-9A-Za-z.+-]*)$")
_VERSION_RE = re.compile(r"^[0-9]+(\.[0-9]+)*$")


def _read_document(root: Path) -> dict:
    """Read the raw manifest mapping (or raise a :class:`PackError`)."""
    path = Path(root) / MANIFEST_NAME
    if not path.is_file():
        raise PackError(f"missing pack manifest: {path}")
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError) as err:
        raise PackError(f"unreadable pack manifest: {path}: {err}") from err
    if not isinstance(doc, Mapping):
        raise PackError(f"pack manifest must be a mapping: {path}")
    return dict(doc)


def _version_tuple(value: str) -> tuple[int, ...]:
    """Dotted integer version -> tuple (raise :class:`PackError` when bad)."""
    text = str(value).strip()
    if not _VERSION_RE.fullmatch(text):
        raise PackError(f"unparseable version {value!r} (expected dotted digits)")
    return tuple(int(part) for part in text.split("."))


def _compare(op: str, left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    """Compare two versions after padding the shorter tuple with zeros."""
    width = max(len(left), len(right))
    a = left + (0,) * (width - len(left))
    b = right + (0,) * (width - len(right))
    return {
        ">=": a >= b,
        "<=": a <= b,
        "==": a == b,
        ">": a > b,
        "<": a < b,
    }[op]


def _check_unigym_requirement(expr: str, path: Path) -> None:
    """Validate ``requires.unigym`` against this checkout's version."""
    import unigym

    current = str(getattr(unigym, "__version__", "unknown"))
    try:
        current_tuple = _version_tuple(current)
    except PackError as err:
        raise PackError(
            f"cannot evaluate requires.unigym {expr!r} in {path}: "
            f"checkout version {current!r} is not dotted digits"
        ) from err
    for part in str(expr).split(","):
        match = _COMPARATOR_RE.match(part.strip())
        if match is None:
            raise PackError(
                f"unsupported requirement form {expr!r} in {path}; " f"expected e.g. '>=0.80,<1.0'"
            )
        op, target = match.group(1), match.group(2)
        try:
            target_tuple = _version_tuple(target)
        except PackError as err:
            raise PackError(
                f"unsupported requirement form {expr!r} in {path}; "
                f"target {target!r} is not dotted digits"
            ) from err
        if not _compare(op, current_tuple, target_tuple):
            raise PackError(
                f"pack requires unigym {expr!r}, this checkout is {current} " f"({path})"
            )


def _face_path(root: Path, value: str, face: str, path: Path) -> Path:
    """Resolve one declared face path and pin it inside the pack root."""
    rel = Path(value)
    if rel.is_absolute():
        raise PackError(
            f"pack face {face!r} must be relative to the pack root: " f"{value!r} ({path})"
        )
    base = Path(root).resolve()
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise PackError(
            f"pack face {face!r} escapes the pack root: {value!r} " f"({path}; root = {base})"
        ) from None
    if not candidate.exists():
        raise PackError(f"pack face {face!r} does not exist: {candidate} ({path})")
    return candidate


def load_manifest(root: Path) -> PackManifest:
    """Load and validate the manifest of the pack rooted at *root*."""
    root = Path(root)
    path = root / MANIFEST_NAME
    doc = _read_document(root)

    name = doc.get("name")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise PackError(f"pack name {name!r} must match [a-z][a-z0-9_]* ({path})")

    version = doc.get("version")
    if not isinstance(version, str) or not version.strip():
        raise PackError(f"pack {name!r} must declare a non-empty version ({path})")

    api = doc.get("api")
    if isinstance(api, bool) or not isinstance(api, int) or api != PACK_API:
        raise PackError(f"unsupported pack api {api!r} in {path}; supported api: {PACK_API}")

    raw_requires = doc.get("requires") or {}
    if not isinstance(raw_requires, Mapping):
        raise PackError(f"pack {name!r} requires must be a mapping ({path})")
    requires = {str(key): str(value) for key, value in raw_requires.items()}
    if "unigym" in requires:
        _check_unigym_requirement(requires["unigym"], path)

    trust = doc.get("trust")
    if trust not in TRUST_LEVELS:
        raise PackError(
            f"pack {name!r} trust {trust!r} must be one of {list(TRUST_LEVELS)} " f"({path})"
        )
    python_dir = root / "python"
    if trust == "content" and python_dir.exists():
        raise PackError(
            f"pack {name!r} declares trust: content but carries python/ "
            f"({python_dir}); use trust: code to import pack code"
        )
    if trust == "code" and not (python_dir / "__init__.py").is_file():
        raise PackError(
            f"pack {name!r} declares trust: code but has no " f"python/__init__.py ({python_dir})"
        )

    raw_faces = doc.get("faces") or {}
    if not isinstance(raw_faces, Mapping):
        raise PackError(f"pack {name!r} faces must be a mapping ({path})")
    faces: dict[str, Path] = {}
    for face, value in raw_faces.items():
        if face not in FACE_NAMES:
            raise PackError(
                f"unknown pack face {face!r} in {path}; known faces: " f"{list(FACE_NAMES)}"
            )
        if not isinstance(value, str) or not value.strip():
            raise PackError(f"pack face {face!r} must be a non-empty relative path ({path})")
        faces[face] = _face_path(root, value, face, path)

    return PackManifest(
        name=name,
        version=version,
        api=api,
        requires=requires,
        trust=trust,
        faces=faces,
        root=Path(root).resolve(),
    )


def _shadow_declared(root: Path) -> bool:
    """True when the manifest explicitly opts into shadowing a same name."""
    try:
        return bool(_read_document(root).get("shadow", False))
    except PackError:
        return False
