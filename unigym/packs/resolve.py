"""Reference resolution over the mounted packs.

Three spellings are accepted: ``pack:<name>/<rel>`` pins one external
pack, ``repo:<rel>`` pins the checkout, and a bare ``<rel>`` searches the
external packs in mount order and falls back to the repository.  Two
external hits are an error (no silent first-match), and every resolved
path must stay inside its pack root, symlinks included.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from . import MountedPack, PackError

_mount = importlib.import_module(f"{__package__}.mount")

PACK_PREFIX = "pack:"
REPO_PREFIX = "repo:"


def _validate_rel(rel: str, ref: str) -> None:
    """Reject empty, absolute and parent-escaping relative paths."""
    if not rel:
        raise PackError(
            f"empty relative path in reference {ref!r}; "
            f"expected 'pack:<name>/<rel>', 'repo:<rel>' or '<rel>'"
        )
    path = Path(rel)
    if path.is_absolute():
        raise PackError(f"absolute path in reference {ref!r}; pack references are relative")
    if any(part == ".." for part in path.parts):
        raise PackError(
            f"'..' component in reference {ref!r}; references stay inside " f"the pack root"
        )


def _safe_join(root: Path, rel: str, ref: str) -> Path:
    """Join *rel* under *root* and reject realpath escapes."""
    _validate_rel(rel, ref)
    base = Path(root).resolve()
    candidate = base / Path(rel)
    real = candidate.resolve()
    try:
        real.relative_to(base)
    except ValueError:
        raise PackError(f"reference {ref!r} escapes its pack root {base} (symlink?)") from None
    return candidate


def _by_name(packs: list[MountedPack], name: str) -> MountedPack:
    """Look up one mounted pack by name (error lists the known names)."""
    for pack in packs:
        if pack.name == name:
            return pack
    known = sorted(pack.name for pack in packs)
    raise PackError(f"unknown pack {name!r}; known packs: {known}")


def _repo(packs: list[MountedPack]) -> MountedPack:
    """The implicit repository mount."""
    return _by_name(packs, "repo")


def _label(pack: MountedPack, rel: object) -> str:
    """Human-readable candidate label for the miss hint."""
    return f"{pack.name}:{rel}"


def _nearest(ref: str, packs: list[MountedPack], limit: int = 5) -> list[str]:
    """Up to *limit* cheap substring/prefix candidates for *ref*."""
    raw = ref.lower()
    token = Path(ref).name.lower()
    hints: list[str] = []
    for pack in packs:
        root = pack.root if pack.root is not None else Path(_mount.repo_root())
        for face_path in pack.faces.values():
            if face_path is None or not face_path.is_dir():
                continue
            for path in sorted(face_path.rglob("*")):
                if not path.is_file():
                    continue
                try:
                    rel = path.relative_to(root)
                except ValueError:
                    continue
                text = str(rel).lower()
                if raw in text or token in text or text.startswith(raw):
                    hints.append(_label(pack, rel))
                    if len(hints) >= limit:
                        return hints
    return hints


def _miss(ref: str, packs: list[MountedPack]) -> PackError:
    """Build the fail-loud miss error (ref, packs, faces, candidates)."""
    names = [pack.name for pack in packs]
    faces = sorted({face for pack in packs for face in pack.faces})
    message = (
        f"pack reference {ref!r} not found; mounted packs: {names}; " f"faces searched: {faces}"
    )
    hints = _nearest(ref, packs)
    if hints:
        message += f"; nearest candidates: {hints}"
    return PackError(message)


def _bare(rel: str, packs: list[MountedPack]) -> Path:
    """Resolve a bare reference across external packs, then the repository."""
    _validate_rel(rel, rel)
    hits: list[tuple[MountedPack, Path]] = []
    for pack in packs:
        if pack.root is None:
            continue
        candidate = _safe_join(pack.root, rel, rel)
        if candidate.exists():
            hits.append((pack, candidate))
    if len(hits) > 1:
        listing = ", ".join(_label(pack, candidate) for pack, candidate in hits)
        raise PackError(
            f"ambiguous bare pack reference {rel!r}: found in {listing}; "
            f"use 'pack:<name>/<rel>'"
        )
    if hits:
        return hits[0][1]
    candidate = _safe_join(Path(_mount.repo_root()), rel, rel)
    if candidate.exists():
        return candidate
    raise _miss(rel, packs)


def resolve(ref: str) -> Path:
    """Resolve ``pack:<name>/<rel>``, ``repo:<rel>`` or a bare relative ref."""
    if not isinstance(ref, str) or not ref.strip():
        raise PackError(f"empty pack reference {ref!r}")
    ref = ref.strip()
    packs = _mount.pack_roots()

    if ref.startswith(PACK_PREFIX):
        name, _, rel = ref[len(PACK_PREFIX) :].partition("/")
        if not name or not rel:
            raise PackError(f"invalid pack reference {ref!r}; expected 'pack:<name>/<rel>'")
        pack = _by_name(packs, name)
        if pack.root is None:
            raise PackError(f"reference {ref!r} names the implicit repo pack; " f"use 'repo:<rel>'")
        candidate = _safe_join(pack.root, rel, ref)
        if candidate.exists():
            return candidate
        raise _miss(ref, packs)

    if ref.startswith(REPO_PREFIX):
        rel = ref[len(REPO_PREFIX) :]
        _validate_rel(rel, ref)
        candidate = _safe_join(Path(_mount.repo_root()), rel, ref)
        if candidate.exists():
            return candidate
        raise _miss(ref, packs)

    return _bare(ref, packs)


def find(face: str, *, suffix: str | None = None) -> list[tuple[MountedPack, Path]]:
    """Every file in *face* across mounted packs, optionally name-filtered.

    ``suffix`` is matched with ``str.endswith`` so both ``".yaml"`` and
    ``"scene.yaml"`` work.  Files are returned in pack mount order, sorted
    within one face.
    """
    packs = _mount.pack_roots()
    known = sorted({name for pack in packs for name in pack.faces})
    if face not in known:
        raise PackError(f"unknown pack face {face!r}; known faces: {known}")
    hits: list[tuple[MountedPack, Path]] = []
    for pack in packs:
        face_path = pack.faces.get(face)
        if face_path is None or not face_path.is_dir():
            continue
        for path in sorted(face_path.rglob("*")):
            if not path.is_file():
                continue
            if suffix is not None and not path.name.endswith(suffix):
                continue
            hits.append((pack, path))
    return hits
