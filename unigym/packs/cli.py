"""``unigym pack`` CLI: list, doctor and path over the mounted packs.

The launcher owns the argparse wiring; this module owns the three
read-only commands.  Nothing here writes a pack: ``list`` renders the
mount table, ``doctor`` re-validates every declared pack (manifest, api,
requires, faces, ``BUILD_MANIFEST.txt`` against the live tree and the
code-face import), and ``path`` prints the resolved face path(s).

Every failure names the legal options (known packs / known faces) so the
CLI stays fail-closed and teachable, matching the resolver's style.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import yaml

from . import (
    PACK_API,
    PackError,
    load_manifest,
    pack_roots,
)
from .load import load_mounted_code, registration_report
from .manifest import FACE_NAMES, MANIFEST_NAME, _check_unigym_requirement
from .mount import _declared_roots

INVENTORY_NAME = "BUILD_MANIFEST.txt"
"""Per-pack tree inventory written by ``tools/build_pack.py``."""

_MISMATCH_LIMIT = 10
"""How many inventory mismatches a doctor report lists."""


SKIP_DIRS = frozenset({".git", "__pycache__"})
"""Directories the pack inventory never counts (VCS and bytecode)."""


def _count_files(path: Path) -> int:
    """Number of regular files under *path* (excluding VCS/bytecode)."""
    total = 0
    for _dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        total += len([name for name in filenames if not name.endswith(".pyc")])
    return total


def _face_counts(pack) -> str:
    """``face=count`` cells for every face the pack provides."""
    cells: list[str] = []
    for face in FACE_NAMES:
        path = pack.faces.get(face)
        if path is None or not path.is_dir():
            continue
        cells.append(f"{face}={_count_files(path)}")
    return " ".join(cells) or "-"


def _manifest_row(pack) -> tuple[str, str]:
    """``(version, api)`` cells; external packs read their manifest."""
    if pack.root is None:
        return "-", "-"
    try:
        manifest = load_manifest(pack.root)
    except PackError:
        return "?", "?"
    return manifest.version, str(manifest.api)


def cmd_list(a, extra: list[str] | None = None) -> int:
    """``unigym pack list`` -- the mount table with per-face file counts."""
    packs = pack_roots()
    rows = []
    for pack in packs:
        version, api = _manifest_row(pack)
        rows.append({
            "name": pack.name,
            "version": version,
            "api": api,
            "trust": pack.trust,
            "origin": pack.origin,
            "faces": _face_counts(pack),
            "root": None if pack.root is None else str(pack.root),
        })
    if getattr(a, "json", False):
        print(json.dumps({"ok": True, "packs": rows}))
        return 0
    print(f"{'name':<16} {'version':<9} {'api':<4} {'trust':<11} "
          f"{'origin':<5} faces")
    for row in rows:
        print(f"{row['name']:<16} {row['version']:<9} {row['api']:<4} "
              f"{row['trust']:<11} {row['origin']:<5} {row['faces']}")
    return 0


def _raw_document(root: Path) -> tuple[dict | None, str | None]:
    """Read a raw manifest mapping; ``(None, error)`` when it is unusable."""
    path = root / MANIFEST_NAME
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError) as err:
        return None, str(err)
    if not isinstance(doc, Mapping):
        return None, f"{path} is not a mapping"
    return dict(doc), None


def _inventory_kind(rel: str) -> str:
    """Inventory kind of one pack-relative path (``manifest`` at the root)."""
    kind = rel.split("/", 1)[0] if "/" in rel else "manifest"
    return kind if kind in (*FACE_NAMES, "python") else "manifest"


def _actual_inventory(root: Path) -> dict[str, tuple[str, int]]:
    """Live tree inventory: pack-relative path -> ``(kind, size)``."""
    actual: dict[str, tuple[str, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames
                             if name not in SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith(".pyc"):
                continue
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if rel == INVENTORY_NAME:
                continue
            actual[rel] = (_inventory_kind(rel), path.stat().st_size)
    return actual


def _declared_inventory(path: Path) -> tuple[dict[str, tuple[str, int]],
                                             list[str]]:
    """Parse ``BUILD_MANIFEST.txt``; return entries plus malformed lines."""
    declared: dict[str, tuple[str, int]] = {}
    bad: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(),
                                   1):
        if not line.strip():
            continue
        parts = line.split(" ", 2)
        if len(parts) != 3:
            bad.append(f"line {lineno}: {line!r}")
            continue
        kind, rel, size = parts
        try:
            declared[rel] = (kind, int(size))
        except ValueError:
            bad.append(f"line {lineno}: bad size {size!r}")
    return declared, bad


def _inventory_diff(root: Path) -> tuple[bool, str, list[str]]:
    """Compare ``BUILD_MANIFEST.txt`` with the live tree."""
    path = root / INVENTORY_NAME
    if not path.is_file():
        return False, f"missing: {path}", []
    declared, bad = _declared_inventory(path)
    actual = _actual_inventory(root)
    mismatches: list[str] = []
    for rel in sorted(declared):
        if rel not in actual:
            mismatches.append(f"missing in tree: {rel}")
        elif actual[rel] != declared[rel]:
            kind, size = declared[rel]
            now_kind, now_size = actual[rel]
            if kind != now_kind:
                mismatches.append(f"kind changed: {rel} {kind}->{now_kind}")
            if size != now_size:
                mismatches.append(f"size changed: {rel} {size}->{now_size}")
    for rel in sorted(set(actual) - set(declared)):
        mismatches.append(f"extra in tree: {rel}")
    mismatches = bad + mismatches
    if not mismatches:
        return True, f"{len(declared)} entries match the tree", []
    detail = f"{len(mismatches)} mismatch(es)"
    return False, detail, mismatches[:_MISMATCH_LIMIT]


def _doctor_one(root: Path) -> tuple[str, list[str], int]:
    """Validate one pack root; return ``(name, report lines, fail count)``."""
    lines: list[str] = []
    fails = 0

    def check(tag: str, ok: bool, detail: str) -> None:
        nonlocal fails
        lines.append(f"  {'ok  ' if ok else 'FAIL'} {tag}: {detail}")
        if not ok:
            fails += 1

    manifest = None
    try:
        manifest = load_manifest(root)
        check("manifest", True,
              f"valid ({manifest.name} {manifest.version})")
    except PackError as err:
        check("manifest", False, str(err))

    raw, raw_error = _raw_document(root)
    name = (manifest.name if manifest is not None
            else str((raw or {}).get("name") or root.name))
    if raw is None:
        check("api", False, f"unreadable manifest: {raw_error}")
        check("requires", False, f"unreadable manifest: {raw_error}")
        check("faces", False, f"unreadable manifest: {raw_error}")
    else:
        api = raw.get("api")
        api_ok = isinstance(api, int) and not isinstance(api, bool) \
            and api == PACK_API
        check("api", api_ok, f"{api!r} (kernel {PACK_API})")

        requires = raw.get("requires") or {}
        expr = requires.get("unigym") if isinstance(requires, Mapping) else None
        if expr is None:
            check("requires", True, "(no unigym constraint)")
        else:
            try:
                _check_unigym_requirement(str(expr), root / MANIFEST_NAME)
            except PackError as err:
                check("requires", False, str(err))
            else:
                check("requires", True, str(expr))

        raw_faces = raw.get("faces") or {}
        if not isinstance(raw_faces, Mapping):
            check("faces", False, "faces must be a mapping")
        else:
            cells: list[str] = []
            ok = True
            for face, value in raw_faces.items():
                if face not in FACE_NAMES:
                    ok = False
                    cells.append(f"unknown {face!r}")
                    continue
                candidate = root / str(value)
                if not isinstance(value, str) or not candidate.exists():
                    ok = False
                    cells.append(f"{face}: missing {value!r}")
                else:
                    cells.append(face)
            check("faces", ok, ", ".join(cells) or "(none)")

    inv_ok, inv_detail, mismatches = _inventory_diff(root)
    check(INVENTORY_NAME, inv_ok, inv_detail)
    for row in mismatches:
        lines.append(f"       {row}")

    trust = manifest.trust if manifest is not None else (raw or {}).get("trust")
    if trust == "code":
        try:
            loaded = load_mounted_code()
        except PackError as err:
            check("code", False, str(err))
        else:
            modules = loaded.get(name, ())
            report = registration_report()
            count = (int(report["scenes"]) + int(report["props"])
                     + int(report["terminations"]))
            check("code", True,
                  f"imported unigym_packs_{name} ({len(modules)} modules, "
                  f"{count} registrations, no hash, internal trust)")
    else:
        check("code", True, f"trust={trust!r} (no import)")
    return name, lines, fails


def _candidate_roots() -> list[Path]:
    """Every declared external pack root in mount order (deduplicated)."""
    roots: list[Path] = []
    seen: set[Path] = set()
    for root in _declared_roots():
        resolved = Path(root).expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(resolved)
    return roots


def _root_label(root: Path) -> str:
    """Best-effort pack name for a root that may carry an invalid manifest."""
    doc, _err = _raw_document(root)
    name = (doc or {}).get("name")
    return str(name) if name else root.name


def cmd_doctor(a, extra: list[str] | None = None) -> int:
    """``unigym pack doctor [--name NAME]`` -- per-pack validation."""
    try:
        roots = _candidate_roots()
    except PackError as err:
        print(f"[unigym pack] FAIL discovery: {err}")
        return 1
    names = [_root_label(root) for root in roots]
    wanted = getattr(a, "name", None)
    if wanted is not None and wanted not in names:
        print(f"[unigym pack] unknown pack {wanted!r}; known packs: {names}")
        return 1
    report = []
    total = 0
    for root, name in zip(roots, names):
        if wanted is not None and name != wanted:
            continue
        pack_name, lines, fails = _doctor_one(root)
        total += fails
        report.append({"name": pack_name, "root": str(root),
                       "ok": fails == 0, "checks": lines})
        if not getattr(a, "json", False):
            print(f"[{pack_name}] {root}")
            print("\n".join(lines))
    if getattr(a, "json", False):
        print(json.dumps({"ok": total == 0, "packs": report}))
        return 1 if total else 0
    verdict = "OK" if total == 0 else "FAIL"
    print(f"[unigym pack] {verdict} ({len(report)} pack(s), {total} failure(s))")
    return 1 if total else 0


def cmd_path(a, extra: list[str] | None = None) -> int:
    """``unigym pack path <name> [face]`` -- print the resolved path(s)."""
    packs = pack_roots()
    by_name = {pack.name: pack for pack in packs}
    pack = by_name.get(a.name)
    if pack is None:
        print(f"[unigym pack] unknown pack {a.name!r}; "
              f"known packs: {sorted(by_name)}")
        return 1
    face = getattr(a, "face", None)
    if face is None:
        for candidate in FACE_NAMES:
            path = pack.faces.get(candidate)
            print(f"{candidate}: {path if path is not None else '(absent)'}")
        return 0
    if face not in FACE_NAMES:
        print(f"[unigym pack] unknown face {face!r}; "
              f"known faces: {list(FACE_NAMES)}")
        return 1
    path = pack.faces.get(face)
    if path is None:
        available = [name for name in FACE_NAMES
                     if pack.faces.get(name) is not None]
        print(f"[unigym pack] pack {a.name!r} has no {face!r} face; "
              f"faces: {available}")
        return 1
    print(path)
    return 0


def add_pack_subparser(sub) -> None:
    """Register the ``pack`` command group on the launcher parser."""
    pack = sub.add_parser("pack", help="pack registry: list / doctor / path")
    pack_sub = pack.add_subparsers(dest="pack_cmd", required=True)

    listing = pack_sub.add_parser(
        "list", help="mounted packs: name, version, api, trust, origin, "
                     "faces with file counts")
    listing.add_argument("--json", action="store_true",
                         help="single JSON document on stdout")
    listing.set_defaults(fn=cmd_list)

    doctor = pack_sub.add_parser(
        "doctor", help="validate packs: manifest, api, requires, faces, "
                       "BUILD_MANIFEST.txt, code import (exit 1 on failure)")
    doctor.add_argument("--name", default=None,
                        help="only this pack (default: every external pack)")
    doctor.add_argument("--json", action="store_true",
                        help="single JSON document on stdout")
    doctor.set_defaults(fn=cmd_doctor)

    path = pack_sub.add_parser("path", help="print resolved pack face path(s)")
    path.add_argument("name", help="pack name (or 'repo')")
    path.add_argument("face", nargs="?", default=None,
                      help=f"one face: {', '.join(FACE_NAMES)}")
    path.set_defaults(fn=cmd_path)


__all__ = [
    "add_pack_subparser",
    "cmd_doctor",
    "cmd_list",
    "cmd_path",
]
