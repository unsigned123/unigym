"""``unigym pack`` CLI: list, path and doctor over the pack runtime.

The launcher owns the argparse wiring; this module owns the three read-only
commands.  Everything goes through :func:`unigym.packs.runtime.discover`, so
the CLI sees the same manifests, resources and activation the rest of the
system uses.  There is no implicit repository pack and no legacy mount table.

``list`` renders the runtime catalog, ``path`` prints a pack root or the
declared resources/entrypoints of one face, and ``doctor`` revalidates each
manifest, checks ``BUILD_MANIFEST.txt`` against the live tree and activates
the pack through the runtime.  Every failure names the legal choices (known
packs / known faces) so the CLI stays fail-closed and teachable.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from . import PackError
from .packset import PackSet
from .runtime import PackRuntime, discover
from .schema import FACE_NAMES, MANIFEST_NAME, PackManifestV1, load_manifest_v1

INVENTORY_NAME = "BUILD_MANIFEST.txt"
"""Per-pack tree inventory written by the external pack build tool."""

_MISMATCH_LIMIT = 10
"""How many inventory mismatches a doctor report lists."""

SKIP_DIRS = frozenset({".git", "__pycache__"})
"""Directories the pack inventory never counts (VCS and bytecode)."""


def _runtime() -> PackRuntime:
    """Discover the configured packs, including installed entry points."""
    return discover(entry_points=True)


def _provides_text(counts: Mapping[str, int]) -> str:
    """``face=count`` cells for every face the manifest provides."""
    return " ".join(f"{face}={count}" for face, count in counts.items()) or "-"


def cmd_list(a, extra: list[str] | None = None) -> int:
    """``unigym pack list`` -- packs with counts; ``--json`` adds provenance.

    The rows are a projection of :class:`~unigym.packs.packset.PackSet` over
    the CLI runtime; there is no second discovery or provenance path.
    """
    try:
        runtime = _runtime()
    except PackError as err:
        print(f"[unigym pack] FAIL discovery: {err}")
        return 1
    pack_set = PackSet.from_runtime(runtime)
    rows = []
    for pack in pack_set.packs():
        rows.append(
            {
                "name": pack.name,
                "version": pack.version,
                "api": pack.api,
                "trust": pack.trust,
                "provides": dict(pack.face_counts),
                "root": str(pack.root),
                "origin": asdict(pack.origin),
            }
        )
    if getattr(a, "json", False):
        print(json.dumps({"ok": True, "packs": rows}))
        return 0
    print(f"{'name':<16} {'version':<9} {'api':<4} {'trust':<8} provides root")
    for row in rows:
        print(
            f"{row['name']:<16} {row['version']:<9} {row['api']:<4} "
            f"{row['trust']:<8} {_provides_text(row['provides'])} {row['root']}"
        )
    return 0


def _inventory_kind(rel: str) -> str:
    """Inventory kind of one pack-relative path (``manifest`` at the root)."""
    if rel == MANIFEST_NAME:
        return "manifest"
    kind = rel.split("/", 1)[0] if "/" in rel else "manifest"
    return kind if kind in (*FACE_NAMES, "python") else "manifest"


def _actual_inventory(root: Path) -> dict[str, tuple[str, int]]:
    """Live tree inventory: pack-relative path -> ``(kind, size)``."""
    actual: dict[str, tuple[str, int]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith(".pyc"):
                continue
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if rel == INVENTORY_NAME:
                continue
            actual[rel] = (_inventory_kind(rel), path.stat().st_size)
    return actual


def _declared_inventory(path: Path) -> tuple[dict[str, tuple[str, int]], list[str]]:
    """Parse ``BUILD_MANIFEST.txt``; return entries plus malformed lines."""
    declared: dict[str, tuple[str, int]] = {}
    bad: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
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


def _doctor_one(manifest: PackManifestV1, runtime: PackRuntime) -> tuple[list[str], int]:
    """Validate one runtime manifest; return ``(report lines, fail count)``."""
    lines: list[str] = []
    fails = 0

    def check(tag: str, ok: bool, detail: str) -> None:
        nonlocal fails
        lines.append(f"  {'ok  ' if ok else 'FAIL'} {tag}: {detail}")
        if not ok:
            fails += 1

    try:
        revalidated = load_manifest_v1(manifest.root)
    except PackError as err:
        check("manifest", False, str(err))
    else:
        check("manifest", True, f"valid ({revalidated.name} {revalidated.version})")

    inv_ok, inv_detail, mismatches = _inventory_diff(manifest.root)
    check(INVENTORY_NAME, inv_ok, inv_detail)
    for row in mismatches:
        lines.append(f"       {row}")

    try:
        handle = runtime.activate(manifest.name)
    except PackError as err:
        check("activation", False, str(err))
    else:
        detail = "no manifest entrypoint" if handle.entrypoint is None else handle.entrypoint
        check("activation", True, detail)
    return lines, fails


def cmd_doctor(a, extra: list[str] | None = None) -> int:
    """``unigym pack doctor [--name NAME]`` -- per-pack runtime validation."""
    try:
        runtime = _runtime()
    except PackError as err:
        print(f"[unigym pack] FAIL discovery: {err}")
        return 1
    names = list(runtime.catalog.packs())
    wanted = getattr(a, "name", None)
    if wanted is not None and wanted not in names:
        print(f"[unigym pack] unknown pack {wanted!r}; known packs: {names}")
        return 1
    report = []
    total = 0
    for manifest in runtime.catalog.manifests:
        if wanted is not None and manifest.name != wanted:
            continue
        lines, fails = _doctor_one(manifest, runtime)
        total += fails
        report.append(
            {
                "name": manifest.name,
                "root": str(manifest.root),
                "ok": fails == 0,
                "checks": lines,
            }
        )
        if not getattr(a, "json", False):
            print(f"[{manifest.name}] {manifest.root}")
            print("\n".join(lines))
    if getattr(a, "json", False):
        print(json.dumps({"ok": total == 0, "packs": report}))
        return 1 if total else 0
    verdict = "OK" if total == 0 else "FAIL"
    print(f"[unigym pack] {verdict} ({len(report)} pack(s), {total} failure(s))")
    return 1 if total else 0


def cmd_path(a, extra: list[str] | None = None) -> int:
    """``unigym pack path <name> [face]`` -- pack root or face resources."""
    try:
        runtime = _runtime()
    except PackError as err:
        print(f"[unigym pack] FAIL discovery: {err}")
        return 1
    try:
        manifest = runtime.catalog.get(a.name)
    except PackError as err:
        print(f"[unigym pack] {err}")
        return 1
    face = getattr(a, "face", None)
    if face is None:
        print(manifest.root)
        return 0
    if face not in FACE_NAMES:
        print(f"[unigym pack] unknown face {face!r}; known faces: {list(FACE_NAMES)}")
        return 1
    items = manifest.provides.get(face, ())
    if not items:
        available = [name for name in FACE_NAMES if name in manifest.provides]
        print(f"[unigym pack] pack {a.name!r} has no {face!r} face; faces: {available}")
        return 1
    for item in items:
        if item.resource is not None:
            print(f"{item.id}: {runtime.resource(manifest.name, item.resource)}")
        else:
            print(f"{item.id}: {item.entrypoint}")
    return 0


def add_pack_subparser(sub) -> None:
    """Register the ``pack`` command group on the launcher parser."""
    pack = sub.add_parser("pack", help="pack registry: list / path / doctor")
    pack_sub = pack.add_subparsers(dest="pack_cmd", required=True)

    listing = pack_sub.add_parser(
        "list",
        help="runtime packs: name, version, api, trust, provided-item counts, "
        "root (--json adds pack-level origin)",
    )
    listing.add_argument("--json", action="store_true", help="single JSON document on stdout")
    listing.set_defaults(fn=cmd_list)

    doctor = pack_sub.add_parser(
        "doctor",
        help="validate packs: manifest, BUILD_MANIFEST.txt, runtime activation "
        "(exit 1 on failure)",
    )
    doctor.add_argument(
        "--name", default=None, help="only this pack (default: every discovered pack)"
    )
    doctor.add_argument("--json", action="store_true", help="single JSON document on stdout")
    doctor.set_defaults(fn=cmd_doctor)

    path = pack_sub.add_parser("path", help="print a pack root or its face resources/entrypoints")
    path.add_argument("name", help="pack name")
    path.add_argument("face", nargs="?", default=None, help=f"one face: {', '.join(FACE_NAMES)}")
    path.set_defaults(fn=cmd_path)


__all__ = [
    "add_pack_subparser",
    "cmd_doctor",
    "cmd_list",
    "cmd_path",
]
