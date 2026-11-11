#!/usr/bin/env python
"""benchmarks subcommands for the unigym launcher.

    unigym benchmarks list [--pack X]    group tasks by pack namespace
    unigym benchmarks validate <dir>     validate one pack dir; exit 0/1

Both accept ``--json``: one JSON document on stdout, success AND failure
alike ({"ok": false, "error": ...} + exit 1; repo --json discipline).

Both commands import unigym.benchmarks.packs lazily (stdlib-only module
import, like client/server in launch.py).  ``list`` builds the configured
pack runtime and discovers the builtin namespace plus every resource-backed
``benchmarks`` catalog entry; ``validate`` loads one explicit directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _first_line(text: str, limit: int = 96) -> str:
    line = " ".join((text or "").split())
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _json_exit(msg: str) -> None:
    print(json.dumps({"ok": False, "error": msg}))
    sys.exit(1)


def cmd_list(a, _pt=None) -> int:
    from unigym.benchmarks.packs import discover_packs
    from unigym.pack_api import discover

    index = discover_packs(discover(entry_points=True))
    if a.pack is not None and a.pack not in index.packs:
        msg = f"unknown pack '{a.pack}' " f"(available: {sorted(index.packs)})"
        if getattr(a, "json", False):
            _json_exit(msg)
        sys.exit(f"[unigym benchmarks] {msg}")
    if getattr(a, "json", False):
        packs = {}
        total = 0
        for name, pack in index.packs.items():
            if a.pack is not None and name != a.pack:
                continue
            packs[name] = {
                "origin": (
                    str(pack.root)
                    if pack.root is not None
                    else "first-party (benchmarks/tasks.py " "TASK_REGISTRY)"
                ),
                "tasks": {
                    t: {"canonical": d.canonical, "kind": d.kind, "description": d.description}
                    for t, d in sorted(pack.tasks.items())
                },
            }
            total += len(pack.tasks)
        print(json.dumps({"ok": True, "packs": packs, "total": total}))
        return 0
    total = 0
    shown = 0
    for name, pack in index.packs.items():
        if a.pack is not None and name != a.pack:
            continue
        shown += 1
        origin = (
            str(pack.root)
            if pack.root is not None
            else "first-party (benchmarks/tasks.py TASK_REGISTRY)"
        )
        print(f"{name}  ({origin})")
        for task in sorted(pack.tasks):
            d = pack.tasks[task]
            total += 1
            print(f"  {d.canonical}  [{d.kind}] {_first_line(d.description)}")
    print(f"{total} task(s) in {shown} pack(s)")
    return 0


def cmd_validate(a, _pt=None) -> int:
    from unigym.benchmarks.packs import load_pack

    root = Path(a.dir)
    json_mode = getattr(a, "json", False)
    try:
        pack = load_pack(root)
    except ValueError as exc:
        lines = [ln for ln in str(exc).splitlines() if ln.startswith("  - ")]
        if json_mode:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": str(exc),
                        "errors": [ln.strip(" -").strip() for ln in lines],
                        "root": str(root),
                    }
                )
            )
            sys.exit(1)
        if lines:
            print(f"INVALID pack ({root}) -- {len(lines)} error(s):")
        else:
            print(f"INVALID pack ({root}):")
        print(exc)
        return 1
    if json_mode:
        print(
            json.dumps(
                {
                    "ok": True,
                    "pack": pack.name,
                    "version": pack.version,
                    "root": str(root),
                    "tasks": len(pack.tasks),
                }
            )
        )
        return 0
    print(f"OK pack '{pack.name}' v{pack.version or '?'} ({root}): " f"{len(pack.tasks)} task(s)")
    for task in sorted(pack.tasks):
        d = pack.tasks[task]
        print(f"  {d.canonical}  [{d.kind}] <- {d.source}")
    return 0


def add_benchmarks_subparser(sub) -> None:
    """Register the ``benchmarks`` command group (lazy CLI dispatch)."""
    bm = sub.add_parser("benchmarks", help="list / validate benchmark packs")
    bms = bm.add_subparsers(dest="benchmarks_cmd", required=True)
    bl = bms.add_parser(
        "list", help="tasks grouped by pack namespace " "(builtin + runtime benchmarks resources)"
    )
    bl.add_argument("--pack", default=None, help="only this pack (default: all)")
    bl.add_argument("--json", action="store_true", help="single JSON document on stdout")
    bl.set_defaults(fn=cmd_list)
    bv = bms.add_parser(
        "validate", help="validate one pack directory " "(exit 0 = valid, 1 = errors)"
    )
    bv.add_argument("dir", help="pack directory (must contain pack.yaml)")
    bv.add_argument(
        "--json", action="store_true", help="single JSON document on stdout (errors too)"
    )
    bv.set_defaults(fn=cmd_validate)
