"""daemon.death -- spawn-child death records.

When a DIRECT Popen child of the daemon (a spawned run) exits non-zero,
the daemon writes one small json file next to the serve registries
(v068 default <runtime_dir>/deaths; UNIGYM_SERVE_DIR stays honored):

    <state_dir>/unigym-death-<pid>.json
    {schema: 1, kind: "death", pid, exit_code, signal, log_path, config,
     out_dir, backend, spawned_ts, reaped_ts, reason}

``reason`` = one line (exit code / signal) + the LAST 20 LINES of the
run's own log (the log file merges stdout and stderr -- the Popen
redirect shares one handle), which is where an NVRM OOM / import error /
config typo actually shows.  ``unigym status --json`` and GET /api/runs
+ /api/status surface the records; a run that died before its serve
registry existed is therefore no longer invisible.

Absent-data honest by construction: installs with no death files
(scan -> empty list); unreadable/partial files are skipped, never raised.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from unigym import runtime_dirs

SCHEMA = 1
REASON_TAIL_LINES = 20
GLOB = "unigym-death-*.json"


def default_dir() -> Path:
    """Directory holding death records.

    UNIGYM_SERVE_DIR stays honored (the legacy entry); otherwise
    ``<runtime_dir>/deaths``.
    """
    return runtime_dirs.deaths_dir()


def death_path(pid: int, state_dir: str | Path | None = None) -> Path:
    """Path of the death record for one pid (does not touch the file)."""
    root = Path(state_dir) if state_dir else default_dir()
    return root / f"unigym-death-{int(pid)}.json"


def log_tail_lines(log: str | Path, n: int = REASON_TAIL_LINES) -> str:
    """Last ``n`` lines of a run log; never raises (unreadable -> note)."""
    try:
        lines = Path(log).read_text(errors="replace").splitlines()
    except OSError:
        return "(log unreadable or gone)"
    tail = lines[-n:] if lines else ["(empty log)"]
    return "\n".join(tail)


def record(
    pid: int,
    exit_code: int | None,
    signal_num: int | None,
    meta: dict,
    state_dir: str | Path | None = None,
    tail_lines: int = REASON_TAIL_LINES,
) -> dict:
    """Write one death record; returns the document (also logged by caller).

    ``meta`` carries the spawn-handle fields (log_path/config/out_dir/
    backend/spawned_ts).  The record for a CLEAN exit (exit_code == 0)
    is never written -- the caller only records non-zero exits.
    """
    log = meta.get("log_path") or ""
    reason_head = (
        f"spawned run pid={pid} exited code={exit_code}"
        if signal_num is None
        else f"spawned run pid={pid} killed by signal {signal_num}"
    )
    doc = {
        "schema": SCHEMA,
        "kind": "death",
        "pid": int(pid),
        "exit_code": exit_code,
        "signal": signal_num,
        "log_path": str(log),
        "config": meta.get("config") or "",
        "out_dir": meta.get("out_dir") or "",
        "backend": meta.get("backend") or "",
        "spawned_ts": meta.get("spawned_ts"),
        "reaped_ts": time.time(),
        "reason": reason_head + "\n" + log_tail_lines(log, tail_lines),
    }
    p = death_path(pid, state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)  # idempotent
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, p)  # atomic publish
    return doc


def _load(p: Path) -> dict | None:
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if isinstance(doc, dict) and doc.get("kind") == "death":
        return doc
    return None


def scan(state_dir: str | Path | None = None) -> list[dict]:
    """All death records, oldest first.

    The scan covers ONLY ``state_dir``/default_dir -- no /tmp-root
    fallback (pre-layout records age out naturally).  Ordering is by
    reaped_ts then pid (lowest-pid-first is a lie for pid reuse);
    unreadable/partial files are skipped, never raised.
    """
    root = Path(state_dir) if state_dir else default_dir()
    paths = sorted(root.glob(GLOB))
    out = [d for p in paths if (d := _load(p)) is not None]
    return sorted(out, key=lambda d: (float(d.get("reaped_ts") or 0.0), int(d.get("pid") or 0)))


def for_pid(pid: int, state_dir: str | Path | None = None) -> dict | None:
    """Death record for one pid, or None (no record / unreadable)."""
    p = death_path(pid, state_dir)
    return _load(p) if p.is_file() else None


def clear(pid: int, state_dir: str | Path | None = None) -> bool:
    """Remove one death record (e.g. after the operator acknowledged it)."""
    try:
        death_path(pid, state_dir).unlink(missing_ok=True)
        return True
    except OSError:
        return False
