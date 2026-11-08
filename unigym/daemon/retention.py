"""daemon.retention -- hourly mtime sweeper for the runtime tree (v068).

Design docs/designs/v068-runtime-layout.md §6: the daemon deletes aged
spawn artifacts so the dedicated tree cannot re-grow the /tmp-root
problem it replaced:

* ``logs/`` ``state/`` ``serve/`` ``deaths/`` -- per-file policy: a file
  whose mtime is older than :data:`MAX_AGE_S` (7 days) is unlinked.
* ``watch/<run_id>/`` -- whole-directory policy: a run's watcher home
  whose NEWEST content (or the dir itself) is older than the cap is
  removed with the run's artifacts together (``shutil.rmtree``).

Protected set (design §6): anything held by a heartbeat-alive run --
the service passes every alive view row's log_path + registry_path +
watch home.  A protected path is never removed no matter its age (a
long-lived run's registry is written once at startup and must not age
out under the run).

Unreadable entries (e.g. root-owned leftovers) are skipped and counted,
never raised -- a cleanup thread must not kill the daemon.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from unigym import runtime_dirs

MAX_AGE_S = 7 * 86400.0           # design §6: one week
INTERVAL_S = 3600.0               # hourly cadence (design §6)
FILE_FAMILIES = ("logs", "state", "serve", "deaths")
WATCH_FAMILY = "watch"


def _log(msg: str) -> None:
    print(f"[unigymd] {msg}", flush=True)


def protected_paths(view_rows, cfg: dict | None = None) -> set[str]:
    """Resolved absolute paths the LIVE runs hold (design §6).

    Alive rows contribute their run log + serve registry + watch home;
    dead rows contribute nothing (their artifacts age out like any
    other).
    """
    prot: set[str] = set()
    for r in view_rows:
        if not r.get("alive"):
            continue
        for key in ("log_path", "registry_path"):
            p = r.get(key)
            if p:
                prot.add(str(Path(p).resolve()))
        rid = r.get("run_id")
        if rid:
            prot.add(str(runtime_dirs.watch_dir(rid, cfg).resolve()))
    return prot


def sweep_service_tree(cfg: dict, view_rows, log=_log) -> dict:
    """The daemon's hourly face: live-view protection + one sweep pass.

    The removal/skip summary is logged HERE (one log line per pass that
    did anything) so the caller stays branch-free.
    """
    report = sweep(runtime_dirs.runtime_root(cfg),
                   protected=protected_paths(view_rows, cfg), log=log)
    if report.get("removed") or report.get("skipped"):
        log(f"retention: removed {len(report['removed'])} "
            f"aged artifacts, skipped {report['skipped']} protected/"
            f"unreadable")
    return report


def service_loop(stop_event, sweep_fn, log=_log) -> None:
    """Hourly sweeper thread body (the service's thread target).

    The catch-all is load-bearing: a bookkeeping error must never kill
    the sweeper thread (same shape as the daemon's counted rebuild-loop
    net; v068 §6 -- counted against the v049 catch-all baseline).
    """
    while not stop_event.wait(INTERVAL_S):
        try:
            sweep_fn()
        except Exception as e:                 # noqa: BLE001 (v068 §6)
            log(f"retention sweep failed: {e!r}")


def _dir_mtime(d: Path) -> float:
    """Newest mtime of a directory's contents (the dir itself included)."""
    newest = d.stat().st_mtime
    try:
        for p in d.rglob("*"):
            try:
                newest = max(newest, p.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        pass
    return newest


def sweep(root: str | Path, protected: set[str] | None = None,
          now: float | None = None, max_age_s: float = MAX_AGE_S,
          log=_log) -> dict:
    """One sweep pass over the runtime tree; returns a report dict.

    ``protected`` -- absolute (resolved) paths + watch-home dirs held by
    live runs; always skipped.  ``now`` injectable for tests.
    """
    root = Path(root)
    prot = {str(p) for p in (Path(x).resolve() for x in (protected or set()))}
    now = time.time() if now is None else float(now)
    removed: list[str] = []
    skipped = 0
    for family in FILE_FAMILIES:
        d = root / family
        if not d.is_dir():
            continue
        try:
            entries = sorted(d.iterdir())
        except OSError as e:
            log(f"retention: {d} unreadable ({e}) -- skipped")
            skipped += 1
            continue
        for p in entries:
            if not p.is_file():
                continue
            if str(p.resolve()) in prot:
                skipped += 1
                continue
            try:
                age = now - p.stat().st_mtime
            except OSError:                    # root-owned etc: skip+count
                skipped += 1
                continue
            if age <= max_age_s:
                continue
            try:
                p.unlink()
                removed.append(str(p))
            except OSError as e:
                skipped += 1
                log(f"retention: unlink {p.name} failed: {e}")
    watch = root / WATCH_FAMILY
    if watch.is_dir():
        try:
            run_dirs = sorted(watch.iterdir())
        except OSError as e:
            log(f"retention: {watch} unreadable ({e}) -- skipped")
            run_dirs = []
        for rd in run_dirs:
            if not rd.is_dir():
                continue
            if str(rd.resolve()) in prot:
                skipped += 1
                continue
            try:
                newest = _dir_mtime(rd)
            except OSError:                    # vanished / unreadable
                skipped += 1
                continue
            if now - newest <= max_age_s:
                continue
            try:
                shutil.rmtree(rd)
                removed.append(str(rd))
            except OSError as e:
                skipped += 1
                log(f"retention: rmtree {rd.name} failed: {e}")
    return {"removed": removed, "skipped": skipped}
