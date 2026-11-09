"""Startup orphan sweep (v0581): stale ``/dev/shm/unigym_*`` segments.

Evidence (the 2026-10-16 12:21 daemon restart): yesterday's quota-dead
collect_cli (pid 2269908, run 69908) survived the restart and lived 14
hours with its segment -- nothing cleaned up after a daemon restart,
and the pid-liveness checks elsewhere only look at rows that are
already in a view.

The sweep runs ONCE per daemon start (after bind, before the ready
line) and answers one question per segment: is anyone still alive and
reachable through a daemon?  A segment is swept when ALL of:

* stale       -- its heartbeat is dead for > ``min_age_s`` (default
  1h; the header hb_ts is the age truth -- a live run bumps it every
  loop iteration, so a healthy run on ANY daemon, on ANY port, is
  never stale enough to sweep; a magic-less/foreign file falls back
  to its mtime);
* unclaimed   -- its run_id is in NO reachable daemon's /api/runs
  registry (the live daemon's own view + every probed peer port),
  AND its owner pid is dead or also unregistered.

Red line (cross-daemon kills): the age gate is the primary protection
and the registry probe the second -- a run claimed by ANY reachable
daemon is kept, and the sweep is skipable entirely with
``UNIGYM_ORPHAN_SWEEP=off``.  Action: SIGTERM -> grace -> SIGKILL ->
unlink the segment, one log line each.
"""
from __future__ import annotations

import os
import signal
import struct
import time

from unigym.runtime.bridge import HDR_SIZE, MAGIC, H_HB_TS, H_PID

__all__ = ["sweep_orphans", "MIN_AGE_S", "TERM_GRACE_S"]

MIN_AGE_S = 3600.0               # 1h stale before anything is touched
TERM_GRACE_S = 5.0
SHM_PREFIX = "unigym_"


def _log(msg: str) -> None:
    print(f"[unigymd] {msg}", flush=True)


def _pid_alive(pid: int | None) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True               # someone else's live process
    except OSError:
        return False


def _header(path: str) -> tuple[float, int]:
    """(hb_ts, owner_pid) of one segment header.

    (0.0, 0) when the file is absent/short/foreign-magic (the mtime
    fallback owns age then; the pid is unknown).
    """
    try:
        with open(path, "rb") as f:
            head = f.read(HDR_SIZE)
        if len(head) == HDR_SIZE and head[:4] == MAGIC:
            hb = struct.unpack_from("<d", head, H_HB_TS)[0]
            pid = int(struct.unpack_from("<Q", head, H_PID)[0])
            return float(hb), pid
    except OSError:
        pass
    return 0.0, 0


def sweep_orphans(shm_dir: str = "/dev/shm",
                  now: float | None = None,
                  min_age_s: float = MIN_AGE_S,
                  registered: set[str] | None = None,
                  pid_alive=None,
                  log=_log, kill: bool = True,
                  term_grace_s: float = TERM_GRACE_S) -> list[dict]:
    """One sweep pass; returns the per-segment action report.

    ``registered`` -- run_ids claimed by reachable daemons (the caller
    assembles: own view + probed peers).  ``pid_alive`` injectable for
    tests.  ``kill=False`` = dry run (report only).
    """
    now = time.time() if now is None else float(now)
    registered = {str(r) for r in (registered or set())}
    alive = pid_alive or _pid_alive
    reports: list[dict] = []
    try:
        names = sorted(os.listdir(shm_dir))
    except OSError as e:
        log(f"orphan sweep skipped: {shm_dir} unreadable ({e})")
        return reports
    for name in names:
        if not name.startswith(SHM_PREFIX):
            continue
        path = os.path.join(shm_dir, name)
        run_id = name[len(SHM_PREFIX):]
        hb, pid = _header(path)
        try:
            mtime = os.stat(path).st_mtime
        except OSError:
            continue                   # vanished mid-sweep
        age = (now - hb) if hb > 0 else (now - mtime)
        rec = {"segment": name, "run_id": run_id, "pid": pid or None,
               "age_s": round(age, 1), "registered": run_id in registered}
        if age <= min_age_s:
            rec["action"] = "keep (fresh)"
            reports.append(rec)
            continue
        if run_id in registered:
            # a reachable daemon claims it -- never touched, however old
            rec["action"] = "keep (registered)"
            reports.append(rec)
            continue
        if pid and alive(pid):
            # live owner that NO daemon claims: the collect_cli ghost
            rec["action"] = "kill (live but unregistered)"
            reports.append(rec)
            if kill:
                _kill_pid(pid, term_grace_s, alive, log)
        else:
            rec["action"] = "unlink (dead owner)"
            reports.append(rec)
        if kill:
            try:
                os.unlink(path)
                log(f"orphan sweep: removed {name} (age {age / 3600:.1f}h"
                    f", pid {pid or '?'}, "
                    f"{'killed' if rec['action'].startswith('kill') else 'owner dead'})")
            except OSError as e:
                log(f"orphan sweep: unlink {name} failed: {e}")
        else:
            log(f"orphan sweep (dry): would remove {name} "
                f"(age {age / 3600:.1f}h, pid {pid or '?'})")
    return reports


def _kill_pid(pid: int, grace_s: float, alive, log=_log) -> None:
    """SIGTERM -> grace -> SIGKILL, logged (never raises)."""
    for sig, label in ((signal.SIGTERM, "TERM"), (signal.SIGKILL, "KILL")):
        try:
            os.kill(pid, sig)
        except OSError as e:
            log(f"orphan sweep: pid {pid} {label} failed: {e}")
            return
        if label == "KILL":
            return
        deadline = time.time() + grace_s
        while time.time() < deadline:
            if not alive(pid):
                return
            time.sleep(0.2)
