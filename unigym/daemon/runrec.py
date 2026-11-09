"""daemon.runrec -- the run ledger: ``<output_dir>/run.json``.

output_dir IS the ledger: every run
records itself in its own output dir -- the daemon writes the spec
segment at spawn, completes exit+results at reap.  No central registry,
no unlink decisions: retention = the user's everyday data_out hygiene,
the filesystem is the database.

Schema ``run.v1`` (contract daemon_api_contract.md §1.21)::

    {"schema": "run.v1", "run_id": str|null,       # null = died pre-bridge
     "run_name": str, "label": str|absent,         # worker-written display tag
     "output_dir": str, "config": str, "backend": str,
     "overrides": {...}, "paused": bool, "started_ts": float, "pid": int,
     "task_face": {...}|null,                       # backfilled at reap
     "exit": {"code": int|null, "reason": str, "wall_time_s": float,
              "finished_ts": float}|absent,         # absent = not reaped
     "results": {task: {"success_rate": x, "n_episodes": n}}|absent}

Lifecycle of the FILE (the adoption protocol, dogfood-proven):

* spawn: the daemon writes ``run.json.pending-<pid>`` into the
  *predicted* out_dir -- a pid-tagged name so a second run of the same
  config can never clobber the previous run's ledger;
* the worker (collect_cli) knows the FINAL dir (it may rotate to
  ``<name>_<ts>`` when real data is present) and ADOPTS the file:
  ``os.replace`` pending -> ``<final>/run.json`` (a file move, content
  stays daemon-authored);
* the daemon follows at backfill/reap (pid-keyed lookups, field
  corrections) -- the invariant is: the ledger lives where the data
  lives.  A run dying pre-adoption keeps its pending file: the scan
  face reads both names and shows it RUNNING(stale) honestly.

Discovery = scanning ``<output_root>/*/run.json`` (+ pending files)
sorted by started_ts (§1): no index, no state, no pruning decisions.
Dirs without a ledger are not listed and not synthesized (§4).
RUNNING-stale honesty (§4): a ledger without an exit segment whose
heartbeat is expired shows state RUNNING + stale=true -- no reaper, no
verdict (liveness is the SHM heartbeat, never the pid).

Stdlib only; importable without mujoco/isaaclab.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import yaml

from ..runtime.bridge import hb_alive, probe_heartbeat

SCHEMA = "run.v1"

# STARTING spawn window: how long a ledger-only row (heartbeat probe 0.0,
# clock never started) may read STARTING after its spawn before it falls
# back to the RUNNING(stale) honesty.  Generous on purpose -- it covers
# the slowest legitimate worker startup (isaaclab scene build) -- and it
# is the ONLY gate, never the pid (the pid is not a liveness signal).
STARTING_WINDOW_S = 600.0

# exit segmentation: a reaped ledger carries exactly these keys
EXIT_KEYS = ("code", "reason", "wall_time_s", "finished_ts")


def pending_name(pid: int) -> str:
    """The pid-tagged spawn-time ledger name (adopted by the worker).

    NOTE: collect_cli mirrors this literal in its adoption step -- a
    static test keeps the two in sync.
    """
    return f"run.json.pending-{int(pid)}"


def _rec_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "run.json"


def _atomic_write(path: Path, doc: dict) -> None:
    """tmp+rename publish (a torn read can never be observed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, path)


def write_spec(
    output_dir: str | Path,
    *,
    run_name: str,
    config: str,
    backend: str,
    overrides: dict,
    paused: bool,
    started_ts: float,
    pid: int,
    task_face: dict | None,
    run_id: str | None = None,
    direct: bool = False,
    adapter_params: dict | None = None,
) -> dict:
    """Spawn-side write of the spec segment, as a PENDING file.

    The worker adopts it into the final dir; a shared-name dir with a
    previous run's ledger is never clobbered.

    ``adapter_params`` (spawn overlay of the yaml adapter section) lands
    verbatim when non-empty -- the ledger is the provenance record of
    which spawn params shaped the run (replay audit chain).
    """
    doc = {
        "schema": SCHEMA,
        "run_id": run_id,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "config": str(config),
        "backend": backend,
        "overrides": dict(overrides or {}),
        "paused": bool(paused),
        "direct": bool(direct),
        "started_ts": float(started_ts),
        "pid": int(pid),
        "task_face": task_face,
    }
    if adapter_params:
        doc["adapter_params"] = dict(adapter_params)
    _atomic_write(Path(output_dir) / pending_name(pid), doc)
    return doc


def read(output_dir: str | Path) -> dict | None:
    """The adopted ledger of a dir (None when absent/corrupt)."""
    try:
        doc = json.loads(_rec_path(output_dir).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def record_path(output_dir: str | Path, pid: int | None = None) -> Path:
    """The ledger FILE of one run in its dir (pid-aware).

    ``run.json`` when it is ours; else the run's pending file.  With a
    pid given, a foreign ``run.json`` is NEVER returned -- the
    pid-tagged pending path (possibly nonexistent) comes back instead,
    so callers see "no ledger here" instead of writing into another
    run's book (dogfood-found: a reap once completed into a previous
    run's ledger this way).
    """
    d = Path(output_dir)
    exact = d / "run.json"
    if pid is None:
        return exact
    rec = read(d)
    if rec is not None and rec.get("pid") == int(pid):
        return exact
    return d / pending_name(pid)


def read_pid(output_dir: str | Path, pid: int) -> dict | None:
    """One run's ledger off its dir, pid-aware (pending fallback)."""
    try:
        doc = json.loads(record_path(output_dir, pid).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _mutate(path: Path, fn) -> dict | None:
    """Read-modify-write helper on an exact file.

    The spec segment is never rewritten by these -- only
    backfill/completion keys change.
    """
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    fn(doc)
    _atomic_write(path, doc)
    return doc


def attach_run_id(output_dir: str | Path, pid: int, run_id: str) -> dict | None:
    """Backfill run_id once the worker names itself.

    A ledger born without one keeps null -- spawn-crash honesty.
    """
    doc = read_pid(output_dir, pid)
    if not run_id or doc is None or doc.get("run_id"):
        return doc

    def fn(d: dict) -> None:
        d["run_id"] = str(run_id)

    return _mutate(record_path(output_dir, pid), fn)


def set_label(output_dir: str | Path, pid: int, label) -> dict | None:
    """The WORKER-side label write.

    The label's single-writer principle: the run process writes its own
    run.json -- the daemon only reads.  Pid-aware like every mutation
    face; a ledgerless dir (CLI runs with no daemon spawn) is an honest
    no-op -- meta.json still carries the label.  Empty label = no-op.
    """
    if not label:
        return None

    def fn(d: dict) -> None:
        d["label"] = str(label)

    return _mutate(record_path(output_dir, pid), fn)


def failure_reason_from_tail(tail_lines: list[str]) -> str:
    """The one-line diagnosis for a failed run's ``exit.reason``.

    Heuristic over the log tail (newest last): the LAST line containing
    "Error"/"error" wins (traceback summaries name the cause); else the
    last non-empty line.  Truncated to 200 chars.
    """
    cands = [ln.strip() for ln in tail_lines if ln.strip()]
    err = [ln for ln in cands if "error" in ln.lower()]
    pick = err[-1] if err else (cands[-1] if cands else "")
    return pick[:200]


def mark_exit(
    output_dir: str | Path,
    *,
    pid: int,
    exit_code: int | None,
    signal_num: int | None = None,
    wall_time_s: float | None,
    finished_ts: float | None = None,
    results: dict | None = None,
    task_face: dict | None = None,
    log_tail: list[str] | None = None,
) -> dict | None:
    """Reap-side completion: exit + results (+task_face backfill).

    ``code`` is the process exit status; a signal death carries
    code=null + a ``reason`` (the ledger stays honest about WHO ended
    the run -- the schema has no signal key by design §1).
    ``log_tail`` (failed runs): the last log lines, kept IN the ledger
    so troubleshooting does not need the /tmp log that may vanish.
    """
    if finished_ts is None:
        finished_ts = time.time()
    if signal_num:
        code, reason = None, f"killed by signal {signal_num}"
    else:
        code, reason = int(exit_code or 0), ""
        if code and not reason and log_tail:
            reason = failure_reason_from_tail(log_tail)

    def fn(d: dict) -> None:
        d["exit"] = {
            "code": code,
            "reason": reason,
            "wall_time_s": round(float(wall_time_s or 0.0), 3),
            "finished_ts": float(finished_ts),
        }
        if log_tail:
            d["exit"]["log_tail"] = log_tail
        if results is not None:
            d["results"] = results
        if task_face is not None:
            d["task_face"] = task_face

    return _mutate(record_path(output_dir, pid), fn)


def set_output_dir(output_dir: str | Path, pid: int, actual: str | Path) -> dict | None:
    """Correct the doc's output_dir field after a relocation.

    The file moved; the field must follow or the scan/post-mortem
    faces would resolve into the stale dir.
    """

    def fn(d: dict) -> None:
        d["output_dir"] = str(actual)

    return _mutate(record_path(output_dir, pid), fn)


def results_from_report(output_dir: str | Path) -> dict:
    """Lightweight results summary off benchmark_report.json.

    Per task: ``{"success_rate": x, "n_episodes": n}`` -- the heavy
    document stays in place, the ledger carries only the verdicts.
    Missing/corrupt report (collect-only run, torn read) -> {} (§1
    tolerance: results are a bonus, never a run's identity).
    """
    try:
        rep = json.loads((Path(output_dir) / "benchmark_report.json").read_text())
        tasks = rep.get("tasks") or {}
        return {
            name: {"success_rate": t.get("success_rate"), "n_episodes": t.get("n_episodes")}
            for name, t in tasks.items()
            if isinstance(t, dict)
        }
    except (OSError, ValueError, AttributeError):
        return {}


# --------------------------------------------------------------------------- #
# spawn-side spec helpers (passthrough -> ledger fields)
# --------------------------------------------------------------------------- #
def overrides_from_passthrough(passthrough: list[str]) -> dict:
    """collect_cli override audit off the spawn argv.

    ``-s key=value`` pairs (repeatable; value kept as the raw string --
    the audit face mirrors what was typed) + ``--steps N`` mapped to
    ``collect.steps`` + the ``--adapter-params`` JSON (kept raw -- the
    spawn overlay's provenance).  Everything else (backend flags,
    cameras...) is not a config override.
    """
    out: dict = {}
    pt = list(passthrough or [])
    for i, tok in enumerate(pt):
        if tok == "-s" and i + 1 < len(pt):
            k, _, v = pt[i + 1].partition("=")
            if k:
                out[k] = v
        elif tok == "--steps" and i + 1 < len(pt):
            out["collect.steps"] = pt[i + 1]
        elif tok == "--adapter-params" and i + 1 < len(pt):
            out["adapter_params"] = pt[i + 1]
    return out


def run_name_of(config_path: str | Path, overrides: dict) -> str:
    """The run's name: yaml ``run_name`` (default: file stem).

    The ``-s run_name=...`` override wins (collect_cli semantics).
    """
    name = None
    try:
        doc = yaml.safe_load(Path(config_path).read_text()) or {}
        name = doc.get("run_name")
    except (OSError, ValueError, yaml.YAMLError):
        # audit (#5): narrowed from except-Exception -- an
        # unreadable/broken yaml falls back to the file stem (defined
        # semantics: naming is best-effort, never a spawn blocker;
        # YAMLError is NOT a ValueError subclass), but a real bug (e.g.
        # a list yaml hitting .get) must surface.
        name = None
    if not name:
        name = Path(config_path).stem
    return str(overrides.get("run_name") or name)


# --------------------------------------------------------------------------- #
# discovery: scan <output_root>/*/run.json (+ pending), design §1
# --------------------------------------------------------------------------- #
def process_alive(pid) -> bool:
    """Process signalability (kill -0) -- an INTERLOCK primitive.

    Not a liveness judgment: run liveness everywhere is the SHM
    heartbeat (bridge.hb_alive); this helper only guards operations
    that signal or delete under a live process (kill/delete
    interlocks).
    """
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True


def _read_all(d: Path) -> list[dict]:
    """Every ledger doc in the dir, one PER ATTEMPT.

    The adopted ``run.json`` first, then pre-adoption
    ``run.json.pending-*`` files (a run_name dir accumulates one
    attempt each spawn -- two failed spawns = two records, neither
    shadows the other).
    """
    docs: list[dict] = []
    seen: set = set()
    for p in sorted(d.glob("run.json*")):
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        key = (doc.get("pid"), doc.get("started_ts"))
        if key in seen:
            continue
        seen.add(key)
        docs.append(doc)
    return docs


def _read_any(d: Path) -> dict | None:
    """First ledger doc (lightweight face for pid-keyed lookups)."""
    docs = _read_all(d)
    return docs[0] if docs else None


def row_of(rec: dict) -> dict:
    """One history row off a ledger record: state derivation (§3/§4).

    Liveness is the SHM heartbeat (probe by run_id, or by owner pid for
    the spawn window where the ledger has no run_id yet): fresh ->
    RUNNING; expired -> RUNNING(stale).  The pid is never the
    judgment (a wedged or SIGSTOP'd worker with a live pid is stale
    all the same).

    STARTING covers the spawn window BEFORE the worker's first
    heartbeat: the ledger exists but the segment does not (python +
    scene startup), so the probe reads 0.0 -- the heartbeat clock has
    never started and "stale" would be a lie (the fresh-spawn bug: a
    newborn run showed RUNNING(stale) until the first bump).  The
    window is TIME-capped off the ledger's started_ts, never pid-gated
    (the pid is not a liveness signal): a run that produces no
    heartbeat within STARTING_WINDOW_S of its spawn reads honestly as
    RUNNING(stale) (no reaper, no verdict -- §4), and every other path
    converges: first bump -> normal freshness judgment, reap -> exit
    segment, registration -> the live view row takes the display.
    """
    exit_seg = rec.get("exit")
    if isinstance(exit_seg, dict) and "code" in exit_seg:
        state = "DONE" if exit_seg.get("code") == 0 else "DEAD"
        stale = False
    else:
        hb = probe_heartbeat(run_id=rec.get("run_id"), pid=rec.get("pid"))
        if hb_alive(hb):
            state, stale = "RUNNING", False
        elif hb == 0.0 and time.time() - float(rec.get("started_ts") or 0.0) < STARTING_WINDOW_S:
            state, stale = "STARTING", False  # heartbeat clock never
            # started: the spawn window before the first bump
        else:
            state, stale = "RUNNING", True  # RUNNING(stale): no reaper,
            # no verdict -- kill -9 keeps the ledger honest (design §4)
    return {
        "run_id": rec.get("run_id"),
        "run_name": rec.get("run_name"),
        "label": rec.get("label"),
        "output_dir": rec.get("output_dir"),
        "config": rec.get("config"),
        "backend": rec.get("backend"),
        "overrides": rec.get("overrides") or {},
        "paused": bool(rec.get("paused")),
        "started_ts": float(rec.get("started_ts") or 0.0),
        "pid": rec.get("pid"),
        "task_face": rec.get("task_face"),
        "exit": exit_seg if isinstance(exit_seg, dict) else None,
        "results": rec.get("results") if isinstance(rec.get("results"), dict) else None,
        "state": state,
        "stale": stale,
    }


def scan(output_root: str | Path, limit: int = 50) -> list[dict]:
    """Recent runs off the ledger scan, per contract §1.21.

    ``<output_root>/*/run.json`` (+ pending files), started_ts desc,
    top N.

    Old dirs without a ledger are silently skipped (§4: not listed, no
    error, no lazy synthesis).  Junk dirs / corrupt records are data,
    not exceptions.
    """
    root = Path(output_root)
    rows: list[dict] = []
    if not root.is_dir():
        return []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        for rec in _read_all(d):  # one row PER ATTEMPT
            rows.append(row_of(rec))
    rows.sort(key=lambda r: r["started_ts"], reverse=True)
    return rows[: max(1, int(limit))]


def find_by_run_id(output_root: str | Path, run_id: str) -> str | None:
    """The directory whose ledger names this run_id.

    Post-mortem artifact resolution.  Returns WHERE THE FILE/DATA SITS
    -- a full single-root scan; the ledger is the database, lookups are
    directory walks (data_out scale: tens of dirs).
    """
    root = Path(output_root)
    if not root.is_dir():
        return None
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        for rec in _read_all(d):
            if rec.get("run_id") == run_id:
                return str(d)
    return None


def find_dir_by_pid(output_root: str | Path, pid: int) -> str | None:
    """The directory whose ledger carries this pid (relocation lookup).

    Returns WHERE THE FILE SITS (the doc's output_dir field may be
    stale until the daemon's backfill corrects it).  The pid is the
    only stable key until the worker names itself; most recent dir
    wins (pid reuse across ancient runs).
    """
    root = Path(output_root)
    if not root.is_dir():
        return None
    for d in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if not d.is_dir():
            continue
        rec = _read_any(d)
        if rec is not None and rec.get("pid") == int(pid):
            return str(d)
    return None


def relocate(old_dir: str | Path, new_dir: str | Path, pid: int) -> bool:
    """Move the ledger file when the run's data home changed.

    Atomic rename; content stays daemon-authored -- this is
    bookkeeping, not authoring.  False when there is nothing to move.
    """
    src = record_path(old_dir, pid)
    dst = _rec_path(new_dir)
    if not src.is_file() or src == dst:
        return False
    try:
        Path(new_dir).mkdir(parents=True, exist_ok=True)
        os.replace(src, dst)
        set_output_dir(new_dir, pid, new_dir)  # the field follows the file
        return True
    except OSError:
        return False
