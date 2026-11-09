"""daemon.queries -- typed control queries + config whitelist .

ONE implementation shared by the control API (8670) and the portal (8600
-- the browser's single source, design §4/§7d): the structured run
queries (metrics / config / report / log), the whitelisted config
listing, and the relative-config start resolution.  Both HTTP faces stay
thin route tables; the business logic lives here (and in unigym.core
below it).  Every function returns ``(code, doc)``; errors keep the
API-wide ``{"ok": false, "error": ..., "hint": ...}`` shape.

Query path (design §2/§3): the daemon attaches ONCE per request via
``unigym.core.session`` and speaks the SHM command slot's typed queries
(cmd 10-12) -- no run-side HTTP, no new channel, same-host promise.
"""
from __future__ import annotations

import json
import mimetypes
import os
import shutil
import signal
from pathlib import Path

import yaml

from unigym.core import registry as core_registry
from unigym.core import run_ops
from unigym.core.session import SessionError, SessionOpError, attach
from unigym.packs import find, pack_roots, repo_pack
from unigym.runtime.bridge import hb_alive, probe_heartbeat

from . import runrec

try:                                   # libyaml C loader: ~10x pure-python
    from yaml import CSafeLoader as _YAML_LOADER
except ImportError:                    # pragma: no cover - env without libyaml
    from yaml import SafeLoader as _YAML_LOADER

# /api/configs summary cache: key=(path, mtime); configs rarely change and
# the wizard hits this face on every portal load (77 yaml parses cost
# ~2s per request on pure-python SafeLoader).
_SUMMARY_CACHE: dict = {}


def delete_run(svc, body) -> tuple[int, dict]:
    """POST /api/runs/delete (contract §1.21 record delete).

    A run record IS its output_dir (the ledger lives inside it), so
    deleting the record removes the whole run directory.  Guardrails:
    path must resolve under the scan root; the dir must carry a run.json
    ledger (refuses to eat non-run data); a live run rejects (heartbeat
    fresh = ticking; a live pid interlock covers the wedged-owner edge
    -- kill it first).
    """
    out = str(body.get("output_dir") or "")
    if not out:
        return 400, {"ok": False, "error": "output_dir required"}
    root = Path(svc.output_root).resolve()
    target = Path(out).expanduser().resolve()
    if target != root and root not in target.parents:
        return 400, {"ok": False,
                     "error": f"output_dir escapes the scan root ({root})"}
    if not target.is_dir():
        return 404, {"ok": False, "error": f"no such run dir: {target}"}
    if not ((target / "run.json").is_file()
            or list(target.glob("run.json.pending-*"))):
        return 400, {"ok": False,
                     "error": "no run.json ledger in this dir -- refusing "
                              "to delete non-run data"}
    doc = runrec.read(target) or {}
    pid = doc.get("pid")
    if hb_alive(probe_heartbeat(run_id=doc.get("run_id"), pid=pid)):
        return 409, {"ok": False,
                     "error": "run is alive (heartbeat fresh) -- "
                              "kill it first"}
    if pid and runrec.process_alive(pid):
        return 409, {"ok": False,
                     "error": f"run pid {pid} is alive -- kill it first"}
    shutil.rmtree(target)
    return 200, {"ok": True, "deleted": str(target)}


REPO_ROOT = Path(__file__).resolve().parents[2]

KINDS = ("metrics", "config", "report", "scene", "snapshots")


def _repo_face(face: str, fallback: str) -> Path:
    """One face of the implicit repo mount (``fallback`` when absent)."""
    path = repo_pack().faces.get(face)
    return path if path is not None else REPO_ROOT / fallback


CONFIG_DIR = _repo_face("runs", "config")
PACKS_DIR = _repo_face("benchmarks", "benchmark_packs")

DEFAULT_LOG_TAIL = 4000          # characters (spawn_spec.log_tail semantic)

# task_face (contract §1.20): the config-snapshot modes collect_cli
# implements; anything else keeps the field honest at null
TASK_FACE_MODES = ("collect", "benchmark", "both")


def _err(msg: str, code: int = 400, hint: str | None = None) -> tuple[int, dict]:
    doc = {"ok": False, "error": msg}
    if hint:
        doc["hint"] = hint
    return code, doc


# --------------------------------------------------------------------------- #
# typed run queries: GET /api/runs/<id>/{metrics,config,report,log}
# --------------------------------------------------------------------------- #
def media_section(svc, run_id: str) -> dict | None:
    """Frame telemetry for one run (design §3).

    THE single implementation behind both faces (§0 one face at both ports): the
    "media" section of GET /api/runs/<id>/status and of the metrics
    query.  State lives in the daemon's shared RunFrames reader (gaps
    and rates are reader observations, not run SHM).  None when the
    reader cannot attach (run gone) -- callers keep the section honest
    by null.
    """
    gw = getattr(svc, "media", None)
    if gw is None:
        return None
    rd = gw.reader(run_id)
    if rd is None:
        return None
    try:
        return rd.media()
    finally:
        gw.release(run_id)


def query_run(svc, run_id: str, kind: str, timeout: float = 10.0,
              params: dict | None = None) -> tuple[int, dict]:
    """One typed query against a live run (command slot, req-ack).

    ``params`` rides the command payload (scene ``full`` escape hatch
    today; the metrics/config/report queries take none).
    """
    if kind not in KINDS:
        return _err(f"unknown query kind: {kind}", hint=" | ".join(KINDS))
    try:
        with attach(run_id, timeout_s=timeout) as s:
            payload = s.query(kind, timeout_s=timeout, params=params)
    except SessionOpError as e:
        msg = str(e)
        if "no ack within" in msg:      # wedged/STOPPED run: poller-friendly
            return 202, {"ok": None, "run_id": run_id, "error": msg,
                         "hint": "run is not answering queries (stalled or "
                                 "SIGSTOPped?); poll /api/runs/<id> and retry"}
        return _err(msg)
    except SessionError as e:
        msg = str(e)
        if ("bad magic" in msg or "bad version" in msg
                or "pre-v0.50 segment" in msg or "segment version" in msg):
            # UNKNOWN segment (exists, unreadable -- a foreign layout or
            # corrupt magic).  The attach face is honestly dead; say so as
            # an error doc (never a 500) and point at the live view.
            return _err(msg, 409,
                        hint="run is UNKNOWN -- its segment is unreadable; "
                             "see the error column of GET /api/runs")
        code = 404 if ("no live bridge run" in msg
                       or "cannot attach" in msg) else 400
        return _err(msg, code,
                    hint="GET /api/runs lists the live runs")
    doc = {"ok": True, "run_id": run_id, kind: payload}
    if kind == "metrics":               # media telemetry section
        doc["media"] = media_section(svc, run_id)
    return 200, doc


def query_status(svc, run_id: str) -> tuple[int, dict]:
    """GET /api/runs/<id>/status (contract §1.15, design §3).

    The media gateway's live status document (SHM status + stale
    marker + publishing camera) plus the frame telemetry section.
    Poller-friendly for the webui.

    Each poll SAMPLES the frame path once (``rd.latest()``) -- the status
    poll is an observation, so the telemetry line stays live even without
    an MJPEG stream; pure reporters (``media()``, the metrics mirror)
    never fabricate observations.
    """
    gw = getattr(svc, "media", None)
    rd = gw.reader(run_id) if gw is not None else None
    if rd is None or gw is None:
        # "no such run" would flatten unreadable into absent --
        # an UNKNOWN view row must say what is actually wrong (v049 §1)
        for r in svc.view():
            if r.get("run_id") == run_id and r.get("phase") == "UNKNOWN":
                return _err(f"run {run_id} is UNKNOWN: {r.get('error')}",
                            409, hint="see GET /api/runs")
        return _err(f"no such run: {run_id}", 404, hint="GET /api/runs")
    try:
        rd.latest()                     # one observation per status poll
        st = rd.status()
        st["media"] = rd.media()
        # watchers brief (per-tick status JSON -> top-level section, the
        # webui watchers card's data face; watchers_contract.md §10)
        brief = (st.get("metrics") or {}).get("watchers")
        if brief is not None:
            st["watchers"] = brief
    finally:
        gw.release(run_id)
    # task face (§1.20): registry output_dir -> meta.json read
    # (pure file side, no reader involvement).  read_entry first (the
    # registry file), then the daemon's cached view row: bridge.close()
    # unlinks the serve-state BEFORE the process exits, so a dying run
    # still has a view row while read_entry already comes back empty --
    # both faces must agree through that window.
    entry = core_registry.read_entry(run_id)
    od = entry.get("output_dir")
    if not od:
        for r in svc.view():
            if r.get("run_id") == run_id:
                od = r.get("output_dir")
                break
    st["task_face"] = task_face_for(od)
    return 200, {"ok": True, **st}


def query_events(svc, run_id: str, tail: int = 50,
                 kind: str | None = None) -> tuple[int, dict]:
    """GET /api/runs/<id>/events (contract §1.18, watchers.md §3).

    The run's events.jsonl tail (timeline).  Pure file read off the
    registry's output_dir -- no attach, works for done runs too
    (post-mortem viewing).
    """
    from unigym.core.session import resolve_run_id
    from unigym.watchers.timeline import read_tail
    try:
        run_id = resolve_run_id(run_id)
    except SessionError as e:
        return _err(str(e), 404, hint="GET /api/runs lists the live runs")
    entry = core_registry.read_entry(run_id)
    od = entry.get("output_dir")
    if not od:
        return _err(f"no output_dir registered for run {run_id}", 404,
                    hint="events.jsonl lives in the run output dir")
    root = Path(od)
    if not root.is_absolute():
        root = REPO_ROOT / root
    path = root / "events.jsonl"
    events = read_tail(path, tail=tail, kind=kind)
    return 200, {"ok": True, "run_id": run_id, "path": str(path),
                 "kind": kind, "events": events}


def query_log(svc, run_id: str, tail: int = DEFAULT_LOG_TAIL) \
        -> tuple[int, dict]:
    """The run's own log file tail (registry v2 log_path; plain file read).

    ``tail`` is in CHARACTERS (spawn_spec.log_tail semantics), clamped to
    1..200000 -- the run.html log pane polls this.
    """
    entry = core_registry.read_entry(run_id)
    log = entry.get("log_path")
    if not log:
        return _err(f"no log path registered for run {run_id}", 404,
                    hint="registry v2 carries log_path; runs spawned by "
                         "older builds may lack it")
    tail = max(1, min(int(tail or DEFAULT_LOG_TAIL), 200000))
    try:
        text = Path(log).read_text(errors="replace")[-tail:]
    except OSError as e:
        return _err(f"log unreadable: {e}", 404,
                    hint="the run may be gone; check GET /api/runs")
    return 200, {"ok": True, "run_id": run_id, "log_path": log,
                 "tail": text}


# --------------------------------------------------------------------------- #
# GET /api/configs: whitelisted spawn choices + summaries
# --------------------------------------------------------------------------- #
def _config_summary(path: Path) -> dict:
    """Parse one config for the wizard summary (guarded; errors are data)."""
    summary: dict = {"backend": None, "mode": None,
                     "benchmark_tasks": [], "cameras": []}
    try:
        mtime = path.stat().st_mtime
        hit = _SUMMARY_CACHE.get(str(path))
        if hit and hit[0] == mtime:
            return dict(summary, **{k: v for k, v in hit[1].items()
                                    if k != "error"})
        cfg = yaml.load(path.read_text(), Loader=_YAML_LOADER) or {}
    except (OSError, ValueError) as e:
        return dict(summary, error=f"unreadable: {e}")
    summary["backend"] = (cfg.get("env") or {}).get("backend")
    summary["mode"] = cfg.get("mode")
    summary["adapter"] = (cfg.get("adapter") or {}).get("name")
    # the wizard's label-input placeholder shows the run_name this
    # config will produce (yaml run_name, else the file stem -- the
    # same resolution runrec.run_name_of applies at spawn)
    summary["run_name"] = (str(cfg.get("run_name"))
                           if cfg.get("run_name") else path.stem)
    bench = cfg.get("benchmark") or {}
    summary["benchmark_tasks"] = [t.get("name") for t in
                                  bench.get("tasks") or []
                                  if isinstance(t, dict) and t.get("name")]
    cams = cfg.get("cameras") or []
    summary["cameras"] = [c.get("name") for c in cams
                          if isinstance(c, dict) and c.get("name")]
    _SUMMARY_CACHE[str(path)] = (mtime, summary)
    return summary


def list_configs(svc) -> tuple[int, dict]:
    """Whitelisted spawn choices + summaries (contract §1.8).

    The ``runs`` face of every mounted pack, enumerated through the pack
    resolver (external packs first, the implicit repo mount last), plus
    the ``benchmarks`` face of every mount.  Same posture as the
    artifact whitelist -- only files the daemon already knows by
    construction (the mounted packs' own faces), resolved against
    realpath; request paths can never escape (nothing user-supplied here
    at all: the listing IS the whitelist).  ``relative`` is the subpath
    under the owning pack's face root (top-level entries keep the bare
    file name); the boundary check is unchanged.

    When an external pack and the repository both provide a run config
    with the same ``run_name`` (yaml ``run_name`` else the file stem),
    the external entry wins and the repository copy is dropped.  The
    dropped count rides the payload as ``shadowed``.
    """
    configs: list[dict] = []
    origins: list[str] = []
    seen: dict[str, int] = {}
    shadowed = 0
    for pack, p in find("runs", suffix=".yaml"):
        face = pack.faces.get("runs")
        if face is None:
            continue
        real = p.resolve()
        if face.resolve() not in real.parents:
            continue
        summary = _config_summary(real)
        run_name = str(summary.get("run_name") or p.stem)
        entry = {"path": str(real),
                 "relative": str(p.relative_to(face)),
                 **summary}
        index = seen.get(run_name)
        if index is None:
            seen[run_name] = len(configs)
            configs.append(entry)
            origins.append(pack.origin)
            continue
        if origins[index] == "pack" and pack.origin == "repo":
            shadowed += 1
            continue
        if origins[index] == "repo" and pack.origin == "pack":
            configs[index] = entry
            origins[index] = pack.origin
            seen[run_name] = index
            shadowed += 1
            continue
        # duplicates inside one mount (e.g. seed variants sharing a
        # run_name) are distinct spawn choices: keep both, change nothing.
        configs.append(entry)
        origins.append(pack.origin)
    packs = []
    for pack in pack_roots():
        face = pack.faces.get("benchmarks")
        if face is None or not face.is_dir():
            continue
        for p in sorted(face.iterdir()):
            meta = p / "pack.yaml"
            if not p.is_dir() or not meta.is_file():
                continue
            try:
                doc = yaml.load(meta.read_text(), Loader=_YAML_LOADER) or {}
            except (OSError, ValueError):
                doc = {}
            tasks = sorted(x.stem for x in
                           (p / "tasks").glob("*.yaml")) \
                if (p / "tasks").is_dir() else []
            packs.append({"name": doc.get("name") or p.name,
                          "version": doc.get("version"),
                          "relative": str(p.relative_to(face)),
                          "tasks": tasks})
    return 200, {"ok": True, "configs": configs,
                 "benchmark_packs": packs, "shadowed": shadowed}


# --------------------------------------------------------------------------- #
# POST /api/start: structured spec body + whitelisted relative config
# --------------------------------------------------------------------------- #
def resolve_whitelisted_config(config: str) -> Path:
    """Relative config paths must resolve inside the repo ``runs`` face.

    Absolute paths keep the established semantics (resolved + existence-checked
    by start_run -- GET /api/configs hands out exactly such real paths).
    Returns the resolved path; raises ValueError on relative escapes
    (``..``, symlinks out of the whitelist).
    """
    p = Path(config).expanduser()
    if p.is_absolute():
        return p.resolve()
    real = (CONFIG_DIR / p).resolve()
    if CONFIG_DIR.resolve() not in real.parents:
        raise ValueError(
            f"config {config!r} resolves outside the whitelist "
            f"({CONFIG_DIR}); pick one from GET /api/configs")
    return real


def start_from_body(svc, body: dict) -> tuple[int, dict]:
    """POST /api/start core (design §3).

    Two body shapes:
    * ``{"spec": {...}}`` -- the RunSpec face (unchanged);
    * ``{"config": <path>, "overrides": {backend, gpu, steps, instruction},
      "debug": bool, ...}`` -- the wizard face: relative paths are
    whitelist-resolved, overrides map onto RunSpec fields + collect_cli
    passthrough flags.
    """
    if "spec" not in body:
        config = (body.get("config") or "").strip()
        if not config:
            return _err("'config' (or 'spec') required",
                        hint="GET /api/configs lists valid choices")
        ov = body.get("overrides") or {}
        passthrough: list[str] = []
        if ov.get("steps") is not None:
            passthrough += ["--steps", str(int(ov["steps"]))]
        if ov.get("instruction"):
            passthrough += ["--instruction", str(ov["instruction"])]
        # run label (designs/v0611): the wizard face carries overrides.label,
        # the replay face a flat top-level body.label (same thread --
        # RunSpec.label -> collect_cli --label; the run process writes
        # it into run.json/meta.json, the daemon only reads it back).
        # Validation lives in RunSpec.validate (<= 80 chars, teaching
        # 400 via the start_run ValueError path).
        label = body.get("label") or ov.get("label") or None
        # gpu override: "auto" -> alloc.pick_gpu (service); a digit -> a
        # HARD CUDA_VISIBLE_DEVICES (was silently dropped at one point
        # -- the spawn wizard could not actually pin a GPU); anything
        # else is an explicit 400, never a silent ignore.
        gpu = str(ov.get("gpu") or "").strip()
        env_hard: dict = {}
        if gpu and gpu.lower() != "auto":
            if not gpu.isdigit():
                return _err(f"overrides.gpu must be 'auto' or a GPU "
                            f"index, got {ov.get('gpu')!r}",
                            hint="e.g. \"auto\" (freest GPU) or \"2\"")
            env_hard["CUDA_VISIBLE_DEVICES"] = gpu
            gpu = None
        body = dict(body, spec={"config": config,
                                "backend": ov.get("backend") or None,
                                "gpu": gpu or None,
                                "passthrough": passthrough,
                                "env": env_hard,
                                "paused": bool(body.get("paused")),
                                "direct": bool(body.get("direct")),
                                "label": label,
                                "adapter_params":
                                    body.get("adapter_params") or {}})
    try:
        spec = run_ops.RunSpec.from_dict(body.get("spec") or {})
    except ValueError as e:
        return _err(str(e), hint="POST /api/start {'spec': {...}} or "
                                 "{'config': ..., 'overrides': {...}}")
    try:
        spec.config = str(resolve_whitelisted_config(spec.config))
    except ValueError as e:
        return _err(str(e), hint="GET /api/configs lists the whitelisted "
                                 "configs")
    warnings: list[str] = []
    if body.get("debug"):
        code, doc = _preflight(spec, warnings)
        if code:
            return code, doc
    try:
        handle, allocd = svc.start_from_spec(spec)
    except ValueError as e:
        return _err(str(e))
    except RuntimeError as e:
        return _err(str(e), 503)
    doc = {"ok": True, "handle": handle, "alloc": allocd,
           "warnings": warnings}
    if body.get("debug"):
        code, doc = _wait_debug(svc, body, handle, doc)
        if code:
            return code, doc
    return 200, doc


def _preflight(spec, warnings) -> tuple[int, dict]:
    """Server-side debug preflight (docs/designs/v026-unigymd.md §9.5).

    Defense in depth: reruns the client's preflight in the daemon
    before spawning.
    """
    from unigym.core.spawn import resolve_backend
    from unigym.scripts.client import DebugPreflightError, debug_preflight
    cfg_path = Path(spec.config).expanduser().resolve()
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except (OSError, ValueError) as e:
        return _err(f"config unreadable: {e}")
    backend = resolve_backend(cfg_path, spec.backend)
    try:
        warnings.extend(debug_preflight(cfg, backend, spec.passthrough,
                                        config_path=cfg_path))
    except DebugPreflightError as e:
        return _err(str(e), hint="fix the preflight failure above")
    return 0, {}


def _wait_debug(svc, body, handle, doc) -> tuple[int, dict]:
    """Debug flow: wait for the worker registry + attach the /info doc."""
    from unigym.runtime.bridge import serve_state_paths
    known = {str(p) for p in serve_state_paths()}
    reg = svc.wait_new_registry(
        known, float(body.get("wait_timeout") or 180.0),
        poll=float(body.get("poll") or 1.0))
    if reg is None:
        return 504, {"ok": False, "handle": handle,
                     "error": f"run worker did not come up within "
                              f"{body.get('wait_timeout') or 180.0}s "
                              f"-- check the run log: {handle['log_path']}"}
    doc["registry"] = reg
    try:
        from .media import run_info_document
        doc["info"] = run_info_document(str(reg["run_id"]), svc.media)
    except Exception as e:                 # noqa: BLE001
        return 502, {"ok": False, "handle": handle,
                     "error": f"run is up but its SHM status is "
                              f"unreadable: {e}"}
    return 0, doc


# --------------------------------------------------------------------------- #
# task face (contract §1.20): badge + benchmark-card data source
# --------------------------------------------------------------------------- #
_TASK_FACE_CACHE: dict[str, tuple[int, dict | None]] = {}


def task_face_for(output_dir: str | None) -> dict | None:
    """One run's task face from its output dir's ``meta.json`` (or None).

    ``{"mode": "collect|benchmark|both", "tasks": [names...]}`` -- read
    from the config snapshot ShardWriter flushes alongside the first
    shard (``config.mode`` absent = the "collect" default; ``tasks`` =
    ``benchmark.tasks[].name``, nameless entries skipped).  ONLY the
    meta.json speaks: old runs, collect runs smaller than one shard and
    benchmark-only runs (no CollectSink -> no flush) have no meta.json --
    those answer None instead of a guess (contract §1.20 null face).
    ``mode`` outside the trio keeps the field honest at None too.
    """
    if not output_dir:
        return None
    root = Path(output_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    meta_path = root / "meta.json"
    try:
        mtime = meta_path.stat().st_mtime_ns
    except OSError:
        return None
    key = str(root)
    cached = _TASK_FACE_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    doc: dict | None = None
    try:
        raw = json.loads(meta_path.read_text())
        cfg = raw.get("config") if isinstance(raw, dict) else None
        if isinstance(cfg, dict):
            mode = cfg.get("mode", "collect")
            bench = cfg.get("benchmark")
            tasks = [t.get("name") for t in (bench or {}).get("tasks") or []
                     if isinstance(t, dict) and t.get("name")]
            if mode in TASK_FACE_MODES:
                doc = {"mode": mode, "tasks": tasks}
    except (OSError, ValueError):
        doc = None
    # cache SUCCESSFUL reads only: meta.json is a plain json.dump (no
    # tmp+rename), so a read racing the one-shot write parses garbage --
    # caching that None would pin it forever (the mtime never changes
    # again); a None answer is one cheap stat+read to redo next poll
    if doc is not None:
        if len(_TASK_FACE_CACHE) > 512:  # bound the cache (dirs churn)
            _TASK_FACE_CACHE.clear()
        _TASK_FACE_CACHE[key] = (mtime, doc)
    return doc


def attach_task_face(rows: list[dict]) -> list[dict]:
    """Additive ``task_face`` key on the /api/runs view rows (§1.2)."""
    for r in rows:
        r["task_face"] = task_face_for(r.get("output_dir"))
    return rows


# --------------------------------------------------------------------------- #
# run history face (contract §1.21): the ledger scan + live merge
# --------------------------------------------------------------------------- #
DEFAULT_HISTORY_LIMIT = 50


def history_rows(svc, limit: int = DEFAULT_HISTORY_LIMIT) -> tuple[int, dict]:
    """GET /api/runs?history=1 (contract §1.21 ledger face).

    Scans ``<output_root>/*/run.json`` (filesystem-as-database, design
    §1) merged with the live view.

    A live run ALWAYS speaks through its SHM phase (``state`` = phase,
    RUNNING/PAUSED/DONE -- the live row is authoritative even when the
    ledger disagrees); the merged ``stale`` flag is the live row's
    heartbeat verdict (expired heartbeat on a RUNNING/PAUSED row ->
    stale -- a wedged/SIGSTOP'd worker reads honestly even with a live
    pid); ledger-only rows derive DONE/DEAD from the exit segment and
    RUNNING+stale from heartbeat expiry without one (§4: no reaper, no
    verdict).  Live rows with no ledger still appear -- they are
    running NOW, that is the more important truth.  Sorted by
    started_ts desc, top ``limit``.
    """
    limit = max(1, min(int(limit or DEFAULT_HISTORY_LIMIT), 1000))
    # v068 §4: the CACHED view (the rebuild loop publishes every 2s) --
    # a request must not pay for a rebuild; fresh=True stays reserved
    # for admin ops.
    live_by_id = {r["run_id"]: r for r in svc.view(fresh=False)
                  if r.get("run_id")}
    rows: list[tuple[float, dict]] = []
    seen: set[str] = set()
    # oversample so the merge keeps `limit` rows after live replacement
    for rec in runrec.scan(svc.output_root, limit=limit * 2 + len(live_by_id)):
        rid = rec.get("run_id")
        if rid:
            seen.add(rid)
        row = _history_row(rec, live_by_id.get(rid))
        rows.append((row["started_ts"], row))
    for rid, live in live_by_id.items():           # live with no ledger
        if rid in seen:
            continue
        rows.append((float(live.get("start_ts") or 0.0),
                     _live_history_row(live)))
    rows.sort(key=lambda t: t[0], reverse=True)
    return 200, {"ok": True, "output_root": str(svc.output_root),
                 "runs": [r for _, r in rows[:limit]]}


def _history_row(rec: dict, live: dict | None) -> dict:
    """Ledger row -> HistoryRow.

    The live row (if any) overrides the observation fields (SHM phase
    is the truth for a live run).
    """
    row = runrec.row_of(rec)
    if live is None:
        # the scan face recomputes the task face from meta.json -- the
        # ledger's spawn-time copy is usually null (flush happens later)
        row["task_face"] = task_face_for(rec.get("output_dir")) \
            if row["task_face"] is None else row["task_face"]
        row["error"] = None
        return row
    row["state"] = live.get("phase") or row["state"]
    row["stale"] = (live.get("phase") in ("RUNNING", "PAUSED")
                    and not live.get("alive"))
    row["task_face"] = live.get("task_face") or row["task_face"]
    row["pid"] = live.get("pid") or row["pid"]
    row["started_ts"] = float(live.get("start_ts") or row["started_ts"])
    # an UNKNOWN live row carries its read-failure text (v049 §1)
    row["error"] = live.get("error") or None
    return row


def _live_history_row(live: dict) -> dict:
    """Build the HistoryRow for a live run with no ledger record.

    Older build / out-of-root output_dir still deserves its row --
    HistoryRow shape, honest nulls.
    """
    out_dir = live.get("output_dir") or ""
    return {"run_id": live.get("run_id"),
            "run_name": Path(out_dir).name if out_dir else None,
            "label": None,
            "output_dir": out_dir, "config": live.get("config") or None,
            "backend": None, "overrides": {},
            "paused": False,
            "started_ts": float(live.get("start_ts") or 0.0),
            "pid": live.get("pid"),
            "task_face": live.get("task_face"),
            "exit": None, "results": None,
            "state": live.get("phase") or "RUNNING",
            "stale": (live.get("phase") in ("RUNNING", "PAUSED")
                      and not live.get("alive")),
            "error": live.get("error") or None}      # UNKNOWN rows carry it


def shutdown_unknown(svc, run: str) -> tuple[int, dict] | None:
    """Shutdown fallback for UNKNOWN runs (design v049 §1).

    The segment exists but cannot attach (older VERSION / corrupt), so
    there is no command slot -- the kill IS the pid signal, with
    pid/output_dir from the registry/ledger-merged view row.

    Returns:
        None when ``run`` is NOT an UNKNOWN row (the caller keeps the
        original attach error).
    """
    from unigym.core.session import resolve_run_id
    try:
        rid = resolve_run_id(run)
    except SessionError:
        return None
    row = next((r for r in svc.view(fresh=True)
                if r.get("run_id") == rid), None)
    if row is None or row.get("phase") != "UNKNOWN":
        return None
    pid = int(row.get("pid") or 0)
    if not pid or not runrec.process_alive(pid):
        return 409, {"ok": False,
                     "error": f"run {rid} is UNKNOWN and its pid is gone "
                              f"-- nothing to kill",
                     "hint": "delete the record instead: POST "
                             "/api/runs/delete {output_dir}"}
    os.kill(pid, signal.SIGTERM)
    return 200, {"ok": True, "run_id": rid, "pid": pid,
                 "shutdown": "sigterm (UNKNOWN run: segment unreadable, "
                             "pid signaled directly)",
                 "error": row.get("error")}


# --------------------------------------------------------------------------- #
# artifact whitelist (semantics; the portal only sends the bytes)
# --------------------------------------------------------------------------- #
def resolve_artifact(svc, run: str, job: str,
                     name: str) -> tuple[int, str, str, bytes | None]:
    """Whitelisted artifact -> (code, error_text, ctype, payload).

    Basename names only -- plus the ONE sanctioned subdirectory
    ``capture/<basename>`` (the flags face's run-side artifact dir:
    capture seg PNGs / webui inspect frames) -- resolved inside a
    directory the daemon already knows (a run's output_dir); no
    free-form paths anywhere, so traversal is impossible by design.
    ``run=`` resolution: live view row -> serve registry -> **run.json
    ledger** (post-mortem: the ledger keeps output_dir resolvable after
    the run leaves the live view -- design §3 "run.json persists").
    (The ``job=`` face is GONE with the submit queue.)
    """
    if name != Path(name).name:
        parts = Path(name).parts
        if (len(parts) != 2 or parts[0] != "capture"
                or parts[1] != Path(parts[1]).name
                or parts[1] in ("", ".", "..")):
            return 400, "bad name", "text/plain", None
    elif name in ("", ".", ".."):
        return 400, "bad name", "text/plain", None
    od = _known_output_dir(svc, run, job)
    if od is None:
        return 404, "unknown run/job", "text/plain", None
    root = Path(od).resolve()
    path = (root / name).resolve()
    if root not in path.parents or not path.is_file():
        return 404, "no such artifact", "text/plain", None
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return 200, "", ctype, path.read_bytes()


def _known_output_dir(svc, run: str, job: str) -> str | None:
    if run:
        for r in svc.view(fresh=True):
            if r.get("run_id") == run and r.get("output_dir"):
                return r["output_dir"]
        entry = core_registry.read_entry(run)
        if entry.get("output_dir"):
            return entry["output_dir"]
        return _ledger_output_dir(svc, run)    # post-mortem
    return None      # no job world -> no job-sourced whitelist


# positive ledger lookups only (run_id -> output_dir); a record never
# moves once written, so the cache cannot go stale -- same bound-and-
# clear pattern as _TASK_FACE_CACHE
_PM_CACHE: dict[str, str] = {}


def _ledger_output_dir(svc, run_id: str) -> str | None:
    """run.json ledger lookup for a finished run (post-mortem face)."""
    hit = _PM_CACHE.get(run_id)
    if hit is not None:
        return hit
    try:
        od = runrec.find_by_run_id(svc.output_root, run_id)
    except OSError:
        return None
    if od:
        if len(_PM_CACHE) > 512:               # bound (dirs churn)
            _PM_CACHE.clear()
        _PM_CACHE[run_id] = od
    return od
