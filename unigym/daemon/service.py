"""daemon.service -- DaemonService: the unigymd process body .

Owns four pieces:

* control API server   (127.0.0.1:<api_port>, routes in api.py)
* webui portal server  (127.0.0.1:<webui_port>, pages in portal.py)
* instance table       (periodic rebuild via core.registry.list_runs --
                        kill -9 safe: the next start rebuilds from
                        SHM + registries, §3)
* spawn scheduling     (/api/start spawns with alloc -> setsid, §1(2):
                        detached sessions, the daemon can never take a
                        benchmark down) + the run.json ledger (output_dir
                        spec at spawn, exit+results at reap)

Supervision is REPORT-ONLY (§1(5)): a dead run is logged and the view
marks it; no restart policy is enabled by default.  (The submit
queue -- pending/running/done jobs + the claim loop -- is GONE.)
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import cast

from unigym.core import alloc
from unigym.core import registry as core_registry
from unigym.core import run_ops, spawn as cspawn
from unigym import runtime_dirs
from unigym.runtime import bridge as _bridge
from unigym.runtime.bridge import hb_alive, probe_heartbeat

from . import death
from . import orphans
from . import queries
from . import retention
from . import runrec
from . import state as dstate
from .events import EventBus

#: a spawn that never registered a fresh heartbeat within this wall
#: budget is half-dead: killed + reaped instead of ghosting (a
#: 13:46 spawn hung unclaimed for 28 minutes)
SPAWN_HANDSHAKE_TIMEOUT_S = 120.0


def _log(msg: str) -> None:
    print(f"[unigymd] {msg}", flush=True)


def _bind_loopback(port: int, scan: bool = False) -> tuple:
    """Bind 127.0.0.1:<port>; scan=True walks upward (portal style).

    Returns (httpd, bound_port, port_note) -- port_note non-None when
    the bound port differs from the requested one.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    ports = [port] if not scan else list(range(port, port + 60))
    last: OSError | None = None
    for p in ports:
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", p), BaseHTTPRequestHandler)
            bound = int(httpd.server_address[1])  # ephemeral-aware
            return httpd, bound, (bound if (scan and bound != port) else None)
        except OSError as e:
            last = e
    raise OSError(f"no bindable port from {ports[0]}: {last}")


class DaemonService:
    """Wires the API + portal servers and the two background loops.

    the portal server also terminates ALL run media (the media
    gateway in media.py -- single browser source, design §3); the auth
    tri-state lives HERE (single point): unigym.yaml ``webui_token`` --
    none = open intranet posture, "auto" = one random token per daemon
    boot, any other non-empty string = fixed per-request token.

    ``reap_in_loop=False`` (test fixtures) hands the periodic loop
    reap's PROCESS-GLOBAL ``waitpid(-1)`` ownership back: when two
    services coexist in one process (embedded daemon + test-forked
    children) the loop reap would steal exits it cannot attribute;
    shutdown and handshake-timeout reaps stay armed.
    """

    def __init__(
        self,
        api_port: int = 0,
        webui_port: int = 0,
        runs_root: str | Path | None = None,
        poll: float = 2.0,
        rebuild_s: float = 2.0,
        write_state: bool = True,
        webui_token: str | None = None,
        stale_after: float | None = None,
        output_root: str | Path | None = None,
        handshake_timeout_s: float = SPAWN_HANDSHAKE_TIMEOUT_S,
        reap_in_loop: bool = True,
    ):
        self.poll = float(poll)
        self.rebuild_s = float(rebuild_s)
        self.handshake_timeout_s = float(handshake_timeout_s)
        # waitpid(-1) is process-global: every direct child of THIS
        # process is reapable, owned or not -- a coexisting daemon must
        # leave that ownership to the process's real parent-of-record
        self._reap_in_loop = bool(reap_in_loop)
        self.started = datetime.now().isoformat(timespec="seconds")
        self.cfg = cspawn.load_cfg()
        self.runs_root = Path(runs_root) if runs_root else cspawn.runs_root(self.cfg)
        # the ledger scan root (single, honest scope -- design §1;
        # None = the daemon cfg's output_root, relative to the repo)
        root = (
            Path(output_root)
            if output_root is not None
            else Path(self.cfg.get("output_root") or "data_out")
        )
        self.output_root = root if root.is_absolute() else queries.REPO_ROOT / root
        self._stop = threading.Event()
        self._view: list[dict] = []
        # the published read cache -- the rebuild loop
        # refreshes it on its own cadence; HTTP read faces consume it
        # WITHOUT rebuilding per request.  `_view` stays the write target
        # of every _rebuild (fresh=True admin ops included).
        self._cached_view: list[dict] = []
        self._view_lock = threading.Lock()
        # direct spawn children (pid -> spawn-handle fields) --
        # non-zero exits surface as death records (daemon/death.py)
        self._spawned: dict[int, dict] = {}
        self._spawned_lock = threading.Lock()
        # control-plane event bus (SSE fan-out, contract §1.22)
        # + the rebuild-loop edge caches (per-run baselines; pruned
        # against each fresh view so they never grow with history)
        self.events = EventBus()
        self._phase_snap: dict[str, str | None] = {}
        self._draft_ep: dict[str, int] = {}
        self._draft_ts: dict[str, float] = {}
        self._tl_off: dict[str, int] = {}

        # -- media gateway (auth single point; media.py is self-contained) --
        from .media import STALE_AFTER_S, MediaGateway

        raw = self.cfg.get("webui_token") if webui_token is None else webui_token
        tok = str(raw or "").strip()
        if tok.lower() == "auto":
            import secrets

            self.control_token: str | None = secrets.token_hex(16)
            self.auth_mode = "auto"
        elif tok:
            self.control_token = tok
            self.auth_mode = "fixed"
        else:
            self.control_token = None
            self.auth_mode = "none"
        self.media = MediaGateway(
            self.auth_mode,
            self.control_token,
            stale_after=(STALE_AFTER_S if stale_after is None else float(stale_after)),
        )
        self._webrtc = None  # lazy: aiortc imports on demand

        self.api_httpd, self.api_port, _ = _bind_loopback(int(api_port))
        from .api import ApiHandler

        self.api_httpd.RequestHandlerClass = ApiHandler
        self.api_httpd.service = self
        self.api_httpd.daemon_threads = True

        self.portal_httpd, self.portal_port, self.port_note = _bind_loopback(
            int(webui_port), scan=True
        )
        from .portal import PortalHandler

        self.portal_httpd.RequestHandlerClass = PortalHandler
        self.portal_httpd.service = self
        self.portal_httpd.daemon_threads = True

        self._write_state = bool(write_state)

    # -- lifecycle ---------------------------------------------------------
    @property
    def state(self) -> dict:
        """The discovery-file dict (pid/ports/runs_root/started)."""
        return {
            "pid": os.getpid(),
            "api_port": self.api_port,
            "webui_port": self.portal_port,
            "runs_root": str(self.runs_root),
            "started": self.started,
            "port_note": self.port_note,
        }

    def start_in_threads(self) -> None:
        """Test/embedded mode: servers + loops on background threads."""
        self._pre_start()
        for name, target in (
            ("api", self.api_httpd.serve_forever),
            ("portal", self.portal_httpd.serve_forever),
        ):
            threading.Thread(
                target=target, kwargs={"poll_interval": 0.2}, name=f"unigymd-{name}", daemon=True
            ).start()
        self._start_loops()

    def run(self) -> None:
        """Foreground mode (nohup/systemd): blocking serve loop + workers.

        The API server owns the MAIN thread; portal + loops run on
        background threads.
        """
        self._pre_start()
        self._start_loops()
        threading.Thread(
            target=self.portal_httpd.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="unigymd-portal",
            daemon=True,
        ).start()
        try:
            self.api_httpd.serve_forever(poll_interval=0.2)
        finally:
            self._shutdown()

    def _backfill_failure_evidence(self) -> None:
        """Startup healing for FAILED ledgers with an empty exit.reason.

        Rows recorded by a pre-log_tail daemon get reason+log_tail from
        their death record (which always carried the tail).  One pass,
        never raises -- bookkeeping must not kill the daemon.
        """
        try:
            deaths = {int(d["pid"]): d for d in death.scan()}
            for row in runrec.scan(self.output_root, limit=500):
                e = row.get("exit") or {}
                pid = row.get("pid")
                if not (e.get("code") or pid is None):
                    continue
                if e.get("reason") or pid not in deaths:
                    continue
                d = deaths[pid]
                lines = str(d.get("reason") or "").splitlines()
                if len(lines) < 2:
                    continue
                runrec.mark_exit(
                    row["output_dir"],
                    pid=pid,
                    exit_code=e.get("code"),
                    signal_num=None,
                    wall_time_s=e.get("wall_time_s"),
                    finished_ts=e.get("finished_ts"),
                    log_tail=lines[1:],
                )
                _log(f"ledger backfill: pid={pid} " f"reason={lines[1][:80]!r}")
        except Exception as e:  # noqa: BLE001
            _log(f"ledger backfill skipped: {e}")

    def _start_loops(self) -> None:
        threading.Thread(target=self._rebuild_loop, name="unigymd-rebuild", daemon=True).start()
        threading.Thread(target=self._retention_loop, name="unigymd-retention", daemon=True).start()

    def _pre_start(self) -> None:
        # (the runs_root pending/running/done subdirs are no
        # longer created -- the submit queue is gone, design §2; runs
        # record themselves in their own output_dir instead)
        # the artifact tree is mkdir -p'ed here (idempotent);
        # failures only log -- per-writer call sites re-run their
        # one-level mkdir, so the daemon still comes up.
        _log(f"runtime tree: {runtime_dirs.ensure_tree_safe(self.cfg, _log)}")
        if self._write_state:
            dstate.write_state(self.state)
        self._sweep_startup_orphans()
        _log(
            f"up: pid={os.getpid()} api=127.0.0.1:{self.api_port} "
            f"portal=127.0.0.1:{self.portal_port}"
            + (f" ({self.port_note})" if self.port_note else "")
            + f" runs_root={self.runs_root}"
        )
        self._backfill_failure_evidence()
        if self.control_token is not None:
            # the single-point token (semantics, one per daemon boot)
            print(
                f"[webui] portal auth={self.auth_mode} " f"token={self.control_token}", flush=True
            )
        else:
            print("[webui] portal auth=none (no token)", flush=True)

    # -- startup orphan sweep -----------------------------------
    def _probe_ports(self) -> list[int]:
        """Peer daemon API ports to consult for the orphan sweep.

        The production default (8670), the state file's claim, and the
        UNIGYM_DAEMON_PROBE_PORTS env (test daemons) -- never our own.
        """
        ports: set[int] = {8670}
        try:
            st = dstate.read_state() or {}
            if st.get("api_port"):
                ports.add(int(st["api_port"]))
        except (OSError, ValueError, TypeError):
            pass
        raw = os.environ.get("UNIGYM_DAEMON_PROBE_PORTS") or ""
        for tok in raw.split(","):
            tok = tok.strip()
            if tok.isdigit():
                ports.add(int(tok))
        ports.discard(int(self.api_port))
        return sorted(ports)

    def _sweep_startup_orphans(self) -> None:
        """Sweep stale unigym_* segments no daemon claims.

        Kill + unlink (the 12:21 restart left a collect_cli ghost
        living 14h with its segment).  Skipped entirely with
        UNIGYM_ORPHAN_SWEEP=off; the age gate (heartbeat dead > 1h)
        is the cross-daemon red line -- a healthy run on any daemon
        bumps its heartbeat every loop iteration.
        """
        if os.environ.get("UNIGYM_ORPHAN_SWEEP", "").lower() in ("0", "off", "false"):
            _log("orphan sweep disabled (UNIGYM_ORPHAN_SWEEP=off)")
            return
        registered: set[str] = set()
        try:
            registered |= {str(r.get("run_id")) for r in self._rebuild() if r.get("run_id")}
        except (OSError, ValueError, TypeError, KeyError) as e:
            # the view scan is file/json work -- a unreadable ledger or
            # registry row only shrinks the registered set (the sweep's
            # age gate protects those runs regardless)
            _log(f"orphan sweep: own view unavailable ({e!r})")
        for port in self._probe_ports():
            try:
                from urllib.request import urlopen
                import json as _json

                with urlopen(f"http://127.0.0.1:{port}/api/runs", timeout=2.0) as resp:
                    doc = _json.loads(resp.read().decode())
                for r in doc.get("runs") or []:
                    if r.get("run_id"):
                        registered.add(str(r["run_id"]))
                _log(
                    f"orphan sweep: probed peer daemon on {port} "
                    f"({len(doc.get('runs') or [])} runs)"
                )
            except (OSError, ValueError, KeyError):  # noqa: BLE001-N-A
                # peer probe: URLError/OSError (peer down), json
                # ValueError, shape KeyError -- a peer that cannot
                # answer contributes nothing to the registry set
                pass
        orphans.sweep_orphans(registered=registered)

    def stop(self) -> None:
        """Stop the daemon (servers closed, children reaped, state cleared)."""
        self._shutdown()

    def _shutdown(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            for httpd in (self.api_httpd, self.portal_httpd):
                with contextlib.suppress(Exception):
                    httpd.shutdown()
                    httpd.server_close()
            with contextlib.suppress(Exception):
                self.media.close()
            if self._webrtc is not None:
                with contextlib.suppress(Exception):
                    self._webrtc.close()
        finally:
            # the reap guarantee: every direct child is waited
            # on the way down -- no spawn dies unobserved because the
            # daemon was shutting down
            with contextlib.suppress(Exception):
                self._reap()
        if self._write_state:
            dstate.clear_state()
        _log("down")

    # -- instance table (rebuildable view) ---------------------------------
    def view(self, fresh: bool = False) -> list[dict]:
        """The instance table.

        ``fresh=True`` forces a rebuild -- ADMIN-OP ONLY (GPU alloc,
        shutdown interlocks); the HTTP read faces read the cached view
        the rebuild loop publishes (a request never pays
        for a rebuild).  ``_rebuild`` maintains ``_cached_view`` and
        ``_view`` together, so the cache alone is authoritative here.
        """
        if fresh:
            self._rebuild()
        with self._view_lock:
            return list(self._cached_view)

    def _rebuild(self) -> list[dict]:
        infos = core_registry.list_runs()
        view = [r.to_dict() for r in infos]
        # attach death records to matching view rows (additive
        # key; clean runs have none) -- old registries simply lack it.
        deaths = {int(d["pid"]): d for d in death.scan()}
        for r in view:
            d = deaths.get(int(r.get("pid") or 0))
            if d is not None:
                r["death"] = {
                    "exit_code": d.get("exit_code"),
                    "signal": d.get("signal"),
                    "reason": d.get("reason"),
                }
        # task face per row (additive key, meta.json-backed;
        # contract daemon_api_contract.md §1.2/§1.20 -- old runs -> null)
        queries.attach_task_face(view)
        self._fill_unknown_from_ledger(view)
        self._backfill_run_ids(view)
        with self._view_lock:
            prev = {r["run_id"]: r for r in self._view}
            for r in view:
                was = prev.get(r["run_id"])
                if was and was.get("alive") and not r.get("alive"):
                    _log(
                        f"run {r['run_id']} is DEAD (pid {r['pid']}) -- "
                        f"report only (no restart policy enabled)"
                    )
            self._view = view
            self._cached_view = list(view)
        return view

    def _fill_unknown_from_ledger(self, view: list[dict]) -> None:
        """Keep UNKNOWN rows actionable.

        A segment that exists but cannot be read cannot speak for
        itself, and its serve registry may already be gone (clean
        removes stale files) -- pid/output_dir then come from the
        run.json ledger.  UNKNOWN rows only (rare), so the directory
        walk stays bounded.
        """
        for r in view:
            if r.get("phase") != "UNKNOWN":
                continue
            if r.get("pid") and r.get("output_dir"):
                continue
            try:
                d = runrec.find_by_run_id(self.output_root, r["run_id"])
                rec = runrec.read(d) if d else None
            except OSError:
                rec = None
            if not rec:
                continue
            if not r.get("pid"):
                r["pid"] = rec.get("pid") or 0
                r["alive"] = hb_alive(probe_heartbeat(run_id=r.get("run_id"), pid=r["pid"] or None))
            if not r.get("output_dir"):
                r["output_dir"] = rec.get("output_dir") or ""

    def _backfill_run_ids(self, view: list[dict]) -> None:
        """Ledger run_id backfill.

        The daemon spawns before the worker names itself, so the ledger
        is born with run_id=null; the first view row that names the pid
        fixes it (one rewrite).  The row's output_dir is ALSO the truth
        for where the data (and thus the ledger) lives -- a collect_cli
        rotation moves the ledger file, so the daemon follows and
        records the new home.
        """
        with self._spawned_lock:
            pending = {pid: meta for pid, meta in self._spawned.items() if not meta.get("run_id")}
        if not pending:
            return
        by_pid = {}
        for r in view:
            try:
                by_pid[int(r.get("pid") or 0)] = r
            except (TypeError, ValueError):
                continue
        with self._spawned_lock:
            for pid, meta in pending.items():
                r = by_pid.get(pid)
                if r is None or not r.get("run_id"):
                    continue
                try:
                    self._relocate_ledger(int(pid), meta, r)
                    meta["run_id"] = r["run_id"]
                    runrec.attach_run_id(cast(str, meta.get("out_dir")), pid, r["run_id"])
                except OSError as e:
                    _log(f"run record backfill failed " f"(pid={pid}): {e!r}")

    def _relocate_ledger(self, pid: int, meta: dict, row: dict) -> None:
        """Follow the worker when the data home changed.

        collect_cli rotation moves the ledger file; the view row's
        output_dir is authoritative.  No-op when the ledger is already
        home (field corrected if the worker's plain move left it stale).
        """
        raw = row.get("output_dir")
        if not raw:
            return
        # registry output_dir is worker-cwd-relative when output_root is
        # -- normalize against the repo (task_face_for/query_events rule)
        p = Path(raw)
        home = str(p) if p.is_absolute() else str(queries.REPO_ROOT / p)
        if home == meta.get("out_dir"):
            return
        cur = runrec.find_dir_by_pid(self.output_root, pid)
        if cur and cur != home:
            runrec.relocate(cur, home, pid)
        elif not cur and meta.get("out_dir"):
            runrec.relocate(meta["out_dir"], home, pid)
        meta["out_dir"] = home
        doc = runrec.read_pid(home, pid)
        if doc is not None and doc.get("output_dir") != home:
            runrec.set_output_dir(home, pid, home)

    def deaths(self) -> list[dict]:
        """Recent spawn-child death records (A4; /api/runs + /api/status)."""
        return death.scan()

    def _rebuild_loop(self) -> None:
        while not self._stop.wait(self.rebuild_s):
            try:
                self._watch_spawn_handshakes()
                if self._reap_in_loop:
                    self._reap()
                self._publish_edges(self._rebuild())
            except Exception as e:  # noqa: BLE001
                _log(f"rebuild failed: {e!r}")

    # -- retention sweeper -------------------------------------
    def _retention_loop(self) -> None:
        """Hourly pass: aged artifacts go, live-run paths are protected.

        The thread body + protection set + summary line all live in
        retention.service_loop / sweep_service_tree (design §6).
        """
        retention.service_loop(
            self._stop,
            lambda: retention.sweep_service_tree(self.cfg, self.view(), log=_log),
            log=_log,
        )

    # -- spawn handshake watchdog -------------------------------
    def _watch_spawn_handshakes(self) -> None:
        """Spawn handshake watchdog.

        A spawned run must register a FRESH heartbeat within
        ``handshake_timeout_s`` -- past that it is a half-dead spawn:
        killed (TERM -> KILL) and reaped with its death record, never a
        ghost child burning the host for hours.
        """
        with self._spawned_lock:
            pending = [(pid, meta) for pid, meta in self._spawned.items() if not meta.get("ready")]
        for pid, meta in pending:
            try:
                fresh = hb_alive(probe_heartbeat(pid=pid))
            except OSError:
                fresh = False  # unreadable segment = not fresh
            if fresh:
                meta["ready"] = True
                continue
            age = time.time() - float(meta.get("spawned_ts") or 0.0)
            if age > self.handshake_timeout_s:
                meta["ready"] = True  # decided; do not re-enter
                _log(
                    f"spawn handshake timeout: pid={pid} never became "
                    f"healthy within {self.handshake_timeout_s:.0f}s -- "
                    f"killing the half-dead spawn (config "
                    f"{meta.get('config')})"
                )
                self._kill_spawn_child(pid)
                self._reap()

    @staticmethod
    def _kill_spawn_child(pid: int) -> None:
        """TERM -> 10s grace -> KILL (never raises)."""
        import signal as _signal

        for sig in (_signal.SIGTERM, _signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except OSError:
                return
            if sig == _signal.SIGTERM:
                deadline = time.time() + 10.0
                while time.time() < deadline:
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        return
                    time.sleep(0.2)

    # -- rebuild-loop event edges (contract §1.22) ------------------- #
    #
    # The bus lives in the daemon; the run-side state changes behind the
    # bench draft and the watcher timeline surface here as registry/file
    # observations (runs are zero-port SHM workers -- there is no
    # run->daemon channel and the design adds none).  Events are
    # NOTIFICATIONS: payloads stay light, clients pull for details.
    @staticmethod
    def _edge_key(r: dict) -> str:
        return str(r.get("run_id") or f"pid:{r.get('pid')}")

    def _publish_edges(self, view: list[dict]) -> None:
        seen = set()
        for r in view:
            key = self._edge_key(r)
            seen.add(key)
            self._phase_edge(key, r)
            self._draft_edge(key, r)
            self._watcher_tail(key, r)
        for key in list(self._phase_snap):
            if key not in seen:
                gone = self._phase_snap.pop(key)
                if gone in ("RUNNING", "PAUSED"):
                    # the row left the live view while alive -> DEAD
                    self.events.publish(
                        "run.phase",
                        run_id=None if key.startswith("pid:") else key,
                        data={"state": "DEAD", "alive": False},
                    )
        for cache in (self._draft_ep, self._draft_ts, self._tl_off):
            for key in list(cache):
                if key not in seen:
                    cache.pop(key, None)

    def _phase_edge(self, key: str, r: dict) -> None:
        """run.phase edge (adjacent-snapshot compare).

        Edge-triggered only, never a tick sample (design §2).
        """
        ph = r.get("phase")
        prev = self._phase_snap.get(key)
        self._phase_snap[key] = ph
        if prev is not None and ph != prev:
            self.events.publish(
                "run.phase",
                run_id=r.get("run_id"),
                data={"state": ph, "alive": bool(r.get("alive"))},
            )

    def _draft_edge(self, key: str, r: dict) -> None:
        """bench.draft edge (the BenchmarkSink per-episode update).

        The daemon-side face: the registry's episodes counter -- fetch
        the live draft once per edge, throttled to >=1s per run.
        """
        if not r.get("alive") or not r.get("run_id"):
            return
        if (r.get("task_face") or {}).get("mode") not in ("benchmark", "both"):
            return
        ep = int(r.get("episodes") or 0)
        prev = self._draft_ep.get(key)
        self._draft_ep[key] = ep
        if prev is None or ep <= prev:
            return
        now = time.monotonic()
        if now - self._draft_ts.get(key, 0.0) < 1.0:
            return  # ≥1s throttle per run (design §2)
        self._draft_ts[key] = now
        draft = self._query_report(r["run_id"])
        if not draft:
            return
        self.events.publish(
            "bench.draft",
            run_id=r["run_id"],
            data={
                "episodes_total": draft.get("episodes_total"),
                "tasks": {
                    name: (t or {}).get("success_rate")
                    for name, t in (draft.get("tasks") or {}).items()
                },
            },
        )

    def _query_report(self, run_id: str) -> dict | None:
        """The live benchmark draft (SHM typed query).

        Best effort: a notification must never kill the rebuild loop.
        """
        try:
            from unigym.core.session import attach

            with attach(run_id, timeout_s=5.0) as s:
                return s.query("report", timeout_s=5.0)
        except Exception:  # noqa: BLE001
            return None

    def _watcher_tail(self, key: str, r: dict) -> None:
        """watcher.fired edge (the watchers timeline append).

        The daemon-side face: a byte-offset tail of the run's
        events.jsonl; only ``kind=watcher`` lines publish (forward-only,
        no history).
        """
        od = r.get("output_dir")
        if not od or not r.get("run_id"):
            return
        p = Path(od)
        path = (p if p.is_absolute() else queries.REPO_ROOT / p) / "events.jsonl"
        try:
            size = path.stat().st_size
        except OSError:
            self._tl_off.pop(key, None)
            return
        off = self._tl_off.get(key)
        if off is None:
            self._tl_off[key] = size  # first sight: notify forward only
            return
        if size <= off:
            self._tl_off[key] = min(off, size)  # truncated: stay honest
            return
        try:
            with open(path, "rb") as f:
                f.seek(off)
                blob = f.read()
        except OSError:
            return
        self._tl_off[key] = off + len(blob)
        for line in blob.splitlines():
            try:
                doc = json.loads(line)
            except ValueError:
                continue
            if doc.get("kind") != "watcher":
                continue
            data = doc.get("data") or {}
            self.events.publish(
                "watcher.fired",
                run_id=r.get("run_id"),
                data={
                    "name": doc.get("name"),
                    "tick": doc.get("tick"),
                    "level": doc.get("level"),
                    "actions": data.get("actions"),
                    "message": data.get("message"),
                },
            )

    def _reap(self) -> None:
        """Reap finished spawn children (non-blocking).

        Detached runs re-parent to init, but the DIRECT Popen children
        of the daemon (run spawns) would otherwise linger as zombies
        and keep pid-liveness checks ambiguous.  A non-zero exit of a
        child WE spawned raises a death record (exit code + log tail);
        every reaped child completes its run.json ledger (exit +
        results) -- the reap point IS the ledger's terminal writer.
        """
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:  # no children left
                return
            except OSError:
                return
            if pid == 0:
                return
            with self._spawned_lock:
                meta = self._spawned.pop(pid, None)
            exit_code = signal_num = None
            if os.WIFEXITED(status):
                exit_code = os.WEXITSTATUS(status)
            elif os.WIFSIGNALED(status):
                signal_num = os.WTERMSIG(status)
            if meta is not None:
                doc = self._complete_ledger(pid, meta, exit_code, signal_num)
                # (contract §1.22): the reap point IS the run.exit
                # notification -- same state change as the ledger's
                # terminal write, announced with its light facts
                d = doc or {}
                self.events.publish(
                    "run.exit",
                    run_id=d.get("run_id"),
                    data={
                        "pid": pid,
                        "exit": d.get("exit")
                        or {
                            "code": None if signal_num else int(exit_code or 0),
                            "reason": (f"killed by signal {signal_num}" if signal_num else ""),
                        },
                        "results": d.get("results") or {},
                    },
                )
            if meta is not None and (exit_code or signal_num):
                doc = death.record(pid, exit_code, signal_num, meta)
                _log(
                    f"spawn child pid={pid} DIED: "
                    f"{doc['reason'].splitlines()[0]} "
                    f"(record {death.death_path(pid)}, "
                    f"log {meta.get('log_path')})"
                )
            else:
                _log(f"reaped spawn child pid={pid}")

    def _complete_ledger(
        self, pid: int, meta: dict, exit_code: int | None, signal_num: int | None
    ) -> dict | None:
        """run.json terminal write at reap.

        Exit segment + the light results summary + the task face
        backfill (meta.json exists by run end).  A ledger missing at
        the spawn-time path (pre-registration death after a
        collect_cli rotation) is located by pid; a ledger that cannot
        be found at all is logged, never raised -- reaping must not
        die on bookkeeping.
        """
        out_dir = meta.get("out_dir")
        if not out_dir:
            _log("run ledger completion skipped: no out_dir (pid meta " "predates)")
            return None
        try:
            doc = runrec.read_pid(out_dir, pid)
            if doc is None:
                moved = runrec.find_dir_by_pid(self.output_root, pid)
                if moved:
                    out_dir = moved
                    meta["out_dir"] = moved
                    if runrec.read_pid(moved, pid) is not None:
                        runrec.set_output_dir(moved, pid, moved)  # field follows
            # a run too short for a rebuild tick never got its run_id
            # backfilled -- the worker announces it in its own log
            # ("[bridge] worker ready run_id=..."); reclaim it here
            doc = runrec.read_pid(out_dir, pid)
            if doc is not None and not doc.get("run_id"):
                rid = self._run_id_from_log(meta.get("log_path"))
                if rid:
                    runrec.attach_run_id(out_dir, pid, rid)
            finished = time.time()
            failed = bool(exit_code or signal_num)
            tail = None
            if failed and meta.get("log_path"):
                # follow-up: failed runs carry their own evidence
                # -- the log tail lives IN the ledger (the /tmp log may
                # vanish; troubleshooting must not depend on it)
                tail = death.log_tail_lines(meta["log_path"]).splitlines()
            doc = runrec.mark_exit(
                out_dir,
                pid=pid,
                exit_code=exit_code,
                signal_num=signal_num,
                wall_time_s=finished - float(meta.get("spawned_ts") or finished),
                finished_ts=finished,
                results=runrec.results_from_report(out_dir),
                log_tail=tail,
                task_face=queries.task_face_for(out_dir),
            )
            if doc is None:
                _log(f"run ledger missing at reap: {out_dir} " f"(spec write failed at spawn?)")
            return doc
        except OSError as e:
            _log(f"run ledger completion failed ({out_dir}): {e!r}")
            return None

    @staticmethod
    def _run_id_from_log(log_path: str | None) -> str | None:
        """The worker's self-announced run_id at reap.

        Parses the ``[bridge] worker ready`` line -- the
        sub-rebuild-tick identity reclaim.
        """
        if not log_path:
            return None
        try:
            import re

            with open(log_path, errors="replace") as f:
                f.seek(0, 2)
                end = f.tell()
                f.seek(max(0, end - 200000))
                tail = f.read()
            m = re.search(r"worker ready run_id=(\S+)", tail)
            return m.group(1) if m else None
        except OSError:
            return None

    # -- spawn scheduling ----------------------------------------------------
    def start_from_spec(self, spec: run_ops.RunSpec) -> tuple[dict, dict]:
        """GPU alloc (when asked) -> setsid spawn.  Returns (handle, alloc).

        NO port allocation anywhere -- runs are pure SHM workers
        with zero listening sockets (design §2/§6b); only the GPU may be
        allocated (spec.gpu == "auto").
        """
        gpu_note = None
        if spec.gpu == "auto" and "CUDA_VISIBLE_DEVICES" not in spec.env:
            occupied: set[int] = set()
            for r in self.view(fresh=True):
                try:
                    if r.get("gpu") is not None:
                        occupied.add(int(r["gpu"]))
                except (TypeError, ValueError):
                    continue
            idx = alloc.pick_gpu(exclude=occupied)
            if idx is not None:
                spec.env["CUDA_VISIBLE_DEVICES"] = str(idx)
                gpu_note = idx
        handle = run_ops.start_run(spec, cfg=self.cfg)
        _log(
            f"spawned run pid={handle.pid} backend={handle.backend}"
            + (f" gpu={gpu_note}" if gpu_note is not None else "")
            + f" config={handle.config}"
        )
        # output_dir is the ledger -- write the spec segment at
        # spawn (task_face reads meta.json, which the worker flushes at
        # its first shard, so it backfills at reap; run_id backfills
        # from the first view row naming the pid).  Bookkeeping must
        # never break a spawn: failures are logged, not raised.
        overrides = runrec.overrides_from_passthrough(spec.passthrough)
        run_id = None
        run_name = None
        try:
            run_name = runrec.run_name_of(handle.config, overrides)
            runrec.write_spec(
                handle.out_dir,
                run_name=run_name,
                config=handle.config,
                backend=handle.backend,
                overrides=overrides,
                paused=bool(spec.paused),
                direct=bool(spec.direct),
                started_ts=time.time(),
                pid=handle.pid,
                task_face=queries.task_face_for(handle.out_dir),
                adapter_params=(spec.adapter_params or None),
            )
        except OSError as e:
            _log(f"run ledger write failed ({handle.out_dir}): {e!r}")
        # A4: remember our direct child so a non-zero exit can raise a
        # death record (exit code + run-log tail) instead of a silent reap
        with self._spawned_lock:
            self._spawned[int(handle.pid)] = {
                "log_path": handle.log_path,
                "config": handle.config,
                "out_dir": handle.out_dir,
                "backend": handle.backend,
                "spawned_ts": time.time(),
                "run_id": run_id,
            }
        # (contract §1.22): spawn notification -- run_id stays null
        # until the first view backfill, exactly as honest as the ledger
        self.events.publish(
            "run.spawned",
            run_id=run_id,
            data={
                "pid": handle.pid,
                "run_name": run_name,
                "config": handle.config,
                "backend": handle.backend,
                "paused": bool(spec.paused),
                "direct": bool(spec.direct),
            },
        )
        hd = handle.to_dict()
        if gpu_note is not None:
            hd["gpu"] = gpu_note
        return hd, {"env": spec.env}

    def webrtc_gateway(self):
        """Lazy WebRTC face (m-3).

        webrtc.py (and thus aiortc) imports on the FIRST
        /run/<id>/offer -- a daemon without aiortc serves everything
        else.  View page toggle calls this path.

        Lifecycle: one gateway per daemon (event-loop thread +
        pump thread started once); ``ice_servers`` rides the daemon
        cfg (unigym.yaml ``webrtc.ice_servers`` -- default
        EMPTY, camera_contract §16.6: the aiortc Google-STUN fallback
        is unreachable on the box and costs every answer a flat 5s
        aioice gather; host-only candidates answer in ~20ms).  The
        gateway reads the media gateway's SHARED RunFrames readers
        (ONE attach per run) and never touches the command slot.
        """
        if self._webrtc is None:
            # stun default EMPTY (camera_contract §16.6) -- the
            # aiortc Google-STUN fallback costs every answer a flat 5s
            from .webrtc import WebRtcGateway

            stun = tuple((self.cfg.get("webrtc") or {}).get("ice_servers") or ())
            self._webrtc = WebRtcGateway(self.media, stun_urls=stun)
            self._webrtc.start()
        return self._webrtc

    def wait_new_registry(self, known: set[str], timeout: float, poll: float = 1.0) -> dict | None:
        """Wait for a NEW serve registry whose heartbeat is fresh.

        Debug flow.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            for p in _bridge.serve_state_paths():
                if str(p) in known:
                    continue
                try:
                    st = json.loads(Path(p).read_text())
                except (OSError, ValueError):
                    continue
                if _bridge.hb_alive(
                    _bridge.probe_heartbeat(run_id=st.get("run_id"), pid=st.get("pid"))
                ):
                    return st
            time.sleep(poll)
        return None
