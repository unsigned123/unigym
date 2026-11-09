"""daemon.api -- the unigymd control API (queries).

127.0.0.1:<api_port> (default 8670, unigym.yaml ``daemon_api_port``),
stdlib http.server in the bridge.py style -- zero new dependencies,
curl-friendly JSON everywhere (docs/designs/v026-unigymd.md §3).

the routing lives in MODULE-LEVEL :func:`api_get` /
:func:`api_post` so the portal (8600 -- the browser's single source)
serves the byte-identical ``/api/*`` face by delegation, not by
re-implementation (design §4/§7d; single-sourcing discipline §9).
docs/contracts/daemon_api_contract.md is the endpoint contract twin --
API changes go through it first.  Business logic lives in unigym.core /
daemon.queries, never here.
"""
from __future__ import annotations

import contextlib
import io
import json
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import urllib.parse
from typing import Any, cast

from unigym.core import registry as core_registry
from unigym.core import run_ops
from unigym.core.session import SessionError, SessionOpError, attach

from . import events
from . import queries


def _err(msg: str, code: int = 400, hint: str | None = None) -> tuple[int, dict]:
    doc = {"ok": False, "error": msg}
    if hint:
        doc["hint"] = hint
    return code, doc


def thread_census() -> dict:
    """Named thread buckets + total (v068 §5 observation face).

    Request/SSE handler threads arrive as generic ``Thread-N`` names;
    every unigym-owned loop carries an ``unigymd-*`` name, so the
    bucket split makes accumulation visible at a glance.
    """
    buckets: dict[str, int] = {}
    for t in threading.enumerate():
        name = t.name
        key = name if name.startswith("unigymd-") else (
            "request-workers" if name.startswith("Thread-") else name)
        buckets[key] = buckets.get(key, 0) + 1
    return {"total": threading.active_count(), "by_name": buckets}


# --------------------------------------------------------------------------- #
# routing (module level -- shared by the 8670 handler and the 8600 mirror)
# --------------------------------------------------------------------------- #
def api_get(svc, path: str, qs: dict) -> tuple[int, dict]:
    """GET /api/... -> (code, doc)."""
    route = path.rstrip("/") or "/"
    if route == "/api/health":
        return 200, {"ok": True, **svc.state, "runs": len(svc.view())}
    if route == "/api/runs":
        # "deaths" = spawn children that exited non-zero
        # (reason = exit code + run-log tail) -- additive key
        # ?history=1 -> the run.json ledger scan face (§1.21)
        # v068 §4: read faces consume the cached view -- a request
        # never triggers a rebuild (the loop publishes every 2s)
        if (qs.get("history") or [""])[0].lower() in ("1", "true"):
            try:
                limit = int((qs.get("limit") or [""])[0])
            except ValueError:
                limit = queries.DEFAULT_HISTORY_LIMIT
            return queries.history_rows(svc, limit=limit)
        return 200, {"ok": True, "runs": svc.view(),
                     "deaths": svc.deaths()}
    if route.startswith("/api/runs/"):
        rest = urllib.parse.unquote(route[len("/api/runs/"):])
        rid, _, sub = rest.partition("/")
        if not sub:
            for r in svc.view():
                if r["run_id"] == rid:
                    return 200, {"ok": True, "run": r}
            return _err(f"no such run: {rid}", 404, hint="GET /api/runs")
        if sub == "log":
            try:
                tail = int((qs.get("tail") or ["4000"])[0])
            except ValueError:
                tail = queries.DEFAULT_LOG_TAIL
            return queries.query_log(svc, rid, tail=tail)
        if sub == "status":                # status + media section
            return queries.query_status(svc, rid)
        if sub == "events":                # timeline tail (events.jsonl, §3)
            try:
                tail = int((qs.get("tail") or ["50"])[0])
            except ValueError:
                tail = 50
            return queries.query_events(svc, rid, tail=tail,
                                        kind=(qs.get("kind") or [None])[0])
        if sub == "scene":                 # curated scene face (?full=1 raw)
            full = (qs.get("full") or [""])[0].lower() in ("1", "true")
            return queries.query_run(svc, rid, "scene",
                                     params={"full": full} if full else None)
        if sub in queries.KINDS:
            return queries.query_run(svc, rid, sub)
        return _err(f"no such endpoint: /api/runs/{rid}/{sub}", 404,
                    hint="typed queries: "
                         "/api/runs/<id>/{status,events,metrics,config,"
                         "report,scene,log}")
    if route.startswith("/api/op/"):
        # rewind face read side (Round B B1): GET /api/op/<run>/snapshots
        # and GET /api/op/<run>/rewind/snapshots -- the same snapshots
        # document; the run id sits left of the trailing segment(s)
        rest = route[len("/api/op/"):]
        for tail in ("/rewind/snapshots", "/snapshots"):
            if rest.endswith(tail):
                run = rest[:len(rest) - len(tail)]
                return queries.query_run(svc, urllib.parse.unquote(run),
                                         "snapshots")
        return _err(f"no such endpoint: {route}", 404,
                    hint="GET /api/op/<run>/snapshots (rewind face list)")
    if route == "/api/configs":
        return queries.list_configs(svc)
    if route == "/api/adapters/params":
        # adapter spawn-param declarations (the registry is the single
        # truth): the webui spawn wizard renders per-param rows from
        # this, agents self-serve the same face via the session schema;
        # undeclared adapters carry declared:false (raw-JSON box there)
        from unigym.adapters import adapter_params_registry

        return 200, {"ok": True, "adapters": adapter_params_registry()}
    if route == "/api/status":
        # the pending/running/done buckets are GONE with the
        # submit queue (design §2) -- live view + deaths only;
        # v068 §5: thread census + live SSE subscription count
        return 200, {"ok": True, "runs_root": str(svc.runs_root),
                     "shm_runs": svc.view(),
                     "deaths": svc.deaths(),
                     "threads": thread_census(),
                     "sse_subscriptions": svc.events.subscriber_count}
    return _err(f"no such endpoint: {route}", 404, hint="GET /api/health")


def api_post(svc, path: str, body: dict) -> tuple[int, dict]:
    """POST /api/... -> (code, doc)."""
    route = path.rstrip("/")
    if route == "/api/start":
        return queries.start_from_body(svc, body)
    if route == "/api/submit":
        # REMOVED (contract §1.11): the queue had zero users and
        # batch semantics live on as `unigym run a.yaml b.yaml ...`
        return 410, {"ok": False,
                     "error": "api removed: the submit job queue is "
                              "gone",
                     "hint": "use `unigym run CONFIG... --bg` for "
                             "detached runs (multiple configs run in "
                             "order)"}
    if route == "/api/runs/delete":
        return queries.delete_run(svc, body)
    if route == "/api/wait":
        return _wait(svc, body)
    if route == "/api/clean":
        return _clean(svc, body)
    if route.startswith("/api/op/"):
        rest = route[len("/api/op/"):]
        for family in ("/direct/", "/watch/", "/rewind/"):
            # two-segment families: /api/op/<run>/<family><sub> -- the
            # run id sits LEFT of the family separator (rpartition below
            # would mis-split them)
            if family in rest:
                run, _, sub = rest.partition(family)
                return _op(svc, urllib.parse.unquote(run),
                           family.strip("/") + "/" + sub, body)
        run, _, op = rest.rpartition("/")
        if not run:
            return _err("expected /api/op/<run>/<op>", 404)
        return _op(svc, urllib.parse.unquote(run), op, body)
    return _err(f"no such endpoint: {route}", 404, hint="GET /api/health")


# --------------------------------------------------------------------------- #
# handlers (plain functions over the service)
# --------------------------------------------------------------------------- #
def _wait(svc, body: dict) -> tuple[int, dict]:
    """Single-shot completion facts (the CLI keeps the loop/format)."""
    from unigym.scripts import spawn_spec as ss
    state = body.get("state")
    state_path = Path(state) if state else ss.newest_state(svc.cfg)
    try:
        st = json.loads(Path(state_path).read_text())
    except (OSError, ValueError) as e:
        return _err(f"unreadable run state {state_path}: {e}")
    name = body.get("container") or st.get("container")
    log = Path(body.get("log") or st.get("log"))
    out_dir = Path(body.get("output_dir") or st.get("output_dir"))
    facts = run_ops.wait_once(log, out_dir, container=name,
                              teardown=bool(body.get("teardown", True)))
    return 200, {"ok": True, "log": str(log), "out_dir": str(out_dir),
                 "container": name, **facts}


def _clean(svc, body: dict) -> tuple[int, dict]:
    dry = bool(body.get("dry_run"))
    force = bool(body.get("force"))        # also version-mismatch/corrupt
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        removed = core_registry.clean(dry_run=dry, force=force)
    return 200, {"ok": True, "removed": removed,
                 "lines": buf.getvalue().splitlines()}


def _unknown_view_row(svc, run: str) -> dict | None:
    """View row of ``run`` when its phase is UNKNOWN, else None.

    Shared by the op pre-check (§1.14 honest 409).
    """
    from unigym.core.session import resolve_run_id
    try:
        rid = resolve_run_id(run)
    except SessionError:
        return None
    # the cached view (the rebuild loop refreshes it every
    # ``rebuild_s``) is the right source for this pre-check: a full
    # rebuild per op made every command pay the scan of the serve
    # directories, and staleness here only degrades the 409 into the
    # attach-time error it replaced -- the op still fails honestly,
    # just one layer deeper (rationale for not going fresh: the
    # command path must stay off the per-request rebuild)
    row = next((r for r in svc.view()
                if r.get("run_id") == rid), None)
    if row is None:
        # cache miss is INCONCLUSIVE: the rebuild loop may lag the serve
        # registry (a just-spawned run writes its serve json before the
        # loop's next tick).  a cheap trace check decides: trace present
        # -> one fresh rebuild for the honest classification (v049 §1);
        # no trace anywhere -> the attach-time 404 stands.
        from unigym.runtime_dirs import serve_registry_trace
        if serve_registry_trace(rid):
            row = next((r for r in svc.view(fresh=True)
                        if r.get("run_id") == rid), None)
    return row


def _op(svc, run: str, op: str, body: dict) -> tuple[int, dict]:
    timeout = float(body.get("timeout") or 10.0)
    if op in queries.KINDS:                # typed-query synonyms
        # scene accepts the {full: true} escape hatch in the op body
        # (§1.14 mirror of the GET ?full=1 face)
        params = ({"full": bool(body.get("full"))}
                  if op == "scene" and body.get("full") else None)
        return queries.query_run(svc, run, op, timeout=timeout,
                                 params=params)
    if op != "shutdown":
        # an UNKNOWN run has no readable command slot -- pause/
        # step/... would fail inside attach anyway; refuse with the
        # honest 409 instead of a generic attach error (v049 §1)
        row = _unknown_view_row(svc, run)
        if row is not None and row.get("phase") == "UNKNOWN":
            return _err(f"run {run} is UNKNOWN: {row.get('error')}", 409,
                        hint="only shutdown and record delete are "
                             "available for UNKNOWN runs")
    if (op == "camera" and body.get("camera_op") == "get"
            and "flags" in body):
        # v054: the flags form is gone at every layer (checked BEFORE
        # the attach -- the removal is structural, not run-dependent;
        # the CLI's --flags is the only surviving alias and it expands
        # before the wire)
        return _err(
            f"'flags' was removed in v054 (RenderSpec face): send "
            f"render={{'view': name}} or a RenderSpec dict -- got "
            f"flags={body.get('flags')!r}",
            hint="POST /api/op/<run>/camera {camera_op:'get', "
                 "render:{...}}")
    try:
        with attach(run, timeout_s=timeout) as s:
            if op == "status":
                return 200, {"ok": True, **s.status()}
            if op == "pause":
                return 200, {"ok": True, **s.pause(timeout)}
            if op == "resume":
                return 200, {"ok": True, **s.resume(timeout)}
            if op == "shutdown":
                return 200, {"ok": True, **s.shutdown(timeout)}
            if op == "step":
                return 200, {"ok": True,
                             **s.step(int(body.get("n") or 1), timeout)}
            if op == "inspect":
                return 200, {"ok": True,
                             **s.inspect(expr=body.get("expr"),
                                         target=body.get("target"),
                                         key=body.get("key"),
                                         timeout_s=timeout)}
            if op == "snapshot":
                return 200, {"ok": True,
                             **s.snapshot(out=body.get("out"),
                                          camera=body.get("camera"),
                                          env=body.get("env"),
                                          pos=body.get("pos"),
                                          lookat=body.get("lookat"),
                                          timeout_s=timeout)}
            if op == "rewind/snapshot":
                # rewind face (Round B B1): capture an engine snapshot
                # (camera-frame `snapshot` keeps its original meaning above)
                return 200, {"ok": True,
                             **s.rewind_snapshot(body.get("label"),
                                                 timeout_s=timeout)}
            if op == "rewind":
                # restore a snapshot; the world auto-pauses on the ack
                sid = body.get("snap_id")
                if isinstance(sid, bool) or not isinstance(sid, int):
                    return _err(
                        f"rewind: 'snap_id' must be an integer, got {sid!r}",
                        hint="POST /api/op/<run>/rewind/snapshot captures "
                             "one; GET .../snapshots lists the slots")
                return 200, {"ok": True,
                             **s.rewind(sid, timeout_s=timeout)}
            if op == "capture":
                return 200, {"ok": True,
                             **s.capture(out=body.get("out"),
                                         npz=body.get("npz"),
                                         camera=body.get("camera"),
                                         env=body.get("env"),
                                         timeout_s=timeout)}
            if op == "camera":
                if body.get("camera_op") == "set":
                    params = dict(body.get("params") or {})
                    # freecam face: top-level pos/lookat (the
                    # §1.14 op-body fields, snapshot-named) merge into
                    # params -- absolute world pose (camera_contract §12.3)
                    if body.get("pos") is not None:
                        params["pos"] = body.get("pos")
                    if body.get("lookat") is not None:
                        params["lookat"] = body.get("lookat")
                    res = s.camera_set(cast(str, body.get("name")),
                                       params,
                                       env=body.get("env"),
                                       timeout_s=timeout)
                elif body.get("camera_op") == "get_pose":
                    res = s.camera_get_pose(name=body.get("name"),
                                            env=body.get("env"),
                                            timeout_s=timeout)
                else:
                    # RenderSpec capture face (camera_contract §14):
                    # render = {"view": name} | spec fragment | full
                    # spec -- the session/manager resolve and fail-fast.
                    # pos/lookat (together, world frame) render the one
                    # frame from a transient world pose (camera aim;
                    # world rigs only)
                    res = s.camera_get(out=body.get("out"),
                                       name=body.get("name"),
                                       env=body.get("env"),
                                       render=body.get("render"),
                                       pos=body.get("pos"),
                                       lookat=body.get("lookat"),
                                       timeout_s=timeout)
                return 200, {"ok": True, **res}
            if op.startswith("direct/"):
                # director op family (/api/op/<run>/direct/<op>;
                # director_contract.md §5).  `timeout` is daemon-side and
                # must not leak into the command payload.
                sub = op[len("direct/"):]
                body = {k: v for k, v in body.items() if k != "timeout"}
                res = s.direct(sub, body, timeout)
                if sub == "status":
                    return 200, {"ok": True, **res}
                return 200, {"ok": True, "directed_op": sub, **res}
            if op.startswith("watch/"):
                # watcher op family (/api/op/<run>/watch/<sub>;
                # watchers_contract.md §7)
                sub = op[len("watch/"):]
                body = {k: v for k, v in body.items() if k != "timeout"}
                res = s.watch(sub, body, timeout)
                if sub in ("add", "update", "clear", "test"):
                    return 200, {"ok": True, "watch_op": sub, **res}
                return 200, {"ok": True, **res}
            return _err(f"unknown op: {op}", 404,
                        hint="pause resume shutdown step inspect "
                             "snapshot capture camera status direct/<op> "
                             "watch/<op> rewind/snapshot rewind snapshots")
    except SessionOpError as e:
        hint = getattr(e, "hint", None)
        return _err(str(e), hint=hint)
    except SessionError as e:
        # UNKNOWN runs keep shutdown available -- the kill is the
        # pid signal (queries.shutdown_unknown; None = not UNKNOWN, keep
        # the plain attach error)
        if op == "shutdown":
            killed = queries.shutdown_unknown(svc, run)
            if killed is not None:
                return killed
        # resolve/attach failures: 'latest' with nothing alive -> 404
        msg = str(e)
        code = 404 if ("no live bridge run" in msg
                       or "cannot attach" in msg) else 400
        return _err(msg, code)


class ApiHandler(BaseHTTPRequestHandler):
    """8670 face: a thin JSON shield over :func:`api_get`/:func:`api_post`.

    The DaemonService is reached via ``self._service()`` (bridge.py's
    pattern).
    """

    def log_message(self, format: str, *args: object) -> None:
        """Quiet per-request access logging (daemon log owns the lines)."""
        pass

    def _service(self) -> Any:
        """The daemon service attached to the enclosing HTTPServer."""
        return getattr(self.server, "service")

    def _json(self, code: int, doc: dict) -> None:
        raw = json.dumps(doc, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n > 65536:                      # same cap as the 8600 mirror:
            raise ValueError("request body too large (65536 bytes max)")
        if n <= 0:
            return {}
        try:
            doc = json.loads(self.rfile.read(n) or b"{}")
            return doc if isinstance(doc, dict) else {}
        except ValueError:
            return {}

    def do_GET(self):                          # noqa: N802 (stdlib name)
        """GET face: /api/events SSE stream, else the api_get JSON shield.

        See docs/contracts/daemon_api_contract.md §1 (endpoints §1.1-§1.25).
        """
        parts = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parts.query)
        if parts.path.rstrip("/") == "/api/events":
            # (contract §1.22): the SSE stream face -- a response
            # with no end, so it bypasses the (code, doc) JSON shield
            return events.serve_sse(self, self._service(), qs)
        try:
            return self._json(*api_get(self._service(), parts.path, qs))
        except Exception as e:                 # noqa: BLE001
            return self._json(*_err(f"{type(e).__name__}: {e}", 500))

    def do_POST(self):                         # noqa: N802 (stdlib name)
        """POST face: the api_post JSON shield (start/wait/clean/op routes).

        See docs/contracts/daemon_api_contract.md §1.
        """
        parts = urllib.parse.urlparse(self.path)
        try:
            return self._json(*api_post(self._service(), parts.path,
                                        self._body()))
        except ValueError as e:                # oversized/malformed body
            return self._json(*_err(str(e)))
        except Exception as e:                 # noqa: BLE001
            return self._json(*_err(f"{type(e).__name__}: {e}", 500))
