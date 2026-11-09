#!/usr/bin/env python
"""client.py -- thin CLI command layer .

Shell/core split end state: every
subcommand is a CONTROL-API REQUEST + output formatting.  There is NO
business logic here and NO unigym.core import -- the CLI never attaches
SHM, never spawns, never reads run state itself; unigymd does all of it
(docs/contracts/launcher_contract.md).  stdout shapes are the
stable contract and are byte-identical to the direct-run CLI face.

The only local process operations in the whole CLI are
`unigym daemon start|stop|status` (unigym.daemon.manage -- daemon
management, not a business path).

Transport: control API on 127.0.0.1 (port resolution in
unigym.daemon.state.api_port: UNIGYM_DAEMON_API_PORT env >
/tmp/unigym-daemon.json > unigym.yaml daemon_api_port > 8670).  When
the daemon is unreachable the CLI exits cleanly with a one-line hint:
`unigym daemon start`.

`debug_preflight` stays here (frozen tests import it): it is pure
read-only validation (registries + port probes, no spawn/attach); the
daemon re-runs the same check server-side before spawning (§9.5).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NoReturn

from unigym.scripts.config_resolve import ConfigResolveError, resolve_config

_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# control-API transport (the ONE way the CLI talks to the world)
# --------------------------------------------------------------------------- #
def _api_base() -> str:
    from unigym.daemon import state as daemon_state
    return f"http://127.0.0.1:{daemon_state.api_port()}"


class _OpError(RuntimeError):
    """A business failure reported by the daemon (message == the
    pre-daemon CLI text)."""

    #: Optional next-action hint attached by rich error faces (director §6).
    hint: str | None = None


def _api(method: str, path: str, body: dict | None = None,
         timeout: float = 180.0, exit_on_unreachable: bool = True) -> dict:
    base = _api_base()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + path, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            doc = json.loads(e.read())
            return doc if isinstance(doc, dict) else \
                {"ok": False, "error": f"HTTP {e.code}"}
        except ValueError:
            return {"ok": False, "error": f"HTTP {e.code}"}
    except (urllib.error.URLError, OSError) as e:
        msg = (f"[unigym] daemon not reachable at {base} ({e}) -- start "
               f"it with: unigym daemon start")
        if not exit_on_unreachable:
            raise _OpError(msg) from e
        sys.exit(msg)


def _resolve_or_exit(doc: dict) -> None:
    """Pre-daemon CLI face: 'latest'/attach failures carry the
    [unigym] prefix."""
    err = doc.get("error", doc)
    if isinstance(err, str) and ("no live bridge run" in err
                                 or "cannot attach" in err):
        sys.exit(f"[unigym] {err}")


def _op(run: str, op: str, body: dict | None = None,
        timeout: float = 15.0) -> dict:
    doc = _api("POST", f"/api/op/{urllib.parse.quote(run)}/{op}",
               body or {}, timeout=timeout + 5.0)
    if not doc.get("ok"):
        _resolve_or_exit(doc)
        e = _OpError(doc.get("error", doc))
        if isinstance(doc, dict) and doc.get("hint"):
            e.hint = doc["hint"]        # next-action face (director §6)
        raise e
    return doc


# --------------------------------------------------------------------------- #
# shared formatting helpers (byte-stable since the daemon migration)
# --------------------------------------------------------------------------- #
_RUN_FIELDS = ("run_id", "phase", "step", "episodes", "port", "pid",
               "camera", "config")


def _rows_from_runs(runs: list[dict]) -> list[tuple]:
    """API run dicts -> live_rows row tuples (table/--json shape)."""
    return [(r.get("run_id", ""), r.get("phase", "?"),
             r.get("step", 0), r.get("episodes", 0), r.get("port", 0),
             r.get("pid", 0), r.get("camera") or "", r.get("config") or "")
            for r in runs]


def live_rows() -> list[tuple]:
    """(run_id, phase, step, episodes, port, pid, camera, config) rows
    via the daemon's merged view."""
    doc = _api("GET", "/api/runs", timeout=15)
    return _rows_from_runs(doc.get("runs") or [])


def _kv_params(pairs: list[str]) -> dict:
    """`k=v...` -> dict; values parsed as JSON, falling back to string."""
    params = {}
    for kv in pairs:
        k, _, v = kv.partition("=")
        if not _:
            sys.exit(f"[unigym camera] expected k=v, got '{kv}'")
        try:
            params[k] = json.loads(v)
        except ValueError:
            params[k] = v
    return params


def _json_exit(a, msg: str) -> NoReturn:
    """ux-9 --json error face: one JSON document on stdout, exit 1."""
    print(json.dumps({"ok": False, "error": msg}))
    sys.exit(1)


def _api_json(a, method: str, path: str, body: dict | None = None,
              timeout: float = 180.0) -> dict:
    """``_api`` for --json faces: business AND transport failures come
    back as one error document (never the plain-text daemon hint)."""
    try:
        return _api(method, path, body, timeout=timeout,
                    exit_on_unreachable=False)
    except _OpError as e:
        _json_exit(a, str(e))


def _op_json(a, run: str, op: str, body: dict | None = None,
             timeout: float = 15.0) -> dict:
    """``_op`` for --json faces: every failure (dead run, op rejection,
    daemon unreachable) -> one {"ok": false, "error", ...} document."""
    try:
        doc = _api("POST", f"/api/op/{urllib.parse.quote(run)}/{op}",
                   body or {}, timeout=timeout + 5.0,
                   exit_on_unreachable=False)
    except _OpError as e:
        _json_exit(a, str(e))
    if not isinstance(doc, dict) or not doc.get("ok"):
        _json_exit(a, str(doc.get("error", doc))
                   if isinstance(doc, dict) else str(doc))
    return doc


def _render_scene(doc: dict, node: str | None = None) -> str:
    """Human face of the scene tree (+ live state).  JSON-adjacent,
    copyable -- agents parse --node/--json, humans eyeball this."""
    tree = doc.get("tree") or {}
    st = doc.get("state") or {}
    out: list[str] = []
    rob_t = tree.get("robot") or {}
    rob_s = st.get("robot") or {}
    if node in (None, "robot"):
        groups = rob_t.get("joint_groups") or {}
        gsum = " / ".join(f"{k} {len(v)}" for k, v in groups.items())
        out.append(f"robot {rob_t.get('name', '?')} · base "
                   f"{rob_t.get('base', '?')} · joints "
                   f"{rob_t.get('n_joints', '?')} ({gsum})")
        if node == "robot":
            for k, v in (rob_s or {}).items():
                out.append(f"  {k}: {v}")
            return "\n".join(out)
    props = tree.get("props") or {}
    pstate = st.get("props") or {}
    show_props = {k: v for k, v in props.items()
                  if node in (None, k)} if node else props
    for name, meta in show_props.items():
        joints = meta.get("joints") or []
        out.append(f"prop {name} · {meta.get('type', '?')}"
                   + (f" · joints: {', '.join(joints)}" if joints else ""))
        ps = (pstate.get(name) or {})
        if ps.get("pos") is not None:
            out.append(f"    pos {['%+.3f' % x for x in ps['pos']]}"
                       + (f"  quat {['%+.2f' % x for x in (ps.get('quat') or [])]}"
                          if ps.get("quat") else ""))
        for jn, jv in (ps.get("joints") or {}).items():
            out.append(f"    {jn} pos={jv.get('pos', 0):+.3f} "
                       f"vel={jv.get('vel', 0):+.3f}")
    cons = tree.get("contacts") or {}
    cstate = st.get("contacts") or {}
    for name, meta in cons.items():
        if node not in (None, name):
            continue
        cs = cstate.get(name) or {}
        out.append(f"contact {name} · on={cs.get('on', False)} · "
                   f"members={len(meta.get('members') or [])}"
                   + (f" · force_z={cs.get('force_z', 0):+.1f}"
                      if "force_z" in cs else ""))
    cams = tree.get("cameras") or []
    if node in (None, "cameras") and cams:
        out.append(f"cameras: {', '.join(cams)}")
    if node and not out:
        out.append(f"no such scene node: {node!r} (known: robot, "
                   + ", ".join(f"props/{p}" for p in props)
                   + ", ".join(f"contacts/{c}" for c in cons) + ", cameras)")
    return "\n".join(out)


def cmd_scene(a, extra: list[str]) -> int:
    """`unigym scene <run> [--full] [--node N] [--json]` -> the curated
    scene tree + live state (agent face of GET /api/runs/<id>/scene;
    PARKED-capable -- the staging workflow reads it while paused)."""
    body = {"full": bool(a.full)}
    doc = _op(a.run, "scene", body, timeout=a.timeout)
    if getattr(a, "json", False):
        print(json.dumps(doc.get("scene") or doc, indent=1, default=str))
        return 0
    node = a.node
    if node and node.startswith("props/"):
        node = node[len("props/"):]
    print(_render_scene(doc.get("scene") or doc, node))
    return 0


def cmd_runs(a, extra: list[str]) -> int:
    if getattr(a, "history", False):
        return _runs_history(a)
    rows = live_rows()
    if getattr(a, "json", False):
        print(json.dumps({"runs": [dict(zip(_RUN_FIELDS, r))
                                   for r in rows]}))
        return 0
    if not rows:
        print("no live bridge runs (start with: unigym run CONFIG --serve)")
        return 0
    w = max(len(r[0]) for r in rows)
    print(f"{'RUN_ID':{w}}  PHASE    STEP      EPISODES  PORT  PID  "
          f"CAMERA  CONFIG")
    for r in rows:
        print(f"{r[0]:{w}}  {r[1]:<8} {r[2]:<9} {r[3]:<9} {r[4]:<5} "
              f"{r[5]:<4} {r[6]:<7} {r[7]}")
    return 0


def _runs_history(a) -> int:
    """`unigym runs --history`: the run.json ledger face
    (GET /api/runs?history=1) -- name/id/state/exit/duration columns;
    --json carries the full HistoryRow documents."""
    limit = int(getattr(a, "limit", 0) or 20)
    doc = _api("GET", f"/api/runs?history=1&limit={limit}", timeout=30)
    runs = doc.get("runs") or []
    if getattr(a, "json", False):
        print(json.dumps(doc, indent=1, default=str))
        return 0
    if not runs:
        print(f"no runs in the ledger (scan root: {doc.get('output_root')}"
              f"; every run records itself in its output_dir/run.json)")
        return 0
    w = max(len(str(r.get("run_name") or r.get("run_id") or "?"))
            for r in runs)
    print(f"{'NAME':{w}}  RUN_ID  STATE           EXIT  DURATION")
    for r in runs:
        name = str(r.get("run_name") or r.get("run_id") or "?")
        rid = str(r.get("run_id") or "-")
        state = r.get("state") or "?"
        if state == "RUNNING" and r.get("stale"):
            state = "RUNNING(stale)"
        ex = r.get("exit")
        if ex:
            exit_s = "sig" if ex.get("code") is None \
                else str(ex.get("code"))
            dur = ex.get("wall_time_s")
            dur_s = f"{dur:.1f}s" if isinstance(dur, (int, float)) else "-"
        else:
            exit_s, dur_s = "-", "-"
        print(f"{name:{w}}  {rid:<20}  {state:<15} {exit_s:<4}  {dur_s}")
    return 0


def cmd_attach(a, extra: list[str]) -> int:
    st = _op(a.run, "status", {}, timeout=getattr(a, "timeout", 10.0))
    last = None
    try:
        while True:
            line = (f"[{st['run_id']}] {st['phase']} step={st['step']} "
                    f"episodes={st['episodes']} frame={st['frame_id']} "
                    f"metrics={st['metrics']}")
            if line != last:
                print("\r" + " " * max(len(last or ""), len(line)),
                      end="", flush=True)
                print("\r" + line, end="", flush=True)
                last = line
            if st["phase"] == "DONE":
                print()
                return 0
            time.sleep(0.5)
            st = _op(st["run_id"], "status", {},
                     timeout=getattr(a, "timeout", 10.0))
    except KeyboardInterrupt:
        print("\n[unigym attach] detached (run keeps going)")
        return 0


def cmd_cam(a, extra: list[str]) -> int:
    if getattr(a, "json", False):
        st = _op_json(a, a.run, "status", {}, timeout=a.timeout)
        base = _portal_base()
        cam = a.camera or st.get("camera") or ""
        enc = urllib.parse.quote
        doc = {"ok": True, "run_id": st.get("run_id"),
               "phase": st.get("phase"), "portal": base,
               "camera": cam or None,
               "publishing_camera": st.get("camera"),
               "status_url": f"{base}/run/{enc(a.run)}/status",
               "stream_url": (f"{base}/run/{enc(a.run)}/cam/{enc(cam)}.mjpg"
                              if cam else None),
               "snapshot_url": (f"{base}/run/{enc(a.run)}/cam/{enc(cam)}/"
                                f"frame.jpg" if cam else None),
               "view_url": f"{base}/view/{enc(a.run)}"}
        print(json.dumps(doc))
        return 0
    st = _op(a.run, "status", {}, timeout=a.timeout)
    base = _portal_base()
    cam = a.camera or st.get("camera") or ""
    print(f"run      : {st['run_id']} (phase {st['phase']})")
    print(f"status   : {base}/run/{st['run_id']}/status")
    if cam:
        print(f"stream   : {base}/run/{st['run_id']}/cam/{cam}.mjpg")
        print(f"snapshot : {base}/run/{st['run_id']}/cam/{cam}/frame.jpg")
        if cam != st.get("camera"):
            print(f"note     : '{cam}' is not the publishing camera "
                  f"('{st.get('camera')}') -- MJPEG streams the publishing "
                  f"camera; switch it from the portal view page")
    else:
        print("cameras  : none in this run")
    print(f"view     : {base}/view/{st['run_id']}   "
          f"(remote: ssh -L <portal-port>:127.0.0.1:<portal-port> <host>)")
    return 0


def _xyz_arg(s, flag: str) -> list[float] | None:
    """`[x,y,z]` JSON list or comma-separated -> float triple."""
    if s is None:
        return None
    try:
        v = json.loads(s)
    except ValueError:
        try:
            v = [float(x) for x in s.split(",")]
        except ValueError:
            v = None
    if not isinstance(v, list) or len(v) != 3 or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool)
            for x in v):
        sys.exit(f"[unigym snapshot] --{flag} expects [x,y,z] (JSON list "
                 f"or comma-separated numbers), got {s!r}")
    return [float(x) for x in v]


def cmd_snapshot(a, extra: list[str]) -> int:
    pos = _xyz_arg(getattr(a, "pos", None), "pos")
    lookat = _xyz_arg(getattr(a, "lookat", None), "lookat")
    if (pos is None) != (lookat is None):
        msg = ("--pos and --lookat must be given together (world-frame "
               "pose, fixed/free rigs only)")
        if getattr(a, "json", False):
            _json_exit(a, msg)
        sys.exit(f"[unigym snapshot] {msg}")
    if getattr(a, "json", False):
        doc = _op_json(a, a.run, "snapshot",
                       {"out": a.out, "camera": a.camera, "env": a.env,
                        "pos": pos, "lookat": lookat, "timeout": a.timeout},
                       timeout=a.timeout)
        _write_photo_sidecar(doc["path"], run=a.run, source="snapshot",
                             camera=doc["camera"], posed=(pos, lookat),
                             timeout=a.timeout)
        out = {k: v for k, v in doc.items() if k != "ok"}
        out["ok"] = True                              # key order
        print(json.dumps(out))
        return 0
    doc = _op(a.run, "snapshot",
              {"out": a.out, "camera": a.camera, "env": a.env,
               "pos": pos, "lookat": lookat, "timeout": a.timeout},
              timeout=a.timeout)
    _write_photo_sidecar(doc["path"], run=a.run, source="snapshot",
                         camera=doc["camera"], posed=(pos, lookat),
                         timeout=a.timeout)
    print(f"[unigym snapshot] {doc['frame_id']}B -> {doc['path']} "
          f"(camera {doc['camera']}, frame_id {doc['ack_frame_id']}"
          f"{', posed' if doc.get('posed') else ''})")
    return 0


# --------------------------------------------------------------------------- #
# photo metadata sidecar (toolkit D item 1; camera_contract §sidecar)
# --------------------------------------------------------------------------- #
def _write_photo_sidecar(jpg_path: str, *, run: str, source: str,
                         camera: str, posed: tuple | None,
                         timeout: float, env: int | None = None) -> None:
    """Attach ``<jpg>.json`` (world pose / convention / intrinsics / tick).

    Best-effort gathering with honest failure recording, but the sidecar
    FILE is always written once the image exists (contract: every photo
    carries metadata).  Posed captures pass the forensic pos/lookat --
    the transient pose is restored inside the run, so the computed pose
    (bit-identical lookat_quat math) is the only record.
    """
    from unigym.cameras.inspect import (build_info_expr,
                                        doc_to_sidecar_facts, unwrap)
    from unigym.cameras.sidecar import posed_world, write_sidecar
    st: dict = {}
    try:
        st = _op(run, "status", {}, timeout=timeout)
    except Exception as e:                          # noqa: BLE001
        st = {"_error": f"{type(e).__name__}: {e}"}
    facts: dict = {}
    try:
        info = unwrap({k: v for k, v in
                       _op(run, "inspect",
                           {"target": "eval",
                            "expr": build_info_expr([camera])},
                           timeout=timeout).items() if k != "ok"})
        facts = doc_to_sidecar_facts(info, camera)
    except Exception as e:                          # noqa: BLE001
        facts = {"world": None, "world_error": f"{type(e).__name__}: {e}"}
    world = posed_world(posed[0], posed[1]) if posed else facts.get("world")
    try:
        write_sidecar(jpg_path, source=source, camera=camera,
                      run_id=st.get("run_id", run), tick=st.get("step"),
                      sim_ts=(float(st.get("ts") or 0.0) or None),
                      phase=st.get("phase"), world=world,
                      intrinsics=facts.get("intrinsics"),
                      convention=facts.get("convention"),
                      rig=facts.get("rig"), mount=facts.get("mount"),
                      world_error=(facts.get("world_error")
                                   or st.get("_error")),
                      extra=({"posed": {"pos": posed[0],
                                        "lookat": posed[1]}}
                             if posed else None))
    except Exception as e:                          # noqa: BLE001 (additive)
        print(f"[unigym] WARNING: sidecar for {jpg_path} failed: "
              f"{type(e).__name__}: {e}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# capture (ux-5) -- one command, one artifact, one stdout line
# --------------------------------------------------------------------------- #
def _capture_exit(a, msg: str) -> NoReturn:
    """Fail-fast that keeps the stdout contract: plain mode -> stderr +
    nonzero; --json -> a single JSON document on stdout, exit 1."""
    if getattr(a, "json", False):
        print(json.dumps({"ok": False, "error": msg}))
        sys.exit(1)
    sys.exit(f"[unigym capture] {msg}")


def cmd_capture(a, extra: list[str]) -> int:
    """ux-5: capture one artifact and print its absolute path.

    Default: one JPEG frame (get_view, the broadcast camera unless
    --camera/--env).  ``--npz obs|extra|all``: a console save_npz dump
    instead.  stdout is EXACTLY ONE line -- the absolute path of an
    artifact that exists on return (--json swaps the line for a single
    structured document; errors never print a path).
    """
    try:
        doc = _op(a.run, "capture",
                  {"out": a.out, "npz": a.npz, "camera": a.camera,
                   "env": a.env, "timeout": a.timeout},
                  timeout=a.timeout)
    except _OpError as e:
        _capture_exit(a, str(e))
    if a.json:
        out = {k: v for k, v in doc.items() if k != "ok"}
        out["ok"] = True                          # key order
        print(json.dumps(out))
    else:
        print(doc["path"])
    if doc.get("kind") == "jpg":
        _write_photo_sidecar(doc["path"], run=a.run, source="capture",
                             camera=doc.get("camera") or "", posed=None,
                             timeout=a.timeout)
    return 0


def cmd_clean(a, extra: list[str]) -> int:
    body = {"dry_run": a.dry_run, "force": bool(getattr(a, "force", False))}
    doc = (_api_json(a, "POST", "/api/clean", body, timeout=60)
           if getattr(a, "json", False)
           else _api("POST", "/api/clean", body, timeout=60))
    if not doc.get("ok"):
        msg = str(doc.get("error", doc))
        if getattr(a, "json", False):
            _json_exit(a, msg)
        sys.exit(f"[unigym clean] {msg}")
    if getattr(a, "json", False):
        print(json.dumps({"ok": True, "dry_run": bool(a.dry_run),
                          "force": bool(getattr(a, "force", False)),
                          "removed": doc.get("removed") or []}))
        return 0
    for line in doc.get("lines") or []:
        print(line)
    removed = doc.get("removed") or []
    print(f"[unigym clean] {len(removed)} segment(s) removed"
          if not a.dry_run else f"[unigym clean] (dry-run) would remove: "
                                f"{removed or 'nothing'}")
    return 0


def cmd_inspect(a, extra: list[str]) -> int:
    """Debug console: eval expr / fixed query / summary."""
    try:
        doc = _op(a.run, "inspect",
                  {"expr": a.expr, "target": a.target, "key": a.key,
                   "timeout": a.timeout}, timeout=a.timeout)
    except _OpError as e:
        msg = str(e)
        if getattr(a, "json", False):
            _json_exit(a, msg)
        sys.exit(f"[unigym inspect] {msg}")
    res = {k: v for k, v in doc.items() if k != "ok"}
    print(json.dumps(res, indent=1, default=str))
    return 0


def cmd_step(a, extra: list[str]) -> int:
    """Debug console: advance n policy steps (auto-pause; stays
    paused afterwards -- step through then `unigym resume`)."""
    try:
        doc = _op(a.run, "step", {"n": a.n, "timeout": a.timeout},
                  timeout=a.timeout)
    except _OpError as e:
        msg = str(e)
        if getattr(a, "json", False):
            _json_exit(a, msg)
        sys.exit(f"[unigym step] {msg}")
    if getattr(a, "json", False):
        print(json.dumps({"ok": True, "auto_paused": doc["auto_paused"],
                          "stepping": doc["stepping"],
                          "hint": doc.get("hint")}))
        return 0
    print(f"[unigym step] {doc['stepping']} policy step(s) "
          f"queued -> engine advances then re-parks "
          f"(hint: {doc.get('hint', '-')})")
    return 0


def cmd_rewind(a, extra: list[str]) -> int:
    """Rewind face (Round B B1): `unigym rewind RUN --label L` captures a
    snapshot; `unigym rewind RUN SNAP_ID` restores it (auto-pauses)."""
    op = "rewind/snapshot" if a.snap_id is None else "rewind"
    body: dict = {"timeout": a.timeout}
    if a.snap_id is None:
        if a.label:
            body["label"] = a.label
    else:
        body["snap_id"] = a.snap_id
    try:
        doc = _op(a.run, op, body, timeout=a.timeout)
    except _OpError as e:
        msg = str(e)
        if getattr(a, "json", False):
            _json_exit(a, msg)
        hint = getattr(e, "hint", None)
        sys.exit(f"[unigym rewind] {msg}"
                 + (f" | hint: {hint}" if hint else ""))
    if getattr(a, "json", False):
        print(json.dumps({"ok": True, **{k: v for k, v in doc.items()
                                         if k != "ok"}}))
        return 0
    if a.snap_id is None:
        line = (f"[unigym rewind] snapshot {doc['snap_id']} @ tick "
                f"{doc['tick']}")
        if doc.get("label"):
            line += f" (label {doc['label']})"
        if doc.get("evicted"):
            ev = doc["evicted"]
            line += (f" | LRU evicted snapshot {ev['snap_id']} "
                     f"(tick {ev['tick']})")
        print(line + "\n  restore with: unigym rewind "
              f"{a.run} {doc['snap_id']}")
    else:
        print(f"[unigym rewind] restored to tick {doc['restored_tick']} "
              f"(snapshot {doc['snap_id']}); world is PAUSED -- resume or "
              f"step to continue on the forked timeline")
    return 0


def cmd_snapshots(a, extra: list[str]) -> int:
    """Rewind face list: live snapshot slots of one run."""
    doc = _api("GET", f"/api/op/{urllib.parse.quote(a.run)}/snapshots",
               timeout=a.timeout + 5.0)
    if not doc.get("ok"):
        msg = str(doc.get("error", doc))
        if getattr(a, "json", False):
            _json_exit(a, msg)
        _resolve_or_exit(doc)
        sys.exit(f"[unigym snapshots] {msg}")
    snaps = doc.get("snapshots") or []
    if getattr(a, "json", False):
        print(json.dumps({"ok": True, "snapshots": snaps,
                          "cap": doc.get("cap")}))
        return 0
    if not snaps:
        print(f"[unigym snapshots] no live snapshots "
              f"(create: unigym rewind {a.run} --label before)")
        return 0
    print(f"[unigym snapshots] {len(snaps)}/{doc.get('cap', 8)} slots:")
    for s in snaps:
        print(f"  {s['snap_id']:>3}  tick {s['tick']:>7}"
              + (f"  label {s['label']}" if s.get("label") else ""))
    return 0


def _simple_cmd(cmd_name: str, ok_hint: str):
    def run(a, extra: list[str]) -> int:
        if getattr(a, "json", False):
            res = _op_json(a, a.run, cmd_name, {"timeout": a.timeout},
                           timeout=a.timeout)
            print(json.dumps({"ok": True, "op": cmd_name,
                              "phase": res.get("phase"),
                              "paused": res.get("paused")}))
            return 0
        res = _op(a.run, cmd_name, {"timeout": a.timeout},
                  timeout=a.timeout)
        print(f"[unigym {cmd_name}] {ok_hint} (phase {res['phase']}, "
              f"paused={res['paused']})")
        return 0
    return run


cmd_pause = _simple_cmd("pause", "paused")
cmd_resume = _simple_cmd("resume", "resumed")


# --------------------------------------------------------------------------- #
# director (docs/contracts/director_contract.md §5) -- thin shell
# --------------------------------------------------------------------------- #
def _vec_arg(s: str, flag: str, n: int) -> list[float]:
    try:
        vals = [float(v) for v in str(s).split(",")]
    except ValueError:
        sys.exit(f"[unigym direct] --{flag} expects {n} comma numbers, "
                 f"got {s!r}")
    if len(vals) != n:
        sys.exit(f"[unigym direct] --{flag} expects {n} numbers, "
                 f"got {vals}")
    return vals


def cmd_direct(a, extra: list[str]) -> int:
    """`unigym direct <sub> <run> ...` -> POST /api/op/<run>/direct/<sub>."""
    payload: dict = {"timeout": a.timeout}
    args = list(a.args or [])
    if a.sub == "contact":
        if len(args) != 2:
            sys.exit("[unigym direct contact] usage: unigym direct "
                     "contact RUN GROUP on|off|clear")
        payload.update({"group": args[0], "state": args[1]})
    elif a.sub == "gripper":
        if not args:
            sys.exit("[unigym direct gripper] usage: unigym direct "
                     "gripper RUN move|grasp|release|where [...]")
        payload["action"] = args[0]
        if a.obj:
            payload["obj"] = a.obj
        if a.mode:
            payload["mode"] = a.mode
        if a.rel:
            payload["rel"] = True
        pose = {}
        if a.pos:
            pose["pos"] = _vec_arg(a.pos, "pos", 3)
        if a.quat:
            pose["quat"] = _vec_arg(a.quat, "quat", 4)
        if pose:
            payload["pose"] = pose
    elif a.sub in ("pin", "release"):
        if not a.name:
            sys.exit(f"[unigym direct {a.sub}] usage: unigym direct "
                     f"{a.sub} RUN --name PROP")
        payload["name"] = a.name
    elif a.sub == "follow":
        if not (a.obj and a.host):
            sys.exit("[unigym direct follow] usage: unigym direct follow "
                     "RUN --obj PROP --host vgrip|hand|prop:NAME|BODYLINK")
        payload["obj"] = a.obj
        payload["host"] = a.host
    elif a.sub == "push":
        if a.name:
            payload["name"] = a.name
        if a.lin is not None:
            payload["lin"] = _vec_arg(a.lin, "lin", 3)
        if a.ang is not None:
            payload["ang"] = _vec_arg(a.ang, "ang", 3)
        if a.mode:
            payload["mode"] = a.mode
        if "lin" not in payload and "ang" not in payload:
            sys.exit("[unigym direct push] usage: unigym direct push RUN "
                     "[--name robot] --lin X,Y,Z [--ang X,Y,Z] "
                     "[--mode set|add]")
    elif a.sub == "joints":
        if not a.joint:
            sys.exit("[unigym direct joints] usage: unigym direct joints "
                     "RUN --joint NAME=VALUE [--joint ...] [--zero-vel]")
        joints: dict = {}
        for pair in a.joint:
            name, sep, val = str(pair).partition("=")
            if not sep or not name.strip():
                sys.exit(f"[unigym direct joints] --joint expects "
                         f"NAME=VALUE, got {pair!r}")
            try:
                joints[name.strip()] = float(val)
            except ValueError:
                sys.exit(f"[unigym direct joints] --joint expects "
                         f"NAME=VALUE (numeric radians), got {pair!r}")
        payload["joints"] = joints
        if a.zero_vel:
            payload["zero_vel"] = True
    elif a.sub in ("teleport", "setvel", "attach", "detach"):
        for key, attr in (("name", "name"), ("obj", "obj"),
                          ("host", "host"), ("mode", "mode")):
            val = getattr(a, attr)
            if val is not None:
                payload[key] = val
        if a.pos is not None:
            payload["pos"] = _vec_arg(a.pos, "pos", 3)
        if a.quat is not None:
            payload["quat"] = _vec_arg(a.quat, "quat", 4)
        if a.lin is not None:
            payload["lin"] = _vec_arg(a.lin, "lin", 3)
        if a.ang is not None:
            payload["ang"] = _vec_arg(a.ang, "ang", 3)
        if a.joints is not None:
            try:
                payload["joints"] = json.loads(a.joints)
            except ValueError:
                sys.exit(f"[unigym direct] --joints must be JSON, got "
                         f"{a.joints!r}")
    # status: payload stays {timeout}
    try:
        doc = _op(a.run, f"direct/{a.sub}", payload, timeout=a.timeout)
    except _OpError as e:
        msg = str(e)
        hint = getattr(e, "hint", None)
        if getattr(a, "json", False):
            print(json.dumps({"ok": False, "error": msg,
                              **({"hint": hint} if hint else {})}))
            sys.exit(1)
        sys.exit(f"[unigym direct {a.sub}] {msg}"
                 + (f"\n  hint: {hint}" if hint else ""))
    res = {k: v for k, v in doc.items() if k not in ("ok",)}
    print(json.dumps(res, indent=1, default=str))
    return 0


def cmd_camera(a, extra: list[str]) -> int:
    if a.op == "inspect":                        # debug toolkit D
        return cmd_camera_inspect(a, extra)
    json_mode = getattr(a, "json", False)
    get_st = _op_json(a, a.run, "status", {}, timeout=a.timeout) \
        if json_mode else _op(a.run, "status", {}, timeout=a.timeout)
    st = get_st
    name = a.camera_name or st.get("camera") or ""
    if not name:
        msg = "no camera specified/available"
        if json_mode:
            _json_exit(a, msg)
        sys.exit(f"[unigym camera] {msg}")
    if a.op == "set":
        params = _kv_params(a.kv)
        if not params:
            msg = "expected k=v pairs (e.g. pose_offset='[0.2,0,0]' yaw_deg=15)"
            if json_mode:
                _json_exit(a, msg)
            sys.exit(f"[unigym camera set] {msg}")
        try:
            if json_mode:
                _op_json(a, a.run, "camera",
                         {"camera_op": "set", "name": name, "params": params,
                          "env": a.env, "timeout": a.timeout},
                         timeout=a.timeout)
            else:
                _op(a.run, "camera",
                    {"camera_op": "set", "name": name, "params": params,
                     "env": a.env, "timeout": a.timeout}, timeout=a.timeout)
        except _OpError as e:
            sys.exit(f"[unigym camera set] {e}")
        if json_mode:
            print(json.dumps({"ok": True, "op": "set", "name": name,
                              "params": params}))
            return 0
        print(f"[unigym camera set] applied {params} to '{name}'")
    else:                                       # get -> frame fetch
        # the RenderSpec capture face is the OUTER layer here: exactly
        # one of --view | --spec | --flags; absent all three = the
        # default view ("visual", bitwise the historical plain frame --
        # X2).  --flags is the v053 DEPRECATED alias: mechanically
        # expanded to a RenderSpec (expand_flags_alias) before the wire
        # -- the flags form never crosses into the daemon.
        view = getattr(a, "view", None)
        spec_raw = getattr(a, "spec", None)
        raw_flags = getattr(a, "flags", None)
        given = [v for v in (view, spec_raw, raw_flags)
                 if v is not None]
        if len(given) > 1:
            sys.exit("[unigym camera get] --view, --spec and --flags are "
                     "mutually exclusive (give exactly one)")
        if raw_flags is not None:
            try:
                flags = json.loads(raw_flags)
            except json.JSONDecodeError as e:
                sys.exit(f"[unigym camera get] --flags must be JSON "
                         f"({e}): {raw_flags!r}")
            if not isinstance(flags, dict):
                sys.exit(f"[unigym camera get] --flags must be a JSON "
                         f"object, got {raw_flags!r}")
            import warnings
            warnings.warn("--flags is deprecated (v053 alias): the "
                          "payload is expanded to a RenderSpec; use "
                          "--view/--spec", DeprecationWarning,
                          stacklevel=1)
            print("[unigym camera] NOTE: --flags is deprecated (v053 "
                  "alias) -- expanded to a RenderSpec; use "
                  "--view/--spec", file=sys.stderr)
            from unigym.cameras.spec import expand_flags_alias
            render = expand_flags_alias(flags)
        elif view is not None:
            render = {"view": view}
        elif spec_raw is not None:
            try:
                render = json.loads(spec_raw)
            except json.JSONDecodeError as e:
                sys.exit(f"[unigym camera get] --spec must be JSON "
                         f"({e}): {spec_raw!r}")
            if not isinstance(render, dict):
                sys.exit(f"[unigym camera get] --spec must be a JSON "
                         f"object, got {spec_raw!r}")
        else:
            render = {"view": "visual"}
        try:
            doc = (_op_json(a, a.run, "camera",
                            {"camera_op": "get", "name": name, "out": a.out,
                             "env": a.env, "render": render,
                             "timeout": a.timeout},
                            timeout=a.timeout)
                   if json_mode else
                   _op(a.run, "camera",
                       {"camera_op": "get", "name": name, "out": a.out,
                        "env": a.env, "render": render,
                        "timeout": a.timeout},
                       timeout=a.timeout))
        except _OpError as e:
            sys.exit(f"[unigym camera get] {e}")
        _write_photo_sidecar(doc["path"], run=a.run, source="camera_get",
                             camera=doc["name"], posed=None,
                             timeout=a.timeout)
        if json_mode:
            out = {k: v for k, v in doc.items() if k != "ok"}
            out["ok"] = True                      # key order
            print(json.dumps(out))
            return 0
        n_legend = len((doc.get("legend") or {}).get("objects", []))
        seg_note = (f", seg {doc['seg_path']} ({n_legend} legend objects)"
                    if doc.get("seg_path") else "")
        src_note = (f", render {doc.get('render_source')}"
                    if doc.get("render_source") else "")
        print(f"[unigym camera get] '{doc['name']}' frame "
              f"{doc['frame_id']} -> {doc['path']}{seg_note}{src_note}")
    return 0


def cmd_camera_inspect(a, extra: list[str]) -> int:
    """Debug toolkit D: one-screen camera forensics (design §5.3).

    Mount chain / resolved world pose / pinhole intrinsics / convention
    line, aggregated by ONE console eval against the live run (no new
    bridge command, no daemon change)."""
    from unigym.cameras.inspect import build_info_expr, render, unwrap
    names = [a.camera_name] if a.camera_name else []
    try:
        doc = _op(a.run, "inspect",
                  {"target": "eval", "expr": build_info_expr(names),
                   "timeout": a.timeout}, timeout=a.timeout)
    except _OpError as e:
        msg = str(e)
        if getattr(a, "json", False):
            _json_exit(a, msg)
        sys.exit(f"[unigym camera inspect] {msg}")
    info = unwrap({k: v for k, v in doc.items() if k != "ok"})
    if getattr(a, "json", False):
        print(json.dumps(info))
        return 0
    text = render(info, a.camera_name)
    if not text:
        print("[unigym camera inspect] no instantiated cameras in this run")
        return 1
    print(text)
    return 0


# --------------------------------------------------------------------------- #
# watchers (watchers_contract.md §7) -- thin shell
# --------------------------------------------------------------------------- #
def _watch_spec(a) -> dict:
    """add/test argparse -> spec dict (one of template|expr|callable)."""
    spec: dict = {"name": getattr(a, "name", None) or "test"}
    given = [x for x in (a.template, a.expr, a.callable) if x]
    if len(given) != 1:
        sys.exit("[unigym watch] exactly one of --template | --expr | "
                 "--callable is required")
    if a.template:
        spec["template"] = a.template
        spec["params"] = _kv_params(a.param or [])
    elif a.expr:
        spec["expr"] = a.expr
    else:
        spec["callable"] = a.callable
    if getattr(a, "on", None):
        spec["on"] = [s.strip() for s in a.on.split(",") if s.strip()]
    if getattr(a, "mode", None):
        spec["mode"] = a.mode
    if getattr(a, "hold", None):
        spec["hold"] = int(a.hold)
    if getattr(a, "once", False):
        spec["once"] = True
    if getattr(a, "cooldown", None) is not None:
        spec["cooldown"] = int(a.cooldown)
    if getattr(a, "every", None):
        spec["every"] = int(a.every)
    if getattr(a, "tier", None):
        spec["tier"] = int(a.tier)
    if getattr(a, "guard", None):
        spec["guard"] = a.guard
    if getattr(a, "note", None):
        spec["note"] = a.note
    if getattr(a, "allow_direct", False):
        spec["allow_direct"] = True
    return spec


def _watch_op(a, sub: str, body: dict, ok_hint) -> int:
    json_mode = getattr(a, "json", False)
    try:
        doc = _op(a.run, f"watch/{sub}", body, timeout=a.timeout)
    except _OpError as e:
        msg = str(e)
        if json_mode:
            print(json.dumps({"ok": False, "error": msg}))
            sys.exit(1)
        sys.exit(f"[unigym watch] {msg}")
    if json_mode:
        out = {k: v for k, v in doc.items() if k != "ok"}
        out["ok"] = True
        print(json.dumps(out, default=str))
        return 0
    if sub == "list":
        rows = doc.get("watchers") or []
        if not rows:
            print("[unigym watch] (no watchers armed)")
            return 0
        w = max(len(r["name"]) for r in rows)
        print(f"{'NAME':{w}}  STATE     TIER  EVERY  FIRED  ON  "
              "PREDICATE")
        for r in rows:
            on = ",".join(r.get("on") or [])
            print(f"{r['name']:{w}}  {r['state']:<8}  {r['tier']}     "
                  f"{r['every']:<5}  {r['fired']:<5}  {on:<8} "
                  f"{r['predicate']}"
                  + (f"  # {r['note']}" if r.get("note") else ""))
        return 0
    if sub == "test":
        print(json.dumps(doc.get("dry_run"), default=str))
        return 0
    print(f"[unigym watch] {ok_hint}")
    return 0


def cmd_watch(a, extra: list[str]) -> int:
    """`unigym watch <run> add|list|clear|update|test|events`."""
    if a.watch_cmd == "add":
        return _watch_op(a, "add", {"spec": _watch_spec(a)},
                         f"armed {a.name} (dry-run: add again with "
                         f"--json to see the trial evaluation)")
    if a.watch_cmd == "test":
        spec = _watch_spec(a)
        spec["name"] = "test"
        return _watch_op(a, "test", {"spec": spec}, "tested")
    if a.watch_cmd == "list":
        return _watch_op(a, "list", {}, "")
    if a.watch_cmd == "clear":
        if not getattr(a, "name", None) and not getattr(a, "all", False):
            sys.exit("[unigym watch clear] a watch name or --all is "
                     "required")
        return _watch_op(a, "clear",
                         {"name": None if getattr(a, "all", False)
                          else a.name}, "cleared")
    if a.watch_cmd == "update":
        patch: dict = {}
        for key in ("on", "mode", "note"):
            v = getattr(a, key, None)
            if v is not None:
                patch[key] = v
        for key in ("hold", "cooldown", "every"):
            v = getattr(a, key, None)
            if v is not None:
                patch[key] = int(v)
        if getattr(a, "once", None) is not None:
            patch["once"] = bool(a.once)
        if getattr(a, "enabled", None) is not None:
            patch["enabled"] = str(a.enabled).lower() in ("1", "true", "yes")
        if not patch:
            sys.exit("[unigym watch update] nothing to change (--on/--mode/"
                     "--hold/--once/--cooldown/--every/--enabled/--note)")
        return _watch_op(a, "update", {"name": a.name, "patch": patch},
                         f"updated {a.name}")
    # events (timeline tail; daemon-side file read, works for done runs)
    qs = f"tail={int(a.tail)}" + (f"&kind={a.kind}" if a.kind else "")
    try:
        doc = _api("GET", f"/api/runs/{urllib.parse.quote(a.run)}/"
                          f"events?{qs}", timeout=15)
    except SystemExit:
        raise
    if not doc.get("ok"):
        msg = str(doc.get("error", doc))
        if getattr(a, "json", False):
            _json_exit(a, msg)
        sys.exit(f"[unigym watch events] {msg}")
    if getattr(a, "json", False):
        out = {k: v for k, v in doc.items() if k != "ok"}
        out["ok"] = True
        print(json.dumps(out, default=str))
        return 0
    events = doc.get("events") or []
    print(f"[unigym watch] {len(events)} event(s) from {doc.get('path')}")
    for e in events:
        data = e.get("data") or {}
        msg = data.get("message") or ""
        extra_keys = {k: v for k, v in data.items() if k != "message"}
        print(f"  t={e.get('tick'):<8} {e.get('kind'):<14} "
              f"{e.get('name'):<20} {e.get('level') or '-':<5} {msg} "
              f"{extra_keys if extra_keys else ''}")
    return 0


def cmd_timeline(a, extra: list[str]) -> int:
    """`unigym timeline <run>` -- post-hoc events.jsonl viewer (§3)."""
    a.json = getattr(a, "json", False)
    a.tail = a.tail
    return cmd_watch(a, extra)


# --------------------------------------------------------------------------- #
# wait (daemon world) -- loop/format here, facts via POST /api/wait
# --------------------------------------------------------------------------- #
def cmd_wait(a) -> int:
    from unigym.scripts import spawn_spec as ss
    cfg = ss.load_cfg()
    q = {"state": a.state, "container": a.container, "log": a.log,
         "output_dir": a.output_dir, "teardown": True}
    json_mode = getattr(a, "json", False)

    def _verdict(verdict: str, ok: bool) -> int:
        print(json.dumps({"ok": ok, "verdict": verdict,
                          "out_dir": out_dir, "container": name,
                          "elapsed_s": round(time.time() - start, 1),
                          "log_tail": (doc.get("log_tail") or "")[-2000:],
                          **({"teardown": doc["teardown"]}
                             if verdict == "complete" and
                             doc.get("teardown") else {})}))
        return {"complete": 0, "pending": 1,
                "container_vanished": 1, "timeout": 2}[verdict]

    doc = _api_json(a, "POST", "/api/wait", q, timeout=30) if json_mode \
        else _api("POST", "/api/wait", q, timeout=30)
    if not doc.get("ok"):
        msg = str(doc.get("error", doc))
        if json_mode:
            _json_exit(a, msg)
        sys.exit(f"[unigym wait] {msg}")
    log, out_dir, name = doc["log"], doc["out_dir"], doc["container"]
    deadline = time.time() + (a.timeout if a.timeout is not None
                              else cfg["wait_timeout"])
    poll = a.poll or cfg["wait_poll"]
    start = time.time()

    while True:
        if doc.get("complete"):
            if json_mode:
                return _verdict("complete", True)
            print(f"[wait] completion criteria met "
                  f"(summary marker + artifacts in {out_dir}) after "
                  f"{time.time() - start:.0f}s")
            if name:
                act = doc.get("teardown")
                if act and (act.startswith("TEARDOWN")
                            or "removed" in act):
                    print(f"[wait] {act}")
            if a.once:
                return 0
            print(doc.get("log_tail", ""))
            return 0
        if a.once:
            if json_mode:
                return _verdict("pending", False)
            print(f"[wait] pending (marker/artifacts not met); "
                  f"log={log} out={out_dir}")
            return 1
        if name and doc.get("container_state") is None:
            if json_mode:
                return _verdict("container_vanished", False)
            print(f"[wait] container {name} vanished BEFORE criteria; "
                  f"log tail:")
            print(doc.get("log_tail", ""))
            return 1
        if time.time() > deadline:
            if json_mode:
                return _verdict("timeout", False)
            print(f"[wait] TIMEOUT after {time.time() - start:.0f}s; "
                  f"log tail:")
            print(doc.get("log_tail", ""))
            return 2
        time.sleep(poll)
        doc = _api_json(a, "POST", "/api/wait", q, timeout=30) if json_mode \
            else _api("POST", "/api/wait", q, timeout=30)
        if not doc.get("ok"):
            msg = str(doc.get("error", doc))
            if json_mode:
                _json_exit(a, msg)
            sys.exit(f"[unigym wait] {msg}")


# --------------------------------------------------------------------------- #
# submit / status / run (server world via the daemon)
# --------------------------------------------------------------------------- #
SUBMIT_REMOVED = ("the submit job queue is gone -- detached "
                  "runs: unigym run CONFIG... --bg (multiple configs "
                  "run in order)")


def cmd_submit(a, extra: list[str]) -> int:
    """REMOVED (design §2): the queue had zero users; keep the
    word in argparse for discovery, fail with the run --bg guidance."""
    if getattr(a, "json", False):
        _json_exit(a, SUBMIT_REMOVED)
    sys.exit(f"[unigym submit] {SUBMIT_REMOVED}")


def _cmd_run_query(a) -> int:
    """`unigym status <run> --metrics|--config|--report`.

    The CLI mirror of the webui run page: same daemon API face
    (POST /api/op/<run>/<kind>), same payloads -- zero core imports.
    """
    kind = ("metrics" if a.metrics else "config" if a.config
            else "report" if a.report else None)
    if not kind:
        print("a run query needs one of: --metrics | --config | --report",
              file=sys.stderr)
        return 2
    doc = _op(a.run, kind, {}, timeout=15.0)
    payload = {"run_id": doc.get("run_id"), kind: doc.get(kind)}
    if doc.get("media") is not None:       # telemetry passes through
        payload["media"] = doc["media"]
    if getattr(a, "json", False):
        print(json.dumps(payload, indent=1, default=str))
    else:
        print(json.dumps(doc.get(kind), indent=1, default=str))
    return 0


def cmd_status(a, extra: list[str]) -> int:
    # `status <run> --metrics|--config|--report` -> typed query
    # (the world view below stays byte-identical when no run is given).
    if getattr(a, "run", None):
        return _cmd_run_query(a)
    # the pending/running/done buckets are GONE with the submit
    # queue (design §2) -- the world view is live SHM runs + deaths.
    # The live SHM view comes from the daemon (empty when unreachable --
    # the CLI's tolerant try/except semantics).
    from unigym.scripts import spawn_spec as ss
    root = ss.runs_root()
    live = []
    if not getattr(a, "no_shm", False):
        try:
            live = [{"run_id": r[0], "phase": r[1], "step": r[2],
                     "episodes": r[3], "port": r[4], "pid": r[5],
                     "camera": r[6], "config": r[7]}
                    for r in live_rows()]
        except Exception as e:                   # noqa: BLE001 (audit #27:
            live = []                            # display face degrades, but
            print(f"[unigym] live view unavailable: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
    # spawn-death records (direct file scan -- works with or
    # without a daemon; old installs have none)
    try:
        from unigym.daemon import death as _death
        deaths = _death.scan()
    except (OSError, ValueError):   # audit #28: narrowed -- pure file scan;
        deaths = []                 # anything else is a bug and must surface
    payload = {"runs_root": str(root), "shm_runs": live,
               "deaths": [
                   {"pid": d.get("pid"), "exit_code": d.get("exit_code"),
                    "signal": d.get("signal"), "config": d.get("config"),
                    "log_path": d.get("log_path"),
                    "reason_head": str(d.get("reason") or "").splitlines()[0]
                    if d.get("reason") else None}
                   for d in deaths]}
    if a.json:
        print(json.dumps(payload, indent=1, default=str))
        return 0
    if deaths:
        print(f"deaths   : {len(deaths)}")
        for d in deaths:
            head = str(d.get("reason") or "").splitlines()[0]
            print(f"  pid={d.get('pid')} {head}  config={d.get('config')}"
                  f"  log={d.get('log_path')}")

    print(f"runs_root: {root} (submit queue removed -- no buckets)")
    print(f"shm runs : {len(live)}")
    for r in live:
        print(f"  {r['run_id']}  {r['phase']} step={r['step']} "
              f"episodes={r['episodes']} port={r['port']} "
              f"config={r['config']}")
    return 0


def _portal_base() -> str:
    """The daemon portal base URL (single browser source, design §3/§4):
    daemon state webui_port > unigym.yaml webui_port > 8600."""
    try:
        from unigym.daemon import state as daemon_state
        st = daemon_state.read_state()
        if st and st.get("webui_port"):
            return f"http://127.0.0.1:{int(st['webui_port'])}"
    except Exception:                          # noqa: BLE001 (display only)
        pass
    try:
        from unigym.scripts.spawn_spec import load_cfg
        port = int(load_cfg().get("webui_port") or 8600)
    except Exception:                          # noqa: BLE001
        port = 8600
    return f"http://127.0.0.1:{port}"


def _warn_retired_flags(a) -> None:
    """Soft landing (design §2): --serve/--webui/--serve-port are
    accepted but no longer mean anything -- one line each, stderr."""
    if getattr(a, "serve", False):
        print("[unigym] NOTE: --serve is a no-op -- every run "
              "is a SHM worker; media/console live in the daemon portal",
              file=sys.stderr)
    if getattr(a, "webui", False):
        print(f"[unigym] NOTE: --webui is a no-op -- the WebUI "
              f"moved to the daemon portal: {_portal_base()}",
              file=sys.stderr)
    if getattr(a, "serve_port", 0):
        print("[unigym] NOTE: --serve-port is a no-op (runs "
              "have no HTTP port)", file=sys.stderr)


# --------------------------------------------------------------------------- #
# run / debug (spawn paths -- the daemon spawns, allocates, supervises)
# --------------------------------------------------------------------------- #
def _run_wait_handle(handle: dict, poll: float = 5.0) -> int:
    """Foreground `unigym run`: block until the daemon reports a verdict
    (completion criteria shared with `wait`; the run itself is detached,
    so a Ctrl-C here never kills it)."""
    q = {"log": handle["log_path"], "output_dir": handle["out_dir"],
         "container": handle.get("container"), "teardown": True}
    missing = 0
    while True:
        doc = _api("POST", "/api/wait", q, timeout=30)
        if doc.get("complete"):
            return 0
        if handle.get("container") and \
                doc.get("container_state") is None:
            # the docker container is created asynchronously by the daemon
            # -- on the first polls it is not visible in `docker ps` yet
            # (isaaclab launch race).  Only a container that NEVER shows
            # up is a verdict; allow ~1 min of spawn latency.
            missing += 1
            if missing > 12:
                return 1
        else:
            missing = 0
        time.sleep(poll)


def cmd_run(a, passthrough: list[str]) -> int:
    codes = []
    _warn_retired_flags(a)
    json_mode = getattr(a, "json", False)
    launched: list = []
    passthrough = list(passthrough)
    if getattr(a, "watch", None):          # born-armed watcher pack
        passthrough += ["--watch", str(a.watch)]
    for c in a.config:
        try:                               # glob / short-name resolution
            resolved = resolve_config(c)
        except ConfigResolveError as e:
            msg = str(e)
            if json_mode:
                print(json.dumps({"ok": False, "error": msg,
                                  "launched": launched}))
                sys.exit(1)
            sys.exit(f"[unigym run] {msg}")
        for cfg_path in resolved:
            payload = {"spec": {"config": str(cfg_path),
                                "backend": a.backend,
                                "passthrough": list(passthrough),
                                "serve_camera": a.serve_camera,
                                "multi": len(a.config) > 1,
                                "paused": bool(getattr(a, "pause", False))}}
            doc = (_api_json(a, "POST", "/api/start", payload, timeout=60)
                   if json_mode
                   else _api("POST", "/api/start", payload, timeout=60))
            if not doc.get("ok"):
                msg = str(doc.get("error", doc))
                if json_mode:
                    print(json.dumps({"ok": False, "error": msg,
                                      "launched": launched}))
                    sys.exit(1)
                sys.exit(f"[unigym run] {msg}")
            h = doc["handle"]
            if json_mode:
                launched.append(h)
                continue
            where = (f"container={h['container']} backend={h['backend']}"
                     if h.get("container") else f"backend={h['backend']}")
            print(f"LAUNCHED pid={h['pid']} {where} log={h['log_path']} "
                  f"state={h.get('state_path')}")
            if h.get("container"):
                print(f"then: unigym wait --state {h.get('state_path')}")
            if not a.bg:
                codes.append(_run_wait_handle(h))
            else:
                codes.append(0)
    if json_mode:
        print(json.dumps({"ok": True, "launched": launched}))
        return 0
    return max(codes) if codes else 0


# --------------------------------------------------------------------------- #
# debug session mode (ux-12, S3) -- preflight (local, read-only) + the
# daemon's /api/start {debug: true} flow (spawn + bridge wait + /info)
# --------------------------------------------------------------------------- #
class DebugPreflightError(ValueError):
    """`unigym debug` preflight failed -- explicit, BEFORE any spawn."""


def debug_preflight(cfg: dict, backend: str, passthrough: list[str],
                    config_path: Path | None = None,
                    ) -> list[str]:
    """Front-load the silent-degradation traps (proposal §1.1/§1.2) into
    hard checks BEFORE the run starts.  Returns non-fatal warnings;
    raises :class:`DebugPreflightError` on anything fatal.

    Checks: backend validity; cameras declared on isaaclab REQUIRE
    --enable_cameras (the kit camera pipeline -- without it the run
    comes up silently pictureless); the same config already running
    (double-start guard via live registries).

    Port checks are GONE -- runs have no HTTP port (pure SHM
    workers; design §2/§5), so there is nothing to probe or collide on.

    read-only validation stays in the CLI for fast-fail UX; the
    daemon re-runs the SAME check server-side before spawning (§9.5).
    """
    from unigym.runtime.bridge import hb_alive, probe_heartbeat
    import os
    import yaml as _yaml

    warnings: list[str] = []
    if backend is None:
        if config_path is None:
            raise DebugPreflightError(
                "backend unknown: pass -b/--backend or a config_path to "
                "read env.backend from")
        peek = _yaml.safe_load(Path(config_path).read_text()) or {}
        backend = (peek.get("env") or {}).get("backend", "isaaclab")
    from unigym.backends.registry import BackendSelectionError, backend_info

    try:
        backend_faces = backend_info(backend)
    except BackendSelectionError as e:
        raise DebugPreflightError(
            f"unknown backend {backend!r} (isaaclab|mujoco)") from e
    cameras = cfg.get("cameras") or []
    if cameras and backend_faces.camera_enable_required and \
            "--enable_cameras" not in (passthrough or []):
        names = [c.get("name", "?") for c in cameras]
        raise DebugPreflightError(
            f"config declares cameras {names}: --enable_cameras is "
            f"REQUIRED (isaaclab camera sensors need the kit camera "
            f"pipeline; without it the run starts with NO picture).  "
            f"Append --enable_cameras to the command line")
    want_cfg = str(config_path) if config_path else None
    for p in _bridge_paths():
        try:
            st = json.loads(Path(p).read_text())
        except (OSError, ValueError):
            continue
        if not hb_alive(probe_heartbeat(run_id=st.get("run_id"),
                                        pid=st.get("pid"))):
            continue
        if want_cfg and st.get("config") and \
                os.path.abspath(st["config"]) == want_cfg:
            raise DebugPreflightError(
                f"this config is ALREADY running: run {st.get('run_id')} "
                f"(pid {st.get('pid')}); debug that run instead of "
                f"double-starting it")
    return warnings


def _bridge_paths():
    """Serve-registry paths (read-only scan; the ONE scanner lives in
    unigym.runtime_dirs -- stdlib glob under the hood)."""
    from unigym import runtime_dirs
    return [str(p) for p in runtime_dirs.serve_registry_paths(
        extra_dirs=[str(_REPO_ROOT)])]


def cmd_debug(a, extra: list[str]) -> int:
    """ux-12 (S3): one command -> run (SHM worker, bg) + console + capture
    session.  Preflight first (fail-fast, NO silent degradation), then the
    daemon spawns + waits for the worker and returns the run info; the CLI
    prints it with the console/capture cheatsheet.  `extra` is the collect
    passthrough (e.g. --enable_cameras).  `--pause` appends
    --serve_pause: the step-gate is born PAUSED and the engine never
    advances until resume/step.
    """
    import yaml

    from unigym.scripts import spawn_spec as ss

    try:                                   # glob / short-name resolution
        resolved = resolve_config(a.config)
    except ConfigResolveError as e:
        sys.exit(f"[unigym debug] {e}")
    if len(resolved) != 1:
        sys.exit(f"[unigym debug] config spec {a.config!r} resolved to "
                 f"{len(resolved)} configs; debug takes exactly one "
                 f"(use `unigym run` for many): "
                 + ", ".join(str(p) for p in resolved[:8]))
    cfg_path = resolved[0]
    if not cfg_path.exists():
        sys.exit(f"[unigym debug] config not found: {cfg_path}")
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except (OSError, ValueError) as e:
        sys.exit(f"[unigym debug] config unreadable: {e}")
    backend = ss.resolve_backend(cfg_path, a.backend)
    extra = list(extra)
    if getattr(a, "pause", False):
        extra.append("--serve_pause")
    _warn_retired_flags(a)

    try:
        warnings = debug_preflight(cfg, backend, extra,
                                   config_path=cfg_path)
    except DebugPreflightError as e:
        sys.exit(f"[unigym debug] preflight FAILED: {e}")
    for w in warnings:
        print(f"[unigym debug] WARNING: {w}", file=sys.stderr)

    doc = _api("POST", "/api/start",
               {"debug": True, "wait_timeout": a.wait_timeout,
                "poll": a.poll,
                "spec": {"config": str(cfg_path), "backend": backend,
                         "passthrough": extra, "multi": False}},
               timeout=float(a.wait_timeout) + 30.0)
    if not doc.get("ok"):
        err = doc.get("error", doc)
        if doc.get("hint") == "fix the preflight failure above":
            sys.exit(f"[unigym debug] preflight FAILED: {err}")
        sys.exit(f"[unigym debug] {err}")
    # (server-side preflight warnings stay in the API response for curl
    # consumers; the CLI already printed its local copy above)

    h = doc["handle"]
    print(f"LAUNCHED pid={h['pid']} backend={h['backend']} "
          f"log={h['log_path']} state={h.get('state_path')}")

    info = doc["info"]
    portal = _portal_base()
    rid = info["run_id"]
    cam = info.get("cameras") or {}
    print("=" * 72)
    print(f"run       : {rid}  (gate {info['gate']}, step {info['step']})")
    if getattr(a, "pause", False) and info.get("gate") == "PAUSED":
        print("paused    : step-gate PAUSED (--pause); release with "
              "`unigym step <run> N` or `unigym resume <run>`")
    print(f"view      : {portal}/view/{rid}   (media + console, "
          f"single source)")
    print(f"cameras   : broadcast={cam.get('broadcast')} "
          f"instantiated={cam.get('instantiated')}")
    print(f"output    : {info.get('output_dir')}")
    print(f"log       : {info.get('log_path') or h['log_path']}")
    print("--- console (single-controller: one writer at a time) -----")
    print("  unigym pause latest && unigym step latest 3   # step through")
    print("  unigym inspect latest --target summary        # state")
    print(f"  curl -s -X POST {portal}/run/{rid}/console "
          f"-H 'Content-Type: application/json' "
          f"-d '{{\"type\": \"eval\", \"expr\": \"np.zeros(3).sum()\"}}'")
    print("--- capture (evidence first, no rerun-without-proof) ------")
    print("  unigym capture latest               # one frame -> path")
    print("  unigym capture latest --npz all     # obs+extra npz -> path")
    print("  unigym snapshot latest --camera free --pos 2,-2,1.5 "
          "--lookat 0,0,0.8   # forensic angle")
    print(f"all endpoints: curl -s {portal}/run/{rid}/info")
    print(f"for HUMANS (browser): {portal}/view/{rid}  |  "
          f"ssh -L <portal>:127.0.0.1:<portal> <host>")
    print("=" * 72)
    return 0
