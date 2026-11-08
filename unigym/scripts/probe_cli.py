"""`unigym probe` -- one-command e2e probe: launch -> wait -> scorecard
(design v040-debug-toolkit-r3 §2).

The campaign burned ~30 calls per probe on the launch->poll->glob->numpy
manual chain; this composes the existing faces instead of adding paths:

    spawn     daemon /api/start (the ONE run face; no new engine path)
    preflight --gpu-preflight: orphan collect_cli abort + VRAM report
    wait      log marker + artifacts via /api/wait (spawn_spec.is_complete),
              --wait-timeout S (default 600); a spawn that dies immediately
              surfaces the A4 death record (exit code + log tail), no dumb
              polling past a corpse
    run_dir   from the spawn handle (NEVER `ls -td` glob guessing)
    scorecard = B2 gate rendering embedded (unigym/analysis/gate.py --
              probe computes nothing itself; launch+wait+analyze sugar)

`--rm` deletes the run dir after the report (default: keep -- probe
artifacts are evidence).  `--json` is the machine face; its `episodes`
key is byte-identical to `unigym analyze <run> --json` gate episodes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from unigym.scripts.config_resolve import ConfigResolveError, resolve_config


def _post(api_port: int, path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{api_port}{path}",
        data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except ValueError:
            return {"ok": False, "error": f"HTTP {e.code}"}
    except (OSError, ValueError) as e:
        return {"ok": False, "error": f"daemon unreachable: {e}",
                "hint": "start it with: unigym daemon start"}


def _api_port() -> int:
    from unigym.daemon import state as dstate
    return dstate.api_port()


def _resolve_output_dir(pid: int) -> str | None:
    """The run's REAL output dir from its serve-registry v2 file (design:
    registry, never a glob).  The spawn handle's value is a yaml-based
    PREDICTION -- a `-s run_name=...` override legally redirects the run.
    Registry files exist from worker mount until the clean-stop unlink,
    so this must be polled DURING the wait window, not after DONE.
    """
    from unigym import runtime_dirs
    paths = runtime_dirs.serve_registry_paths()
    seen: dict[str, str] = {}
    for p in paths:
        try:
            st = json.loads(Path(p).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(st, dict) and st.get("pid") is not None:
            seen[str(st.get("run_id"))] = (str(st.get("output_dir")),
                                           int(st["pid"]))
    for out, entry_pid in seen.values():
        if entry_pid == int(pid) and out:
            return out
    return None


def nvidia_rows() -> tuple[list[str], str]:
    """(csv rows, error) -- the ONE nvidia-smi query, shared with `unigym
    doctor` (single source: probe --gpu-preflight + doctor GPU item)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=10)
    except (OSError, subprocess.SubprocessError) as e:
        return [], str(e)
    if out.returncode != 0:
        return [], (out.stderr or "").strip() or f"exit {out.returncode}"
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()], ""


def gpu_preflight() -> tuple[list[str], list[str]]:
    """(fatal_errors, warnings): orphan collect_cli aborts; VRAM advises."""
    fatal: list[str] = []
    warns: list[str] = []
    try:
        out = subprocess.run(["pgrep", "-af", "unigym.scripts.collect_cli"],
                             capture_output=True, text=True, timeout=10)
        orphans = [ln for ln in out.stdout.splitlines() if ln.strip()]
    except (OSError, subprocess.SubprocessError) as e:
        orphans = []
        warns.append(f"orphan scan unavailable: {e}")
    if orphans:
        fatal.append(
            f"{len(orphans)} orphan collect_cli process(es) -- they hold "
            "GL contexts/GPU memory and corrupt probe timing; kill or "
            "wait for them first:\n    " + "\n    ".join(orphans[:5]))
    rows, err = nvidia_rows()
    if err:
        warns.append(f"nvidia-smi unavailable: {err}")
    else:
        low = [ln for ln in rows if ln and _free_mb(ln) < 2000]
        if low:
            warns.append("low VRAM (<2 GB free): " + "; ".join(low))
        elif rows:
            warns.append(f"GPUs: {' | '.join(rows)}")
    return fatal, warns


def _free_mb(csv_row: str) -> int:
    try:
        parts = [p.strip() for p in csv_row.split(",")]
        used, total = int(parts[1].split(" ")[0]), int(parts[2].split(" ")[0])
        return total - used
    except (IndexError, ValueError):
        return 1 << 30


def cmd_probe(a: argparse.Namespace, extra: list[str]) -> int:
    t0 = time.time()
    # born-paused refusal: probe waits for DONE -- a born-paused
    # run would stall here to --wait-timeout.  Fail-fast instead.
    bad = [f for f in (extra or [])
           if f in ("--serve_pause", "--serve-pause")]
    if bad:
        return _fail(a,
                     "born-paused spawn (--serve_pause) is not supported "
                     "on probe: probe waits for the run to COMPLETE, so a "
                     "run parked at the first step boundary would stall to "
                     "--wait-timeout",
                     hint="start it paused with: unigym run CONFIG --pause "
                          "(then: unigym resume latest / step latest N)")
    try:                                   # glob / short-name resolution
        resolved = resolve_config(a.config)
    except ConfigResolveError as e:
        print(json.dumps({"ok": False, "error": str(e)})
              if a.json else f"[unigym probe] {e}",
              file=None if a.json else sys.stderr)
        return 1
    if len(resolved) != 1:
        msg = (f"config spec {a.config!r} resolved to {len(resolved)} "
               f"configs; probe takes exactly one (use `unigym run` for "
               f"many): " + ", ".join(str(p) for p in resolved[:8]))
        print(json.dumps({"ok": False, "error": msg})
              if a.json else f"[unigym probe] {msg}",
              file=None if a.json else sys.stderr)
        return 1
    cfg_path = resolved[0]
    if a.gpu_preflight:
        fatal, warns = gpu_preflight()
        for w in warns:
            print(f"[unigym probe] preflight: {w}", file=sys.stderr)
        if fatal:
            msg = fatal[0]
            if a.json:
                print(json.dumps({"ok": False, "error": msg,
                                  "phase": "preflight"}))
            else:
                print(f"[unigym probe] PREFLIGHT FAILED: {msg}",
                      file=sys.stderr)
            return 2
    from unigym.scripts import spawn_spec as ss
    try:
        backend = ss.resolve_backend(cfg_path, a.backend)
    except OSError as e:
        print(f"[unigym probe] unreadable config: {e}", file=sys.stderr)
        return 1
    passthrough: list[str] = []
    if a.steps is not None:
        passthrough += ["--steps", str(int(a.steps))]
    if a.episodes is not None:
        passthrough += ["-s", f"benchmark.min_episodes={int(a.episodes)}"]
    for item in a.set or []:
        passthrough += ["-s", str(item)]
    passthrough += list(extra)
    api = _api_port()
    doc = _post(api, "/api/start",
                {"spec": {"config": str(cfg_path), "backend": backend,
                          "passthrough": passthrough, "multi": False}},
                timeout=60)
    if not doc.get("ok"):
        return _fail(a, f"spawn failed: {doc.get('error', doc)}",
                     hint=doc.get("hint"), phase="spawn")
    h = doc["handle"]
    pid, log = int(h["pid"]), h["log_path"]
    run_dir = handle_out = h["out_dir"]
    if not a.json:
        print(f"[unigym probe] launched pid={pid} backend={backend} "
              f"log={log}")
    # ---- wait DONE (with A4 death fast-fail) ----
    from unigym.daemon import death
    deadline = time.time() + float(a.wait_timeout)
    real_dir = None
    while True:
        d = death.for_pid(pid)
        if d is not None:
            return _fail(a, f"run died: {d['reason'].splitlines()[0]}",
                         hint=f"log tail in the death record "
                              f"({death.death_path(pid)}); full log: {log}",
                         phase="run", handle=h)
        w = _post(api, "/api/wait",
                  {"log": log, "output_dir": real_dir or run_dir,
                   "teardown": False},
                  timeout=30)
        if w.get("complete"):
            break
        # the registry carries the REAL output dir (-s run_name redirects
        # it); it is unlinked on clean stop -> poll inside this window
        real_dir = _resolve_output_dir(pid) or real_dir
        if time.time() >= deadline:
            return _fail(a, f"not done within --wait-timeout "
                            f"{a.wait_timeout}s",
                         hint=f"tail the log: tail -40 {log}",
                         phase="wait", handle=h)
        time.sleep(2.0)
    real_dir = _resolve_output_dir(pid) or real_dir
    renamed = real_dir is not None and real_dir != run_dir
    run_dir = real_dir or run_dir
    # ---- scorecard = B2 gate rendering (zero re-computation here) ----
    from unigym.analysis import gate
    try:
        gdoc = gate.gate_doc(run_dir)
    except gate.GateError as e:
        return _fail(a, f"scorecard failed: {e}", hint=e.hint,
                     phase="analyze", handle=h)
    seconds = time.time() - t0
    if a.json:
        print(json.dumps({"ok": True, "config": str(cfg_path),
                          "backend": backend, "pid": pid, "log": log,
                          "run_dir": run_dir,
                          "run_dir_predicted": handle_out,
                          "seconds": round(seconds, 1),
                          "gate": gdoc, "episodes": gdoc["episodes"]}))
    else:
        print(f"[unigym probe] DONE in {seconds:.0f}s -> {run_dir}"
              + ("  (redirected by -s run_name)" if renamed else ""))
        if gdoc.get("available"):
            print(f"  gate: {gdoc['summary']}")
            for row in gdoc["rows"]:
                cells = "  ".join(
                    _cell(v) for v in row["values"])
                print(f"  {row['label']:<44} {cells}")
        else:
            print(f"  gate: {gdoc['status']}")
        print(f"  log: {log}")
    if a.rm:
        shutil.rmtree(run_dir, ignore_errors=True)
        if not a.json:
            print(f"[unigym probe] --rm: removed {run_dir}")
    return 0


def _cell(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "T" if v else "F"
    return f"{v:.4g}"


def _fail(a: argparse.Namespace, msg: str, hint: str | None = None,
          phase: str = "?", handle: dict | None = None) -> int:
    doc = {"ok": False, "error": msg, "phase": phase}
    if hint:
        doc["hint"] = hint
    if handle:
        doc["handle"] = handle
    if a.json:
        print(json.dumps(doc))
    else:
        print(f"[unigym probe] FAILED ({phase}): {msg}"
              + (f"\n  hint: {hint}" if hint else ""), file=sys.stderr)
    return 1


def add_probe_subparser(sub) -> None:
    """Wire the B1 `unigym probe` into the launcher (launch.py)."""
    p = sub.add_parser("probe", help="one-command e2e probe: launch -> "
                                     "wait -> gate scorecard; "
                                     "spawn via the daemon run face)",
                       description="probe = launch + wait-for-DONE + gate "
                                   "scorecard.  Born-paused spawn "
                                   "(--serve_pause / `unigym run --pause`) "
                                   "is REJECTED here on purpose: probe "
                                   "waits for completion, so a run parked "
                                   "PAUSED at the first step boundary "
                                   "would stall to --wait-timeout.  Start "
                                   "it with `unigym run CONFIG --pause` "
                                   "instead (release: `unigym resume "
                                   "latest`).")
    p.add_argument("config", help="run yaml (mujoco recommended for probes)")
    p.add_argument("-b", "--backend", choices=["isaaclab", "mujoco"],
                   default=None, help="override yaml env.backend")
    p.add_argument("--steps", type=int, default=None,
                   help="passthrough --steps (collect.steps override)")
    p.add_argument("--episodes", type=int, default=None,
                   help="benchmark.min_episodes override (rides -s, B4)")
    p.add_argument("--wait-timeout", type=float, default=600.0,
                   dest="wait_timeout",
                   help="seconds to wait for DONE (default 600)")
    p.add_argument("--gpu-preflight", action="store_true",
                   dest="gpu_preflight",
                   help="abort on orphan collect_cli processes; report "
                        "per-GPU VRAM before spawning")
    p.add_argument("-s", "--set", dest="set", action="append", default=None,
                   metavar="KEY=VALUE",
                   help="config override passthrough (-s key=val, "
                        "repeatable)")
    p.add_argument("--rm", action="store_true",
                   help="remove the run dir after the report (default: "
                        "keep -- probe artifacts are evidence)")
    p.add_argument("--json", action="store_true",
                   help="single JSON document on stdout (errors too)")
    p.set_defaults(fn=cmd_probe)
