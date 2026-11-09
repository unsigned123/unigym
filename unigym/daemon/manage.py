"""daemon.manage -- `unigym daemon start|stop|status` .

The CLI's ONLY local process operations (design §1(6)/§4): start spawns
the daemon detached (setsid, own log), stop TERM-then-KILLs it via the
state-file pid, status reports identity + a live health probe.  The
daemon itself never owns run instances (they are setsid-detached one
further layer down), so a daemon restart can never take a benchmark
down.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import state as dstate


def _log_path() -> str:
    return os.environ.get("UNIGYM_DAEMON_LOG", "/tmp/unigym-daemon.log")


def _health(api_port: int, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{api_port}/api/health", timeout=timeout
        ) as r:
            import json

            return json.loads(r.read())
    except (OSError, ValueError, urllib.error.URLError):
        return None


def pid_alive(pid: int | None) -> bool:
    """True when the pid exists (signal-0 probe; EPERM counts as alive)."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True


def start(
    api_port: int | None = None,
    webui_port: int | None = None,
    foreground: bool = False,
    wait_s: float = 30.0,
) -> dict:
    """Start unigymd (nohup-style detached by default).

    Returns the daemon state ({"already": true} when one is healthy).

    ``foreground=True`` is the systemd user-unit mode: no already-
    running check (the service manager owns the lifecycle), immediate
    exec into the daemon process.
    """
    argv = [sys.executable, "-m", "unigym.daemon"]
    if api_port:
        argv += ["--api-port", str(int(api_port))]
    if webui_port:
        argv += ["--webui-port", str(int(webui_port))]
    if foreground:
        os.execvp(argv[0], argv)  # systemd mode (ReplaceProcess)
    st = dstate.read_state()
    if st and _health(int(st.get("api_port") or 0)):
        return {**st, "already": True}
    if st and pid_alive(st.get("pid")):
        # a live pid that does not answer: report honestly, do not guess
        return {**st, "already": True, "unhealthy": True}
    log = _log_path()
    with open(log, "a") as f:
        proc = subprocess.Popen(argv, stdout=f, stderr=f, start_new_session=True)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        st = dstate.read_state()
        if st and st.get("pid") == proc.pid:
            health = _health(int(st.get("api_port") or 0))
            if health:
                return {**st, "already": False}
        elif not pid_alive(proc.pid):
            break
        time.sleep(0.3)
    raise RuntimeError(
        f"daemon pid {proc.pid} did not become healthy within {wait_s}s " f"-- check the log: {log}"
    )


def stop(timeout_s: float = 15.0) -> dict:
    """TERM the daemon via the state-file pid, KILL after the grace.

    Returns:
        {"ok", "stopped", "note"[/ "error"]} -- a stale or absent state
        file is reported, never raised.
    """
    st = dstate.read_state()
    if not st:
        return {"ok": True, "stopped": None, "note": "no daemon state file (nothing to stop)"}
    pid = st.get("pid")
    if not isinstance(pid, int) or pid <= 0 or not pid_alive(pid):
        dstate.clear_state()
        return {"ok": True, "stopped": pid, "note": f"stale state file (pid {pid} already dead)"}
    try:
        os.kill(int(pid), signal.SIGTERM)
    except OSError as e:
        return {"ok": False, "stopped": None, "error": f"kill: {e}"}
    deadline = time.time() + timeout_s
    stopped = False
    while time.time() < deadline:
        if not pid_alive(pid):
            stopped = True
            break
        time.sleep(0.2)
    if not stopped:
        with contextlib.suppress(OSError):
            os.kill(int(pid), signal.SIGKILL)
        note = f"pid {pid} ignored TERM; SIGKILLed"
    else:
        note = f"pid {pid} stopped cleanly"
    dstate.clear_state()
    return {"ok": True, "stopped": pid, "note": note}


def status() -> dict:
    """Daemon identity + a live /api/health probe (down daemon -> health None)."""
    st = dstate.read_state() or {}
    health = _health(int(st.get("api_port") or 0)) if st else None
    return {
        "state": st,
        "health": health,
        "alive": bool(health and health.get("ok")),
        "pid_alive": pid_alive(st.get("pid")),
    }


# --------------------------------------------------------------------------- #
# CLI entry (argparse namespace from launch.py `unigym daemon ...`)
# --------------------------------------------------------------------------- #
def cmd_daemon(a) -> int:
    """`unigym daemon start|stop|status` subcommand dispatch.

    The thin CLI's ONLY local process commands.

    Args:
        a: argparse namespace from launch.py (`a.daemon_cmd`,
            `a.api_port`, `a.webui_port`, `a.foreground`, `a.json`).

    Returns:
        Process exit code (0 ok; 1 start failure / unhealthy daemon).
    """
    if a.daemon_cmd == "start":
        try:
            st = start(api_port=a.api_port, webui_port=a.webui_port, foreground=a.foreground)
        except RuntimeError as e:
            print(f"[unigym daemon] {e}", file=sys.stderr)
            return 1
        if st.get("already"):
            unhealthy = " (unhealthy: no /api/health answer)" if st.get("unhealthy") else ""
            print(
                f"[unigym daemon] already running pid={st.get('pid')} "
                f"api=127.0.0.1:{st.get('api_port')} "
                f"portal=127.0.0.1:{st.get('webui_port')}{unhealthy}"
            )
        else:
            print(
                f"DAEMON pid={st['pid']} "
                f"api=http://127.0.0.1:{st['api_port']} "
                f"portal=http://127.0.0.1:{st['webui_port']} "
                f"(log: {_log_path()})"
            )
        return 0
    if a.daemon_cmd == "stop":
        doc = stop(timeout_s=getattr(a, "timeout", 15.0))
        if not doc.get("ok"):
            print(f"[unigym daemon] {doc.get('error', doc)}", file=sys.stderr)
            return 1
        if doc.get("stopped") is None:
            print(f"[unigym daemon] {doc.get('note', 'nothing to stop')}")
        else:
            print(f"[unigym daemon] stopped ({doc['note']})")
        return 0
    # status
    doc = status()
    if getattr(a, "json", False):
        print(json.dumps(doc, default=str))
        return 0
    st = doc.get("state") or {}
    health = doc.get("health") or {}
    if doc.get("alive"):
        print(f"daemon    : up (pid {st.get('pid')}, " f"started {st.get('started', '?')})")
        print(f"api       : http://127.0.0.1:{st.get('api_port')}  " f"(health ok)")
        print(f"portal    : http://127.0.0.1:{st.get('webui_port')}")
        print(f"runs      : {health.get('runs', 0)} in view")
    elif doc.get("pid_alive"):
        print(
            f"daemon    : UNHEALTHY (pid {st.get('pid')} alive but no "
            f"health answer on {st.get('api_port')})"
        )
        return 1
    else:
        print("daemon    : down (start it with: unigym daemon start)")
    return 0
