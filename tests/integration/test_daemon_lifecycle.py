"""Daemon process lifecycle over real loopback ports and a temporary state file."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests._support.envs import require_optional_module

pytestmark = [pytest.mark.subprocess, pytest.mark.network, pytest.mark.serial]

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def _daemon_extra() -> None:
    """Skip locally or fail in a strict lane when the daemon extra is absent."""
    require_optional_module(
        "zmq",
        reason="the daemon lifecycle lane requires the unigym[daemon] extra",
    )


def _free_ports(count: int) -> list[int]:
    """Return ``count`` distinct free loopback ports from ephemeral probes."""
    ports: list[int] = []
    while len(ports) < count:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in ports:
            ports.append(port)
    return ports


def _daemon_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    """Build an isolated daemon environment rooted at ``tmp_path``."""
    deps_root = tmp_path / "deps"
    deps_root.mkdir()
    state_path = tmp_path / "state.json"
    log_path = tmp_path / "daemon.log"

    env = os.environ.copy()
    for name in (
        "UNIGYM_DEPS_ROOT",
        "UNIGYM_DAEMON_STATE",
        "UNIGYM_DAEMON_LOG",
        "UNIGYM_DAEMON_API_PORT",
        "UNIGYM_ORPHAN_SWEEP",
    ):
        env.pop(name, None)
    env["UNIGYM_DEPS_ROOT"] = str(deps_root)
    env["UNIGYM_DAEMON_STATE"] = str(state_path)
    env["UNIGYM_DAEMON_LOG"] = str(log_path)
    env["UNIGYM_ORPHAN_SWEEP"] = "off"
    return env, state_path, log_path


def _run_cli(
    *args: str,
    env: dict[str, str],
    timeout: float = 60,
) -> subprocess.CompletedProcess[str]:
    """Run the launcher entry point as a real subprocess from the repo root."""
    return subprocess.run(
        [sys.executable, "-m", "unigym.scripts.launch", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _read_state(path: Path) -> dict[str, object] | None:
    """Read the daemon discovery file, or None when absent/unreadable."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _stdout_json(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    """Parse the JSON document that a ``--json`` status command prints."""
    doc = json.loads(result.stdout)
    assert isinstance(doc, dict), result.stdout
    return doc


def _pid_alive(pid: object) -> bool:
    """True when ``pid`` names a live process (signal-0 probe)."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def _diagnostics(result: subprocess.CompletedProcess[str], log_path: Path) -> str:
    """Collect CLI output and daemon log tail for a loud start failure."""
    parts = [f"exit={result.returncode}", result.stdout, result.stderr]
    if log_path.exists():
        parts.append(f"--- {log_path} ---")
        parts.append(log_path.read_text(errors="replace")[-4000:])
    return "\n".join(part for part in parts if part)


def test_daemon_start_status_stop_lifecycle(tmp_path: Path) -> None:
    """The daemon starts healthy on isolated ports and stops through its state file."""
    env, state_path, log_path = _daemon_env(tmp_path)
    api_port, webui_port = _free_ports(2)
    known_pid: object = None

    try:
        start = _run_cli(
            "daemon",
            "start",
            "--api-port",
            str(api_port),
            "--webui-port",
            str(webui_port),
            env=env,
        )
        assert start.returncode == 0, _diagnostics(start, log_path)
        assert state_path.exists(), f"daemon start did not write {state_path}"

        state = _read_state(state_path)
        assert state is not None, state_path.read_text(errors="replace")
        pid = state.get("pid")
        assert isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
        known_pid = pid
        assert state.get("api_port") == api_port
        assert state.get("webui_port") == webui_port

        status = _run_cli("daemon", "status", "--json", env=env)
        assert status.returncode == 0, status.stderr
        doc = _stdout_json(status)
        assert doc.get("alive") is True, doc
        health = doc.get("health")
        assert isinstance(health, dict), doc
        assert health.get("ok") is True, health
        status_state = doc.get("state")
        assert isinstance(status_state, dict), doc
        assert status_state.get("pid") == pid
        assert status_state.get("api_port") == state.get("api_port")
        assert status_state.get("webui_port") == state.get("webui_port")
        assert _pid_alive(pid)

        stop = _run_cli("daemon", "stop", "--timeout", "15", env=env)
        assert stop.returncode == 0, stop.stderr
        assert not state_path.exists(), "daemon stop left its state file behind"
        assert not _pid_alive(pid), f"daemon pid {pid} survived stop"

        down = _run_cli("daemon", "status", "--json", env=env)
        assert down.returncode == 0, down.stderr
        down_doc = _stdout_json(down)
        assert down_doc.get("alive") is False, down_doc
        assert down_doc.get("pid_alive") is False, down_doc
    finally:
        leftover = _read_state(state_path)
        if leftover is not None and isinstance(leftover.get("pid"), int):
            known_pid = leftover["pid"]
        if state_path.exists() or known_pid is not None:
            _run_cli("daemon", "stop", "--timeout", "15", env=env)
        if _pid_alive(known_pid):
            raise AssertionError(
                f"daemon pid {known_pid} is still alive after cleanup; " f"daemon log: {log_path}"
            )


def test_daemon_stop_when_down_is_honest(tmp_path: Path) -> None:
    """Stopping with no state file succeeds and reports that nothing was running."""
    env, state_path, _log_path = _daemon_env(tmp_path)

    result = _run_cli("daemon", "stop", env=env)

    assert result.returncode == 0, result.stderr
    assert "nothing to stop" in result.stdout, result.stdout
    assert not state_path.exists(), "a down stop must not create daemon state"
    assert "Traceback" not in result.stderr, result.stderr
