"""Daemon CLI boundary: a base install reports the missing extra, not a crash."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]

_DISPATCH_ENV_VARS = (
    "UNIGYM_DEPS_ROOT",
    "UNIGYM_DAEMON_STATE",
    "UNIGYM_DAEMON_API_PORT",
)


def test_daemon_status_reports_missing_zmq_extra(tmp_path: Path) -> None:
    """`daemon status` without the daemon extra is a clean teaching error."""
    if importlib.util.find_spec("zmq") is not None:
        pytest.skip("daemon extra is installed; the missing-extra branch does not apply")

    deps_root = tmp_path / "deps"
    deps_root.mkdir()
    state_path = tmp_path / "state.json"

    env = os.environ.copy()
    for name in _DISPATCH_ENV_VARS:
        env.pop(name, None)
    env["UNIGYM_DEPS_ROOT"] = str(deps_root)
    env["UNIGYM_DAEMON_STATE"] = str(state_path)

    result = subprocess.run(
        [sys.executable, "-m", "unigym.scripts.launch", "daemon", "status"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 2, result.stderr
    assert "optional dependency 'zmq'" in result.stderr, result.stderr
    assert "unigym[daemon]" in result.stderr, result.stderr
    assert "Traceback" not in result.stderr, result.stderr
    assert "Traceback" not in result.stdout, result.stdout
    assert not state_path.exists(), "a missing extra must not write daemon state"
    assert "DAEMON pid=" not in result.stdout, result.stdout
