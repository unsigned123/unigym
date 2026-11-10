"""CLI process-boundary smoke tests: help, usage errors and lazy dispatch."""

from __future__ import annotations

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


def clean_env() -> dict[str, str]:
    """Return the current environment without dispatch/machine overrides."""
    env = os.environ.copy()
    for name in _DISPATCH_ENV_VARS:
        env.pop(name, None)
    return env


def run_cli(
    *args: str,
    env: dict[str, str] | None = None,
    timeout: float = 20,
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


def assert_no_traceback(result: subprocess.CompletedProcess[str]) -> None:
    """A CLI help/usage path may not leak an interpreter traceback."""
    assert "Traceback" not in result.stdout, result.stdout
    assert "Traceback" not in result.stderr, result.stderr


def test_top_level_help_exits_cleanly() -> None:
    """Top-level help renders the command surface without importing engines."""
    result = run_cli("--help", env=clean_env())

    assert result.returncode == 0, result.stderr
    assert "usage: unigym" in result.stdout
    for token in ("run", "daemon", "describe"):
        assert token in result.stdout, f"expected {token!r} in top-level help"
    assert_no_traceback(result)


def test_unknown_command_is_a_usage_error() -> None:
    """An unknown subcommand is an argparse usage error, not a crash."""
    result = run_cli("nope", env=clean_env())

    assert result.returncode == 2, result.stdout
    assert "usage: unigym" in result.stderr
    assert "invalid choice" in result.stderr
    assert_no_traceback(result)


def test_run_help_is_available_without_asset_root() -> None:
    """`run --help` renders backend choices without an asset/deps root."""
    result = run_cli("run", "--help", env=clean_env())

    assert result.returncode == 0, result.stderr
    assert "--backend" in result.stdout
    for token in ("isaaclab", "mujoco"):
        assert token in result.stdout, f"expected backend {token!r} in run help"
    assert_no_traceback(result)


def test_daemon_help_is_available_without_daemon_extra() -> None:
    """`daemon --help` renders its subcommands without starting a daemon."""
    result = run_cli("daemon", "--help", env=clean_env())

    assert result.returncode == 0, result.stderr
    for token in ("start", "stop", "status"):
        assert token in result.stdout, f"expected daemon command {token!r}"
    assert_no_traceback(result)
