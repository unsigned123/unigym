"""daemon.state -- daemon identity/discovery file .

One small json file (default /tmp/unigym-daemon.json, override with
UNIGYM_DAEMON_STATE) written by the daemon at startup and removed on a
clean stop; CLI transport resolves the control API port from it:

    UNIGYM_DAEMON_API_PORT env  >  state file api_port  >
    unigym.yaml daemon_api_port  >  8670

stdlib only; safe to read from any process (missing/corrupt file ->
None, never an error).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml      # api_port()'s yaml fallback layer (YAMLError != ValueError)

DEFAULT_STATE_PATH = "/tmp/unigym-daemon.json"
DEFAULT_API_PORT = 8670
DEFAULT_WEBUI_PORT = 8600


def state_path() -> Path:
    """Discovery-file path (UNIGYM_DAEMON_STATE or the /tmp default)."""
    return Path(os.environ.get("UNIGYM_DAEMON_STATE",
                               DEFAULT_STATE_PATH))


def read_state() -> dict | None:
    """The daemon state dict, or None (absent/unreadable/corrupt)."""
    try:
        st = json.loads(state_path().read_text())
        return st if isinstance(st, dict) and st.get("pid") else None
    except (OSError, ValueError):
        return None


def write_state(d: dict) -> Path:
    """Atomically publish the daemon state dict; returns the file path."""
    p = state_path()
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, p)                          # atomic publish
    return p


def clear_state() -> None:
    """Remove the discovery file (missing file is not an error)."""
    try:
        state_path().unlink(missing_ok=True)
    except OSError:
        pass


def api_port() -> int:
    """Control-API port resolution order (see module docstring)."""
    env = os.environ.get("UNIGYM_DAEMON_API_PORT")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    st = read_state()
    if st and st.get("api_port"):
        return int(st["api_port"])
    try:
        from unigym.core.spawn import load_cfg
        v = load_cfg().get("daemon_api_port")
        if v:
            return int(v)
    except (OSError, ValueError, yaml.YAMLError):
        # audit (#3): narrowed from except-Exception -- unreadable
        # yaml takes the documented fallback chain (state > yaml > 8670;
        # YAMLError is NOT a ValueError subclass), but a real bug
        # (AttributeError etc.) must surface, not silently move the
        # daemon to the default port.
        pass
    return DEFAULT_API_PORT
