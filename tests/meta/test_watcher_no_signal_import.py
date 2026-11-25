"""Import isolation: the watcher stack never imports the signal module."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = (
    "import sys;"
    "import unigym.watchers;"
    "import unigym.watchers.base;"
    "import unigym.watchers.engine;"
    "import unigym.watchers.templates as t;"
    "assert 'unigym.signals' not in sys.modules, sorted(sys.modules);"
    "assert not hasattr(t, '_CURRENT_ENGINE')"
)


@pytest.mark.subprocess
def test_watcher_modules_do_not_import_the_signal_module():
    """A fresh interpreter imports the watcher stack without unigym.signals."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        env=env,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
