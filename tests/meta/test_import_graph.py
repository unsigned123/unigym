"""Capability-layer import graph: pure leaves and the registry façade."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a subprocess that resolves ``unigym`` from this worktree."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.subprocess
def test_capabilities_import_no_backend_or_run_layer() -> None:
    """Importing the pure capability leaves pulls in no backend/run module."""
    code = (
        "import sys\n"
        "import unigym.capabilities.traits\n"
        "import unigym.capabilities.versions\n"
        "import unigym.capabilities.selection\n"
        "forbidden = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name == 'unigym.backends'\n"
        "    or name.startswith('unigym.backends.')\n"
        "    or name == 'unigym.registration'\n"
        "    or name.startswith('unigym.registration.')\n"
        "    or name == 'unigym.run'\n"
        "    or name.startswith('unigym.run.')\n"
        ")\n"
        "assert not forbidden, forbidden\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr


@pytest.mark.subprocess
def test_bindings_and_compose_import_no_engine() -> None:
    """The composition modules keep every engine import inside a factory."""
    code = (
        "import sys\n"
        "import unigym.backends.compose\n"
        "import unigym.backends.bindings\n"
        "heavy = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name.split('.')[0] in {'mujoco', 'mujoco_warp', 'warp', 'isaaclab', 'torch'}\n"
        ")\n"
        "assert not heavy, heavy\n"
    )
    result = _run_python(code)
    assert result.returncode == 0, result.stderr


def test_registry_facade_is_removed() -> None:
    """The temporary pure-vocabulary façade is gone after the P1c cutover."""
    import importlib.util

    assert importlib.util.find_spec("unigym.backends.registry") is None


def test_backend_runtime_combines_the_canonical_spec_and_bindings_tables() -> None:
    """``BackendRuntime`` is exactly the canonical spec + bindings pair."""
    from unigym.backends.bindings import BACKEND_BINDINGS
    from unigym.backends.compose import backend_runtime
    from unigym.capabilities.specs import BACKEND_SPECS
    from unigym.capabilities.traits import CANONICAL_BACKENDS

    for name in CANONICAL_BACKENDS:
        runtime = backend_runtime(name)
        assert runtime.spec is BACKEND_SPECS[name]
        assert runtime.bindings is BACKEND_BINDINGS[name]
        assert runtime.spec.name == name
