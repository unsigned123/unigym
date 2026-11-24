"""The obsolete signal API is absent and no migrated consumer imports it."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_REMOVED_NAMES = ("SignalBus", "SignalView", "SignalPublisher")
_FACADES = (
    "unigym",
    "unigym.sink",
    "unigym.run.sink",
    "unigym.run.instance",
    "unigym.values",
    "unigym.commands",
    "unigym.evaluation",
    "unigym.runner",
)
_CONSUMERS = (
    "unigym.run.sink",
    "unigym.run.instance",
    "unigym.runner.sync",
    "unigym.evaluation.step",
    "unigym.commands.base",
    "unigym.values",
    "unigym.env.obs",
    "unigym.watchers.engine",
)

#: Drop stale editable-install mappings (for example a shared venv pointing at
#: another checkout) so the subprocess imports exactly this checkout.  Without
#: this, a deleted submodule can still resolve through the other tree.
_PRELUDE = (
    "import sys\n"
    "sys.meta_path[:] = [f for f in sys.meta_path "
    "if not getattr(f, '__module__', '').startswith('__editable__')]\n"
    "import unigym\n"
    "from pathlib import Path\n"
    "assert Path(unigym.__file__).resolve() == Path("
    f"{str(_ROOT / 'unigym' / '__init__.py')!r}).resolve(), unigym.__file__\n"
)


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh interpreter rooted at this checkout."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-c", _PRELUDE + code],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        env=env,
        timeout=120,
    )


@pytest.mark.subprocess
def test_signal_module_is_not_importable() -> None:
    """A fresh interpreter cannot find or import ``unigym.signals``."""
    code = (
        "import importlib.util\n"
        "assert importlib.util.find_spec('unigym.signals') is None\n"
        "try:\n"
        "    import unigym.signals\n"
        "except ModuleNotFoundError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('unigym.signals is importable')\n"
    )

    result = _run(code)

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("facade_name", _FACADES)
def test_public_facades_expose_no_signal_names(facade_name: str) -> None:
    """No public facade advertises or exposes a removed signal symbol."""
    facade = importlib.import_module(facade_name)
    exported = getattr(facade, "__all__", ())
    for name in _REMOVED_NAMES:
        assert name not in exported, f"{facade_name}.__all__ still exports {name}"
        assert not hasattr(facade, name), f"{facade_name} still exposes {name}"


@pytest.mark.subprocess
def test_migration_consumers_import_without_a_signal_module() -> None:
    """Every migrated production consumer imports with ``unigym.signals`` gone."""
    imports = "".join(f"import {name}\n" for name in _CONSUMERS)
    code = (
        imports
        + "import sys\n"
        + "assert 'unigym.signals' not in sys.modules, sorted(sys.modules)\n"
    )

    result = _run(code)

    assert result.returncode == 0, result.stderr
