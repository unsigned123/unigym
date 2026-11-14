"""Import isolation: ``unigym`` and ``unigym.sim`` stay engine-free."""

from __future__ import annotations

import subprocess
import sys

import pytest

_HEAVY_ROOTS = frozenset({"torch", "mujoco", "isaaclab", "omni", "warp"})


@pytest.mark.subprocess
def test_import_unigym_and_sim_load_no_heavy_backend() -> None:
    """A fresh interpreter importing both namespaces loads no engine/torch."""
    code = (
        "import sys\n"
        "import unigym\n"
        "import unigym.sim\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in "
        f"{sorted(_HEAVY_ROOTS)!r})\n"
        "assert not heavy, heavy\n"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


@pytest.mark.subprocess
def test_lazy_public_sim_names_resolve_without_a_backend() -> None:
    """The top-level lazy facade names resolve without importing an engine."""
    code = (
        "import sys\n"
        "import unigym\n"
        "assert unigym.Control.position_pd is not None\n"
        "assert unigym.SimulationBuilder is not None\n"
        "assert unigym.sim.SIM_API_VERSION == 1\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in "
        f"{sorted(_HEAVY_ROOTS)!r})\n"
        "assert not heavy, heavy\n"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
