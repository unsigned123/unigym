"""Importing core's step boundary pulls in no benchmark or pack package."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.subprocess
def test_evaluation_import_has_no_benchmark_or_pack_effects() -> None:
    """``unigym.evaluation`` imports no benchmark pack layer."""
    code = (
        "import sys\n"
        "import unigym.evaluation\n"
        "import unigym.evaluation.step\n"
        "bench = sorted(m for m in sys.modules if m == 'unigym.benchmarks' "
        "or m.startswith('unigym.benchmarks.'))\n"
        "assert not bench, bench\n"
        "assert 'unigym_bench' not in sys.modules, 'unigym_bench imported'\n"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
