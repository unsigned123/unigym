"""Importing core's step boundary pulls in no pack package."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.subprocess
def test_evaluation_import_has_no_pack_effects() -> None:
    """``unigym.evaluation`` imports no pack layer."""
    code = (
        "import sys\n"
        "import unigym.evaluation\n"
        "import unigym.evaluation.step\n"
        "packs = sorted(m for m in sys.modules if m == 'unigym.packs' "
        "or m.startswith('unigym.packs.'))\n"
        "assert not packs, packs\n"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
