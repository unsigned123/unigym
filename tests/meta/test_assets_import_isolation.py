"""Import isolation: the light asset submodule never imports yaml."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.subprocess
def test_asset_ref_import_does_not_load_yaml():
    """A fresh interpreter importing unigym.assets.asset_ref keeps yaml unloaded."""
    code = (
        "import sys; import unigym.assets.asset_ref; "
        "assert 'yaml' not in sys.modules, sorted(sys.modules)"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_assets_manifest_names_still_resolve_lazily():
    """The public manifest names resolve through the lazy package surface."""
    import unigym.assets as assets

    assert "SCHEMA" in assets.__all__
    assert assets.SCHEMA == "unigym.asset/1"
    assert callable(assets.load_manifest)
