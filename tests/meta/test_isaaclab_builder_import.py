"""Import isolation for the generic IsaacLab builder and the asset-scene engine."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.subprocess
def test_builder_import_does_not_load_isaaclab():
    """A fresh interpreter imports the builder without loading isaaclab."""
    code = (
        "import sys; import unigym.backends.isaaclab2.robots.builder; "
        "loaded = sorted(m for m in sys.modules if m == 'isaaclab' or m.startswith('isaaclab.')); "
        "assert not loaded, loaded"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


@pytest.mark.subprocess
def test_asset_scene_engine_import_loads_no_isaaclab_pxr_or_omni():
    """The asset-scene engine imports no engine/OpenUSD/Kit module."""
    code = (
        "import sys; import unigym.backends.isaaclab2.asset_scene.engine; "
        "roots = ('isaaclab', 'pxr', 'omni'); "
        "loaded = sorted(m for m in sys.modules "
        "if any(m == r or m.startswith(r + '.') for r in roots)); "
        "assert not loaded, loaded"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


@pytest.mark.subprocess
def test_asset_scene_leaf_imports_do_not_register_the_engine():
    """Leaf imports neither register isaaclab2 nor load a heavy engine root."""
    code = (
        "import sys; "
        "import unigym.backends.isaaclab2.asset_scene.names; "
        "import unigym.backends.isaaclab2.static_shapes; "
        "from unigym.scenes.engine import registered_scene_engines; "
        "registered = registered_scene_engines(); "
        "assert registered == [], registered; "
        "roots = ('isaaclab', 'pxr', 'omni'); "
        "loaded = sorted(m for m in sys.modules "
        "if any(m == r or m.startswith(r + '.') for r in roots)); "
        "assert not loaded, loaded"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_build_robot_realization_rejects_a_profile_without_an_asset():
    """A profile without an effective asset is rejected by asset, not by name."""
    from unigym.assets.asset_ref import AssetRefError
    from unigym.backends.isaaclab2.robots import build_robot_realization
    from unigym.scenes.articulation import ArticulationSpec
    from tests._support.profiles import fixture_profile

    profile = fixture_profile(name="arm")

    with pytest.raises(AssetRefError, match="no asset reference is declared"):
        build_robot_realization(profile, spec=ArticulationSpec(name="arm"), params={})
