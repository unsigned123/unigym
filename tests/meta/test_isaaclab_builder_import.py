"""Import isolation for the generic IsaacLab builder and asset rejection."""

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


def test_build_robot_realization_rejects_a_profile_without_usd():
    """A profile without an effective USD is rejected by asset, not by name."""
    from unigym.backends.isaaclab2.robots import build_robot_realization
    from unigym.scenes.articulation import ArticulationSpec
    from tests._support.profiles import fixture_profile

    profile = fixture_profile(name="arm")

    with pytest.raises(NotImplementedError, match="no effective USD"):
        build_robot_realization(profile, spec=ArticulationSpec(name="arm"), params={})
