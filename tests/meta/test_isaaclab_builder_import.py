"""Import isolation for the generic IsaacLab builder and the GR1 seam."""

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


def test_build_robot_realization_still_rejects_gr1():
    """GR1 is rejected by its missing effective USD asset, not by name."""
    from unigym.backends.isaaclab2.robots import build_robot_realization
    from unigym.robots import build_robot_profile
    from unigym.scenes.articulation import ArticulationSpec

    profile = build_robot_profile({"name": "gr1"})

    with pytest.raises(NotImplementedError, match="no effective USD"):
        build_robot_realization(profile, spec=ArticulationSpec(name="gr1"), params={})
