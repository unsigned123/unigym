"""The isaaclab robot realizer seam rejects robots with no realization."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.sim, pytest.mark.usefixtures("simulation_app")]


def test_isaaclab_realizer_rejects_a_profile_without_a_realization():
    """GR1 has no isaaclab realization and the seam fails fast."""
    from unigym.backends.isaaclab2.robots import build_robot_realization
    from unigym.robots import build_robot_profile
    from unigym.scenes.articulation import ArticulationSpec

    profile = build_robot_profile({"name": "gr1"})

    with pytest.raises(NotImplementedError, match="no robot realizer.*gr1"):
        build_robot_realization(profile, spec=ArticulationSpec(name="gr1"), params={})
