"""The isaaclab robot realizer rejects robots with no effective USD asset."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.sim, pytest.mark.usefixtures("simulation_app")]


def test_isaaclab_realizer_rejects_a_profile_without_an_effective_usd_asset():
    """GR1 declares no USD asset, so the generic realizer rejects it by asset."""
    from unigym.backends.isaaclab2.robots import build_robot_realization
    from unigym.robots import build_robot_profile
    from unigym.scenes.articulation import ArticulationSpec

    profile = build_robot_profile({"name": "gr1"})

    with pytest.raises(NotImplementedError, match="no effective USD"):
        build_robot_realization(profile, spec=ArticulationSpec(name="gr1"), params={})


def test_isaaclab_realizer_does_not_fall_back_from_a_model_without_usd():
    """An explicit spec.model without USD never falls back to the profile USD."""
    from unigym.assets.asset_ref import AssetRef
    from unigym.backends.isaaclab2.robots import build_robot_realization
    from unigym.robots.g1 import make_g1_profile
    from unigym.scenes.articulation import ArticulationSpec

    profile = make_g1_profile(usd_path="/profile/g1.usd")
    spec = ArticulationSpec(name="g1", model=AssetRef(mjcf="model.xml"))

    with pytest.raises(NotImplementedError, match="spec.model does not fall back"):
        build_robot_realization(profile, spec=spec, params={})
