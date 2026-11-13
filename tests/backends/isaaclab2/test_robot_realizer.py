"""The isaaclab robot realizer rejects robots with no effective USD asset."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.sim, pytest.mark.usefixtures("simulation_app")]


def test_isaaclab_realizer_rejects_a_profile_without_an_effective_usd_asset():
    """A profile without USD is rejected by asset, not by robot name."""
    from unigym.backends.isaaclab2.robots import build_robot_realization
    from unigym.scenes.articulation import ArticulationSpec
    from tests._support.profiles import fixture_profile

    profile = fixture_profile(name="arm")

    with pytest.raises(NotImplementedError, match="no effective USD"):
        build_robot_realization(profile, spec=ArticulationSpec(name="arm"), params={})


def test_isaaclab_realizer_does_not_fall_back_from_a_model_without_usd():
    """An explicit spec.model without USD never falls back to the profile USD."""
    from unigym.assets.asset_ref import AssetRef
    from unigym.backends.isaaclab2.robots import build_robot_realization
    from unigym.scenes.articulation import ArticulationSpec
    from tests._support.profiles import fixture_profile

    profile = fixture_profile(name="arm", usd_path="/profile/arm.usd")
    spec = ArticulationSpec(name="arm", model=AssetRef(mjcf="model.xml"))

    with pytest.raises(NotImplementedError, match="spec.model does not fall back"):
        build_robot_realization(profile, spec=spec, params={})
