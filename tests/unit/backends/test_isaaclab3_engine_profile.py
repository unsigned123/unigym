"""The IsaacLab 3 engine profile is the only engine-divergent contract."""

from __future__ import annotations

import dataclasses

import pytest

from unigym.backends.shared.isaaclab3_common import EngineProfile


def _profile() -> EngineProfile:
    return EngineProfile(
        name="test",
        build_physics_cfg=lambda: object(),
        renderer_cfg=None,
        depth_background=0.0,
        default_joint_ordering=None,
        set_body_com=lambda asset, com, env_ids: None,
    )


def test_engine_profile_names_the_divergence_points() -> None:
    names = {field.name for field in dataclasses.fields(EngineProfile)}
    assert names == {
        "name",
        "build_physics_cfg",
        "renderer_cfg",
        "depth_background",
        "default_joint_ordering",
        "set_body_com",
    }


def test_engine_profile_is_frozen() -> None:
    profile = _profile()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(profile, "name", "other")
