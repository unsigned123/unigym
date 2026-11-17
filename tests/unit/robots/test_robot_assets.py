"""Backend-aware RobotProfile asset selection."""

from __future__ import annotations

import pytest

from unigym.assets.asset_ref import AssetRef, AssetRefError
from unigym.robots.assets import asset_options_for, select_backend_asset, select_robot_asset
from unigym.robots.base import RobotProfile


def _profile(
    assets: AssetRef | None,
    *,
    asset_backends: dict[str, tuple[str, ...]] | None = None,
    asset_options: dict[str, dict[str, object]] | None = None,
) -> RobotProfile:
    return RobotProfile(
        name="robot",
        joint_sets={"body": ("j",)},
        default_joint_pos={"j": 0.0},
        assets=assets,
        extra_action_sets={},
        base_body=None,
        foot_bodies=(),
        body_sets={},
        body_bindings={},
        asset_backends=asset_backends or {},
        asset_options=asset_options or {},
    )


def test_mujoco_selects_the_declared_mjcf_resource() -> None:
    profile = _profile(
        AssetRef(mjcf="robot.xml", usd="robot.usd"),
        asset_backends={"mjcf": ("mujoco", "mjwarp"), "usd": ("isaaclab2",)},
    )

    assert select_robot_asset(profile, backend="mujoco", backend_formats=("mjcf",)) == (
        "mjcf",
        "robot.xml",
    )


def test_isaaclab2_selects_the_declared_usd_resource() -> None:
    profile = _profile(
        AssetRef(mjcf="robot.xml", usd="robot.usd"),
        asset_backends={"mjcf": ("mujoco", "mjwarp"), "usd": ("isaaclab2",)},
    )

    assert select_robot_asset(profile, backend="isaaclab2", backend_formats=("usd", "urdf")) == (
        "usd",
        "robot.usd",
    )


def test_a_format_restricted_to_other_backends_is_skipped() -> None:
    profile = _profile(
        AssetRef(mjcf="robot.xml", usd="robot.usd"),
        asset_backends={"mjcf": ("mjwarp",), "usd": ("mujoco",)},
    )

    assert select_robot_asset(profile, backend="mujoco", backend_formats=("mjcf", "usd")) == (
        "usd",
        "robot.usd",
    )


def test_no_compatible_format_reports_declared_formats_and_restrictions() -> None:
    profile = _profile(
        AssetRef(mjcf="robot.xml"),
        asset_backends={"mjcf": ("isaaclab2",)},
    )

    with pytest.raises(AssetRefError, match=r"\['mjcf'\].*'mujoco'.*restrictions"):
        select_robot_asset(profile, backend="mujoco", backend_formats=("mjcf",))


def test_empty_restrictions_keep_the_backend_preference_order() -> None:
    profile = _profile(AssetRef(mjcf="robot.xml", usd="robot.usd"))

    assert select_robot_asset(profile, backend="isaaclab2", backend_formats=("usd", "urdf")) == (
        "usd",
        "robot.usd",
    )
    assert select_robot_asset(profile, backend="isaaclab2", backend_formats=("urdf", "usd")) == (
        "usd",
        "robot.usd",
    )


def test_asset_options_are_metadata_for_the_selected_format() -> None:
    profile = _profile(
        AssetRef(usd="robot.usd"),
        asset_options={"usd": {"scale": 1.0}},
    )

    assert dict(asset_options_for(profile, "usd")) == {"scale": 1.0}
    assert dict(asset_options_for(profile, "mjcf")) == {}
    with pytest.raises(AssetRefError, match="unknown asset format"):
        asset_options_for(profile, "obj")


def test_select_backend_asset_rejects_a_bare_string_allow_list() -> None:
    """A string must not substring-match a backend name in the public helper."""
    with pytest.raises(AssetRefError, match="must be a sequence"):
        select_backend_asset(
            AssetRef(mjcf="robot.xml"),
            asset_backends={"mjcf": "mujoco"},
            backend="mujoco",
            backend_formats=("mjcf",),
        )


@pytest.mark.parametrize(
    ("backends", "match"),
    [
        ((), "must not be empty"),
        (("mujoco", "mujoco"), "must be unique"),
        (("mujoco", ""), "non-empty strings"),
    ],
)
def test_select_backend_asset_validates_allow_list_entries(
    backends: tuple[str, ...], match: str
) -> None:
    """Empty, duplicated and blank backend names fail instead of matching."""
    with pytest.raises(AssetRefError, match=match):
        select_backend_asset(
            AssetRef(mjcf="robot.xml"),
            asset_backends={"mjcf": backends},
            backend="mujoco",
            backend_formats=("mjcf",),
        )
