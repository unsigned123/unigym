"""Engine-free checks that the isaaclab2 realizer honours ``asset_backends``."""

from __future__ import annotations

from dataclasses import replace

import pytest

from unigym.actuators.ir import ActuationGroup
from unigym.assets.asset_ref import AssetRefError
from unigym.scenes.articulation import ArticulationSpec
from tests._support.profiles import fixture_profile


def _spec() -> ArticulationSpec:
    return ArticulationSpec(
        name="arm",
        actuation=(
            ActuationGroup(
                name="body",
                law="standard_pd",
                joint_set="body",
                gains={"kp": 1.0, "kd": 0.1},
            ),
        ),
    )


def _patch_realizer(monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]) -> None:
    """Replace the cfg builder and the backend probe (no engine import)."""
    import unigym.backends.isaaclab2.robots.builder as builder

    def _fake_build(profile: object, **kwargs: object) -> str:
        captured["asset_format"] = kwargs["asset_format"]
        return "cfg"

    monkeypatch.setattr(builder, "build_articulation_cfg", _fake_build)


def test_isaaclab2_realizer_selects_the_declared_format_allowed_for_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """USD is preferred by the backend but restricted elsewhere, so URDF wins."""
    from unigym.backends.isaaclab2.robots import build_robot_realization

    captured: dict[str, object] = {}
    _patch_realizer(monkeypatch, captured)
    profile = replace(
        fixture_profile(name="arm", usd_path="/assets/arm.usd", urdf_path="/assets/arm.urdf"),
        asset_backends={"usd": ("mujoco",), "urdf": ("isaaclab2",)},
    )

    assert build_robot_realization(profile, spec=_spec(), params={}) == "cfg"
    assert captured["asset_format"] == "urdf"


def test_isaaclab2_realizer_rejects_a_format_restricted_to_other_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No declared format is allowed for isaaclab2, so selection fails closed."""
    from unigym.backends.isaaclab2.robots import build_robot_realization

    captured: dict[str, object] = {}
    _patch_realizer(monkeypatch, captured)
    profile = replace(
        fixture_profile(name="arm", usd_path="/assets/arm.usd"),
        asset_backends={"usd": ("mujoco",)},
    )

    with pytest.raises(AssetRefError, match="restrictions"):
        build_robot_realization(profile, spec=_spec(), params={})
    assert captured == {}
