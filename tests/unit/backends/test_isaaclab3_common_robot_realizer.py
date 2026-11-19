"""Engine-free checks that the shared IsaacLab 3 realizer honours ``asset_backends``."""

from __future__ import annotations

from dataclasses import replace

import pytest

from unigym.actuators.ir import ActuationGroup
from unigym.assets.asset_ref import AssetRefError
from unigym.backends.registry import BackendInfo
from unigym.scenes.articulation import ArticulationSpec
from tests._support.profiles import fixture_profile

_ISAACLAB3 = BackendInfo("isaaclab3_physx", (3, 1), frozenset(), asset_formats=("usd", "urdf"))


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
    import unigym.backends.shared.isaaclab3_common.robots.builder as builder
    import unigym.backends.registry as registry

    def _fake_build(profile: object, **kwargs: object) -> str:
        captured["asset_format"] = kwargs["asset_format"]
        return "cfg"

    monkeypatch.setattr(registry, "backend_info", lambda _name: _ISAACLAB3)
    monkeypatch.setattr(builder, "build_articulation_cfg", _fake_build)


def test_isaaclab3_realizer_selects_the_declared_format_allowed_for_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """USD is preferred by the backend but restricted elsewhere, so URDF wins."""
    from unigym.backends.shared.isaaclab3_common.robots import build_robot_realization

    captured: dict[str, object] = {}
    _patch_realizer(monkeypatch, captured)
    profile = replace(
        fixture_profile(name="arm", usd_path="/assets/arm.usd", urdf_path="/assets/arm.urdf"),
        asset_backends={"usd": ("mujoco",), "urdf": ("isaaclab3_physx",)},
    )

    assert (
        build_robot_realization(profile, spec=_spec(), params={}, backend="isaaclab3_physx")
        == "cfg"
    )
    assert captured["asset_format"] == "urdf"


def test_isaaclab3_realizer_rejects_a_format_restricted_to_other_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No declared format is allowed for isaaclab3_physx, so selection fails closed."""
    from unigym.backends.shared.isaaclab3_common.robots import build_robot_realization

    captured: dict[str, object] = {}
    _patch_realizer(monkeypatch, captured)
    profile = replace(
        fixture_profile(name="arm", usd_path="/assets/arm.usd"),
        asset_backends={"usd": ("mujoco",)},
    )

    with pytest.raises(AssetRefError, match="restrictions"):
        build_robot_realization(profile, spec=_spec(), params={}, backend="isaaclab3_physx")
    assert captured == {}


def test_realizer_for_returns_a_registry_shaped_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``realizer_for`` forwards the explicit backend and the registry signature."""
    from unigym.backends.shared.isaaclab3_common.robots import realizer_for

    captured: dict[str, object] = {}
    _patch_realizer(monkeypatch, captured)
    profile = fixture_profile(name="arm", usd_path="/assets/arm.usd", urdf_path="/assets/arm.urdf")

    realize = realizer_for("isaaclab3_physx")

    assert callable(realize)
    assert realize(profile, spec=_spec(), params={}, asset_plan=None) == "cfg"
    assert captured["asset_format"] == "usd"


def test_realizer_for_accepts_both_canonical_isaaclab3_backends() -> None:
    """The shared closure is parameterised by the explicit canonical backend."""
    from unigym.backends.shared.isaaclab3_common.robots import realizer_for

    assert callable(realizer_for("isaaclab3_physx"))
    assert callable(realizer_for("isaaclab3_newton"))
