"""Contract for the external robot factory registry and profile resolution."""

from __future__ import annotations

from typing import Any

import pytest

from unigym.robots import (
    ROBOT_REGISTRY,
    RobotProfile,
    build_robot_profile,
    register_robot,
    restore_robot_registry,
    restore_robots,
    robot_origin,
    snapshot_robot_registry,
    snapshot_robots,
)
from unigym.provenance import ProviderOrigin


@pytest.fixture(autouse=True)
def _isolate_robot_registry():
    """Restore the process-global registry and its origins after every test."""
    snapshot = snapshot_robot_registry()
    yield
    restore_robot_registry(snapshot)


def _arm_profile(name: str = "acme.arm", *, shoulder: float = 0.0) -> RobotProfile:
    return RobotProfile(
        name=name,
        joint_sets={"body": ("shoulder",)},
        default_joint_pos={"shoulder": shoulder},
        assets=None,
        action_faces={},
        base_body=None,
        foot_bodies=(),
        body_sets={},
        body_bindings={},
    )


def make_arm(*, shoulder: float = 0.0) -> RobotProfile:
    return _arm_profile(shoulder=shoulder)


def make_other() -> RobotProfile:
    return _arm_profile("acme.other")


def test_register_robot_builds_a_non_builtin_profile() -> None:
    """A registered external factory is resolved by build_robot_profile."""
    assert register_robot("acme.arm", make_arm) is make_arm

    profile = build_robot_profile({"name": "acme.arm", "params": {"shoulder": 0.25}})

    assert isinstance(profile, RobotProfile)
    assert profile.name == "acme.arm"
    assert profile.default_joint_pos["shoulder"] == pytest.approx(0.25)
    assert "acme.arm" in ROBOT_REGISTRY


def test_register_robot_rejects_duplicate_external_name() -> None:
    """A duplicate external name fails and keeps the original owner."""
    register_robot("acme.arm", make_arm)

    with pytest.raises(ValueError, match="already registered"):
        register_robot("acme.arm", make_other)

    assert ROBOT_REGISTRY["acme.arm"] is make_arm


def test_register_robot_rejects_one_factory_under_two_names() -> None:
    """One factory owns exactly one name; the error names both names."""
    register_robot("acme.arm", make_arm)

    with pytest.raises(ValueError) as info:
        register_robot("acme.other", make_arm)

    message = str(info.value)
    assert "acme.arm" in message
    assert "acme.other" in message
    assert "acme.other" not in ROBOT_REGISTRY


@pytest.mark.parametrize(
    "name",
    [
        "",
        " Acme.arm",
        "acme.arm ",
        "Acme.arm",
        "acme.Arm",
        "acme..arm",
        "acme-arm",
        "9acme",
        None,
        3,
    ],
)
def test_register_robot_rejects_malformed_names(name: Any) -> None:
    """Names use the adapter grammar verbatim and are never trimmed."""
    with pytest.raises(ValueError, match="lowercase snake_case"):
        register_robot(name, make_arm)

    assert ROBOT_REGISTRY == {}


@pytest.mark.parametrize(
    "factory",
    [_arm_profile(), object(), 3, "factory"],
)
def test_register_robot_rejects_non_callable_factories(factory: object) -> None:
    """A profile instance and every other non-callable is not a factory."""
    with pytest.raises(ValueError, match="must be callable"):
        register_robot("acme.arm", factory)

    assert "acme.arm" not in ROBOT_REGISTRY


def test_build_robot_profile_rejects_a_non_profile_factory_result() -> None:
    """A callable that does not return a RobotProfile fails at the boundary."""

    def wrong() -> object:
        return object()

    register_robot("acme.wrong", wrong)

    with pytest.raises(TypeError, match="must return a RobotProfile"):
        build_robot_profile({"name": "acme.wrong"})


def test_build_robot_profile_rejects_a_mismatched_profile_name() -> None:
    """The returned profile name must equal the registry name."""

    def wrong() -> RobotProfile:
        return _arm_profile("acme.other")

    register_robot("acme.arm", wrong)

    with pytest.raises(ValueError, match="registry name and the profile name"):
        build_robot_profile({"name": "acme.arm"})


def test_unknown_robot_is_an_attribute_error() -> None:
    """An unregistered name keeps the existing teaching error family."""
    with pytest.raises(AttributeError, match="unknown robot 'acme.missing'"):
        build_robot_profile({"name": "acme.missing"})

    assert "acme.missing" not in ROBOT_REGISTRY


def test_snapshot_restore_returns_the_exact_registry() -> None:
    """Restoring a snapshot removes later names and keeps the prior entries."""
    register_robot("acme.arm", make_arm)
    snapshot = snapshot_robots()
    register_robot("acme.extra", make_other)

    restore_robots(snapshot)

    assert ROBOT_REGISTRY == {"acme.arm": make_arm}
    assert "acme.extra" not in ROBOT_REGISTRY


def test_snapshot_is_an_independent_copy() -> None:
    """Mutating the live registry does not mutate an earlier snapshot."""
    snapshot = snapshot_robots()
    register_robot("acme.arm", make_arm)

    assert snapshot == {}
    assert "acme.arm" in ROBOT_REGISTRY


def test_origin_aware_snapshot_restores_values_and_origins_together() -> None:
    """The origin-aware pair is total; the legacy pair stays factory-only."""
    origin = ProviderOrigin(kind="manifest", pack="acme", pack_version="1.0.0", item_id="acme.arm")
    register_robot("acme.arm", make_arm, origin=origin)
    snapshot = snapshot_robot_registry()
    register_robot("acme.extra", make_other)

    restore_robot_registry(snapshot)

    assert ROBOT_REGISTRY == {"acme.arm": make_arm}
    assert robot_origin("acme.arm") == origin

    legacy = snapshot_robots()
    register_robot("acme.extra", make_other)
    restore_robots(legacy)

    assert ROBOT_REGISTRY == {"acme.arm": make_arm}
    assert robot_origin("acme.arm") == origin
