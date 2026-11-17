"""RobotProfile body-binding completeness and body-set invariants."""

from __future__ import annotations

import pytest

from dataclasses import FrozenInstanceError

from unigym.assets.asset_ref import AssetRef
from unigym.actuators.ir import ActuationGroup
from unigym.robots.base import BodyBinding, BodyBindingError, RobotProfile


def _profile(**overrides):
    base = {
        "name": "arm",
        "joint_sets": {"body": ("j",)},
        "default_joint_pos": {"j": 0.0},
        "assets": None,
        "extra_action_sets": {},
        "base_body": None,
        "foot_bodies": (),
        "body_sets": {},
        "body_bindings": {},
    }
    base.update(overrides)
    return RobotProfile(**base)


def test_profile_accepts_none_or_mapping_assets():
    """Assets accept an AssetRef, an AssetRef mapping, or None."""
    assert _profile(assets=None).assets is None
    assert _profile(assets=AssetRef(usd="robot.usd")).assets == AssetRef(usd="robot.usd")
    assert _profile(assets={"mjcf": "robot.xml"}).assets == AssetRef(mjcf="robot.xml")


def test_base_foot_and_body_set_members_require_bindings():
    """Every canonical body referenced by the profile must be bound."""
    binding = BodyBinding("base", {"mjcf": "root"})

    with pytest.raises(BodyBindingError, match=r"missing required ids \['foot'\]"):
        _profile(base_body="base", foot_bodies=("foot",), body_bindings={"base": binding})


def test_body_bindings_validate_with_formats_none_and_allow_multiple_formats():
    """Profile construction does not reject a binding that carries several formats."""
    profile = _profile(
        base_body="base",
        body_bindings={
            "base": BodyBinding("base", {"mjcf": "root", "usd": "Root"}),
        },
    )

    assert profile.body_bindings["base"].names == {"mjcf": "root", "usd": "Root"}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"body_sets": {"grasp": ()}}, "is empty"),
        ({"body_sets": {"grasp": ("base", "base")}}, "duplicate body ids"),
        ({"body_sets": {"": ("base",)}}, "names must be non-empty"),
        ({"base_body": "left/base"}, "base_body"),
    ],
)
def test_profile_rejects_malformed_body_sets_and_ids(kwargs, match):
    """Body sets are non-empty, unique and named; ids are canonical."""
    bindings = {"base": BodyBinding("base", {"mjcf": "root"})}
    with pytest.raises((ValueError, BodyBindingError), match=match):
        _profile(body_bindings=bindings, **kwargs)


def test_profile_base_height_is_neutral_optional_data():
    """base_height is None or a finite float; malformed values fail fast."""
    assert _profile().base_height is None
    assert _profile(base_height=0.78).base_height == pytest.approx(0.78)

    with pytest.raises(ValueError, match="finite number or None"):
        _profile(base_height=True)
    with pytest.raises(ValueError, match="finite"):
        _profile(base_height=float("nan"))


def test_profile_has_no_body_parts_or_robot_asset_fields():
    """The hard cutover leaves no compatibility fields."""
    import dataclasses

    from unigym.robots import base as robot_base

    fields = {item.name for item in dataclasses.fields(RobotProfile)}
    assert "body_parts" not in fields
    assert "assets" in fields
    assert not hasattr(robot_base, "RobotAsset")


def _actuation(**overrides: object) -> dict[str, object]:
    """One valid standard-PD body group for the two-joint test profile."""
    entry: dict[str, object] = {
        "name": "body",
        "law": "standard_pd",
        "joint_set": "body",
        "gains": {"kp": 20.0, "kd": 1.0},
    }
    entry.update(overrides)
    return entry


def test_profile_actuation_accepts_a_covering_group_and_stays_frozen() -> None:
    """A valid group becomes normalized ActuationGroup data on a frozen profile."""
    profile = _profile(
        joint_sets={"body": ("a", "b")},
        default_joint_pos={"a": 0.0, "b": 0.0},
        actuation=[_actuation()],
    )
    assert len(profile.actuation) == 1
    assert isinstance(profile.actuation[0], ActuationGroup)
    assert profile.actuation[0].joint_set == "body"
    with pytest.raises(FrozenInstanceError):
        profile.actuation = ()  # type: ignore[misc]


def test_profile_empty_actuation_is_legal() -> None:
    """A passive profile may declare no actuation."""
    assert _profile().actuation == ()
    assert _profile(actuation=()).actuation == ()


@pytest.mark.parametrize(
    ("actuation", "match"),
    [
        ([_actuation(), _actuation()], "must be unique"),
        ([_actuation(name="bad", joint_set="missing")], "unknown"),
        ([_actuation(joints=("missing",), joint_set=None)], "unknown joints"),
    ],
)
def test_profile_actuation_rejects_invalid_groups(actuation, match: str) -> None:
    """Duplicate names, unresolved sets and unknown joints fail fast."""
    with pytest.raises(ValueError, match=match):
        _profile(
            joint_sets={"body": ("a", "b")},
            default_joint_pos={"a": 0.0, "b": 0.0},
            actuation=actuation,
        )


def test_profile_actuation_rejects_incomplete_coverage() -> None:
    """A non-empty actuation tuple must cover every declared non-passive joint."""
    entry = _actuation(joints=("a",), joint_set=None)
    with pytest.raises(ValueError, match=r"do not cover joints \['b'\]"):
        _profile(
            joint_sets={"body": ("a", "b")},
            default_joint_pos={"a": 0.0, "b": 0.0},
            actuation=[entry],
        )
