"""Identity-scoped handles, body vocabulary and ordered action-group validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from unigym.camera import Camera
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.runtime.actions import OrderedActionGroup
from unigym.sim import (
    ArticulationHandle,
    Control,
    DeferredFeatureError,
    ForeignHandleError,
    SimulationBuilder,
    SimulationConfigError,
)
from tests._support.sim_fixtures import write_sim_pack


def _profile() -> RobotProfile:
    """A tiny profile covering every declared body vocabulary source."""
    return RobotProfile(
        name="fixture_arm",
        joint_sets={"body": ("shoulder", "elbow"), "hand": ("finger",)},
        default_joint_pos={"shoulder": 0.0, "elbow": 0.0, "finger": 0.0},
        assets=None,
        extra_action_sets={},
        base_body="base",
        foot_bodies=("left_foot",),
        body_sets={"grasp.left": ("left_hand",)},
        body_bindings={
            "base": BodyBinding("base", {"mjcf": "base"}),
            "left_foot": BodyBinding("left_foot", {"mjcf": "left_foot"}),
            "left_hand": BodyBinding("left_hand", {"mjcf": "left_hand"}),
        },
    )


def _group(
    joints: tuple[str, ...], name: str = "body", articulation: str = "arm"
) -> OrderedActionGroup:
    """One declared position group."""
    return OrderedActionGroup(
        name=name,
        articulation=articulation,
        joints=joints,
        target="position",
    )


def _handle(*, owner: object | None = None) -> ArticulationHandle:
    """A directly constructed handle with two disjoint action groups."""
    return ArticulationHandle(
        name="arm",
        ref=None,
        profile=_profile(),
        control=None,
        action_groups={
            "body": _group(("shoulder", "elbow")),
            "finger": _group(("finger",), name="finger"),
        },
        _owner=object() if owner is None else owner,
    )


def test_handles_compare_and_hash_by_identity() -> None:
    """Two equal-looking handles from different owners stay distinct keys."""
    first = _handle()
    second = _handle()

    assert first == first
    assert first != second
    assert len({first, second}) == 2
    assert first.body("base") != second.body("base")


def test_body_accepts_every_declared_vocabulary_source() -> None:
    """Base, feet, body-set members and binding keys are all legal body ids."""
    handle = _handle()

    assert handle.body("base").canonical_id == "base"
    assert handle.body("left_foot").articulation is handle
    assert handle.body("left_hand").canonical_id == "left_hand"


def test_body_rejects_an_undeclared_canonical_id() -> None:
    """An unknown body id fails with the declared vocabulary named."""
    with pytest.raises(SimulationConfigError, match="no body id 'pelvis'"):
        _handle().body("pelvis")


def test_action_groups_are_read_only_and_keep_the_template() -> None:
    """The handle stores the ordered groups by name in a read-only mapping."""
    handle = _handle()

    assert set(handle.action_groups) == {"body", "finger"}
    assert handle.action_groups["body"].joints == ("shoulder", "elbow")
    assert handle.action_groups["body"].target == "position"
    with pytest.raises(TypeError):
        handle.action_groups["other"] = _group(("finger",), name="other")


def test_action_group_key_must_match_the_group_name() -> None:
    """A mapping key that disagrees with its group is a declaration error."""
    with pytest.raises(SimulationConfigError, match="does not match its group name"):
        ArticulationHandle(
            name="arm",
            ref=None,
            profile=_profile(),
            control=None,
            action_groups={"other": _group(("shoulder",))},
            _owner=object(),
        )


def test_action_group_must_belong_to_the_enclosing_articulation() -> None:
    """A group declared for another articulation cannot be re-homed by the key."""
    with pytest.raises(SimulationConfigError, match="declared for articulation"):
        ArticulationHandle(
            name="arm",
            ref=None,
            profile=_profile(),
            control=None,
            action_groups={"body": _group(("shoulder",), articulation="other")},
            _owner=object(),
        )


def test_body_camera_attach_is_foreign_or_deferred_across_builders(
    tmp_path: Path,
) -> None:
    """Own body handles reach the deferred feature; foreign ones are refused."""
    pack_set, ref = write_sim_pack(tmp_path)
    control = Control.position_pd(joint_set="body", kp=10.0, kd=0.5)
    first = SimulationBuilder(backend="mujoco", dt=0.02, physics_dt=0.005, pack_set=pack_set)
    second = SimulationBuilder(backend="mujoco", dt=0.02, physics_dt=0.005, pack_set=pack_set)
    own_body = first.add_articulation(ref, control=control).body("base")
    other_body = second.add_articulation(ref, control=control).body("base")
    camera = Camera.world("top", pos=(1.0, 0.0, 2.0), lookat=(0.0, 0.0, 0.0))

    with pytest.raises(ForeignHandleError):
        first.add_camera(camera, attach=other_body)
    with pytest.raises(DeferredFeatureError, match="Camera.attached"):
        first.add_camera(camera, attach=own_body)
