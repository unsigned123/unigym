"""Identity-scoped handles, body vocabulary and action-binding validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from unigym.camera import Camera
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.sim import (
    ArticulationHandle,
    BindingError,
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
        action_faces={},
        base_body="base",
        foot_bodies=("left_foot",),
        body_sets={"grasp.left": ("left_hand",)},
        body_bindings={
            "base": BodyBinding("base", {"mjcf": "base"}),
            "left_foot": BodyBinding("left_foot", {"mjcf": "left_foot"}),
            "left_hand": BodyBinding("left_hand", {"mjcf": "left_hand"}),
        },
    )


def _handle(*, owner: object | None = None) -> ArticulationHandle:
    """A directly constructed handle with two disjoint faces."""
    return ArticulationHandle(
        name="arm",
        ref=None,
        profile=_profile(),
        control=None,
        faces={"body": ("shoulder", "elbow"), "hand": ("finger",)},
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


def test_bind_action_uses_the_face_order_by_default() -> None:
    """The default binding order is the declared face joint order."""
    binding = _handle().bind_action("body", kind="position")

    assert binding.face == "body"
    assert binding.joint_names == ("shoulder", "elbow")
    assert binding.kind == "position"
    assert binding.dim == 2


def test_bind_action_accepts_an_explicit_permutation() -> None:
    """An explicit order is the tensor column order, not a face redefinition."""
    binding = _handle().bind_action("body", kind="position", order=["elbow", "shoulder"])

    assert binding.joint_names == ("elbow", "shoulder")


@pytest.mark.parametrize(
    "order",
    [
        ["shoulder"],
        ["shoulder", "shoulder"],
        ["shoulder", "finger"],
    ],
)
def test_bind_action_rejects_a_non_permutation_order(order: list[str]) -> None:
    """An explicit order must be a duplicate-free permutation of the face."""
    with pytest.raises(BindingError, match="must be a permutation"):
        _handle().bind_action("body", kind="position", order=order)


def test_bind_action_rejects_an_unknown_face() -> None:
    """The face must exist, and the error lists the declared faces."""
    with pytest.raises(BindingError, match="no action face"):
        _handle().bind_action("legs", kind="position")


def test_bind_action_rejects_an_illegal_kind() -> None:
    """The kind vocabulary is owned by the runtime action contract."""
    with pytest.raises(BindingError, match="kind must be one of"):
        _handle().bind_action("body", kind="position_pd")


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
