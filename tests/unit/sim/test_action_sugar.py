"""The named action sugar: coverage, face selection, order and shape."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
import torch

from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run.api import Run
from unigym.runner.state import read_extra
from unigym.sim import (
    ArticulationHandle,
    BindingError,
    Control,
    ControlSpec,
    ForeignHandleError,
    PendingActionError,
    Simulation,
)
from unigym.world import World
from tests._support.fakes.runner import FakeRunner


def _handle(
    *,
    faces: Mapping[str, Sequence[str]] | None = None,
    name: str = "arm",
    owner: object | None = None,
) -> ArticulationHandle:
    """One real handle with explicit control faces over its profile."""
    declared = {"arm": ("shoulder", "elbow")} if faces is None else dict(faces)
    joints: list[str] = []
    for face_joints in declared.values():
        for joint in face_joints:
            if joint not in joints:
                joints.append(joint)
    token = object() if owner is None else owner
    profile = RobotProfile(
        name=name,
        joint_sets={"body": tuple(joints)},
        default_joint_pos={joint: 0.0 for joint in joints},
        assets=None,
        action_faces={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
    )
    specs = tuple(
        Control.position_pd(joints=tuple(face_joints), face=face_name, kp=1.0, kd=0.1).faces[
            face_name
        ]
        for face_name, face_joints in declared.items()
    )
    control = ControlSpec.from_faces(specs)
    return ArticulationHandle(
        name=name,
        ref=None,
        profile=profile,
        control=control,
        faces={face_name: tuple(face_joints) for face_name, face_joints in declared.items()},
        _owner=token,
    )


def _sim(handle: ArticulationHandle, runner: FakeRunner) -> Simulation:
    """Construct the facade directly over the typed runner double."""
    run = Run({"run_name": "facade-sugar", "env": {"backend": "mujoco", "num_envs": 1}})
    return Simulation(
        backend="mujoco",
        num_envs=runner.env.num_envs,
        world=World.base(mjcf="world.xml"),
        articulations=(handle,),
        cameras=(),
        runner=runner,
        env=runner.env,
        run=run,
        owner=handle._owner,
        state=read_extra(runner.env),
        obs_cache=dict(runner.env.obs_cache),
        t=0,
    )


def test_named_mapping_order_becomes_the_named_action_column_order() -> None:
    """The mapping key order is the lowered column order; names resolve the rest."""
    handle = _handle()
    runner = FakeRunner(joint_names=("shoulder", "elbow"))
    sim = _sim(handle, runner)

    sim.act(handle, {"elbow": 0.7, "shoulder": 0.3})
    obs = sim.step()

    action = runner.last_actions
    assert action is not None
    assert action["arm"].joint_names == ["elbow", "shoulder"]
    assert action["arm"].values.shape == (1, 2)
    assert torch.allclose(action["arm"].values, torch.tensor([[0.7, 0.3]]))
    assert obs.articulations[handle].joint_names == ("shoulder", "elbow")
    assert torch.allclose(obs.articulations[handle].joint_pos, torch.tensor([[0.3, 0.7]]))


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"shoulder": 0.1}, r"missing \['elbow'\]"),
        (
            {"shoulder": 0.1, "elbow": 0.2, "wrist": 0.3},
            r"unexpected \['wrist'\]",
        ),
    ],
)
def test_named_mapping_must_cover_every_face_joint(values: dict[str, float], match: str) -> None:
    """A partial or over-specified mapping names both sets and stages nothing."""
    handle = _handle()

    with pytest.raises(BindingError, match=match) as excinfo:
        _sim(handle, FakeRunner(joint_names=("shoulder", "elbow"))).act(handle, values)

    message = str(excinfo.value)
    assert "face joints ['shoulder', 'elbow']" in message


def test_multiple_faces_require_an_explicit_face() -> None:
    """An ambiguous handle lists its declared faces instead of guessing."""
    handle = _handle(faces={"body": ("shoulder",), "hand": ("elbow",)})
    sim = _sim(handle, FakeRunner(joint_names=("shoulder", "elbow")))

    with pytest.raises(BindingError, match="needs face=") as excinfo:
        sim.act(handle, {"shoulder": 0.1})

    message = str(excinfo.value)
    assert "['body', 'hand']" in message


def test_explicit_face_selects_one_of_many() -> None:
    """face= stages the named face even when the handle declares several."""
    handle = _handle(faces={"body": ("shoulder",), "hand": ("elbow",)})
    sim = _sim(handle, FakeRunner(joint_names=("shoulder", "elbow")))

    sim.act(handle, {"shoulder": 0.1}, face="body")

    with pytest.raises(PendingActionError, match="already staged"):
        sim.act(handle, {"shoulder": 0.2}, face="body")


def test_explicit_face_is_accepted_on_the_single_face_form() -> None:
    """The explicit face argument matches the one declared face."""
    handle = _handle()
    runner = FakeRunner(joint_names=("shoulder", "elbow"))
    sim = _sim(handle, runner)

    sim.act(handle, {"shoulder": 0.3, "elbow": -0.4}, face="arm")
    obs = sim.step()

    assert torch.allclose(obs.articulations[handle].joint_pos, torch.tensor([[0.3, -0.4]]))


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"shoulder": "high", "elbow": 0.1}, "must be a finite number"),
        ({"shoulder": float("nan"), "elbow": 0.1}, "must be finite"),
        ({"shoulder": True, "elbow": 0.1}, "must be a finite number"),
    ],
)
def test_named_mapping_values_must_be_finite_numbers(values: dict[str, object], match: str) -> None:
    """A non-numeric, non-finite or bool value is refused before staging."""
    handle = _handle()

    with pytest.raises(BindingError, match=match):
        _sim(handle, FakeRunner(joint_names=("shoulder", "elbow"))).act(handle, values)


def test_named_mapping_is_single_env_only() -> None:
    """The scalar-per-joint mapping documents and enforces its N == 1 scope."""
    handle = _handle()
    sim = _sim(handle, FakeRunner(num_envs=2, joint_names=("shoulder", "elbow")))

    with pytest.raises(BindingError, match="single-env"):
        sim.act(handle, {"shoulder": 0.1, "elbow": 0.2})


def test_foreign_handle_is_refused_before_staging() -> None:
    """A handle minted under another owner cannot use the named form here."""
    handle = _handle()
    foreign = _handle(owner=object())
    sim = _sim(handle, FakeRunner(joint_names=("shoulder", "elbow")))

    with pytest.raises(ForeignHandleError, match="another builder/simulation"):
        sim.act(foreign, {"shoulder": 0.1, "elbow": 0.2})


def test_tensor_values_are_not_the_named_mapping_form() -> None:
    """A handle target with a tensor is refused instead of guessed."""
    handle = _handle()

    with pytest.raises(BindingError, match="requires a mapping"):
        _sim(handle, FakeRunner(joint_names=("shoulder", "elbow"))).act(handle, torch.zeros(1, 2))
