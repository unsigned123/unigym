"""The action surface: group templates, named values, coverage and shapes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
import torch

from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run.api import Run
from unigym.runtime.actions import OrderedActionGroup, OrderedActionSpec
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
    groups: Mapping[str, Sequence[str]] | None = None,
    name: str = "arm",
    owner: object | None = None,
) -> ArticulationHandle:
    """One real handle with explicit ordered action groups over its profile."""
    declared = {"arm": ("shoulder", "elbow")} if groups is None else dict(groups)
    joints: list[str] = []
    for group_joints in declared.values():
        for joint in group_joints:
            if joint not in joints:
                joints.append(joint)
    token = object() if owner is None else owner
    profile = RobotProfile(
        name=name,
        joint_sets={"body": tuple(joints)},
        default_joint_pos={joint: 0.0 for joint in joints},
        assets=None,
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
    )
    groups_spec = tuple(
        Control.position_pd(joints=tuple(group_joints), group=group_name, kp=1.0, kd=0.1).groups[
            group_name
        ]
        for group_name, group_joints in declared.items()
    )
    control = ControlSpec.from_groups(groups_spec)
    action_groups = {
        group_name: OrderedActionGroup(
            name=group_name,
            articulation=name,
            joints=tuple(group_joints),
            target="position",
        )
        for group_name, group_joints in declared.items()
    }
    return ArticulationHandle(
        name=name,
        profile=profile,
        control=control,
        action_groups=action_groups,
        _owner=token,
    )


def _spec(handle: ArticulationHandle) -> OrderedActionSpec:
    """The global action template of one handle."""
    return OrderedActionSpec(tuple(handle.action_groups.values()))


def _sim(handle: ArticulationHandle, runner: FakeRunner) -> Simulation:
    """Construct the facade directly over the typed runner double."""
    run = Run({"run_name": "facade-sugar", "env": {"backend": "mujoco", "num_envs": 1}})
    return Simulation(
        backend="mujoco",
        num_envs=runner.env.num_envs,
        world=World.base(mjcf="world.xml"),
        articulations=(handle,),
        cameras=(),
        action_spec=_spec(handle),
        runner=runner,
        env=runner.env,
        run=run,
        owner=handle._owner,
        state=runner.env.record(),
        obs=runner.env.obs.snapshot(),
        t=0,
    )


def test_named_mapping_values_are_reordered_into_group_joint_order() -> None:
    """The mapping key order never changes the group's declared column order."""
    handle = _handle()
    runner = FakeRunner(joint_names=("shoulder", "elbow"))
    sim = _sim(handle, runner)

    sim.act(handle, {"elbow": 0.7, "shoulder": 0.3})
    obs = sim.step()

    action = runner.last_actions
    assert action is not None
    assert len(action) == 1
    assert action[0].articulation == "arm"
    assert action[0].joint_names == ("shoulder", "elbow")
    assert action[0].values.shape == (1, 2)
    assert torch.allclose(action[0].values, torch.tensor([[0.3, 0.7]]))
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
def test_named_mapping_must_cover_every_group_joint(values: dict[str, float], match: str) -> None:
    """A partial or over-specified mapping names both sets and stages nothing."""
    handle = _handle()

    with pytest.raises(BindingError, match=match) as excinfo:
        _sim(handle, FakeRunner(joint_names=("shoulder", "elbow"))).act(handle, values)

    message = str(excinfo.value)
    assert "the group joints ['shoulder', 'elbow']" in message


def test_multiple_groups_require_the_group_target() -> None:
    """A multi-group handle lists its identities instead of guessing."""
    handle = _handle(groups={"body": ("shoulder",), "hand": ("elbow",)})
    sim = _sim(handle, FakeRunner(joint_names=("shoulder", "elbow")))

    with pytest.raises(BindingError, match="needs a single-group handle") as excinfo:
        sim.act(handle, {"shoulder": 0.1})

    message = str(excinfo.value)
    assert "('arm', 'body')" in message
    assert "('arm', 'hand')" in message


def test_explicit_group_stages_one_of_many() -> None:
    """Passing the group object stages that group even on a multi-group handle."""
    handle = _handle(groups={"body": ("shoulder",), "hand": ("elbow",)})
    sim = _sim(handle, FakeRunner(joint_names=("shoulder", "elbow")))
    body = handle.action_groups["body"]

    sim.act(body, {"shoulder": 0.1})

    with pytest.raises(PendingActionError, match="already staged"):
        sim.act(body, {"shoulder": 0.2})


def test_single_group_handle_accepts_a_tensor_in_group_order() -> None:
    """A single-group handle convenience accepts the explicit (N, dim) tensor."""
    handle = _handle()
    runner = FakeRunner(joint_names=("shoulder", "elbow"))
    sim = _sim(handle, runner)

    sim.act(handle, torch.tensor([[0.3, -0.4]], dtype=torch.float32))
    obs = sim.step()

    assert torch.allclose(obs.articulations[handle].joint_pos, torch.tensor([[0.3, -0.4]]))


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"shoulder": "high", "elbow": 0.1}, "must be a finite number"),
        ({"shoulder": float("nan"), "elbow": 0.1}, "must be finite"),
        ({"shoulder": True, "elbow": 0.1}, "must be a finite number"),
        ({"shoulder": torch.tensor([[0.1]]), "elbow": 0.1}, r"not a legal \(1,\)"),
        ({"shoulder": torch.tensor([True]), "elbow": 0.1}, "must be a floating tensor"),
    ],
)
def test_named_mapping_values_must_be_finite_numbers(values: dict[str, object], match: str) -> None:
    """A non-numeric, non-finite or bool value is refused before staging."""
    handle = _handle()

    with pytest.raises(BindingError, match=match):
        _sim(handle, FakeRunner(joint_names=("shoulder", "elbow"))).act(handle, values)


def test_named_mapping_accepts_numbers_and_vectors_at_one_env() -> None:
    """N == 1 mixes finite numbers with floating (1,) tensors."""
    handle = _handle()
    runner = FakeRunner(joint_names=("shoulder", "elbow"))
    sim = _sim(handle, runner)

    sim.act(handle, {"shoulder": 0.3, "elbow": torch.tensor([0.7])})
    obs = sim.step()

    assert torch.allclose(obs.articulations[handle].joint_pos, torch.tensor([[0.3, 0.7]]))


def test_named_mapping_requires_per_env_tensors() -> None:
    """At N > 1 every joint needs a floating (N,) tensor, never a number."""
    handle = _handle()
    runner = FakeRunner(num_envs=2, joint_names=("shoulder", "elbow"))
    sim = _sim(handle, runner)

    with pytest.raises(
        BindingError, match=r"joint 'shoulder'.*torch\.full\(\(2,\), 0\.1\)"
    ) as excinfo:
        sim.act(handle, {"shoulder": 0.1, "elbow": torch.zeros(2)})

    assert "num_envs=2" in str(excinfo.value)
    with pytest.raises(
        BindingError,
        match=r"joint 'shoulder'.*shape \(1,\).*expected \(num_envs,\)=\(2,\)",
    ):
        sim.act(handle, {"shoulder": torch.zeros(1), "elbow": torch.zeros(2)})
    with pytest.raises(
        BindingError,
        match=r"joint 'elbow'.*shape \(2, 1\).*expected \(num_envs,\)=\(2,\)",
    ):
        sim.act(handle, {"shoulder": torch.zeros(2), "elbow": torch.zeros(2, 1)})
    assert runner.ticks_done == 0


def test_named_mapping_requires_uniform_dtype_at_many_envs() -> None:
    """A mixed per-env dtype is refused naming the joint and both dtypes."""
    handle = _handle()
    sim = _sim(handle, FakeRunner(num_envs=2, joint_names=("shoulder", "elbow")))

    with pytest.raises(
        BindingError,
        match=r"joint 'elbow'.*dtype torch\.float64.*joint 'shoulder'.*dtype torch\.float32",
    ):
        sim.act(
            handle,
            {
                "shoulder": torch.zeros(2, dtype=torch.float32),
                "elbow": torch.zeros(2, dtype=torch.float64),
            },
        )


def test_named_mapping_applies_per_env_values() -> None:
    """Each column lands in the same row across the whole mapping."""
    handle = _handle()
    runner = FakeRunner(num_envs=2, joint_names=("shoulder", "elbow"))
    sim = _sim(handle, runner)

    sim.act(
        handle,
        {
            "elbow": torch.tensor([0.3, 0.4]),
            "shoulder": torch.tensor([0.1, 0.2]),
        },
    )
    obs = sim.step()

    action = runner.last_actions
    assert action is not None
    assert len(action) == 1
    assert action[0].joint_names == ("shoulder", "elbow")
    assert action[0].values.shape == (2, 2)
    assert torch.allclose(
        action[0].values,
        torch.tensor([[0.1, 0.3], [0.2, 0.4]]),
    )
    assert torch.allclose(
        obs.articulations[handle].joint_pos,
        torch.tensor([[0.1, 0.3], [0.2, 0.4]]),
    )


def test_foreign_handle_is_refused_before_staging() -> None:
    """A handle minted under another owner cannot use the named form here."""
    handle = _handle()
    foreign = _handle(owner=object())
    sim = _sim(handle, FakeRunner(joint_names=("shoulder", "elbow")))

    with pytest.raises(ForeignHandleError, match="another builder/simulation"):
        sim.act(foreign, {"shoulder": 0.1, "elbow": 0.2})


def test_unknown_group_is_refused_instead_of_staging() -> None:
    """A group identity the action_spec does not declare is not accepted."""
    handle = _handle()
    sim = _sim(handle, FakeRunner(joint_names=("shoulder", "elbow")))
    unknown = OrderedActionGroup(
        name="legs",
        articulation="arm",
        joints=("shoulder", "elbow"),
        target="position",
    )

    with pytest.raises(BindingError, match="is not declared by this simulation"):
        sim.act(unknown, torch.zeros(1, 2))
