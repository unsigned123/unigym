"""The Simulation pending/clock/fault state machine against a typed runner double."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from unigym.env.obs import ObsSnapshot, ObservationSchema
from unigym.run.api import Run
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.runner.base import Runner, StepResult
from unigym.runtime.actions import NamedAction, OrderedActionGroup, OrderedActionSpec
from unigym.sim import (
    ArticulationHandle,
    Control,
    ForeignHandleError,
    MissingActionError,
    PendingActionError,
    Simulation,
    SimulationClosedError,
    SimulationFaultedError,
    StaleStateError,
)
from unigym.sim.errors import BindingError
from unigym.sim.simulation import _StepRunner
from unigym.world import World
from tests._support.fakes.runner import FakeRunner


def _handle(owner: object, name: str = "arm", group: str = "arm") -> ArticulationHandle:
    """One real handle with one single-joint action group (owner-scoped)."""
    joints = ("shoulder",)
    profile = RobotProfile(
        name=name,
        joint_sets={"body": joints},
        default_joint_pos={"shoulder": 0.0},
        assets=None,
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
    )
    control = Control.position_pd(joints=joints, group=group, kp=1.0, kd=0.1)
    template = OrderedActionGroup(name=group, articulation=name, joints=joints, target="position")
    return ArticulationHandle(
        name=name,
        profile=profile,
        control=control,
        action_groups={group: template},
        _owner=owner,
    )


def _singleton_spec(handle: ArticulationHandle) -> OrderedActionSpec:
    """The global template of one handle."""
    return OrderedActionSpec(tuple(handle.action_groups.values()))


def _sim(
    handles: tuple[ArticulationHandle, ...] | None = None,
    *,
    runner: FakeRunner | None = None,
    owner: object | None = None,
    num_envs: int = 1,
) -> Simulation:
    """Construct the facade directly over the typed runner double."""
    token = object() if owner is None else owner
    declared = (_handle(token),) if handles is None else handles
    live = FakeRunner(num_envs=num_envs) if runner is None else runner
    spec = OrderedActionSpec(
        tuple(group for handle in declared for group in handle.action_groups.values())
    )
    run = Run({"run_name": "facade-unit", "env": {"backend": "mujoco", "num_envs": num_envs}})
    return Simulation(
        backend="mujoco",
        num_envs=num_envs,
        world=World.base(mjcf="world.xml"),
        articulations=declared,
        cameras=(),
        action_spec=spec,
        runner=live,
        env=live.env,
        run=run,
        owner=token,
        state=live.env.record(),
        obs=live.env.obs.snapshot(),
        t=0,
    )


def _group(handle: ArticulationHandle, group: str = "arm") -> OrderedActionGroup:
    """The handle's declared action group."""
    return handle.action_groups[group]


def _values(value: float = 0.5) -> torch.Tensor:
    """A single-env, one-column floating action tensor."""
    return torch.tensor([[value]], dtype=torch.float32)


def test_runner_property_is_the_neutral_runner_seam() -> None:
    """The fake is served at the neutral seam, not cast to ExternalRunner."""
    runner = FakeRunner()
    sim = _sim(runner=runner)

    assert sim.runner is runner
    assert isinstance(sim.runner, Runner)
    assert isinstance(sim.runner, _StepRunner)


def test_step_without_pending_actions_is_refused() -> None:
    """Nothing is reused or defaulted; an empty pending table raises."""
    sim = _sim()

    with pytest.raises(PendingActionError, match="needs a staged action"):
        sim.step()

    assert sim.runner.ticks_done == 0


def test_act_stages_without_ticking_and_observe_serves_the_cache() -> None:
    """act changes no clock/state; observe builds the cached spawn observation."""
    runner = FakeRunner()
    sim = _sim(runner=runner)
    handle = sim.articulations[0]
    group = _group(handle)

    sim.act(group, _values())
    obs = sim.observe()

    assert runner.ticks_done == 0
    assert obs.t == 0
    assert "reset_env_ids" not in obs.record
    assert "should_reset" not in obs.record
    assert obs.obs.robot("arm").joint_pos.tolist() == [[0.0]]
    record = obs.articulations[handle]
    assert record.joint_names == ("shoulder",)
    assert record.joint_pos.tolist() == [[0.0]]
    assert dict(obs.cameras) == {}
    assert dict(obs.contacts) == {}


def test_duplicate_group_act_keeps_the_first_staged_value() -> None:
    """A second act for one identity raises and does not overwrite the target."""
    sim = _sim()
    group = _group(sim.articulations[0])
    sim.act(group, _values(0.25))

    with pytest.raises(PendingActionError, match="already staged"):
        sim.act(group, _values(0.75))

    obs = sim.step()
    assert obs.obs.robot("arm").joint_pos.tolist() == [[0.25]]


def test_step_requires_every_declared_group() -> None:
    """A missing required group raises before the engine tick."""
    token = object()
    handles = (_handle(token, name="x", group="x"), _handle(token, name="y", group="y"))
    runner = FakeRunner()
    sim = _sim(handles, runner=runner, owner=token)
    sim.act(_group(handles[0], "x"), _values())

    with pytest.raises(MissingActionError, match="missing staged action") as excinfo:
        sim.step()

    message = str(excinfo.value)
    assert "('y', 'y')" in message
    assert "('x', 'x')" in message
    assert runner.ticks_done == 0


def test_same_group_name_on_two_articulations_is_distinct() -> None:
    """Both ``body`` groups stage and lower by their (articulation, group) identity."""
    token = object()
    left = _handle(token, name="left", group="body")
    right = _handle(token, name="right", group="body")
    runner = _EchoRunner((left, right))
    sim = Simulation(
        backend="mujoco",
        num_envs=1,
        world=World.base(mjcf="world.xml"),
        articulations=(left, right),
        cameras=(),
        action_spec=OrderedActionSpec((left.action_groups["body"], right.action_groups["body"])),
        runner=runner,
        env=runner.env,
        run=_run(),
        owner=token,
        state=_two_state((left, right)),
        obs=ObsSnapshot({}, schema=ObservationSchema({}), num_envs=1, device="cpu"),
        t=0,
    )

    sim.act(left.action_groups["body"], _values(0.25))
    sim.act(right.action_groups["body"], _values(0.75))
    obs = sim.step()

    lowered = runner.last_actions
    assert lowered is not None
    assert [entry.articulation for entry in lowered] == ["left", "right"]
    assert [entry.values.tolist() for entry in lowered] == [[[0.25]], [[0.75]]]
    assert obs.articulations[left].joint_pos.tolist() == [[0.25]]
    assert obs.articulations[right].joint_pos.tolist() == [[0.75]]


def test_step_applies_and_consumes_the_staged_action() -> None:
    """A successful tick caches the result and leaves an empty pending table."""
    sim = _sim()
    group = _group(sim.articulations[0])
    sim.act(group, _values(0.5))

    obs = sim.step()

    assert obs.t == 0
    assert obs.obs.robot("arm").joint_pos.tolist() == [[0.5]]
    assert sim.t == 0
    assert sim.runner.ticks_done == 1
    with pytest.raises(PendingActionError, match="needs a staged action"):
        sim.step()


def test_step_actions_sugar_stages_then_steps() -> None:
    """The one-shot mapping form is the canonical stage+step path."""
    sim = _sim()
    group = _group(sim.articulations[0])

    obs = sim.step({group: _values(0.5)})

    assert obs.obs.robot("arm").joint_pos.tolist() == [[0.5]]
    assert sim.runner.ticks_done == 1


def test_step_actions_sugar_refuses_an_existing_pending_table() -> None:
    """The sugar never merges with or overwrites staged actions."""
    sim = _sim()
    group = _group(sim.articulations[0])
    sim.act(group, _values(0.1))

    with pytest.raises(PendingActionError, match="empty pending table"):
        sim.step({group: _values(0.2)})

    assert sim.runner.ticks_done == 0


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("kind", "does not match the declared template"),
        ("shape", "expected \\(num_envs, dim\\)=\\(1, 1\\)"),
        ("foreign", "another builder/simulation"),
    ],
)
def test_step_actions_sugar_is_atomic_when_a_later_entry_fails(case: str, match: str) -> None:
    """A failed entry clears every staged entry and leaves the tick untouched."""
    token = object()
    handles = (_handle(token, name="x", group="x"), _handle(token, name="y", group="y"))
    sim = _sim(handles, owner=token)
    first = _group(handles[0], "x")
    if case == "kind":
        second = OrderedActionGroup(
            name="y", articulation="y", joints=("shoulder",), target="torque"
        )
        values = _values()
    elif case == "shape":
        second = _group(handles[1], "y")
        values = torch.zeros(2, 1)
    else:
        second = _handle(object(), name="z", group="z")
        values = _values()

    with pytest.raises((BindingError, ForeignHandleError), match=match):
        sim.step({first: _values(), second: values})

    assert sim.runner.ticks_done == 0
    with pytest.raises(PendingActionError, match="needs a staged action"):
        sim.step()


def test_engine_exception_faults_and_clears_pending() -> None:
    """A runner exception is fatal for the live sim until close()."""
    runner = FakeRunner()
    sim = _sim(runner=runner)
    group = _group(sim.articulations[0])
    sim.act(group, _values())
    runner.fail_next_tick = True

    with pytest.raises(RuntimeError, match="engine failure"):
        sim.step()

    assert sim.faulted is True
    for call in (sim.observe, sim.reset, lambda: sim.act(group, _values())):
        with pytest.raises(SimulationFaultedError):
            call()

    sim.close()
    assert sim.closed is True


def test_out_of_band_tick_raises_stale_and_clears_pending() -> None:
    """A runner tick outside the facade invalidates the staged target."""
    runner = FakeRunner()
    sim = _sim(runner=runner)
    handle = sim.articulations[0]
    group = _group(handle)
    sim.act(group, _values(0.3))
    runner.step((NamedAction("arm", _values(0.9), ("shoulder",), "position"),))

    with pytest.raises(StaleStateError) as excinfo:
        sim.step()

    message = str(excinfo.value)
    assert "facade clock 0" in message
    assert "ticks_done 1" in message
    with pytest.raises(StaleStateError):
        sim.observe()


def test_reset_clears_pending_and_refreshes_the_state() -> None:
    """Reset never consumes an action; it restores the spawn state and clock."""
    sim = _sim()
    group = _group(sim.articulations[0])
    sim.act(group, _values(0.7))
    sim.step()
    sim.act(group, _values(0.7))

    obs = sim.reset()

    assert obs.t == sim.runner.ticks_done == 1
    assert obs.obs.robot("arm").joint_pos.tolist() == [[0.0]]
    with pytest.raises(PendingActionError, match="needs a staged action"):
        sim.step()


def test_reset_uses_runner_ticks_done_not_the_result_tick() -> None:
    """The facade clock follows ticks_done even when reset reports another tick."""
    runner = FakeRunner()
    sim = _sim(runner=runner)
    runner.reset_t = 7

    obs = sim.reset()

    assert runner.ticks_done == 0
    assert obs.t == 0
    assert sim.t == 0


def test_reset_reports_the_rows_the_runner_reset() -> None:
    """reset_env_ids is the reset StepResult's rows, not the extra empty default."""
    sim = _sim()

    obs = sim.reset()

    assert obs.reset_env_ids.tolist() == [0]
    assert obs.terminated is None


def test_close_is_idempotent_and_refuses_live_calls() -> None:
    """Close is safe twice; every later live call names the closed state."""
    sim = _sim()
    group = _group(sim.articulations[0])

    sim.close()
    sim.close()

    assert sim.closed is True
    for call in (
        lambda: sim.act(group, _values()),
        sim.step,
        sim.observe,
        sim.reset,
    ):
        with pytest.raises(SimulationClosedError):
            call()


def test_context_manager_closes_on_exit() -> None:
    """Leaving the with block runs the same idempotent close path."""
    sim = _sim()

    with sim as entered:
        assert entered is sim
        assert sim.closed is False

    assert sim.closed is True


def test_foreign_group_is_refused_before_staging() -> None:
    """A group minted for another owner cannot stage here."""
    sim = _sim()
    other = _handle(object())

    with pytest.raises(ForeignHandleError, match="another builder/simulation"):
        sim.act(other, {"shoulder": 0.1})


@pytest.mark.parametrize(
    ("kind", "values", "match"),
    [
        ("torque", torch.tensor([[0.0]]), "does not match the declared template"),
        ("position", torch.zeros(2, 1), "expected \\(num_envs, dim\\)=\\(1, 1\\)"),
        ("position", torch.zeros(1, 1, dtype=torch.int64), "must be floating"),
    ],
)
def test_act_rejects_bad_kind_shape_or_dtype(kind, values, match) -> None:
    """Group identity, shape and dtype are validated before anything is staged."""
    sim = _sim()
    handle = sim.articulations[0]
    group = OrderedActionGroup(name="arm", articulation="arm", joints=("shoulder",), target=kind)

    with pytest.raises(BindingError, match=match):
        sim.act(group if kind != "position" else handle, values)

    assert sim.runner.ticks_done == 0


def test_tensor_action_applies_every_row_at_n2() -> None:
    """A (2, 1) action lands in both rows and still stages once per group."""
    runner = FakeRunner(num_envs=2)
    sim = _sim(num_envs=2, runner=runner)
    group = _group(sim.articulations[0])

    sim.act(group, torch.tensor([[0.25], [0.75]], dtype=torch.float32))
    with pytest.raises(PendingActionError, match="already staged"):
        sim.act(group, torch.tensor([[9.0], [9.0]], dtype=torch.float32))
    obs = sim.step()

    assert obs.obs.robot("arm").joint_pos.tolist() == [[0.25], [0.75]]


def test_tensor_action_refuses_a_single_row_at_n2() -> None:
    """A (1, 1) action is refused against the exact (num_envs, dim)."""
    runner = FakeRunner(num_envs=2)
    sim = _sim(num_envs=2, runner=runner)
    group = _group(sim.articulations[0])

    with pytest.raises(BindingError, match=r"expected \(num_envs, dim\)=\(2, 1\)") as excinfo:
        sim.act(group, torch.zeros(1, 1))

    assert "(1, 1)" in str(excinfo.value)
    assert runner.ticks_done == 0


def test_group_with_different_joints_is_refused_as_template_mismatch() -> None:
    """A same-identity group with other joints is not silently accepted."""
    sim = _sim()
    wrong = OrderedActionGroup(name="arm", articulation="arm", joints=("elbow",), target="position")

    with pytest.raises(BindingError, match="does not match the declared template"):
        sim.act(wrong, torch.zeros(1, 1))
    assert sim.runner.ticks_done == 0


def test_partial_group_is_refused_before_the_tick() -> None:
    """A group value tensor must cover exactly the group joint count."""
    sim = _sim()
    handle = sim.articulations[0]

    with pytest.raises(BindingError, match=r"expected \(num_envs, dim\)=\(1, 1\)"):
        sim.act(handle, torch.zeros(1, 2))
    assert sim.runner.ticks_done == 0


class _EchoEnv:
    """Minimal env face for the two-articulation same-name test."""

    def __init__(self) -> None:
        self.num_envs = 1
        self.device = "cpu"


class _EchoRunner:
    """Records the lowered actions and returns a complete two-articulation state."""

    def __init__(self, handles: Sequence[ArticulationHandle]) -> None:
        self.env = _EchoEnv()
        self.ticks_done = 0
        self.last_actions: tuple[NamedAction, ...] | None = None
        self._handles = tuple(handles)

    def step(self, actions: Sequence[NamedAction]) -> StepResult:
        self.last_actions = tuple(actions)
        values = {entry.articulation: entry.values.clone() for entry in self.last_actions}
        extra = _two_state(self._handles, values=values)
        result = StepResult(
            t=self.ticks_done,
            obs=ObsSnapshot({}, schema=ObservationSchema({}), num_envs=1, device="cpu"),
            record=extra,
            reset_env_ids=torch.empty(0, dtype=torch.int64),
            terminated=None,
        )
        self.ticks_done += 1
        return result


def _run() -> Run:
    """One Run shell for the direct Simulation seam."""
    return Run({"run_name": "facade-multi", "env": {"backend": "mujoco", "num_envs": 1}})


def _two_state(
    handles: Sequence[ArticulationHandle],
    *,
    values: dict[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    """A complete one-row state for two one-joint articulations."""
    state: dict[str, object] = {}
    for handle in handles:
        value = torch.zeros(1, 1)
        if values is not None:
            value = values[handle.name]
        prefix = f"articulation/{handle.name}"
        zeros3 = torch.zeros(1, 3)
        state[f"{prefix}/joint_pos"] = value
        state[f"{prefix}/joint_vel"] = torch.zeros_like(value)
        state[f"{prefix}/applied_torque"] = torch.zeros_like(value)
        state[f"{prefix}/gravity_b"] = torch.tensor([[0.0, 0.0, -1.0]])
        state[f"{prefix}/base_pos_w"] = zeros3.clone()
        state[f"{prefix}/base_quat_w"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        state[f"{prefix}/base_lin_vel_b"] = zeros3.clone()
        state[f"{prefix}/base_ang_vel_b"] = zeros3.clone()
    return state
