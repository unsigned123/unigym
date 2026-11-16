"""The Simulation multi-env reset, due/terminated and observation semantics."""

from __future__ import annotations

import pytest
import torch

from unigym.run.api import Run
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.runner.base import StepResult
from unigym.runner.state import read_extra
from unigym.sim import (
    ActionBinding,
    ArticulationHandle,
    BindingOrder,
    Control,
    ObservationError,
    PendingActionError,
    Simulation,
    SimulationConfigError,
    StaleStateError,
)
from unigym.world import World
from tests._support.fakes.runner import FakeRunner


def _handle(owner: object) -> ArticulationHandle:
    """One real one-joint handle scoped to *owner*."""
    joints = ("shoulder",)
    profile = RobotProfile(
        name="arm",
        joint_sets={"body": joints},
        default_joint_pos={"shoulder": 0.0},
        assets=None,
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
    )
    control = Control.position_pd(joints=joints, face="arm", kp=1.0, kd=0.1)
    return ArticulationHandle(
        name="arm",
        ref=None,
        profile=profile,
        control=control,
        faces={"arm": joints},
        _owner=owner,
    )


def _sim(*, runner: FakeRunner | None = None, num_envs: int = 2) -> Simulation:
    """Construct the facade directly over the typed runner double."""
    owner = object()
    handle = _handle(owner)
    live = FakeRunner(num_envs=num_envs) if runner is None else runner
    run = Run({"run_name": "facade-multienv", "env": {"backend": "mujoco", "num_envs": num_envs}})
    return Simulation(
        backend="mujoco",
        num_envs=num_envs,
        world=World.base(mjcf="world.xml"),
        articulations=(handle,),
        cameras=(),
        runner=live,
        env=live.env,
        run=run,
        owner=owner,
        state=read_extra(live.env),
        obs_cache=dict(live.env.obs_cache),
        t=0,
    )


def _binding(sim: Simulation) -> ActionBinding:
    """The one declared action binding of the fixture handle."""
    return sim.articulations[0].bind_action("arm", kind="position", order=BindingOrder.DEFAULT)


def _values() -> torch.Tensor:
    """Distinct per-row position targets for the two-env fixture."""
    return torch.tensor([[0.25], [0.75]], dtype=torch.float32)


class _BadResetRunner(FakeRunner):
    """A runner double whose reset result reports an invalid id vector."""

    def __init__(self, bad_ids: torch.Tensor, num_envs: int) -> None:
        super().__init__(num_envs=num_envs)
        self.bad_ids = bad_ids

    def reset(self, env_ids=None) -> StepResult:
        """Return the shared reset result with the configured bad ids."""
        result = super().reset(env_ids)
        return StepResult(
            t=result.t,
            obs=result.obs,
            extra=result.extra,
            reset_env_ids=self.bad_ids,
            terminated=result.terminated,
        )


class _BadTerminatedRunner(FakeRunner):
    """A runner double whose step result reports an invalid termination mask."""

    def __init__(self, bad_terminated: torch.Tensor | None, num_envs: int) -> None:
        super().__init__(num_envs=num_envs)
        self.bad_terminated = bad_terminated

    def tick(self, actions_override=None) -> StepResult:
        """Return the shared tick result with the configured bad mask."""
        result = super().tick(actions_override)
        return StepResult(
            t=result.t,
            obs=result.obs,
            extra=result.extra,
            reset_env_ids=result.reset_env_ids,
            terminated=self.bad_terminated,
        )


def test_spawn_reports_empty_masks_and_no_termination() -> None:
    """Spawn masks are explicit empties; the state mapping is never read as one."""
    sim = _sim()

    obs = sim.observe()

    assert obs.num_envs == 2
    assert obs.reset_env_ids.dtype == torch.int64
    assert obs.reset_env_ids.tolist() == []
    assert obs.reset_due_env_ids.dtype == torch.int64
    assert obs.reset_due_env_ids.tolist() == []
    assert obs.terminated is None


def test_step_reports_due_and_terminated_without_resetting_rows() -> None:
    """A step performs no reset: due/terminated are reported, the rows survive."""
    runner = FakeRunner(num_envs=2, due_rows=(0, 1), terminated_rows=(1,))
    sim = _sim(runner=runner)
    binding = _binding(sim)
    sim.act(binding, _values())

    obs = sim.step()

    assert obs.reset_env_ids.tolist() == []
    assert obs.reset_due_env_ids.tolist() == [0, 1]
    assert obs.terminated is not None
    assert obs.terminated.tolist() == [False, True]
    assert obs.articulations[sim.articulations[0]].joint_pos.tolist() == [[0.25], [0.75]]
    time_in_episode = obs.raw["time_in_episode"]
    assert isinstance(time_in_episode, torch.Tensor)
    assert time_in_episode.tolist() == [1.0, 1.0]


def test_reset_after_a_due_tick_clears_due_and_episode_clock() -> None:
    """The caller's reset clears the due mask and the per-row episode clock."""
    runner = FakeRunner(num_envs=2, due_rows=(0, 1))
    sim = _sim(runner=runner)
    sim.act(_binding(sim), _values())
    sim.step()

    obs = sim.reset(env_ids=[0, 1])

    assert obs.reset_env_ids.tolist() == [0, 1]
    assert obs.reset_due_env_ids.tolist() == []
    assert obs.terminated is None
    time_in_episode = obs.raw["time_in_episode"]
    assert isinstance(time_in_episode, torch.Tensor)
    assert time_in_episode.tolist() == [0.0, 0.0]


def test_reset_subset_restores_only_the_named_row() -> None:
    """A subset reset re-reads the full state and leaves the other rows untouched."""
    sim = _sim()
    sim.act(_binding(sim), _values())
    sim.step()

    obs = sim.reset(env_ids=[1])

    handle = sim.articulations[0]
    assert obs.reset_env_ids.tolist() == [1]
    assert obs.reset_due_env_ids.tolist() == []
    assert obs.terminated is None
    assert obs.articulations[handle].joint_pos.tolist() == [[0.25], [0.0]]
    time_in_episode = obs.raw["time_in_episode"]
    assert isinstance(time_in_episode, torch.Tensor)
    assert time_in_episode.tolist() == [1.0, 0.0]
    episode_id = obs.raw["episode_id"]
    assert isinstance(episode_id, torch.Tensor)
    assert episode_id.tolist() == [0, 1]


def test_reset_all_reports_every_row() -> None:
    """reset() without ids keeps the v1 all-rows behavior."""
    sim = _sim()

    obs = sim.reset()

    assert obs.reset_env_ids.tolist() == [0, 1]
    assert obs.reset_due_env_ids.tolist() == []
    assert obs.terminated is None


def test_reset_never_advances_the_tick() -> None:
    """Reset re-synchronises t with ticks_done and never ticks the runner."""
    sim = _sim()
    sim.act(_binding(sim), _values())
    sim.step()
    ticks = sim.runner.ticks_done

    obs = sim.reset(env_ids=[0])

    assert sim.runner.ticks_done == ticks
    assert obs.t == ticks
    assert sim.t == ticks


def test_reset_clears_the_whole_pending_table() -> None:
    """A staged action may belong to a reset row, so reset clears every face."""
    sim = _sim()
    sim.act(_binding(sim), _values())

    sim.reset(env_ids=[1])

    with pytest.raises(PendingActionError, match="needs a staged action"):
        sim.step()


@pytest.mark.parametrize(
    ("env_ids", "match"),
    [
        ([], "at least one row"),
        ([2], "outside the legal range"),
        ([-1], "outside the legal range"),
        ([0, 0], "duplicates"),
        ([True], "must be ints"),
        ([0.0], "must be ints"),
        ([0, "1"], "must be ints"),
        (torch.tensor([0.0]), "integer id tensor"),
        (torch.tensor([True, False]), "integer id tensor"),
        (torch.tensor([[0, 1]]), "1-D id tensor"),
        ("01", "sequence of ints"),
        (1, "sequence of ints"),
    ],
)
def test_reset_ids_fail_fast_naming_the_field_and_range(env_ids: object, match: str) -> None:
    """Bad caller ids raise SimulationConfigError and touch nothing."""
    runner = FakeRunner(num_envs=2)
    sim = _sim(runner=runner)

    with pytest.raises(SimulationConfigError, match=match) as excinfo:
        sim.reset(env_ids=env_ids)

    message = str(excinfo.value)
    assert "env_ids" in message
    assert "[0, 2)" in message
    assert runner.ticks_done == 0
    assert sim.faulted is False


@pytest.mark.parametrize(
    ("bad_ids", "match"),
    [
        (torch.tensor([0, 0], dtype=torch.int64), "duplicate"),
        (torch.tensor([2], dtype=torch.int64), "outside the legal range"),
        (torch.tensor([0.0]), "must be int64"),
        (torch.tensor([0], dtype=torch.int32), "must be int64"),
        (torch.tensor([[0]], dtype=torch.int64), "must be 1-D"),
    ],
)
def test_reset_rejects_a_bad_runner_result(bad_ids: torch.Tensor, match: str) -> None:
    """A runner result outside the id contract is a faulted ObservationError."""
    runner = _BadResetRunner(bad_ids, num_envs=2)
    sim = _sim(runner=runner)

    with pytest.raises(ObservationError, match=match):
        sim.reset()

    assert sim.faulted is True


def test_reset_rejects_rows_outside_the_request() -> None:
    """The runner may reset fewer rows, never a row the caller did not request."""
    runner = _BadResetRunner(torch.tensor([0], dtype=torch.int64), num_envs=2)
    sim = _sim(runner=runner)

    with pytest.raises(ObservationError, match="not a subset"):
        sim.reset(env_ids=[1])


@pytest.mark.parametrize(
    ("bad_terminated", "match"),
    [
        (torch.tensor([0.0, 0.0]), "must be bool"),
        (torch.zeros(1, dtype=torch.bool), r"shape \(2,\)"),
        (torch.zeros(2, 2, dtype=torch.bool), r"shape \(2,\)"),
        (torch.zeros(2, dtype=torch.int64), "must be bool"),
    ],
)
def test_step_rejects_a_bad_terminated_mask(bad_terminated: torch.Tensor, match: str) -> None:
    """A malformed termination mask is not adopted; the facade stays unfaulted."""
    runner = _BadTerminatedRunner(bad_terminated, num_envs=2)
    sim = _sim(runner=runner)
    sim.act(_binding(sim), _values())

    with pytest.raises(ObservationError, match=match):
        sim.step()

    assert sim.faulted is False
    assert sim.t == 0
    assert sim.runner.ticks_done == 1
    with pytest.raises(StaleStateError):
        sim.observe()
