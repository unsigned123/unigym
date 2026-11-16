"""The Simulation pending/clock/fault state machine against a typed runner double."""

from __future__ import annotations

import pytest
import torch

from unigym.run.api import Run
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.runner.base import Runner
from unigym.runner.state import read_extra
from unigym.runtime.actions import NamedAction
from unigym.sim import (
    ActionBinding,
    ArticulationHandle,
    BindingOrder,
    Control,
    ForeignHandleError,
    MissingActionError,
    ObservationError,
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


def _handle(owner: object, name: str = "arm", face: str = "arm") -> ArticulationHandle:
    """One real handle with one one-joint face (identity-scoped by *owner*)."""
    joints = ("shoulder",)
    profile = RobotProfile(
        name=name,
        joint_sets={"body": joints},
        default_joint_pos={"shoulder": 0.0},
        assets=None,
        action_faces={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
    )
    control = Control.position_pd(joints=joints, face=face, kp=1.0, kd=0.1)
    return ArticulationHandle(
        name=name,
        ref=None,
        profile=profile,
        control=control,
        faces={face: joints},
        _owner=owner,
    )


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
    run = Run({"run_name": "facade-unit", "env": {"backend": "mujoco", "num_envs": num_envs}})
    return Simulation(
        backend="mujoco",
        num_envs=num_envs,
        world=World.base(mjcf="world.xml"),
        articulations=declared,
        cameras=(),
        runner=live,
        env=live.env,
        run=run,
        owner=token,
        state=read_extra(live.env),
        obs_cache=dict(live.env.obs_cache),
        t=0,
    )


def _binding(
    handle: ArticulationHandle, face: str = "arm", kind: str = "position"
) -> ActionBinding:
    """An action binding for the handle's declared face."""
    return handle.bind_action(face, kind=kind, order=BindingOrder.DEFAULT)


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
    binding = _binding(handle)

    sim.act(binding, _values())
    obs = sim.observe()

    assert runner.ticks_done == 0
    assert obs.t == 0
    assert "reset_env_ids" not in obs.raw
    assert "should_reset" not in obs.raw
    assert obs.obs_cache["joint_pos"].tolist() == [[0.0]]
    record = obs.articulations[handle]
    assert record.joint_names == ("shoulder",)
    assert record.joint_pos.tolist() == [[0.0]]
    assert dict(obs.cameras) == {}
    assert dict(obs.contacts) == {}


def test_duplicate_act_keeps_the_first_staged_value() -> None:
    """A second act for one face raises and does not overwrite the target."""
    sim = _sim()
    binding = _binding(sim.articulations[0])
    sim.act(binding, _values(0.25))

    with pytest.raises(PendingActionError, match="already staged"):
        sim.act(binding, _values(0.75))

    obs = sim.step()
    assert obs.obs_cache["joint_pos"].tolist() == [[0.25]]


def test_step_requires_every_declared_face() -> None:
    """A missing required face raises before the engine tick."""
    token = object()
    handles = (_handle(token, name="x", face="x"), _handle(token, name="y", face="y"))
    runner = FakeRunner()
    sim = _sim(handles, runner=runner, owner=token)
    sim.act(_binding(handles[0], face="x"), _values())

    with pytest.raises(MissingActionError, match="missing staged action") as excinfo:
        sim.step()

    message = str(excinfo.value)
    assert "'y'" in message
    assert "'x'" in message
    assert runner.ticks_done == 0


def test_step_applies_and_consumes_the_staged_action() -> None:
    """A successful tick caches the result and leaves an empty pending table."""
    sim = _sim()
    binding = _binding(sim.articulations[0])
    sim.act(binding, _values(0.5))

    obs = sim.step()

    assert obs.t == 0
    assert obs.obs_cache["joint_pos"].tolist() == [[0.5]]
    assert sim.t == 0
    assert sim.runner.ticks_done == 1
    with pytest.raises(PendingActionError, match="needs a staged action"):
        sim.step()


def test_step_actions_sugar_stages_then_steps() -> None:
    """The one-shot mapping form is the canonical stage+step path."""
    sim = _sim()
    binding = _binding(sim.articulations[0])

    obs = sim.step({binding: _values(0.5)})

    assert obs.obs_cache["joint_pos"].tolist() == [[0.5]]
    assert sim.runner.ticks_done == 1


def test_step_actions_sugar_refuses_an_existing_pending_table() -> None:
    """The sugar never merges with or overwrites staged actions."""
    sim = _sim()
    binding = _binding(sim.articulations[0])
    sim.act(binding, _values(0.1))

    with pytest.raises(PendingActionError, match="empty pending table"):
        sim.step({binding: _values(0.2)})

    assert sim.runner.ticks_done == 0


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("kind", "control target 'position'"),
        ("shape", "expected \\(num_envs, dim\\)=\\(1, 1\\)"),
        ("foreign", "another builder/simulation"),
    ],
)
def test_step_actions_sugar_is_atomic_when_a_later_entry_fails(case: str, match: str) -> None:
    """A failed entry clears every staged entry and leaves the tick untouched."""
    token = object()
    handles = (_handle(token, name="x", face="x"), _handle(token, name="y", face="y"))
    sim = _sim(handles, owner=token)
    first = _binding(handles[0], face="x")
    if case == "kind":
        second = _binding(handles[1], face="y", kind="torque")
        values = _values()
    elif case == "shape":
        second = _binding(handles[1], face="y")
        values = torch.zeros(2, 1)
    else:
        second = _binding(_handle(object(), name="z", face="z"), face="z")
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
    binding = _binding(sim.articulations[0])
    sim.act(binding, _values())
    runner.fail_next_tick = True

    with pytest.raises(RuntimeError, match="engine failure"):
        sim.step()

    assert sim.faulted is True
    for call in (sim.observe, sim.reset, lambda: sim.act(binding, _values())):
        with pytest.raises(SimulationFaultedError):
            call()

    sim.close()
    assert sim.closed is True


def test_out_of_band_tick_raises_stale_and_clears_pending() -> None:
    """A runner tick outside the facade invalidates the staged target."""
    runner = FakeRunner()
    sim = _sim(runner=runner)
    handle = sim.articulations[0]
    binding = _binding(handle)
    sim.act(binding, _values(0.3))
    runner.step((NamedAction("arm", "arm", _values(0.9), ("shoulder",), "position"),))

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
    binding = _binding(sim.articulations[0])
    sim.act(binding, _values(0.7))
    sim.step()
    sim.act(binding, _values(0.7))

    obs = sim.reset()

    assert obs.t == sim.runner.ticks_done == 1
    assert obs.obs_cache["joint_pos"].tolist() == [[0.0]]
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


def test_step_keeps_a_non_tensor_cache_entry_exactly() -> None:
    """A record-like obs entry crosses the step cache boundary verbatim."""
    runner = FakeRunner()
    payload = {"probe": [1, 2]}
    runner.cache_extra = {"contact_ops/foo": payload}
    sim = _sim(runner=runner)
    sim.act(_binding(sim.articulations[0]), _values(0.5))

    obs = sim.step()

    assert obs.obs_cache["contact_ops/foo"] is payload
    assert dict(obs.obs_cache)["contact_ops/foo"] == {"probe": [1, 2]}


def test_bad_expected_cache_key_is_reported_after_the_tick_is_adopted() -> None:
    """A non-tensor projected key fails without faulting or rewinding the clock."""
    runner = FakeRunner()
    runner.cache_extra = {"articulation/arm/prev_action": "not-a-tensor"}
    sim = _sim(runner=runner)
    sim.act(_binding(sim.articulations[0]), _values(0.5))

    with pytest.raises(ObservationError, match="articulation/arm/prev_action"):
        sim.step()

    assert sim.faulted is False
    assert sim.t == 0
    assert sim.runner.ticks_done == 1
    with pytest.raises(ObservationError, match="articulation/arm/prev_action"):
        sim.observe()


def test_close_is_idempotent_and_refuses_live_calls() -> None:
    """Close is safe twice; every later live call names the closed state."""
    sim = _sim()
    binding = _binding(sim.articulations[0])

    sim.close()
    sim.close()

    assert sim.closed is True
    for call in (
        lambda: sim.act(binding, _values()),
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


def test_foreign_binding_is_refused_before_staging() -> None:
    """A binding minted under another owner cannot stage here."""
    sim = _sim()
    other = _handle(object())

    with pytest.raises(ForeignHandleError, match="another builder/simulation"):
        sim.act(_binding(other), _values())


@pytest.mark.parametrize(
    ("binding_kind", "values", "match"),
    [
        ("torque", torch.tensor([[0.0]]), "control target 'position'"),
        ("position", torch.zeros(2, 1), "expected \\(num_envs, dim\\)=\\(1, 1\\)"),
        ("position", torch.zeros(1, 1, dtype=torch.int64), "must be floating"),
    ],
)
def test_act_rejects_bad_kind_shape_or_dtype(binding_kind, values, match) -> None:
    """Kind, shape and dtype are validated before anything is staged."""
    sim = _sim()
    binding = _binding(sim.articulations[0], kind=binding_kind)

    with pytest.raises(BindingError, match=match):
        sim.act(binding, values)

    assert sim.runner.ticks_done == 0


def test_tensor_action_applies_every_row_at_n2() -> None:
    """A (2, 1) action lands in both rows and still stages once per face."""
    runner = FakeRunner(num_envs=2)
    sim = _sim(num_envs=2, runner=runner)
    binding = _binding(sim.articulations[0])

    sim.act(binding, torch.tensor([[0.25], [0.75]], dtype=torch.float32))
    with pytest.raises(PendingActionError, match="already staged"):
        sim.act(binding, torch.tensor([[9.0], [9.0]], dtype=torch.float32))
    obs = sim.step()

    assert obs.obs_cache["joint_pos"].tolist() == [[0.25], [0.75]]


def test_tensor_action_refuses_a_single_row_at_n2() -> None:
    """A (1, 1) action is refused against the exact (num_envs, dim)."""
    runner = FakeRunner(num_envs=2)
    sim = _sim(num_envs=2, runner=runner)
    binding = _binding(sim.articulations[0])

    with pytest.raises(BindingError, match=r"expected \(num_envs, dim\)=\(2, 1\)") as excinfo:
        sim.act(binding, torch.zeros(1, 1))

    assert "(1, 1)" in str(excinfo.value)
    assert runner.ticks_done == 0
