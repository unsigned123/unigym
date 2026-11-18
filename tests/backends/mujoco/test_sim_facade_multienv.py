"""The shared facade script and the per-env reset contract on real MuJoCo."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from unigym.packs import discover_set
from unigym.packs.packset import PackSet
from unigym.robots.base import RobotProfile
from unigym.robots import restore_robot_registry, snapshot_robot_registry
from unigym.runner.external import ExternalRunner
from unigym.sim import (
    ArticulationHandle,
    Control,
    Simulation,
    SimulationBuilder,
)
from unigym.world import World
from tests._support.sim_facade_script import ARTICULATION_LABEL, run_facade_acceptance

FACADE_PACK = Path(__file__).resolve().parents[2] / "_support" / "packs" / "facade_mjcf"
ROBOT_ID = "facade_mjcf.tiny_arm"
JOINT_ORDER = ("shoulder", "elbow")
PROJECTED_KEYS = (
    "joint_pos",
    "joint_vel",
    "applied_torque",
    "gravity_b",
    "base_pos_w",
    "base_quat_w",
    "base_lin_vel_b",
    "base_ang_vel_b",
)


@pytest.fixture(autouse=True)
def _isolate_robot_registry() -> Iterator[None]:
    """Restore the process-wide robot registry after the facade build."""
    snapshot = snapshot_robot_registry()
    yield
    restore_robot_registry(snapshot)


def _facade_pack() -> tuple[PackSet, RobotProfile]:
    """The lane-owned acceptance pack and its one tiny-arm robot ref."""
    pack_set = discover_set([FACADE_PACK])
    return pack_set, pack_set.pack("facade_mjcf").robot(ROBOT_ID)


def _build_sim(
    tiny_mjcf: Path,
    *,
    num_envs: int,
    episode_length_s: float = 20.0,
) -> tuple[Simulation, ArticulationHandle]:
    """One real multi-env facade build over the shared tiny MJCF asset."""
    pack_set, ref = _facade_pack()
    builder = SimulationBuilder(
        backend="mujoco",
        dt=0.02,
        physics_dt=0.005,
        num_envs=num_envs,
        episode_length_s=episode_length_s,
        pack_set=pack_set,
    )
    builder.world = World.base(mjcf=str(tiny_mjcf))
    handle = builder.add_articulation(
        ref,
        name=ARTICULATION_LABEL,
        control=Control.position_pd(joint_set="body", kp=20.0, kd=1.0),
    )
    return builder.build(), handle


def _targets(num_envs: int) -> torch.Tensor:
    """Distinct per-row position targets for the tiny two-joint arm."""
    pattern = torch.linspace(-0.2, 0.2, len(JOINT_ORDER), dtype=torch.float32)
    offsets = 0.25 * torch.arange(num_envs, dtype=torch.float32).reshape(num_envs, 1)
    return pattern.reshape(1, len(JOINT_ORDER)) + offsets


def _joint_pos(obs, prefix: str) -> torch.Tensor:
    """The raw production-position tensor of one observation."""
    value = obs.record[f"{prefix}/joint_pos"]
    assert isinstance(value, torch.Tensor)
    return value


@pytest.mark.parametrize("num_envs", [2, 3])
def test_shared_script_builds_distinct_rows(num_envs: int, tiny_mjcf: Path) -> None:
    """The same acceptance body drives N explicit, distinct rows."""
    pack_set, ref = _facade_pack()

    evidence = run_facade_acceptance(
        pack_set=pack_set,
        robot_id=ref.name,
        world=World.base(mjcf=str(tiny_mjcf)),
        backend="mujoco",
        expected_joint_names=JOINT_ORDER,
        steps=3,
        num_envs=num_envs,
    )

    assert evidence["backend"] == "mujoco"
    assert evidence["num_envs"] == num_envs
    assert evidence["joint_names"] == list(JOINT_ORDER)
    assert evidence["t"] == evidence["steps"] - 1
    rows = evidence["final_joint_pos"]
    assert isinstance(rows, list)
    assert len(rows) == num_envs
    assert all(isinstance(row, list) and len(row) == len(JOINT_ORDER) for row in rows)
    assert (
        len({tuple(row) for row in rows}) == num_envs
    ), f"the {num_envs} env rows did not diverge: {rows}"


def test_reset_subset_restores_only_the_named_row(tiny_mjcf: Path) -> None:
    """reset(env_ids=[1]) rolls row 1 back to spawn and leaves row 0 untouched."""
    sim, handle = _build_sim(tiny_mjcf, num_envs=2)
    prefix = f"articulation/{handle.name}"
    try:
        spawn = _joint_pos(sim.observe(), prefix).clone()
        sim.act(
            handle.action_groups["body"],
            _targets(2),
        )
        stepped = sim.step()
        post_step = _joint_pos(stepped, prefix).clone()
        assert not torch.equal(
            post_step[0], post_step[1]
        ), f"the per-row action targets did not separate the rows: {post_step.tolist()}"
        assert not torch.equal(
            post_step[1], spawn[1]
        ), f"row 1 did not move from spawn on the stepped tick: {post_step[1].tolist()}"

        obs = sim.reset(env_ids=[1])

        assert obs.num_envs == 2
        assert obs.reset_env_ids.tolist() == [1]
        assert obs.reset_due_env_ids.tolist() == []
        assert obs.terminated is None
        reset_pos = _joint_pos(obs, prefix)
        assert torch.equal(reset_pos[0], post_step[0]), "row 0 was touched by a row-1 reset"
        assert torch.equal(reset_pos[1], spawn[1]), "row 1 was not restored to its spawn state"
        time_in_episode = obs.record["time_in_episode"]
        assert isinstance(time_in_episode, torch.Tensor)
        assert torch.allclose(time_in_episode, torch.tensor([0.02, 0.0]))
    finally:
        sim.close()


def test_due_tick_reports_without_resetting_then_explicit_reset_clears(
    tiny_mjcf: Path,
) -> None:
    """episode_length_s=0.04 makes both rows due on tick 2, reported not applied."""
    sim, handle = _build_sim(tiny_mjcf, num_envs=2, episode_length_s=0.04)
    binding = handle.action_groups["body"]
    try:
        sim.act(binding, _targets(2))
        first = sim.step()
        assert first.reset_due_env_ids.tolist() == []
        assert first.reset_env_ids.tolist() == []
        time_in_episode = first.record["time_in_episode"]
        assert isinstance(time_in_episode, torch.Tensor)
        assert torch.allclose(time_in_episode, torch.full((2,), 0.02))

        sim.act(binding, _targets(2))
        second = sim.step()

        assert second.reset_due_env_ids.tolist() == [0, 1]
        assert second.reset_env_ids.tolist() == []
        due_clock = second.record["time_in_episode"]
        assert isinstance(due_clock, torch.Tensor)
        assert torch.allclose(
            due_clock, torch.full((2,), 0.04)
        ), "a due row must keep stepping until the caller resets it"

        cleared = sim.reset(env_ids=[0, 1])

        assert cleared.reset_env_ids.tolist() == [0, 1]
        assert cleared.reset_due_env_ids.tolist() == []
        assert cleared.terminated is None
        cleared_clock = cleared.record["time_in_episode"]
        assert isinstance(cleared_clock, torch.Tensor)
        assert cleared_clock.tolist() == [0.0, 0.0]
    finally:
        sim.close()


def test_reader_after_subset_reset_agrees_row_wise(tiny_mjcf: Path) -> None:
    """A post-subset-reset read reports one honest, row-ordered state record."""
    sim, handle = _build_sim(tiny_mjcf, num_envs=2)
    prefix = f"articulation/{handle.name}"
    runner = sim.runner
    assert isinstance(runner, ExternalRunner)
    try:
        spawn = _joint_pos(sim.observe(), prefix).clone()
        sim.act(
            handle.action_groups["body"],
            _targets(2),
        )
        stepped = sim.step()
        post_step = _joint_pos(stepped, prefix).clone()
        step_state = runner.env.record()
        for key in PROJECTED_KEYS:
            read = step_state[f"{prefix}/{key}"]
            assert isinstance(read, torch.Tensor)
            assert torch.equal(
                stepped.record[f"{prefix}/{key}"], read
            ), f"the state reader disagrees with the step extra on {key}"

        obs = sim.reset(env_ids=[1])
        state = runner.env.record()

        for key in PROJECTED_KEYS:
            raw = obs.record[f"{prefix}/{key}"]
            read = state[f"{prefix}/{key}"]
            assert isinstance(raw, torch.Tensor)
            assert isinstance(read, torch.Tensor)
            assert raw.shape == (2, raw.shape[1])
            assert torch.equal(raw, read), f"reset observation and reader disagree on {key}"
        reset_pos = _joint_pos(obs, prefix)
        assert torch.equal(
            reset_pos[0], post_step[0]
        ), "the reader lost row 0's stepped state after a row-1 reset"
        assert torch.equal(reset_pos[1], spawn[1]), "the reader did not re-read row 1's reset state"
    finally:
        sim.close()
