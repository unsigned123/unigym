"""Facade observation tensors equal the producing step extra column-for-column."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from unigym.robots import restore_robot_registry, snapshot_robot_registry
from unigym.sim import Control, SimulationBuilder
from unigym.world import World
from tests._support.sim_fixtures import write_sim_pack


@pytest.fixture(autouse=True)
def _isolate_robot_registry() -> Iterator[None]:
    """Restore the process-wide robot registry after the facade build."""
    snapshot = snapshot_robot_registry()
    yield
    restore_robot_registry(snapshot)


def test_observed_joint_tensors_match_the_step_extra(tmp_path: Path, tiny_mjcf: Path) -> None:
    """joint_pos/joint_vel columns come from the step extra, unchanged."""
    pack_set, ref = write_sim_pack(
        tmp_path / "pack",
        joint_sets={"body": ("shoulder", "elbow")},
        default_joint_pos={"shoulder": 0.1, "elbow": -0.2},
    )
    builder = SimulationBuilder(backend="mujoco", dt=0.02, physics_dt=0.005, pack_set=pack_set)
    builder.world = World.base(mjcf=str(tiny_mjcf))
    handle = builder.add_articulation(
        ref,
        name="body",
        control=Control.position_pd(joint_set="body", kp=20.0, kd=1.0),
    )
    binding = handle.action_groups["body"]

    sim = builder.build()
    try:
        sim.act(binding, torch.tensor([[0.3, -0.4]], dtype=torch.float32))
        obs = sim.step()
    finally:
        sim.close()

    prefix = f"articulation/{handle.name}"
    record = obs.articulations[handle]
    joint_pos = obs.raw[f"{prefix}/joint_pos"]
    joint_vel = obs.raw[f"{prefix}/joint_vel"]
    assert isinstance(joint_pos, torch.Tensor)
    assert isinstance(joint_vel, torch.Tensor)

    assert record.joint_names == ("shoulder", "elbow")
    assert record.joint_pos.shape == joint_pos.shape == (1, 2)
    assert record.joint_vel.shape == joint_vel.shape == (1, 2)
    for column in range(record.joint_pos.shape[1]):
        assert torch.equal(record.joint_pos[:, column], joint_pos[:, column])
        assert torch.equal(record.joint_vel[:, column], joint_vel[:, column])
    q0 = torch.tensor([0.1, -0.2], dtype=record.joint_pos.dtype)
    assert torch.equal(record.joint_pos_rel, record.joint_pos - q0)


def test_reset_observation_reports_the_rows_the_runner_reset(
    tmp_path: Path, tiny_mjcf: Path
) -> None:
    """reset_env_ids reflects the real reset StepResult, not an empty default."""
    pack_set, ref = write_sim_pack(
        tmp_path / "pack",
        joint_sets={"body": ("shoulder", "elbow")},
        default_joint_pos={"shoulder": 0.1, "elbow": -0.2},
    )
    builder = SimulationBuilder(backend="mujoco", dt=0.02, physics_dt=0.005, pack_set=pack_set)
    builder.world = World.base(mjcf=str(tiny_mjcf))
    builder.add_articulation(
        ref,
        name="body",
        control=Control.position_pd(joint_set="body", kp=20.0, kd=1.0),
    )

    sim = builder.build()
    try:
        obs = sim.reset()
    finally:
        sim.close()

    assert obs.reset_env_ids.dtype == torch.int64
    assert obs.reset_env_ids.tolist() == [0]
    assert obs.terminated is None
