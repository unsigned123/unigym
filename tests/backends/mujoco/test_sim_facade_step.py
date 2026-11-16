"""Facade stepping is bit-identical to a raw ExternalRunner loop on tiny_arm."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from unigym.robots import restore_robot_registry, snapshot_robot_registry
from unigym.run.api import Run
from unigym.runtime.actions import NamedAction
from unigym.sim import Control, SimulationBuilder
from unigym.world import World
from tests._support.sim_fixtures import write_sim_pack


@pytest.fixture(autouse=True)
def _isolate_robot_registry() -> Iterator[None]:
    """Restore the process-wide robot registry after the facade build."""
    snapshot = snapshot_robot_registry()
    yield
    restore_robot_registry(snapshot)


def test_facade_step_matches_the_raw_external_runner(tmp_path: Path, tiny_mjcf: Path) -> None:
    """Same seed/actions produce bit-identical joint positions and velocities."""
    pack_set, ref = write_sim_pack(
        tmp_path / "pack",
        joint_sets={"body": ("shoulder", "elbow")},
        default_joint_pos={"shoulder": 0.0, "elbow": 0.0},
    )
    builder = SimulationBuilder(backend="mujoco", dt=0.02, physics_dt=0.005, pack_set=pack_set)
    builder.world = World.base(mjcf=str(tiny_mjcf))
    handle = builder.add_articulation(
        ref,
        name="body",
        control=Control.position_pd(joint_set="body", kp=20.0, kd=1.0),
    )
    group = handle.action_groups["body"]
    actions = (
        torch.tensor([[0.2, -0.1]], dtype=torch.float32),
        torch.tensor([[0.0, 0.3]], dtype=torch.float32),
        torch.tensor([[-0.2, 0.1]], dtype=torch.float32),
    )
    joint_pos_key = f"articulation/{handle.name}/joint_pos"
    joint_vel_key = f"articulation/{handle.name}/joint_vel"

    sim = builder.build()
    raw_run: Run | None = None
    try:
        assert sim.runner is sim.run.runner()
        facade = []
        for action in actions:
            sim.act(group, action)
            obs = sim.step()
            facade.append((obs.raw[joint_pos_key], obs.raw[joint_vel_key]))

        raw_run = Run(sim.run.spec.cfg, no_artifacts=True, device="cpu")
        raw_runner = raw_run.runner("external")
        raw = []
        for action in actions:
            result = raw_runner.step(
                (
                    NamedAction(
                        group.articulation,
                        group.name,
                        action,
                        tuple(group.joints),
                        "position",
                    ),
                )
            )
            raw.append((result.extra[joint_pos_key], result.extra[joint_vel_key]))

        assert len(facade) == len(raw) == len(actions)
        for index, ((facade_pos, facade_vel), (raw_pos, raw_vel)) in enumerate(
            zip(facade, raw, strict=True)
        ):
            assert torch.equal(facade_pos, raw_pos), f"joint_pos diverged at step {index}"
            assert torch.equal(facade_vel, raw_vel), f"joint_vel diverged at step {index}"
    finally:
        sim.close()
        if raw_run is not None:
            raw_run.close()
