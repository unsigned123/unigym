"""A facade config loads to the same declared control and action groups."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import yaml

from unigym.robots import restore_robot_registry, snapshot_robot_registry
from unigym.runtime.actions import OrderedActionGroup
from unigym.sim import BindingError, SimulationBuilder
from tests._support.sim_fixtures import write_sim_pack


@pytest.fixture(autouse=True)
def _isolate_robot_registry() -> Iterator[None]:
    """Restore the process-wide robot registry after each facade build."""
    snapshot = snapshot_robot_registry()
    yield
    restore_robot_registry(snapshot)


def _write_config(tmp_path: Path, tiny_mjcf: Path) -> tuple[Path, str]:
    """Write one config that references the two-joint pack robot."""
    pack_set, ref = write_sim_pack(
        tmp_path / "pack",
        joint_sets={"body": ("shoulder", "elbow")},
        default_joint_pos={"shoulder": 0.1, "elbow": -0.2},
    )
    cfg = {
        "run_name": "facade_roundtrip",
        "seed": 0,
        "env": {
            "backend": "mujoco",
            "num_envs": 1,
            "episode_length_s": 20.0,
            "decimation": 4,
            "sim_dt": 0.005,
        },
        "world_base": {"mjcf": str(tiny_mjcf)},
        "runner": {"auto_reset": False},
        "adapter": {"name": "manual"},
        "command": {"name": "fixed", "dim": 0, "values": []},
        "collect": {"steps": 1},
        "articulations": [
            {
                "name": "body",
                "params": {"profile": ref.name},
                "actuation": [
                    {
                        "name": "body",
                        "law": "standard_pd",
                        "joints": ["shoulder", "elbow"],
                        "gains": {"kp": 20.0, "kd": 1.0},
                    }
                ],
            }
        ],
        "action_spec": [
            {
                "name": "body",
                "articulation": "body",
                "joints": ["shoulder", "elbow"],
                "target": "position",
                "role": "body",
            }
        ],
        "policies": [{"adapter": "manual", "groups": [["body", "body"]]}],
    }
    path = tmp_path / "sim.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path, pack_set


def test_config_mode_acts_through_the_runtime_target(tmp_path: Path, tiny_mjcf: Path) -> None:
    """A control=None config handle stages the RunSpec action group."""
    path, pack_set = _write_config(tmp_path, tiny_mjcf)

    loaded = SimulationBuilder.from_config(path, pack_set=pack_set)
    sim = loaded.build()
    try:
        handle = sim.articulations[0]
        group = handle.action_groups["body"]
        sim.act(group, torch.tensor([[0.3, -0.4]], dtype=torch.float32))
        obs = sim.step()
        assert obs.raw[f"articulation/{handle.name}/joint_pos"].shape == (1, 2)
        with pytest.raises(BindingError, match="does not match the declared template"):
            sim.act(
                OrderedActionGroup(
                    name="body", articulation="body", joints=("shoulder", "elbow"), target="torque"
                ),
                torch.zeros(1, 2),
            )
    finally:
        sim.close()
