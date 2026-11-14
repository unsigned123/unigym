"""A facade config round-trip reproduces the same reset joint state and actions."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from unigym.packs.packset import PackSet
from unigym.robots import restore_robot_registry, snapshot_robot_registry
from unigym.sim import BindingError, Control, SimulationBuilder
from unigym.world import World
from tests._support.sim_fixtures import write_sim_pack


@pytest.fixture(autouse=True)
def _isolate_robot_registry() -> Iterator[None]:
    """Restore the process-wide robot registry after each facade build."""
    snapshot = snapshot_robot_registry()
    yield
    restore_robot_registry(snapshot)


def _source_builder(tmp_path: Path, tiny_mjcf: Path) -> tuple[SimulationBuilder, PackSet]:
    """A declared builder over the two-joint pack robot and tiny MJCF world."""
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
    return builder, pack_set


def _write_roundtrip(source: SimulationBuilder, path: Path) -> None:
    """Build, write the authoring config and close the source simulation."""
    sim = source.build()
    try:
        sim.to_config(path)
    finally:
        sim.close()


def test_config_round_trip_reproduces_the_reset_joint_positions(
    tmp_path: Path, tiny_mjcf: Path
) -> None:
    """to_config -> from_config -> build -> reset is bit-identical to the source."""
    builder, pack_set = _source_builder(tmp_path, tiny_mjcf)
    source = builder.build()
    try:
        source_obs = source.reset()
        source_pos = source_obs.articulations[source.articulations[0]].joint_pos.clone()
        written = source.to_config(tmp_path / "roundtrip.yaml")
    finally:
        source.close()

    loaded = SimulationBuilder.from_config(written, pack_set=pack_set)
    sim = loaded.build()
    try:
        loaded_obs = sim.reset()
        loaded_pos = loaded_obs.articulations[sim.articulations[0]].joint_pos
    finally:
        sim.close()

    assert sim.backend == source.backend == "mujoco"
    assert tuple(handle.name for handle in loaded.articulations) == ("body",)
    assert tuple(loaded.articulations[0].faces) == ("body",)
    assert torch.equal(source_pos, loaded_pos)


def test_config_mode_acts_through_the_runtime_target(tmp_path: Path, tiny_mjcf: Path) -> None:
    """A control=None config handle binds with the live ControlRuntime target."""
    builder, pack_set = _source_builder(tmp_path, tiny_mjcf)
    path = tmp_path / "roundtrip.yaml"
    _write_roundtrip(builder, path)

    loaded = SimulationBuilder.from_config(path, pack_set=pack_set)
    sim = loaded.build()
    try:
        handle = sim.articulations[0]
        binding = handle.bind_action("body", kind="position")
        sim.act(binding, torch.tensor([[0.3, -0.4]], dtype=torch.float32))
        obs = sim.step()
        assert obs.raw[f"articulation/{handle.name}/joint_pos"].shape == (1, 2)
        with pytest.raises(BindingError, match="control target 'position'"):
            sim.act(handle.bind_action("body", kind="torque"), torch.zeros(1, 2))
    finally:
        sim.close()
