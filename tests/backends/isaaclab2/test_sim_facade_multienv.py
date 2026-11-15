"""The shared facade script and per-env reset on the real IsaacLab 2 G1."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import yaml

from unigym.packs import discover_set
from unigym.packs.packset import PackSet, RobotRef
from unigym.robots import restore_robot_registry, snapshot_robot_registry
from unigym.scenes import Scene
from unigym.scenes.terrain import Terrain
from unigym.sim import Control, SimulationBuilder
from unigym.world import World
from tests._support.sim_facade_script import ARTICULATION_LABEL, run_facade_acceptance

pytestmark = [
    pytest.mark.sim,
    pytest.mark.usefixtures("simulation_app"),
    pytest.mark.requires_test_packs("g1"),
]


@pytest.fixture(autouse=True)
def _fresh_stage(simulation_app: object) -> Iterator[None]:
    """Start every test from a new USD stage (the lane's per-build idiom).

    Kit keeps one app for the whole lane process.  The facade build does not
    create a stage itself, while the lane's other env fixtures open a fresh
    stage per construction; without one, a second facade build in the same
    process reuses the playing stage and IsaacLab's second SimulationContext
    stops the timeline from inside the previous context's stop callback.
    """
    import isaaclab.sim as sim_utils

    sim_utils.create_new_stage()
    yield


@pytest.fixture(autouse=True)
def _isolate_robot_registry() -> Iterator[None]:
    """Restore the process-wide robot registry after the facade build."""
    snapshot = snapshot_robot_registry()
    yield
    restore_robot_registry(snapshot)


def _token(root: Path) -> str:
    """A pack/robot token unique to the caller-owned temporary root."""
    token = "".join(character if character.isalnum() else "_" for character in root.name.lower())
    return token.strip("_")[:42] or "facade_g1"


def _write_fixture_pack(
    root: Path,
    *,
    source_robot: Path,
    usd: Path,
) -> tuple[PackSet, RobotRef, tuple[str, ...]]:
    """Write one lane-owned pack whose G1 profile carries the resolved USD.

    This is the same lane-owned fixture pack the recorded ``N == 1`` evidence
    file writes; the multi-env test must not fork the asset or the joint order.
    """
    token = _token(root)
    pack_name = f"pack_{token}"
    robot_id = f"robot.{token}"
    document = yaml.safe_load(source_robot.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    document["name"] = robot_id
    document["assets"] = {"usd": str(usd)}
    body_order = tuple(str(joint) for joint in document["joint_sets"]["body"])

    robot_path = root / "robots" / f"{token}.yaml"
    robot_path.parent.mkdir(parents=True, exist_ok=True)
    robot_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    manifest = {
        "schema": "unigym.pack/1",
        "name": pack_name,
        "version": "1.0.0",
        "api": 1,
        "trust": "content",
        "provides": {
            "providers": [
                {
                    "id": robot_id,
                    "kind": "robot",
                    "resource": f"robots/{token}.yaml",
                }
            ]
        },
    }
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    pack_set = discover_set([root])
    return pack_set, pack_set.pack(pack_name).robot(robot_id), body_order


def _world(tmp_path: Path) -> World:
    """The lane-owned inline terrain world (no pack ground asset)."""
    terrain = Terrain.heightmap([[0.0] * 8 for _ in range(8)], size=(8.0, 8.0))
    return World.inline(Scene(name=f"terrain_{_token(tmp_path)}", terrain=terrain))


def test_isaaclab2_runs_the_shared_facade_script_for_two_envs(
    tmp_path: Path,
    pack_runtime,
    g1_usd: Path,
) -> None:
    """The shared body builds, observes, acts and steps two real G1 clones."""
    source_robot = pack_runtime("g1").resource_for("providers", "g1")
    pack_set, robot, body_order = _write_fixture_pack(
        tmp_path, source_robot=source_robot, usd=g1_usd
    )

    evidence = run_facade_acceptance(
        pack_set=pack_set,
        robot_id=robot.id,
        world=_world(tmp_path),
        backend="isaaclab2",
        expected_joint_names=body_order,
        steps=3,
        device="cuda:0",
        num_envs=2,
    )

    assert evidence["backend"] == "isaaclab2"
    assert evidence["articulation"] == ARTICULATION_LABEL
    assert evidence["joint_names"] == list(body_order)
    assert evidence["num_envs"] == 2
    assert evidence["t"] == evidence["steps"] - 1
    rows = evidence["final_joint_pos"]
    assert isinstance(rows, list)
    assert len(rows) == 2
    assert all(isinstance(row, list) and len(row) == len(body_order) for row in rows)
    assert len({tuple(row) for row in rows}) == 2, f"the two G1 rows did not diverge: {rows}"


def test_isaaclab2_reset_subset_restores_only_the_named_row(
    tmp_path: Path,
    pack_runtime,
    g1_usd: Path,
) -> None:
    """reset(env_ids=[1]) rolls one G1 clone back and leaves the other alone."""
    source_robot = pack_runtime("g1").resource_for("providers", "g1")
    pack_set, robot, body_order = _write_fixture_pack(
        tmp_path, source_robot=source_robot, usd=g1_usd
    )
    builder = SimulationBuilder(
        backend="isaaclab2",
        dt=0.02,
        physics_dt=0.005,
        num_envs=2,
        pack_set=pack_set,
        device="cuda:0",
    )
    builder.world = _world(tmp_path)
    handle = builder.add_articulation(
        robot,
        name=ARTICULATION_LABEL,
        control=Control.position_pd(joint_set="body", kp=20.0, kd=1.0),
    )
    prefix = f"articulation/{handle.name}"
    pattern = torch.linspace(-0.1, 0.1, len(body_order), dtype=torch.float32)
    targets = pattern.reshape(1, len(body_order)) + 0.05 * torch.arange(
        2, dtype=torch.float32
    ).reshape(2, 1)

    sim = builder.build()
    try:
        spawn = sim.observe().raw[f"{prefix}/joint_pos"]
        assert isinstance(spawn, torch.Tensor)
        spawn = spawn.clone()
        sim.act(handle.bind_action("body", kind="position"), targets)
        stepped = sim.step()
        post_step = stepped.raw[f"{prefix}/joint_pos"]
        assert isinstance(post_step, torch.Tensor)
        post_step = post_step.clone()
        assert not torch.equal(
            post_step[1], spawn[1]
        ), f"row 1 did not move from spawn on the stepped tick: {post_step[1].tolist()}"

        obs = sim.reset(env_ids=[1])

        assert obs.reset_env_ids.tolist() == [1]
        assert obs.reset_due_env_ids.tolist() == []
        assert obs.terminated is None
        reset_pos = obs.raw[f"{prefix}/joint_pos"]
        assert isinstance(reset_pos, torch.Tensor)
        assert torch.equal(reset_pos[0], post_step[0]), "row 0 was touched by a row-1 reset"
        assert torch.allclose(reset_pos[1], spawn[1], atol=1e-5), (
            f"row 1 was not restored to its spawn state: {reset_pos[1].tolist()} "
            f"vs {spawn[1].tolist()}"
        )
    finally:
        sim.close()
