"""The shared facade script and per-env reset on the real IsaacLab 2 G1."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import yaml

from unigym.packs import discover_set
from unigym.packs.loader import PackLoader
from unigym.packs.packset import PackSet
from unigym.robots.base import RobotProfile
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
    """Start every test from a fresh USD stage (lane stage provisioning).

    Kit keeps one app for the whole lane process, and the facade build does not
    create a stage itself, so each test needs one to spawn into.  This fixture
    is stage provisioning only: it is not the sequential-build fix, and the
    same-process rebuild test below deliberately builds twice without it.
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
) -> tuple[PackSet, RobotProfile, tuple[str, ...]]:
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
    loader = PackLoader.from_pack_set(pack_set)
    return pack_set, loader.robot_profile(pack_name, robot_id), body_order


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
    source_robot = pack_runtime("g1").resolve("providers", "g1")
    pack_set, robot, body_order = _write_fixture_pack(
        tmp_path, source_robot=source_robot, usd=g1_usd
    )

    evidence = run_facade_acceptance(
        pack_set=pack_set,
        robot_id=robot.name,
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
    source_robot = pack_runtime("g1").resolve("providers", "g1")
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
        spawn = sim.observe().record[f"{prefix}/joint_pos"]
        assert isinstance(spawn, torch.Tensor)
        spawn = spawn.clone()
        sim.act(
            handle.action_groups["body"],
            targets,
        )
        stepped = sim.step()
        post_step = stepped.record[f"{prefix}/joint_pos"]
        assert isinstance(post_step, torch.Tensor)
        post_step = post_step.clone()
        assert not torch.equal(
            post_step[1], spawn[1]
        ), f"row 1 did not move from spawn on the stepped tick: {post_step[1].tolist()}"

        obs = sim.reset(env_ids=[1])

        assert obs.reset_env_ids.tolist() == [1]
        assert obs.reset_due_env_ids.tolist() == []
        assert obs.terminated is None
        reset_pos = obs.record[f"{prefix}/joint_pos"]
        assert isinstance(reset_pos, torch.Tensor)
        assert torch.equal(reset_pos[0], post_step[0]), "row 0 was touched by a row-1 reset"
        assert torch.allclose(reset_pos[1], spawn[1], atol=1e-5), (
            f"row 1 was not restored to its spawn state: {reset_pos[1].tolist()} "
            f"vs {spawn[1].tolist()}"
        )
    finally:
        sim.close()


def _clear_scene_prims() -> None:
    """Remove the previous build's scene prims so the same declarations respawn.

    The facade does not own stage teardown, and Isaac Lab's shape and terrain
    spawners refuse a prim path that already exists.  This is a stage content
    reset only: it never opens a new stage and never touches the Kit timeline,
    so a ``close()`` that left the timeline playing would still hang the next
    build in ``SimulationContext.__init__``.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    world = stage.GetPrimAtPath("/World")
    for child in list(world.GetChildren()):
        stage.RemovePrim(child.GetPath())


def test_isaaclab2_rebuilds_the_facade_in_one_process(
    tmp_path: Path,
    pack_runtime,
    g1_usd: Path,
) -> None:
    """A closed facade build leaves the Kit timeline paused, so a rebuild works.

    IsaacLab's ``SimulationContext.__init__`` registers its STOP callback
    before the Isaac Sim parent constructor, and the parent stops a playing
    timeline.  If ``IsaaclabEnv.close()`` left the timeline playing, the second
    build in this process would fire that fresh callback and spin in
    ``render()`` forever in headless mode.  The assertion after the first close
    is the fast failure guard; the second build is the proof.  No new stage is
    created between the builds.
    """
    import omni.timeline

    source_robot = pack_runtime("g1").resolve("providers", "g1")
    pack_set, robot, body_order = _write_fixture_pack(
        tmp_path, source_robot=source_robot, usd=g1_usd
    )
    world = _world(tmp_path)

    first = run_facade_acceptance(
        pack_set=pack_set,
        robot_id=robot.name,
        world=world,
        backend="isaaclab2",
        expected_joint_names=body_order,
        steps=1,
        device="cuda:0",
        num_envs=1,
    )
    timeline = omni.timeline.get_timeline_interface()
    assert first["t"] == 0
    assert not timeline.is_playing(), (
        "IsaaclabEnv.close() left the Kit timeline playing; the second build in "
        "this process would hang in SimulationContext.__init__"
    )

    _clear_scene_prims()
    second = run_facade_acceptance(
        pack_set=pack_set,
        robot_id=robot.name,
        world=world,
        backend="isaaclab2",
        expected_joint_names=body_order,
        steps=1,
        device="cuda:0",
        num_envs=1,
    )

    assert second["backend"] == "isaaclab2"
    assert second["joint_names"] == list(body_order)
    assert second["t"] == 0
    assert not timeline.is_playing(), "the rebuilt env's close left the Kit timeline playing"
