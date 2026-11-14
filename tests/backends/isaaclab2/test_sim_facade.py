"""The shared facade acceptance script runs unchanged on the real IsaacLab 2 G1."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from unigym.packs import discover_set
from unigym.packs.packset import PackSet, RobotRef
from unigym.robots import restore_robot_registry, snapshot_robot_registry
from unigym.scenes import Scene
from unigym.scenes.terrain import Terrain
from unigym.world import World
from tests._support.sim_facade_script import ARTICULATION_LABEL, run_facade_acceptance

pytestmark = [
    pytest.mark.sim,
    pytest.mark.usefixtures("simulation_app"),
    pytest.mark.requires_test_packs("g1"),
]


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
    """Write one lane-owned pack whose G1 profile carries the resolved USD."""
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


def test_isaaclab2_runs_the_shared_facade_acceptance_script(
    tmp_path: Path,
    pack_runtime,
    g1_usd: Path,
) -> None:
    """The shared body builds, observes, acts and steps on the real G1 USD."""
    source_robot = pack_runtime("g1").resource_for("providers", "g1")
    pack_set, robot, body_order = _write_fixture_pack(
        tmp_path, source_robot=source_robot, usd=g1_usd
    )
    terrain = Terrain.heightmap([[0.0] * 8 for _ in range(8)], size=(8.0, 8.0))
    world = World.inline(Scene(name=f"terrain_{_token(tmp_path)}", terrain=terrain))

    evidence = run_facade_acceptance(
        pack_set=pack_set,
        robot_id=robot.id,
        world=world,
        backend="isaaclab2",
        expected_joint_names=body_order,
        steps=5,
        device="cuda:0",
    )

    assert evidence["backend"] == "isaaclab2"
    assert evidence["articulation"] == ARTICULATION_LABEL
    assert evidence["joint_names"] == list(body_order)
    assert evidence["steps"] == 5
    assert evidence["t"] == evidence["steps"] - 1
    final_joint_pos = evidence["final_joint_pos"]
    assert isinstance(final_joint_pos, list)
    assert len(final_joint_pos) == len(body_order)
