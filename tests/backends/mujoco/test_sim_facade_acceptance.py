"""The shared acceptance script runs unchanged on the real MuJoCo backend."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from unigym.packs import discover_set
from unigym.robots import restore_robot_registry, snapshot_robot_registry
from unigym.world import World
from tests._support.sim_facade_script import ARTICULATION_LABEL, run_facade_acceptance

FACADE_PACK = Path(__file__).resolve().parents[2] / "_support" / "packs" / "facade_mjcf"
ROBOT_ID = "facade_mjcf.tiny_arm"
JOINT_ORDER = ("shoulder", "elbow")


@pytest.fixture(autouse=True)
def _isolate_robot_registry() -> Iterator[None]:
    """Restore the process-wide robot registry after the facade build."""
    snapshot = snapshot_robot_registry()
    yield
    restore_robot_registry(snapshot)


def test_mujoco_runs_the_shared_facade_acceptance_script(tiny_mjcf: Path) -> None:
    """The shared body builds, observes, acts and steps on the MJCF fixture."""
    pack_set = discover_set([FACADE_PACK])
    evidence = run_facade_acceptance(
        pack_set=pack_set,
        robot_id=ROBOT_ID,
        world=World.base(mjcf=str(tiny_mjcf)),
        backend="mujoco",
        expected_joint_names=JOINT_ORDER,
        steps=5,
    )

    assert evidence["backend"] == "mujoco"
    assert evidence["articulation"] == ARTICULATION_LABEL
    assert evidence["joint_names"] == list(JOINT_ORDER)
    assert evidence["steps"] == 5
    assert evidence["t"] == evidence["steps"] - 1
    final_joint_pos = evidence["final_joint_pos"]
    assert isinstance(final_joint_pos, list)
    assert len(final_joint_pos) == len(JOINT_ORDER)
