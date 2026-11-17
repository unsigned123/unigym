"""The policy clock and world articulation actuation on the real MuJoCo lane."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from unigym.robots import register_robot, restore_robot_registry, snapshot_robot_registry
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run.api import Run
from unigym.world import World

ACTUATION = [
    {
        "name": "body",
        "law": "standard_pd",
        "joint_set": "body",
        "gains": {"kp": 20.0, "kd": 1.0},
    }
]


def _tiny_profile() -> RobotProfile:
    """The tiny-arm profile matching the shared primitives-only MJCF."""
    return RobotProfile(
        name="tiny",
        joint_sets={"body": ("shoulder", "elbow")},
        default_joint_pos={"shoulder": 0.0, "elbow": 0.0},
        assets=None,
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
    )


@pytest.fixture(autouse=True)
def _isolate_robot_registry() -> Iterator[None]:
    """Register the tiny-arm profile for one test, then restore the registry."""
    snapshot = snapshot_robot_registry()
    register_robot("tiny", _tiny_profile)
    yield
    restore_robot_registry(snapshot)


def _clock_cfg(mjcf: Path, control_dt: float | None) -> dict:
    policy: dict = {"adapter": "manual", "groups": [["tiny", "tiny"]]}
    if control_dt is not None:
        policy["control_dt"] = control_dt
    return {
        "run_name": "clock",
        "seed": 0,
        "mode": "collect",
        "env": {
            "backend": "mujoco",
            "num_envs": 1,
            "episode_length_s": 1.0,
            "decimation": 4,
            "sim_dt": 0.005,
        },
        "collect": {"steps": 2, "shard_steps": 100},
        "world_base": {"mjcf": str(mjcf)},
        "adapter": {"name": "manual"},
        "command": {"name": "fixed", "dim": 0, "values": []},
        "articulations": [{"name": "tiny", "params": {"profile": "tiny"}, "actuation": ACTUATION}],
        "action_spec": [
            {
                "name": "tiny",
                "articulation": "tiny",
                "joints": ["shoulder", "elbow"],
                "target": "position",
                "role": "body",
            }
        ],
        "policies": [policy],
    }


def test_run_owned_articulation_supplies_actuation_on_flat_world(tiny_mjcf: Path) -> None:
    """The run (never the World) owns the robot asset and its actuation."""
    run = Run(
        world=World.flat(),
        robot={
            "name": "tiny",
            "model": {"mjcf": str(tiny_mjcf)},
            "actuation": ACTUATION,
        },
        adapter="manual",
        inspect=True,
    )
    try:
        instance = run.build()

        emitted = run.cfg["articulations"][0]["actuation"]
        assert emitted == ACTUATION
        assert run.cfg["world_base"]["mjcf"].endswith("flat_ground.xml")
        assert ("tiny", "tiny") in instance.env.controls
        assert instance.env.controls[("tiny", "tiny")].drive == "position_pd"
    finally:
        run.close()


def test_policy_control_dt_is_authoritative_in_the_meta_and_runner(tiny_mjcf: Path) -> None:
    """A declared policy period overrides the adapter period (0.02) everywhere."""
    run = Run(_clock_cfg(tiny_mjcf, 0.1), inspect=True)
    try:
        instance = run.build()

        assert instance.writer.meta["policy_dt"] == 0.1
        assert instance.env.controls[("tiny", "tiny")].control_dt == 0.1
        assert run.runner("sync").control_dt == 0.1
    finally:
        run.close()


def test_policy_control_dt_falls_back_to_the_adapter_period(tiny_mjcf: Path) -> None:
    """Without a binding period the adapter's explicit value is the fallback."""
    run = Run(_clock_cfg(tiny_mjcf, None), inspect=True)
    try:
        instance = run.build()

        assert instance.writer.meta["policy_dt"] == 0.02
        assert instance.env.controls[("tiny", "tiny")].control_dt == 0.02
    finally:
        run.close()
