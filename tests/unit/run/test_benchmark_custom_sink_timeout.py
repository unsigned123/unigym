"""Custom-sink benchmark runs reject the built-in deadline configuration.

The core episode clock is the one physical deadline authority.  A custom sink
ends episodes through its terminals, so ``benchmark.timeout_s`` /
``benchmark.timeout_steps`` must fail at the build boundary before any env or
benchmark task exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from unigym.actuators.ir import ActuationGroup
from unigym.assets.asset_ref import AssetRef
from unigym.errors import ConfigError
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run.api import Run
from unigym.run.sink import SinkSpec
from unigym.runtime.actions import OrderedActionSpec, PolicyBinding
from unigym.world import World
from tests._support.fakes.external_sink import make_external_sink

TINY_ARM = Path(__file__).resolve().parents[2] / "_support" / "data" / "tiny_arm.xml"
JOINTS = ("shoulder", "elbow")
ACTION_SPEC = OrderedActionSpec.from_mapping(
    [
        {
            "name": "tiny",
            "articulation": "tiny",
            "joints": list(JOINTS),
            "target": "position",
            "role": "body",
        }
    ]
)
POLICIES = (PolicyBinding.from_mapping({"adapter": "manual", "groups": [["tiny", "tiny"]]}),)


def _profile() -> RobotProfile:
    return RobotProfile(
        name="tiny",
        joint_sets={"body": JOINTS},
        default_joint_pos={"shoulder": 0.1, "elbow": -0.2},
        assets=AssetRef(mjcf=str(TINY_ARM)),
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
        actuation=(
            ActuationGroup(
                name="body",
                law="standard_pd",
                joint_set="body",
                gains={"kp": 20.0, "kd": 1.0},
            ),
        ),
    )


def _run(tmp_path: Path, *, mode: str, benchmark: dict[str, object]) -> Run:
    # The world is deliberately missing: the timeout rejection must win before
    # env construction, so a correctly rejected run never touches this path.
    return Run(
        world=World.base(mjcf=tmp_path / "missing_world.xml"),
        robot=_profile(),
        backend="mujoco",
        observations={"robot": ["tiny"]},
        action_spec=ACTION_SPEC,
        policies=POLICIES,
        steps=1,
        inspect=True,
        sink=SinkSpec(factory=make_external_sink),
        mode=mode,
        benchmark=benchmark,
        pack_runtime=None,
        discover_runtime=False,
    )


@pytest.mark.parametrize(
    ("mode", "benchmark", "field"),
    [
        ("benchmark", {"tasks": [], "timeout_s": 0.04}, "benchmark.timeout_s"),
        ("benchmark", {"tasks": [], "timeout_steps": 2}, "benchmark.timeout_steps"),
        ("both", {"tasks": [], "timeout_steps": 2}, "benchmark.timeout_steps"),
    ],
    ids=["benchmark/timeout_s", "benchmark/timeout_steps", "both/timeout_steps"],
)
def test_custom_sink_benchmark_rejects_timeout_before_env_build(
    tmp_path: Path, mode: str, benchmark: dict[str, object], field: str
) -> None:
    """A custom benchmark/both deadline key is a teaching ConfigError."""
    run = _run(tmp_path, mode=mode, benchmark=benchmark)
    try:
        with pytest.raises(ConfigError, match=rf"{re.escape(field)}.*core episode clock"):
            run.build(inspect=True)
    finally:
        run.close()


def test_custom_sink_benchmark_timeout_is_rejected_from_config(tmp_path: Path) -> None:
    """The config path reaches the same build-boundary rejection."""
    path = tmp_path / "run.yaml"
    path.write_text(
        "\n".join(
            [
                "run_name: c6",
                "seed: 0",
                "env: {backend: mujoco, num_envs: 1}",
                "world_base: {mjcf: missing.xml}",
                "collect: {steps: 1}",
                "articulations: [{name: arm, joints: [{name: shoulder}]}]",
                "action_spec:",
                "  - {name: arm, articulation: arm, joints: [shoulder],",
                "     target: position, role: body}",
                "observations: {robot: [arm]}",
                "policies:",
                "  - {adapter: manual, groups: [[arm, arm]]}",
                "mode: benchmark",
                "benchmark: {tasks: [], min_episodes: 1, max_steps: 1, timeout_s: 0.02}",
                "sink:",
                "  factory: tests._support.fakes.external_sink:make_external_sink",
            ]
        ),
        encoding="utf-8",
    )

    run = Run.from_config(path, discover_runtime=False)
    try:
        with pytest.raises(ConfigError, match=r"benchmark\.timeout_s.*core episode clock"):
            run.build(inspect=True)
    finally:
        run.close()
