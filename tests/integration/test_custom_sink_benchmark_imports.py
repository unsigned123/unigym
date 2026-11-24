"""A custom-sink benchmark run leaves ``unigym.benchmarks`` unimported.

The custom sink owns benchmark control flow, so the built-in task provider,
buffer, evaluator, reporter and session must not load on that path.  The run
happens in a subprocess so an earlier test cannot pollute ``sys.modules``; no
code pack is activated.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from unigym.run.api import Run

_TINY_ARM = Path(__file__).resolve().parents[1] / "_support" / "data" / "tiny_arm.xml"
_FLOOR_XML = (
    '<mujoco model="floor"><worldbody>'
    '<geom name="world_floor" type="plane" size="2 2 0.1"/>'
    "</worldbody></mujoco>"
)
_JOINTS = ("shoulder", "elbow")
_STEPS = 20
_EPISODE_LENGTH_S = 0.04


def _benchmark_custom_sink_run(world_dir: Path, benchmark: dict[str, object]) -> Run:
    """Build the tiny-arm custom-sink benchmark declaration without running it."""
    from unigym.actuators.ir import ActuationGroup
    from unigym.assets.asset_ref import AssetRef
    from unigym.robots.base import BodyBinding, RobotProfile
    from unigym.run.api import Run
    from unigym.run.sink import SinkSpec
    from unigym.runtime.actions import OrderedActionSpec, PolicyBinding
    from unigym.world import World

    from tests._support.fakes.external_sink import make_external_sink

    world = world_dir / "world.xml"
    world.write_text(_FLOOR_XML, encoding="utf-8")
    profile = RobotProfile(
        name="tiny",
        joint_sets={"body": _JOINTS},
        default_joint_pos={"shoulder": 0.1, "elbow": -0.2},
        assets=AssetRef(mjcf=str(_TINY_ARM)),
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
    action_spec = OrderedActionSpec.from_mapping(
        [
            {
                "name": "tiny",
                "articulation": "tiny",
                "joints": list(_JOINTS),
                "target": "position",
                "role": "body",
            }
        ]
    )
    return Run(
        world=World.base(mjcf=world),
        robot=profile,
        backend="mujoco",
        observations={"robot": ["tiny"]},
        action_spec=action_spec,
        policies=(PolicyBinding.from_mapping({"adapter": "manual", "groups": [["tiny", "tiny"]]}),),
        mode="benchmark",
        steps=_STEPS,
        episode_length_s=_EPISODE_LENGTH_S,
        inspect=True,
        sink=SinkSpec(factory=make_external_sink, options={"target_episodes": 2}),
        benchmark=benchmark,
        pack_runtime=None,
        discover_runtime=False,
    )


def build_and_run_benchmark_custom_sink(world_dir: Path) -> None:
    """Build and run one tiny-arm ``benchmark`` run driven by a custom sink."""
    from unigym.runtime.actions import NamedAction

    run = _benchmark_custom_sink_run(
        world_dir, {"tasks": [], "min_episodes": 1, "max_steps": _STEPS}
    )

    def zero_actions(_t: int) -> tuple[NamedAction, ...]:
        return (NamedAction.zeros(1, "tiny", _JOINTS, "position"),)

    try:
        instance = run.build(inspect=True)
        runner = run.runner("external", action_fn=zero_actions)
        runner.run(stop_when=instance.stop_when)
    finally:
        run.close()


def reject_benchmark_custom_sink_timeout(world_dir: Path) -> None:
    """Prove a custom benchmark timeout fails before any built-in benchmark."""
    from unigym.errors import ConfigError

    run = _benchmark_custom_sink_run(
        world_dir,
        {"tasks": [], "min_episodes": 1, "max_steps": _STEPS, "timeout_s": 0.04},
    )
    try:
        with pytest.raises(ConfigError, match=r"benchmark\.timeout_s.*core episode clock"):
            run.build(inspect=True)
    finally:
        run.close()


@pytest.mark.subprocess
def test_custom_sink_benchmark_run_imports_no_benchmarks(tmp_path: Path) -> None:
    """A custom-sink benchmark run never imports an ``unigym.benchmarks`` module."""
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from tests.integration.test_custom_sink_benchmark_imports import "
        "build_and_run_benchmark_custom_sink\n"
        "build_and_run_benchmark_custom_sink(Path(sys.argv[1]))\n"
        "leaked = sorted(m for m in sys.modules if m.startswith('unigym.benchmarks'))\n"
        "assert not leaked, leaked\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.subprocess
def test_custom_sink_benchmark_timeout_rejection_imports_no_benchmarks(tmp_path: Path) -> None:
    """The custom-sink timeout rejection precedes every ``unigym.benchmarks`` import."""
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from tests.integration.test_custom_sink_benchmark_imports import "
        "reject_benchmark_custom_sink_timeout\n"
        "reject_benchmark_custom_sink_timeout(Path(sys.argv[1]))\n"
        "leaked = sorted(m for m in sys.modules if m.startswith('unigym.benchmarks'))\n"
        "assert not leaked, leaked\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
