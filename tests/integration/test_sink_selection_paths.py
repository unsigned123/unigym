"""Every selected-sink path builds and runs a tiny-arm real backend run.

A run whose selection is absent, the built-in collect selection, a custom
``sinks=`` selection and the ``sink:`` shorthand must all build and step a
real MuJoCo env.  Each case runs in a subprocess so the backend lifecycle
cannot leak between parametrized cases; no code pack is activated.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from unigym.run import CollectSinkSpec, SinkSpec
    from unigym.run.api import Run

_TINY_ARM = Path(__file__).resolve().parents[1] / "_support" / "data" / "tiny_arm.xml"
_FLOOR_XML = (
    '<mujoco model="floor"><worldbody>'
    '<geom name="world_floor" type="plane" size="2 2 0.1"/>'
    "</worldbody></mujoco>"
)
_JOINTS = ("shoulder", "elbow")
_STEPS = 20


def _run_declaration(
    world_dir: Path,
    *,
    sink: SinkSpec | None = None,
    sinks: Sequence[CollectSinkSpec | SinkSpec] | None = None,
) -> Run:
    """Build a tiny-arm declaration without running it."""
    from unigym.actuators.ir import ActuationGroup
    from unigym.assets.asset_ref import AssetRef
    from unigym.robots.base import BodyBinding, RobotProfile
    from unigym.run.api import Run
    from unigym.runtime.actions import OrderedActionSpec, PolicyBinding
    from unigym.world import World

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
        steps=_STEPS,
        inspect=True,
        sink=sink,
        sinks=sinks,
        discover_runtime=False,
    )


def _run_selected_sink(run: Run) -> None:
    from unigym.runtime.actions import NamedAction

    def zero_actions(_t: int) -> tuple[NamedAction, ...]:
        return (NamedAction.zeros(1, "tiny", _JOINTS, "position"),)

    try:
        instance = run.build(inspect=True)
        runner = run.runner("external", action_fn=zero_actions)
        runner.run(stop_when=instance.stop_when)
    finally:
        run.close()


def build_and_run_default_collect(world_dir: Path) -> None:
    """Run the absent-selection default (one built-in collect sink)."""
    _run_selected_sink(_run_declaration(world_dir))


def build_and_run_collect_selection(world_dir: Path) -> None:
    """Run one tiny-arm collect selection (``sinks=[CollectSinkSpec()]``)."""
    from unigym.run import CollectSinkSpec

    _run_selected_sink(_run_declaration(world_dir, sinks=[CollectSinkSpec()]))


def build_and_run_custom_selection(world_dir: Path) -> None:
    """Run one tiny-arm custom selection through the ``sinks=`` path."""
    from unigym.run import SinkSpec

    from tests._support.fakes.external_sink import make_external_sink

    _run_selected_sink(
        _run_declaration(
            world_dir,
            sinks=[SinkSpec(factory=make_external_sink, options={"target_episodes": 2})],
        )
    )


def build_and_run_custom_shorthand(world_dir: Path) -> None:
    """Run one tiny-arm custom sink through the ``sink=`` shorthand."""
    from unigym.run import SinkSpec

    from tests._support.fakes.external_sink import make_external_sink

    _run_selected_sink(
        _run_declaration(
            world_dir,
            sink=SinkSpec(factory=make_external_sink, options={"target_episodes": 2}),
        )
    )


def _run_helper(helper: str, world_dir: Path) -> None:
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from tests.integration.test_sink_selection_paths import "
        f"{helper}\n"
        f"{helper}(Path(sys.argv[1]))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code, str(world_dir)], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.subprocess
@pytest.mark.parametrize(
    "helper",
    [
        "build_and_run_default_collect",
        "build_and_run_collect_selection",
        "build_and_run_custom_selection",
        "build_and_run_custom_shorthand",
    ],
    ids=["default", "collect", "custom", "shorthand"],
)
def test_selected_sink_paths_build_and_run(tmp_path: Path, helper: str) -> None:
    """Every selection path builds and steps the tiny-arm run."""
    _run_helper(helper, tmp_path)
