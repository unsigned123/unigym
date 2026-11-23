"""An external Sink drives a real tiny-arm run through the public seam.

The sink is built by a factory outside core code, injected by API and by config,
and sees the full runner lifecycle.  The manual adapter has no policy forward,
so the full loop runs as ``ExternalRunner.run`` (the same ``SyncRunner`` tick
engine with a caller action function), never a hand-rolled simulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from unigym.actuators.ir import ActuationGroup
from unigym.assets.asset_ref import AssetRef
from unigym.pack_api import discover
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run.api import Run
from unigym.run.sink import SinkSpec
from unigym.runtime.actions import NamedAction, OrderedActionSpec, PolicyBinding
from unigym.world import World
from tests._support.fakes.external_sink import ExternalSink, make_external_sink

if TYPE_CHECKING:
    from unigym.run.instance import RunInstance
    from unigym.runner.base import Runner

TINY_ARM = Path(__file__).resolve().parents[1] / "_support" / "data" / "tiny_arm.xml"
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
EPISODE_LENGTH_S = 0.04
STEPS = 20


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


def _floor_world(tmp_path: Path) -> Path:
    path = tmp_path / "world.xml"
    path.write_text(
        '<mujoco model="floor"><worldbody>'
        '<geom name="world_floor" type="plane" size="2 2 0.1"/>'
        "</worldbody></mujoco>",
        encoding="utf-8",
    )
    return path


def _zero_actions(_t: int) -> tuple[NamedAction, ...]:
    return (NamedAction.zeros(1, "tiny", JOINTS, "position"),)


def _write_tiny_pack(root: Path) -> None:
    """Write one content pack whose robot profile matches the tiny-arm MJCF."""
    robot = {
        "schema": "unigym.robot/1",
        "name": "tiny_pack.tiny_arm",
        "joint_sets": {"body": list(JOINTS)},
        "default_joint_pos": {"shoulder": 0.1, "elbow": -0.2},
        "base_body": "base",
        "foot_bodies": [],
        "body_sets": {},
        "body_bindings": {"base": {"mjcf": "base"}},
        "extra_action_sets": {},
        "actuation": [
            {
                "name": "body",
                "law": "standard_pd",
                "joint_set": "body",
                "gains": {"kp": 20.0, "kd": 1.0},
            }
        ],
    }
    robot_path = root / "robots" / "tiny_arm.yaml"
    robot_path.parent.mkdir(parents=True, exist_ok=True)
    robot_path.write_text(yaml.safe_dump(robot, sort_keys=False), encoding="utf-8")
    manifest = {
        "schema": "unigym.pack/1",
        "name": "tiny_pack",
        "version": "1.0.0",
        "api": 1,
        "trust": "content",
        "provides": {
            "providers": [
                {
                    "id": "tiny_pack.tiny_arm",
                    "kind": "robot",
                    "resource": "robots/tiny_arm.yaml",
                }
            ]
        },
    }
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )


def _external_sink(instance: RunInstance) -> ExternalSink:
    """The custom sink is its own provider, so discovery proves it was mounted."""
    providers = [
        provider
        for provider in instance.rewind_providers
        if getattr(provider, "name", None) == ExternalSink.name
    ]
    assert len(providers) == 1, [getattr(p, "name", None) for p in instance.rewind_providers]
    sink = providers[0]
    assert isinstance(sink, ExternalSink)
    return sink


def _assert_lifecycle(sink: ExternalSink, runner: Runner) -> None:
    assert sink.step_calls > 0
    assert sink.step_end_calls == sink.step_calls
    assert sink.episode_end_calls == 2
    assert sink.episodes_completed == 2
    assert sink.episode_env_ids == [(0,), (0,)]
    assert sink.run_end_calls == 1
    assert sink.stop_polls >= 1
    assert sink.stop_when() is True
    # The public boundary view is the observation surface; the retired
    # bundle env key is absent and no view is exposed at on_step_end.
    assert sink.view_seen is True
    assert sink.env_key_seen is False
    assert sink.step_end_view_none is True
    # The loop stopped at the tick that closed episode 2, not at the budget.
    assert sink.last_episode_end_tick is not None
    assert sink.last_episode_end_tick == runner.ticks_done - 1
    assert runner.ticks_done < STEPS


def test_collect_custom_sink_api_runs_the_full_lifecycle(tmp_path: Path) -> None:
    """An API-injected sink sees every callback and stops after two episodes."""
    run = Run(
        world=World.base(mjcf=_floor_world(tmp_path)),
        robot=_profile(),
        backend="mujoco",
        observations={"robot": ["tiny"]},
        action_spec=ACTION_SPEC,
        policies=POLICIES,
        mode="collect",
        steps=STEPS,
        episode_length_s=EPISODE_LENGTH_S,
        inspect=True,
        sink=SinkSpec(factory=make_external_sink, options={"target_episodes": 2}),
    )
    try:
        instance = run.build(inspect=True)
        declaration = instance.spec.custom_sink
        assert declaration is not None
        assert declaration.factory is make_external_sink
        sink = _external_sink(instance)

        runner = run.runner("external", action_fn=_zero_actions)
        runner.run(stop_when=instance.stop_when)

        _assert_lifecycle(sink, runner)
    finally:
        run.close()


def test_collect_custom_sink_config_factory_runs_the_full_lifecycle(
    tmp_path: Path, tiny_mjcf: Path
) -> None:
    """A YAML ``sink.factory`` path builds the same sink outside core code."""
    pack_root = tmp_path / "tiny_pack"
    _write_tiny_pack(pack_root)
    runtime = discover([pack_root])
    config_path = tmp_path / "run.yaml"
    config = {
        "run_name": "external_sink_config",
        "seed": 0,
        "mode": "collect",
        "env": {
            "backend": "mujoco",
            "num_envs": 1,
            "episode_length_s": EPISODE_LENGTH_S,
            "decimation": 4,
            "sim_dt": 0.005,
        },
        "collect": {"steps": STEPS, "shard_steps": 100},
        "world_base": {"mjcf": str(tiny_mjcf)},
        "adapter": {"name": "manual"},
        "command": {"name": "fixed", "dim": 0, "values": []},
        "articulations": [{"name": "tiny", "params": {"profile": "tiny_pack.tiny_arm"}}],
        "action_spec": [
            {
                "name": "tiny",
                "articulation": "tiny",
                "joints": list(JOINTS),
                "target": "position",
                "role": "body",
            }
        ],
        "policies": [{"adapter": "manual", "groups": [["tiny", "tiny"]]}],
        "observations": {"robot": ["tiny"]},
        "sink": {
            "factory": "tests._support.fakes.external_sink:make_external_sink",
            "options": {"target_episodes": 2},
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    run = Run.from_config(
        config_path,
        pack_runtime=runtime,
        discover_runtime=False,
        inspect=True,
    )
    try:
        instance = run.build(inspect=True)
        declaration = instance.spec.custom_sink
        assert declaration is not None
        assert declaration.factory == "tests._support.fakes.external_sink:make_external_sink"
        sink = _external_sink(instance)

        runner = run.runner("external", action_fn=_zero_actions)
        runner.run(stop_when=instance.stop_when)

        _assert_lifecycle(sink, runner)
    finally:
        run.close()


def test_benchmark_custom_sink_builds_without_a_pack_runtime(tmp_path: Path) -> None:
    """A custom benchmark sink owns tasks, so the mode needs no pack runtime."""
    run = Run(
        world=World.base(mjcf=_floor_world(tmp_path)),
        robot=_profile(),
        backend="mujoco",
        observations={"robot": ["tiny"]},
        action_spec=ACTION_SPEC,
        policies=POLICIES,
        mode="benchmark",
        steps=STEPS,
        episode_length_s=EPISODE_LENGTH_S,
        inspect=True,
        sink=SinkSpec(factory=make_external_sink, options={"target_episodes": 2}),
        benchmark={"tasks": [], "min_episodes": 1, "max_steps": STEPS, "timeout_steps": 2},
        pack_runtime=None,
        discover_runtime=False,
    )
    try:
        instance = run.build(inspect=True)
        assert instance.spec.pack_runtime is None
        assert instance.env.run_identity is not None
        assert getattr(instance.env, "timeout_steps", None) == 2
        sink = _external_sink(instance)

        runner = run.runner("external", action_fn=_zero_actions)
        runner.run(stop_when=instance.stop_when)

        _assert_lifecycle(sink, runner)
    finally:
        run.close()
