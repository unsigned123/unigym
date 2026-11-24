"""Build wiring: the value store exists before watcher attach and is shared."""

from __future__ import annotations

from pathlib import Path

from tests._support.fakes.command_sampler import RecordingSampler
from unigym.actuators.ir import ActuationGroup
from unigym.assets.asset_ref import AssetRef
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run.api import Run
from unigym.runtime.actions import OrderedActionSpec, PolicyBinding
from unigym.world import World

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
SAMPLER_FACTORY = "tests._support.fakes.command_sampler:make_sampler"


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


def _run(tmp_path: Path) -> Run:
    return Run(
        world=World.base(mjcf=_floor_world(tmp_path)),
        robot=_profile(),
        backend="mujoco",
        observations={"robot": ["tiny"]},
        action_spec=ACTION_SPEC,
        policies=POLICIES,
        steps=1,
        inspect=True,
        command={"module": SAMPLER_FACTORY, "dim": 1, "params": {"seats": 2}},
    )


def test_watcher_engine_and_sampler_share_the_instance_store(tmp_path):
    """The build attaches watchers without a signal transport and binds values."""
    run = _run(tmp_path)
    try:
        instance = run.build(inspect=True)

        assert not hasattr(instance, "signals")
        providers = {provider.name: provider for provider in instance.rewind_providers}
        assert "signals" not in providers
        assert providers["values"] is instance.values
        watchers = providers["watchers"]
        assert getattr(watchers, "signals") is None

        sampler = instance.sampler
        assert isinstance(sampler, RecordingSampler)
        assert sampler.value_views == [instance.values]
    finally:
        run.close()
