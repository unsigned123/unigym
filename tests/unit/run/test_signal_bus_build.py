"""The build seam creates one run SignalBus and binds it once."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from tests._support.fakes.command_sampler import RecordingSampler
from tests._support.fakes.external_sink import SignalPublishingSink, make_signal_publishing_sink
from unigym.actuators.ir import ActuationGroup
from unigym.assets.asset_ref import AssetRef
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run.api import Run
from unigym.run.sink import SinkSpec
from unigym.runner import make_runner
from unigym.runtime.actions import OrderedActionSpec, PolicyBinding
from unigym.signals import SignalBus
from unigym.world import World

if TYPE_CHECKING:
    from unigym.run.instance import RunInstance

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


def _run(tmp_path: Path, *, custom_sink: bool = False) -> Run:
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
        sink=SinkSpec(factory=make_signal_publishing_sink) if custom_sink else None,
    )


def _signal_sink(instance: RunInstance) -> SignalPublishingSink:
    """The sink is its own provider; discovery proves the built instance."""
    providers = [
        provider
        for provider in instance.rewind_providers
        if getattr(provider, "name", None) == SignalPublishingSink.name
    ]
    assert len(providers) == 1, [getattr(p, "name", None) for p in instance.rewind_providers]
    sink = providers[0]
    assert isinstance(sink, SignalPublishingSink)
    return sink


def test_build_creates_one_bus_and_binds_it_to_the_sampler(tmp_path):
    """The instance stores the bus and hands the same view to the sampler."""
    run = _run(tmp_path)
    try:
        instance = run.build(inspect=True)
        sampler = instance.sampler

        assert isinstance(sampler, RecordingSampler)
        assert isinstance(instance.signals, SignalBus)
        assert instance.signals.num_envs == instance.env.num_envs
        assert sampler.signal_views == [instance.signals]

        instance.signals.publish("user/value", torch.zeros(instance.signals.num_envs))
        assert sampler.signal_views[0].get("user/value") is not None
    finally:
        run.close()


def test_the_bus_joins_rewind_providers_exactly_once(tmp_path):
    """The instance-owned bus is a single named rewind provider."""
    run = _run(tmp_path)
    try:
        instance = run.build(inspect=True)

        names = [provider.name for provider in instance.rewind_providers]
        assert names.count("signals") == 1
        provider = next(p for p in instance.rewind_providers if p.name == "signals")
        assert provider is instance.signals
    finally:
        run.close()


def test_two_runners_over_one_instance_do_not_rebind_signals(tmp_path):
    """Runner reuse never calls bind_signals a second time."""
    run = _run(tmp_path)
    try:
        instance = run.build(inspect=True)
        sampler = instance.sampler
        assert isinstance(sampler, RecordingSampler)

        first = make_runner(instance, "sync")
        second = make_runner(instance, "sync")

        assert first is not second
        assert sampler.signal_views == [instance.signals]
    finally:
        run.close()


def test_custom_sink_factory_receives_the_instance_bus(tmp_path):
    """The construction context publisher is the run's one bus, not a copy."""
    run = _run(tmp_path, custom_sink=True)
    try:
        instance = run.build(inspect=True)
        sampler = instance.sampler
        sink = _signal_sink(instance)

        assert sink.publisher is instance.signals
        assert isinstance(sampler, RecordingSampler)
        assert sampler.signal_views == [instance.signals]
        names = [provider.name for provider in instance.rewind_providers]
        assert names.count("signals") == 1
        assert names.count(SignalPublishingSink.name) == 1
    finally:
        run.close()


def test_sink_publication_and_rewind_share_one_bus(tmp_path):
    """A value published through the sink face is restored by the bus provider."""
    run = _run(tmp_path, custom_sink=True)
    try:
        instance = run.build(inspect=True)
        sink = _signal_sink(instance)
        provider = next(p for p in instance.rewind_providers if p.name == "signals")
        assert sink.publisher is instance.signals
        assert provider is instance.signals

        sink.publisher.publish(sink.signal, torch.tensor([[1.0]]))
        state = provider.rewind_state()
        sink.publisher.publish(sink.signal, torch.tensor([[2.0]]))

        provider.apply_rewind_state(state)

        restored = instance.signals.get(sink.signal)
        assert restored is not None
        assert torch.equal(restored, torch.tensor([[1.0]]))
    finally:
        run.close()
