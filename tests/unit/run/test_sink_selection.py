"""Mode-aware sink composition and stop-predicate wiring for custom sinks."""

from __future__ import annotations

from pathlib import Path

import pytest

from unigym.actuators.ir import ActuationGroup
from unigym.assets.asset_ref import AssetRef
from unigym.errors import ConfigError
from unigym.packs.runtime import discover
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run.api import Run
from unigym.run.build import (
    _combine_stop,
    _compose_primary_sink,
    _metrics_source_for,
    _sink_stop_when,
)
from unigym.run.sink import SinkSpec
from unigym.runner.sinks import CollectSink, FanOutSink
from unigym.runtime.actions import OrderedActionSpec, PolicyBinding
from unigym.world import World
from tests._support.fakes.external_sink import (
    MetricsFeedSink,
    NonCallableMetricsSink,
    RecordingSink,
    StopPollingSink,
)

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


@pytest.mark.parametrize(
    ("mode", "use_custom", "first", "second"),
    [
        ("collect", False, "collect", None),
        ("collect", True, "collect", "custom"),
        ("benchmark", False, "session", None),
        ("benchmark", True, "custom", None),
        ("both", False, "collect", "session"),
        ("both", True, "collect", "custom"),
    ],
    ids=[
        "collect/default",
        "collect/custom",
        "benchmark/default",
        "benchmark/custom",
        "both/default",
        "both/custom",
    ],
)
def test_primary_sink_selection_matrix(
    mode: str, use_custom: bool, first: str, second: str | None
) -> None:
    """The 3x2 mode/custom matrix composes exactly the planned sinks."""
    parts = {
        "collect": RecordingSink("collect"),
        "session": RecordingSink("session"),
        "custom": RecordingSink("custom"),
    }
    custom_sink = parts["custom"] if use_custom else None

    primary = _compose_primary_sink(
        mode,
        collect_sink=parts["collect"],
        eval_session=parts["session"],
        custom_sink=custom_sink,
    )

    if second is None:
        assert primary is parts[first]
    else:
        assert isinstance(primary, FanOutSink)
        assert primary.sinks == [parts[first], parts[second]]


def test_primary_sink_selection_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="unsupported run mode"):
        _compose_primary_sink(
            "unknown",
            collect_sink=RecordingSink("collect"),
            eval_session=None,
            custom_sink=None,
        )


def test_fanout_stop_when_ors_children_and_ignores_plain_sinks() -> None:
    """Children without the capability contribute nothing to the OR."""
    first = StopPollingSink(False)
    second = StopPollingSink(True)
    fanout = FanOutSink(RecordingSink("plain"), first, second)

    assert fanout.stop_when() is True
    assert first.polls == 1
    assert second.polls == 1


def test_fanout_stop_when_short_circuits_at_the_first_true_child() -> None:
    first = StopPollingSink(True)
    second = StopPollingSink(False)

    assert FanOutSink(first, second).stop_when() is True
    assert second.polls == 0


def test_fanout_stop_when_participates_through_a_nested_fanout() -> None:
    """The composite does not flatten children; nested fan-out recurses itself."""
    inner = FanOutSink(StopPollingSink(True))

    assert FanOutSink(RecordingSink("plain"), inner).stop_when() is True


def test_sink_stop_when_detects_the_optional_capability_only() -> None:
    assert _sink_stop_when(None) is None
    assert _sink_stop_when(RecordingSink("plain")) is None

    stopper = StopPollingSink(False)
    predicate = _sink_stop_when(stopper)

    assert predicate is not None
    assert predicate() is False
    assert stopper.polls == 1


def test_combine_stop_keeps_the_single_predicate_unchanged() -> None:
    predicate = StopPollingSink(False).stop_when

    assert _combine_stop(None, None) is None
    assert _combine_stop(predicate, None) is predicate
    assert _combine_stop(None, predicate) is predicate


def test_combine_stop_ors_in_argument_order_with_short_circuit() -> None:
    calls: list[str] = []

    def first() -> bool:
        calls.append("first")
        return True

    def second() -> bool:
        calls.append("second")
        return False

    combined = _combine_stop(first, second)

    assert combined is not None
    assert combined() is True
    assert calls == ["first"]


def test_combine_stop_evaluates_the_second_predicate_when_needed() -> None:
    calls: list[str] = []

    def first() -> bool:
        calls.append("first")
        return False

    def second() -> bool:
        calls.append("second")
        return True

    combined = _combine_stop(first, second)

    assert combined is not None
    assert combined() is True
    assert calls == ["first", "second"]


def test_metrics_source_requires_callable_feed_members() -> None:
    assert _metrics_source_for(None) is None
    assert _metrics_source_for(RecordingSink("plain")) is None
    assert _metrics_source_for(NonCallableMetricsSink()) is None

    feed = MetricsFeedSink()
    assert _metrics_source_for(feed) is feed


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


def _run(tmp_path: Path, *, mode: str, sink: SinkSpec | None = None) -> Run:
    runtime = discover([]) if mode != "collect" else None
    return Run(
        world=World.base(mjcf=_floor_world(tmp_path)),
        robot=_profile(),
        backend="mujoco",
        observations={"robot": ["tiny"]},
        action_spec=ACTION_SPEC,
        policies=POLICIES,
        steps=1,
        inspect=True,
        sink=sink,
        mode=mode,
        benchmark={"tasks": [], "min_episodes": 1, "max_steps": 10, "timeout_steps": 5},
        pack_runtime=runtime,
        discover_runtime=False,
    )


def _primary_sink(instance) -> object:
    """Strip the unchanged timeline/bridge tail to reach the primary composition."""
    outer = instance.sink
    assert isinstance(outer, FanOutSink)
    timeline_chain = outer.sinks[0]
    assert isinstance(timeline_chain, FanOutSink)
    return timeline_chain.sinks[0]


def test_build_injects_a_custom_sink_after_collect(tmp_path: Path) -> None:
    contexts = []
    custom_sink = StopPollingSink(True, label="custom")

    def factory(ctx):
        contexts.append(ctx)
        return custom_sink

    run = _run(
        tmp_path,
        mode="collect",
        sink=SinkSpec(factory=factory, options={"target_episodes": 2}),
    )
    try:
        instance = run.build(inspect=True)
        spec = run.spec
        assert len(contexts) == 1
        ctx = contexts[0]
        assert ctx.spec is spec
        assert ctx.mode == "collect"
        assert ctx.options == {"target_episodes": 2}
        assert ctx.run_identity is None

        primary = _primary_sink(instance)
        assert isinstance(primary, FanOutSink)
        assert isinstance(primary.sinks[0], CollectSink)
        assert primary.sinks[1] is custom_sink
        assert instance.stop_when() is True
    finally:
        run.close()


def test_build_benchmark_custom_sink_replaces_the_builtin_session(tmp_path: Path) -> None:
    contexts = []
    custom_sink = StopPollingSink(False, label="custom")

    def factory(ctx):
        contexts.append(ctx)
        return custom_sink

    run = _run(tmp_path, mode="benchmark", sink=SinkSpec(factory=factory))
    try:
        instance = run.build(inspect=True)
        assert len(contexts) == 1
        ctx = contexts[0]
        assert ctx.mode == "benchmark"
        assert ctx.run_identity is not None
        assert ctx.run_identity is getattr(instance.env, "run_identity", None)
        assert getattr(instance.env, "timeout_steps", None) == 5

        assert _primary_sink(instance) is custom_sink
        assert instance.stop_when() is False
    finally:
        run.close()


def test_build_both_custom_sink_keeps_collect_first(tmp_path: Path) -> None:
    contexts = []
    custom_sink = StopPollingSink(True, label="custom")

    def factory(ctx):
        contexts.append(ctx)
        return custom_sink

    run = _run(tmp_path, mode="both", sink=SinkSpec(factory=factory))
    try:
        instance = run.build(inspect=True)
        assert len(contexts) == 1
        ctx = contexts[0]
        assert ctx.mode == "both"
        assert ctx.run_identity is not None
        assert getattr(instance.env, "timeout_steps", None) == 5

        primary = _primary_sink(instance)
        assert isinstance(primary, FanOutSink)
        assert isinstance(primary.sinks[0], CollectSink)
        assert primary.sinks[1] is custom_sink
        assert instance.stop_when() is True
    finally:
        run.close()


def test_build_rejects_an_invalid_factory_before_constructing_the_env(tmp_path: Path) -> None:
    """The factory resolves before the missing world asset can reach env build."""
    run = Run(
        world=World.base(mjcf=tmp_path / "missing_world.xml"),
        robot=_profile(),
        backend="mujoco",
        observations={"robot": ["tiny"]},
        action_spec=ACTION_SPEC,
        policies=POLICIES,
        steps=1,
        inspect=True,
        sink="no_such_unigym_sink_factory:make",
        discover_runtime=False,
    )
    try:
        with pytest.raises(ConfigError, match="sink.factory.*could not be imported"):
            run.build(inspect=True)
    finally:
        run.close()


def test_build_keeps_the_collect_sink_when_no_custom_sink_is_configured(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, mode="collect")
    try:
        instance = run.build(inspect=True)
        assert isinstance(_primary_sink(instance), CollectSink)
        assert instance.stop_when() is False
    finally:
        run.close()
