"""The custom-sink construction data contract: SinkSpec, config and API parsing."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path

import pytest

from unigym._typing import JsonValue
from unigym.actuators.ir import ActuationGroup
from unigym.errors import ConfigError
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run import SinkContext, SinkFactory, SinkSpec
from unigym.run.api import Run, load
from unigym.run.sink import build_custom_sink, resolve_sink_factory
from unigym.run.spec import build_spec
from unigym.runtime.actions import OrderedActionSpec, PolicyBinding
from unigym.values import ValueStore
from unigym.sink import Sink
from unigym.world import World
from tests._support.fakes.external_sink import (
    DirectProviderSink,
    IncompleteRewindProvider,
    NamelessRewindProvider,
    NonCallableDrainSink,
    NonCallableRewindSink,
    NonCallableStopSink,
    PartialSink,
    ProviderSink,
    RecordingSink,
    RewindProvider,
)


def _cfg(**overrides: JsonValue) -> dict[str, JsonValue]:
    cfg: dict[str, JsonValue] = {
        "run_name": "r",
        "seed": 0,
        "env": {"backend": "mujoco", "num_envs": 1},
        "world_base": {"mjcf": "model.xml"},
        "collect": {"steps": 1},
        "articulations": [{"name": "arm", "joints": [{"name": "shoulder"}]}],
        "action_spec": [
            {
                "name": "arm",
                "articulation": "arm",
                "joints": ["shoulder"],
                "target": "position",
                "role": "body",
            }
        ],
        "policies": [{"adapter": "manual", "groups": [["arm", "arm"]]}],
        "observations": {"robot": ["arm"]},
    }
    cfg.update(overrides)
    return cfg


def _args() -> Namespace:
    return Namespace(config="run.yaml")


def _make_sink() -> object:
    """A never-called factory body; only the callable object matters here."""
    raise AssertionError("a sink factory must not be invoked at the spec boundary")


def test_sink_spec_accepts_import_paths_and_callables() -> None:
    """Both declaration forms are validated data, with no import or call."""
    by_path = SinkSpec(factory="acme.sinks:make_sink")
    by_callable = SinkSpec(factory=_make_sink)

    assert by_path.factory == "acme.sinks:make_sink"
    assert by_callable.factory is _make_sink
    assert SinkFactory == Callable[[SinkContext], Sink]


def test_sink_spec_copies_options_into_an_immutable_mapping() -> None:
    """Later mutation of the caller's mapping cannot change the run intent."""
    options: dict[str, JsonValue] = {"target_episodes": 3}
    spec = SinkSpec(factory=_make_sink, options=options)

    options["target_episodes"] = 99

    assert spec.options == {"target_episodes": 3}
    with pytest.raises(TypeError):
        spec.options["target_episodes"] = 1  # type: ignore[index]


def test_sink_context_carries_construction_metadata_and_the_publisher() -> None:
    """The context fields are exactly the accepted plan surface; values is write-only."""
    run_spec = build_spec(_cfg(), _args())
    options: dict[str, JsonValue] = {"report_name": "acme.json"}
    context = SinkContext(
        spec=run_spec,
        mode=run_spec.mode,
        out_dir="data_out/r",
        num_envs=1,
        policy_dt=0.02,
        seed=0,
        action_groups=(),
        run_identity=None,
        options=options,
    )

    assert context.spec is run_spec
    assert (context.mode, context.out_dir, context.num_envs) == ("collect", "data_out/r", 1)
    assert (context.policy_dt, context.seed, context.action_groups) == (0.02, 0, ())
    assert context.run_identity is None
    assert context.options is options
    assert context.values is None
    assert tuple(field.name for field in fields(SinkContext)) == (
        "spec",
        "mode",
        "out_dir",
        "num_envs",
        "policy_dt",
        "seed",
        "action_groups",
        "run_identity",
        "options",
        "values",
    )
    for live_field in ("env", "writer", "logger", "bridge"):
        assert not hasattr(context, live_field)

    store = ValueStore(num_envs=1, device="cpu")
    publisher = store.publisher("sink")
    assert replace(context, values=publisher).values is publisher


def test_resolve_sink_factory_returns_the_callable_unchanged() -> None:
    """An API-supplied callable is never imported or wrapped."""
    assert resolve_sink_factory(SinkSpec(factory=_make_sink), "run.yaml") is _make_sink


def test_resolve_sink_factory_imports_a_dotted_attribute() -> None:
    """A config path imports the target but does not call it."""
    factory = resolve_sink_factory(SinkSpec(factory="unigym.run.sink:SinkSpec"), "run.yaml")
    nested = resolve_sink_factory(SinkSpec(factory="unigym.run.sink:SinkSpec.__init__"), "run.yaml")

    assert factory is SinkSpec
    assert nested is SinkSpec.__init__


def test_build_spec_parses_a_top_level_sink_declaration() -> None:
    """``sink:`` is parsed into construction data; ``sink`` still mirrors mode."""
    cfg = _cfg(
        mode="benchmark",
        sink={"factory": "acme.sinks:make_sink", "options": {"target_episodes": 2}},
    )

    spec = build_spec(cfg, _args())

    assert spec.custom_sink is not None
    assert spec.custom_sink.factory == "acme.sinks:make_sink"
    assert spec.custom_sink.options == {"target_episodes": 2}
    assert spec.sinks == spec.mode == "benchmark"


def test_build_spec_keeps_no_custom_sink_paths_unchanged() -> None:
    """Without ``sink:`` the spec stays inert and ``sinks`` mirrors ``mode``."""
    spec = build_spec(_cfg(mode="both"), _args())

    assert spec.custom_sink is None
    assert spec.sinks == spec.mode == "both"


def test_run_api_accepts_a_sink_spec_string_or_callable() -> None:
    """The facade normalizes all three forms without resolving the factory."""
    from_spec = Run(_cfg(), sink=SinkSpec(factory=_make_sink)).spec
    from_path = Run(_cfg(), sink="acme.sinks:make_sink").spec
    from_callable = Run(_cfg(), sink=_make_sink).spec

    assert from_spec.custom_sink is not None
    assert from_spec.custom_sink.factory is _make_sink
    assert from_path.custom_sink is not None
    assert from_path.custom_sink.factory == "acme.sinks:make_sink"
    assert from_callable.custom_sink is not None
    assert from_callable.custom_sink.factory is _make_sink


def test_run_minimal_accepts_a_sink() -> None:
    """``Run.minimal`` threads the additive keyword into the one spec path."""
    profile = RobotProfile(
        name="tiny",
        joint_sets={"body": ("shoulder",)},
        default_joint_pos={"shoulder": 0.0},
        assets=None,
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
                "joints": ["shoulder"],
                "target": "position",
                "role": "body",
            }
        ]
    )
    policies = (PolicyBinding.from_mapping({"adapter": "manual", "groups": [["tiny", "tiny"]]}),)

    run = Run.minimal(
        world=World.flat(),
        robot=profile,
        action_spec=action_spec,
        policies=policies,
        observations={"robot": ["tiny"]},
        backend="mujoco",
        sink=_make_sink,
    )

    try:
        assert run.spec.custom_sink is not None
        assert run.spec.custom_sink.factory is _make_sink
    finally:
        run.close()


def _write_run_yaml(tmp_path: Path, *, sink_factory: str | None = None) -> Path:
    """Write a small valid run config, optionally declaring ``sink:``."""
    lines = [
        "run_name: r",
        "seed: 0",
        "env: {backend: mujoco, num_envs: 1}",
        "world_base: {mjcf: model.xml}",
        "collect: {steps: 1}",
        "articulations: [{name: arm, joints: [{name: shoulder}]}]",
        "action_spec:",
        "  - {name: arm, articulation: arm, joints: [shoulder],",
        "     target: position, role: body}",
        "observations: {robot: [arm]}",
        "policies:",
        "  - {adapter: manual, groups: [[arm, arm]]}",
    ]
    if sink_factory is not None:
        lines.extend(
            [
                "sink:",
                f"  factory: {sink_factory}",
                "  options: {target_episodes: 2}",
            ]
        )
    path = tmp_path / "run.yaml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_from_config_reads_a_sink_section_into_the_spec(tmp_path: Path) -> None:
    """A YAML ``sink:`` declaration reaches ``Run.spec.custom_sink``."""
    path = _write_run_yaml(tmp_path, sink_factory="acme.config:make_sink")

    run = Run.from_config(path)
    try:
        assert run.spec.custom_sink is not None
        assert run.spec.custom_sink.factory == "acme.config:make_sink"
        assert run.spec.custom_sink.options == {"target_episodes": 2}
    finally:
        run.close()


def test_from_config_and_load_accept_a_sink(tmp_path: Path) -> None:
    """Both lazy loaders share the additive keyword on the public surface."""
    path = _write_run_yaml(tmp_path)

    from_config = Run.from_config(path, sink=_make_sink)
    loaded = load(path, sink="acme.sinks:make_sink")
    try:
        assert from_config.spec.custom_sink is not None
        assert from_config.spec.custom_sink.factory is _make_sink
        assert loaded.spec.custom_sink is not None
        assert loaded.spec.custom_sink.factory == "acme.sinks:make_sink"
    finally:
        from_config.close()
        loaded.close()


def test_explicit_sink_keyword_overrides_a_config_sink(tmp_path: Path) -> None:
    """The API declaration wins over the same config's ``sink:`` section."""
    path = _write_run_yaml(tmp_path, sink_factory="acme.config:make_sink")

    from_config = Run.from_config(path, sink="acme.api:make_sink")
    loaded = load(path, sink=_make_sink)
    try:
        assert from_config.spec.custom_sink is not None
        assert from_config.spec.custom_sink.factory == "acme.api:make_sink"
        assert loaded.spec.custom_sink is not None
        assert loaded.spec.custom_sink.factory is _make_sink
    finally:
        from_config.close()
        loaded.close()


@pytest.mark.parametrize(
    ("sink", "match"),
    [
        ("not-a-mapping", "sink must be a mapping with factory/options"),
        ({"factory": "acme.sinks:make_sink", "extra": 1}, r"sink\.extra is not supported"),
        ({}, r"sink\.factory must be a non-empty"),
        ({"factory": ""}, r"sink\.factory must be a non-empty"),
        ({"factory": 5}, r"sink\.factory=5 must be a non-empty"),
        ({"factory": "bad-path"}, r"must be 'module:callable'"),
        ({"factory": "a:b:c"}, r"must be 'module:callable'"),
        (
            {"factory": "acme.sinks:make_sink", "options": [1]},
            r"sink\.options must be a string-keyed mapping",
        ),
        (
            {"factory": "acme.sinks:make_sink", "options": {1: "x"}},
            r"sink\.options must be a string-keyed mapping",
        ),
    ],
    ids=[
        "not-a-mapping",
        "unknown-key",
        "factory-missing",
        "factory-empty",
        "factory-non-string",
        "factory-no-colon",
        "factory-two-colons",
        "options-not-a-mapping",
        "options-non-string-key",
    ],
)
def test_build_spec_rejects_invalid_sink_sections(sink: JsonValue, match: str) -> None:
    """Every config-shape error names the sink field and the accepted form."""
    with pytest.raises(ConfigError, match=match) as info:
        build_spec(_cfg(sink=sink), _args())

    assert "run.yaml" in str(info.value)


def test_build_spec_rejects_a_callable_sink_factory_from_config() -> None:
    """A config file declares only a ``module:callable`` string; callables are API-only."""
    cfg = _cfg()
    cfg["sink"] = {"factory": _make_sink}  # type: ignore[assignment]

    with pytest.raises(ConfigError, match="must be a non-empty 'module:callable' string") as info:
        build_spec(cfg, _args())

    message = str(info.value)
    assert "run.yaml" in message
    assert "or a callable" not in message


def test_build_spec_rejects_non_string_sink_keys() -> None:
    """Mixed key types fail with a path-bearing ConfigError, not a raw TypeError."""
    cfg = _cfg()
    cfg["sink"] = {"factory": "acme.sinks:make_sink", 1: "x"}  # type: ignore[assignment]

    with pytest.raises(ConfigError, match="sink keys must be strings") as info:
        build_spec(cfg, _args())

    assert "run.yaml" in str(info.value)


def test_build_spec_rejects_benchmark_sink_with_a_hint() -> None:
    """A custom sink is mode-independent; ``benchmark.sink`` has one home."""
    cfg = _cfg(benchmark={"sink": {"factory": "acme.sinks:make_sink"}})

    with pytest.raises(ConfigError) as info:
        build_spec(cfg, _args())

    message = str(info.value)
    assert "run.yaml" in message
    assert "benchmark.sink is not supported" in message
    assert "top-level sink:" in message


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        ("no_such_module_xyz:make_sink", "could not be imported"),
        ("unigym.run.sink:not_there", "has no attribute 'not_there'"),
        ("unigym.run.sink:__doc__", "is not callable"),
    ],
    ids=["import-failure", "missing-attribute", "not-callable"],
)
def test_resolve_sink_factory_reports_teaching_errors(factory: str, match: str) -> None:
    """Import/attribute/callable failures name the config path and value."""
    with pytest.raises(ConfigError, match=match) as info:
        resolve_sink_factory(SinkSpec(factory=factory), "run.yaml")

    assert "run.yaml" in str(info.value)
    assert factory in str(info.value)


def test_sink_spec_rejects_a_non_callable_non_string_factory() -> None:
    """The direct data boundary shares the config error vocabulary."""
    with pytest.raises(ConfigError, match="must be a non-empty"):
        SinkSpec(factory=5)  # type: ignore[arg-type]


def test_run_api_rejects_an_unsupported_sink_input() -> None:
    """The facade names the accepted forms instead of coercing silently."""
    with pytest.raises(TypeError, match="accepts a SinkSpec"):
        Run(_cfg(), sink=123)  # type: ignore[arg-type]


def test_run_api_rejects_a_malformed_sink_path() -> None:
    """A malformed API path fails at SinkSpec creation, before any build."""
    with pytest.raises(ConfigError, match="must be 'module:callable'"):
        Run(_cfg(), sink="bad-path")


def _context() -> SinkContext:
    spec = build_spec(_cfg(), _args())
    return SinkContext(
        spec=spec,
        mode=spec.mode,
        out_dir="data_out/r",
        num_envs=1,
        policy_dt=0.02,
        seed=0,
        action_groups=(),
        run_identity=None,
        options={},
    )


def test_build_custom_sink_calls_the_factory_once_and_returns_its_sink() -> None:
    context = _context()
    sink = RecordingSink()
    calls: list[SinkContext] = []

    def factory(ctx: SinkContext) -> RecordingSink:
        calls.append(ctx)
        return sink

    assert build_custom_sink(factory, context) is sink
    assert calls == [context]


def test_build_custom_sink_reports_missing_required_callbacks() -> None:
    def factory(ctx: SinkContext) -> PartialSink:
        del ctx
        return PartialSink()

    with pytest.raises(TypeError, match="missing callbacks") as info:
        build_custom_sink(factory, _context())  # type: ignore[arg-type]

    message = str(info.value)
    assert "PartialSink" in message
    assert "on_episode_end" in message
    assert "on_step_end" in message
    assert "on_run_end" in message


@pytest.mark.parametrize(
    ("sink", "field"),
    [
        (NonCallableStopSink(), "stop_when"),
        (NonCallableRewindSink(), "on_rewind"),
        (NonCallableDrainSink(), "drain_terminations"),
    ],
    ids=["stop-when", "on-rewind", "drain-terminations"],
)
def test_build_custom_sink_requires_callable_optional_capabilities(sink, field) -> None:
    def factory(ctx: SinkContext) -> object:
        del ctx
        return sink

    with pytest.raises(TypeError, match=rf"{field} must be callable"):
        build_custom_sink(factory, _context())  # type: ignore[arg-type]


def test_build_custom_sink_accepts_a_valid_rewind_provider() -> None:
    sink = ProviderSink(RewindProvider())

    def factory(ctx: SinkContext) -> ProviderSink:
        del ctx
        return sink

    assert build_custom_sink(factory, _context()) is sink
    assert sink.supplier_calls == 0


def test_build_custom_sink_accepts_a_direct_rewind_provider_shape() -> None:
    sink = DirectProviderSink(RewindProvider())

    def factory(ctx: SinkContext) -> DirectProviderSink:
        del ctx
        return sink

    assert build_custom_sink(factory, _context()) is sink


@pytest.mark.parametrize(
    ("provider", "missing"),
    [
        (IncompleteRewindProvider(), "apply_rewind_state"),
        (NamelessRewindProvider(), "name"),
    ],
    ids=["missing-apply", "missing-name"],
)
def test_build_custom_sink_reports_an_invalid_rewind_provider(provider, missing) -> None:
    def factory(ctx: SinkContext) -> DirectProviderSink:
        del ctx
        return DirectProviderSink(provider)

    with pytest.raises(TypeError, match="rewind_provider is missing") as info:
        build_custom_sink(factory, _context())

    assert missing in str(info.value)
