"""The explicit ``sinks:`` selection surface: config, API and validation."""

from __future__ import annotations

from argparse import Namespace

import pytest

from unigym._typing import JsonValue
from unigym.errors import ConfigError
from unigym.run import CollectSinkSpec, SinkSpec
from unigym.run.api import Run
from unigym.run.spec import build_spec


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


def test_absent_sinks_leaves_the_default_collect_selection_to_the_builder() -> None:
    spec = build_spec(_cfg(), _args())

    assert spec.sinks is None
    assert spec.custom_sink is None


def test_config_sinks_parse_to_an_ordered_tuple_of_declarations() -> None:
    spec = build_spec(
        _cfg(
            sinks=[
                {"kind": "collect"},
                {
                    "name": "bench",
                    "factory": "acme.sinks:make_sink",
                    "options": {"target_episodes": 2},
                },
            ]
        ),
        _args(),
    )

    assert isinstance(spec.sinks, tuple)
    assert isinstance(spec.sinks[0], CollectSinkSpec)
    assert (spec.sinks[0].kind, spec.sinks[0].name) == ("collect", "collect")
    assert isinstance(spec.sinks[1], SinkSpec)
    assert spec.sinks[1].name == "bench"
    assert spec.sinks[1].options == {"target_episodes": 2}
    assert spec.custom_sink is None


def test_single_custom_entry_may_omit_its_name() -> None:
    spec = build_spec(_cfg(sinks=[{"factory": "acme.sinks:make_sink"}]), _args())

    assert spec.sinks is not None
    assert len(spec.sinks) == 1
    assert isinstance(spec.sinks[0], SinkSpec)
    assert spec.sinks[0].name is None


def test_collect_entry_name_can_be_overridden() -> None:
    spec = build_spec(_cfg(sinks=[{"kind": "collect", "name": "recorded"}]), _args())

    assert spec.sinks is not None
    assert spec.sinks[0] == CollectSinkSpec(name="recorded")


def test_sink_shorthand_still_populates_the_custom_sink_slot() -> None:
    spec = build_spec(_cfg(sink={"factory": "acme.sinks:make_sink"}), _args())

    assert spec.sinks is None
    assert spec.custom_sink is not None
    assert spec.custom_sink.factory == "acme.sinks:make_sink"


@pytest.mark.parametrize(
    "sink",
    [{"factory": "acme.sinks:make_sink"}, None],
    ids=["mapping", "null"],
)
def test_declaring_both_sink_and_sinks_is_an_error(sink: JsonValue) -> None:
    with pytest.raises(ConfigError, match="not both"):
        build_spec(_cfg(sink=sink, sinks=[{"kind": "collect"}]), _args())


def test_config_mode_and_sinks_rejects_the_removed_mode() -> None:
    with pytest.raises(ConfigError, match="top-level `mode:` was removed"):
        build_spec(_cfg(mode="collect", sinks=[{"kind": "collect"}]), _args())


def test_api_mode_and_sinks_rejects_the_removed_mode() -> None:
    with pytest.raises(ConfigError, match="top-level `mode:` was removed"):
        Run(_cfg(), mode="collect", sinks=[CollectSinkSpec()]).spec


def test_api_sinks_with_a_config_mode_rejects_the_removed_mode() -> None:
    with pytest.raises(ConfigError, match="top-level `mode:` was removed"):
        Run(_cfg(mode="collect"), sinks=[CollectSinkSpec()]).spec


def test_only_one_builtin_collect_entry_is_allowed() -> None:
    with pytest.raises(ConfigError, match="only one collect entry is allowed"):
        build_spec(
            _cfg(
                sinks=[
                    {"kind": "collect", "name": "first"},
                    {"kind": "collect", "name": "second"},
                ]
            ),
            _args(),
        )

    with pytest.raises(ConfigError, match="only one collect entry is allowed"):
        Run(
            _cfg(),
            sinks=[CollectSinkSpec(name="first"), CollectSinkSpec(name="second")],
        )


@pytest.mark.parametrize(
    ("sinks", "match"),
    [
        (None, "sinks must be a list"),
        ({}, "sinks must be a list"),
        ([], "at least one entry"),
        ([{"kind": "benchmark"}], "only core built-in kind"),
        (
            [{"kind": "collect", "factory": "acme.sinks:make_sink"}],
            "both 'kind' and 'factory'",
        ),
        ([{"name": "x"}], "needs one of"),
        ([{"factory": "acme.sinks:make_sink", "extra": 1}], r"\.extra is not supported"),
        ([{"kind": "collect", "options": {}}], r"\.options is not supported"),
        ([{"kind": "collect", "name": ""}], "name must be a non-empty string"),
        ([{"factory": "bad-path", "name": "x"}], "module:callable"),
        (
            [{"factory": "acme.sinks:make_sink", "options": [1]}],
            "options must be a string-keyed mapping",
        ),
        (
            [{"factory": "acme.sinks:make_sink", "options": {1: "x"}}],
            "options must be a string-keyed mapping",
        ),
        (["not-a-mapping"], "must be a mapping"),
    ],
    ids=[
        "null",
        "not-a-list",
        "empty",
        "unknown-kind",
        "kind-and-factory",
        "no-selector",
        "unknown-key",
        "collect-options",
        "empty-name",
        "bad-factory",
        "options-not-mapping",
        "options-non-string-key",
        "entry-not-mapping",
    ],
)
def test_invalid_sinks_fail_fast(sinks: JsonValue, match: str) -> None:
    with pytest.raises(ConfigError, match=match) as info:
        build_spec(_cfg(sinks=sinks), _args())

    assert "run.yaml" in str(info.value)


def test_multi_entry_requires_every_name() -> None:
    with pytest.raises(ConfigError, match="needs an explicit name") as info:
        build_spec(
            _cfg(
                sinks=[
                    {"kind": "collect"},
                    {"factory": "acme.sinks:make_sink"},
                ]
            ),
            _args(),
        )

    assert "run.yaml" in str(info.value)


def test_multi_entry_requires_unique_names() -> None:
    with pytest.raises(ConfigError, match="share the name 'collect'") as info:
        build_spec(
            _cfg(
                sinks=[
                    {"kind": "collect"},
                    {"name": "collect", "factory": "acme.sinks:make_sink"},
                ]
            ),
            _args(),
        )

    assert "run.yaml" in str(info.value)


def test_run_accepts_the_explicit_sinks_shape() -> None:
    run = Run(
        _cfg(),
        sinks=[
            CollectSinkSpec(),
            SinkSpec(factory="acme.sinks:make_sink", name="bench"),
        ],
    )
    try:
        spec = run.spec
        assert spec.sinks == (
            CollectSinkSpec(),
            SinkSpec(factory="acme.sinks:make_sink", name="bench"),
        )
        assert spec.custom_sink is None
    finally:
        run.close()


def test_run_sinks_override_a_config_sink_section() -> None:
    run = Run(
        _cfg(sink={"factory": "acme.config:make_sink"}),
        sinks=[CollectSinkSpec()],
    )
    try:
        spec = run.spec
        assert spec.sinks == (CollectSinkSpec(),)
        assert spec.custom_sink is None
    finally:
        run.close()


def test_run_rejects_sink_and_sinks_together() -> None:
    with pytest.raises(ConfigError, match="not both"):
        Run(
            _cfg(),
            sink="acme.sinks:make_sink",
            sinks=[CollectSinkSpec()],
        )


def test_run_rejects_an_empty_sinks_sequence() -> None:
    with pytest.raises(ConfigError, match="at least one"):
        Run(_cfg(), sinks=[])


def test_run_rejects_non_declaration_entries() -> None:
    with pytest.raises(TypeError, match="CollectSinkSpec or SinkSpec"):
        Run(_cfg(), sinks=[object()])  # type: ignore[list-item]


def test_run_rejects_multientry_missing_names() -> None:
    with pytest.raises(ConfigError, match="needs an explicit name"):
        Run(
            _cfg(),
            sinks=[CollectSinkSpec(), SinkSpec(factory="acme.sinks:make_sink")],
        )
