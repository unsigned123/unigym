"""RunSpec normalization: frozen intent, CLI precedence and rejection paths."""

from __future__ import annotations

import copy
from argparse import Namespace

import pytest

from unigym.errors import ConfigError
from unigym.run.spec import build_spec, requires_pack_runtime


def _cfg(**overrides):
    cfg = {
        "run_name": "r",
        "seed": 7,
        "env": {"backend": "isaaclab2", "num_envs": 2},
        "collect": {"steps": 100},
        "articulations": [
            {"name": "arm", "joints": [{"name": "shoulder"}, {"name": "elbow"}]},
        ],
        "controls": [
            {
                "name": "arm",
                "adapter": "manual",
                "articulation": "arm",
                "joints": ["shoulder", "elbow"],
                "drive": "position_pd",
                "target": "position",
            },
        ],
    }
    cfg.update(overrides)
    return cfg


def _args(**overrides):
    return Namespace(config="run.yaml", **overrides)


def test_build_spec_normalizes_minimal_run_config():
    """A minimal config freezes into typed articulations, controls and output."""
    spec = build_spec(_cfg(), _args())

    assert spec.backend == "isaaclab2"
    assert spec.mode == spec.sinks == "collect"
    assert (spec.num_envs, spec.total_steps, spec.seed) == (2, 100, 7)
    assert [articulation.name for articulation in spec.articulations] == ["arm"]
    assert [group.name for group in spec.controls] == ["arm"]
    assert spec.controls[0].joints == ("shoulder", "elbow")
    assert spec.output.out_dir == "data_out/r"


def test_build_spec_rejects_a_content_fill_without_a_scene():
    """A run-level objects fill needs a preset/provider scene before env construction."""
    cfg = _cfg(
        env={"backend": "mujoco", "num_envs": 1},
        objects={"filler": "acme_filler", "sets": {"cup": 1}},
    )

    with pytest.raises(ConfigError, match="needs a composed scene"):
        build_spec(cfg, _args())


def test_build_spec_applies_cli_overrides_without_mutating_input():
    """CLI overrides win and the caller's config document stays untouched."""
    cfg = _cfg()
    before = copy.deepcopy(cfg)

    spec = build_spec(cfg, _args(num_envs=8, steps=9, seed=3, device="cuda:1"))

    assert (spec.num_envs, spec.total_steps, spec.seed, spec.device) == (8, 9, 3, "cuda:1")
    assert cfg == before


def test_build_spec_scene_preset_without_runtime_fails_fast():
    """A catalog scene-preset config without a pack runtime is a teaching error."""
    cfg = _cfg(scene_preset="acme_scene.tabletop")

    with pytest.raises(ConfigError, match="acme_scene.tabletop.*requires a pack runtime"):
        build_spec(cfg, _args())


@pytest.mark.parametrize("mode", ["train", "", 1], ids=["unknown", "empty", "integer"])
def test_build_spec_rejects_unsupported_mode(mode):
    """Only collect / benchmark / both are implemented run modes."""
    with pytest.raises(ValueError, match="is not implemented"):
        build_spec(_cfg(mode=mode), _args())


@pytest.mark.parametrize("controls", [None, []], ids=["absent", "empty"])
def test_build_spec_requires_controls_declaration(controls):
    """A run with articulations must declare at least one controls group."""
    with pytest.raises(ConfigError, match="no controls declared"):
        build_spec(_cfg(controls=controls), _args())


@pytest.mark.parametrize(
    ("world_base", "match"),
    [
        (None, "world_base is required"),
        ({}, "must be a mapping declaring"),
        ({"mjcf": ""}, "must be a mapping declaring"),
    ],
    ids=["absent", "empty-mapping", "empty-path"],
)
def test_build_spec_requires_valid_world_base_for_mujoco(world_base, match):
    """The mujoco backend rejects a missing or malformed world_base."""
    cfg = _cfg()
    cfg["env"]["backend"] = "mujoco"
    if world_base is not None:
        cfg["world_base"] = world_base

    with pytest.raises(ConfigError, match=match):
        build_spec(cfg, _args())


def test_build_spec_rejects_world_base_for_a_backend_that_does_not_consume_it():
    """isaaclab2 has no world_base consumer, so a declared world base is refused."""
    spec_cfg = _cfg(world_base={"usd": "w.usd"})

    with pytest.raises(ConfigError, match="does not consume a world base"):
        build_spec(spec_cfg, _args())


@pytest.mark.parametrize("section", ["env", "collect", "run_name"])
def test_build_spec_rejects_missing_required_section(section):
    """A missing env/collect/run_name is a teaching ConfigError, not a KeyError."""
    cfg = _cfg()
    del cfg[section]

    with pytest.raises(ConfigError, match=rf"{section} must be"):
        build_spec(cfg, _args())


@pytest.mark.parametrize(
    ("section", "value"),
    [("env", ["isaaclab2"]), ("collect", "100")],
    ids=["env-list", "collect-string"],
)
def test_build_spec_rejects_non_mapping_section(section, value):
    """env and collect must be mappings, not arbitrary JSON values."""
    cfg = _cfg()
    cfg[section] = value

    with pytest.raises(ConfigError, match=rf"{section} must be a mapping"):
        build_spec(cfg, _args())


@pytest.mark.parametrize(
    ("section", "field"),
    [("env", "num_envs"), ("collect", "steps")],
)
def test_build_spec_rejects_missing_required_count(section, field):
    """Missing env.num_envs / collect.steps names the field instead of KeyError."""
    cfg = _cfg()
    del cfg[section][field]

    with pytest.raises(ConfigError, match=rf"{section}\.{field} must be an integer"):
        build_spec(cfg, _args())


@pytest.mark.parametrize(
    ("section", "field"),
    [("env", "num_envs"), ("collect", "steps")],
)
def test_build_spec_rejects_non_numeric_required_count(section, field):
    """A non-numeric env.num_envs / collect.steps is a ConfigError naming the field."""
    cfg = _cfg()
    cfg[section][field] = "many"

    with pytest.raises(ConfigError, match=rf"{section}\.{field} must be an integer"):
        build_spec(cfg, _args())


def test_build_spec_rejects_non_string_run_name():
    """run_name must be a string; a non-string value fails fast with ConfigError."""
    cfg = _cfg()
    cfg["run_name"] = 123

    with pytest.raises(ConfigError, match="run_name must be a string"):
        build_spec(cfg, _args())


def test_build_spec_rejects_empty_config_with_teaching_error():
    """An empty config fails on the required env section, not a KeyError."""
    with pytest.raises(ConfigError, match="env must be a mapping"):
        build_spec({}, _args())


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("env", "num_envs", True),
        ("env", "num_envs", float("inf")),
        ("env", "num_envs", float("nan")),
        ("collect", "steps", "5.5"),
    ],
    ids=["bool", "inf", "nan", "fractional-string"],
)
def test_build_spec_rejects_non_integer_required_count(section, field, value):
    """Required counts reject bool, non-finite and non-integral values."""
    cfg = _cfg()
    cfg[section][field] = value

    with pytest.raises(ConfigError, match=rf"{section}\.{field} must be an integer"):
        build_spec(cfg, _args())


@pytest.mark.parametrize("value", [0, -2], ids=["zero", "negative"])
def test_build_spec_rejects_non_positive_num_envs(value):
    """A run must have at least one environment."""
    cfg = _cfg()
    cfg["env"]["num_envs"] = value

    with pytest.raises(ConfigError, match=r"env\.num_envs must be positive"):
        build_spec(cfg, _args())


@pytest.mark.parametrize("value", [0, -5], ids=["zero", "negative"])
def test_build_spec_rejects_non_positive_collect_steps(value):
    """A run must collect at least one step."""
    cfg = _cfg()
    cfg["collect"]["steps"] = value

    with pytest.raises(ConfigError, match=r"collect\.steps must be positive"):
        build_spec(cfg, _args())


def test_build_spec_rejects_benchmark_packs():
    """benchmark.packs is removed; packs come from the runtime benchmarks face."""
    cfg = _cfg(benchmark={"packs": ["benchmark_packs/acme"]})

    with pytest.raises(ConfigError) as info:
        build_spec(cfg, _args())

    message = str(info.value)
    assert "benchmark.packs" in message
    assert "benchmarks" in message
    assert "runtime" in message


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, False),
        ({"scene_preset": "acme.tabletop"}, True),
        ({"mode": "benchmark"}, True),
        ({"mode": "both"}, True),
        ({"props": {"types": [{"name": "cup", "asset": "pack:acme/cup.urdf"}]}}, True),
        ({"props": {"types": [{"name": "cup", "asset": "${UNIGYM_ASSET_PACK}/cup.urdf"}]}}, True),
        ({"objects": {"types": [{"name": "cup", "template": "acme.cup"}]}}, True),
        ({"adapter": {"name": "manual"}}, False),
        ({"adapter": {}}, False),
        ({"adapter": {"name": "acme.policy"}}, True),
    ],
    ids=[
        "plain",
        "scene_preset",
        "benchmark",
        "both",
        "pack",
        "removed-anchor",
        "template",
        "manual-adapter",
        "adapter-without-name",
        "external-adapter",
    ],
)
def test_requires_pack_runtime_boundary_decision(overrides, expected):
    """The helper detects scenes, benchmark modes, asset refs and external adapters."""
    assert requires_pack_runtime(_cfg(**overrides)) is expected


def test_requires_pack_runtime_detects_an_unregistered_inline_provider():
    """An inline provider that is not process-local needs the pack runtime."""
    cfg = _cfg(scene_preset={"provider": "acme.tabletop", "params": {"seed": 1}})

    assert requires_pack_runtime(cfg) is True


def test_requires_pack_runtime_detects_a_world_provider_fragment():
    """A World.from_provider fragment lands the scene_preset inline form."""
    from unigym.world import World

    cfg = _cfg()
    World.from_provider("acme.tabletop", params={"seed": 1}).apply(cfg)

    assert cfg["scene_preset"] == {"provider": "acme.tabletop", "params": {"seed": 1}}
    assert requires_pack_runtime(cfg) is True


def test_run_spec_has_no_legacy_scene_field():
    """RunSpec no longer exposes a dead/mistyped ``scene`` mapping."""
    spec = build_spec(_cfg(), _args())

    assert not hasattr(spec, "scene")


def test_build_spec_selects_the_backend_before_preset_resolution():
    """Backend-specific preset blocks resolve against the selected backend."""
    cfg = _cfg(
        env={"backend": "mjwarp", "num_envs": 1},
        scene_preset={
            "provider": "acme.tabletop",
            "world": {"mjcf": "scene.xml"},
            "backend_params": {
                "mujoco": {"initializer": {"type": "a", "params": {"seed": 1}}},
                "mjwarp": {"initializer": {"type": "b", "params": {"seed": 2}}},
            },
        },
    )

    spec = build_spec(cfg, _args())

    assert spec.backend_params["initializer"]["type"] == "b"
    assert spec.backend_params["initializer"]["params"] == {"seed": 2}


def test_build_spec_rejects_an_unknown_backend_param_key():
    """The selected backend block is gated by backend_param_keys (no fallback)."""
    cfg = _cfg(
        env={"backend": "mujoco", "num_envs": 1},
        scene_preset={
            "provider": "acme.tabletop",
            "world": {"mjcf": "scene.xml"},
            "backend_params": {"mujoco": {"nope": 1}},
        },
    )

    with pytest.raises(ConfigError, match="unsupported key"):
        build_spec(cfg, _args())


def test_build_spec_rejects_direct_internal_scene_key_declaration():
    """scene_provider* are internal routing keys, not authoring keys."""
    cfg = _cfg()
    cfg["env"]["params"] = {"scene_provider": "acme.tabletop"}

    with pytest.raises(ConfigError, match="scene_preset"):
        build_spec(cfg, _args())
