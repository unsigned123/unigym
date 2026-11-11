"""RunSpec normalization: frozen intent, CLI precedence and rejection paths."""

from __future__ import annotations

import copy
from argparse import Namespace

import pytest

from unigym.errors import ConfigError
from unigym.run.spec import build_spec


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


def test_build_spec_applies_cli_overrides_without_mutating_input():
    """CLI overrides win and the caller's config document stays untouched."""
    cfg = _cfg()
    before = copy.deepcopy(cfg)

    spec = build_spec(cfg, _args(num_envs=8, steps=9, seed=3, device="cuda:1"))

    assert (spec.num_envs, spec.total_steps, spec.seed, spec.device) == (8, 9, 3, "cuda:1")
    assert cfg == before


def test_build_spec_scene_without_runtime_fails_fast():
    """A scene-bearing config without a pack runtime is a teaching error."""
    cfg = _cfg(scene="acme_scene.tabletop")

    with pytest.raises(ValueError, match="acme_scene.tabletop.*requires a pack runtime"):
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
