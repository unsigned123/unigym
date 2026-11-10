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
