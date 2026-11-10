"""Run construction requires an explicit robot for minimal/world runs."""

from __future__ import annotations

import pytest

from unigym.errors import ConfigError
from unigym.run.api import Run
from unigym.world import World


def _world() -> World:
    """A world declaration that never touches the engine or the filesystem."""
    return World.base(mjcf="model.xml")


def test_minimal_construction_without_robot_fails():
    """The world/minimal form has no implicit default robot."""
    with pytest.raises(ConfigError, match="explicit robot"):
        Run(world=_world())
    with pytest.raises(ConfigError, match="explicit robot"):
        Run.minimal(world=_world())


def test_minimal_construction_accepts_explicit_robot():
    """An explicit registered robot name remains valid."""
    run = Run(world=_world(), robot="g1")

    assert run.cfg["articulations"][0]["name"] == "g1"


def test_cfg_with_articulations_needs_no_robot_injection():
    """A config that declares articulations is used as-is when robot is omitted."""
    cfg = {
        "run_name": "r",
        "seed": 7,
        "env": {"backend": "isaaclab2", "num_envs": 1},
        "collect": {"steps": 1},
        "articulations": [{"name": "arm", "joints": [{"name": "shoulder"}]}],
        "controls": [
            {
                "name": "arm",
                "adapter": "manual",
                "articulation": "arm",
                "joints": ["shoulder"],
                "drive": "position_pd",
                "target": "position",
            }
        ],
    }

    run = Run(cfg)

    assert [art["name"] for art in run.cfg["articulations"]] == ["arm"]
    assert run.cfg["controls"][0]["articulation"] == "arm"
