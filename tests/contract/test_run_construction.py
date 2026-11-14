"""Run construction requires an explicit robot for minimal/world runs."""

from __future__ import annotations

import pytest

from unigym.errors import ConfigError
from unigym.robots import register_robot, restore_robot_registry, snapshot_robot_registry
from unigym.run.api import Run
from unigym.world import World
from tests._support.profiles import fixture_profile


@pytest.fixture(autouse=True)
def _fixture_robot():
    """Expose one real profile to the programmatic builder tests."""
    snapshot = snapshot_robot_registry()
    register_robot("fixture_arm", fixture_profile)
    yield
    restore_robot_registry(snapshot)


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
    run = Run(world=_world(), robot="fixture_arm")

    assert run.cfg["articulations"][0]["name"] == "fixture_arm"


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


def test_robot_declaration_dict_passes_actuation_through():
    """The programmatic declaration dict carries the neutral actuation IR."""
    actuation = [
        {
            "name": "body",
            "law": "standard_pd",
            "joint_set": "body",
            "gains": {"kp": 1.0, "kd": 1.0},
        }
    ]

    run = Run(world=_world(), robot={"name": "fixture_arm", "actuation": actuation})

    assert run.cfg["articulations"][0]["actuation"] == actuation


def test_bare_robot_string_live_build_fails_the_actuation_hard_cut():
    """Run(robot=...) stays representable but carries no actuation to build."""
    run = Run(world=_world(), robot="fixture_arm")

    with pytest.raises(ValueError, match="no `actuation`"):
        run.build()


def _live_cfg(**env_params):
    return {
        "run_name": "r",
        "seed": 7,
        "env": {"backend": "isaaclab2", "num_envs": 1, "params": env_params},
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


def test_live_build_rejects_top_level_actuator():
    """The removed top-level actuator override fails before construction."""
    cfg = _live_cfg()
    cfg["actuator"] = {"type": "implicit_pd"}

    with pytest.raises(ValueError, match="top-level `actuator:`"):
        Run(cfg).build()


@pytest.mark.parametrize("key", ["actuators", "kp", "kd", "hand_kp", "hand_kd"])
def test_live_build_rejects_removed_gain_env_params(key):
    """The removed env actuator/gain channels fail before construction."""
    with pytest.raises(ValueError, match="removed actuator/gain keys"):
        Run(_live_cfg(**{key: {}})).build()


def test_spec_rejects_the_removed_camera_depth_key():
    """The retired top-level `camera_depth:` key fails at spec construction."""
    cfg = _live_cfg()
    cfg["camera_depth"] = {"width": 106, "height": 60}

    with pytest.raises(ConfigError, match="camera_depth"):
        Run(cfg).spec


def test_spec_keeps_a_renamed_profile_backed_articulation():
    """A renamed articulation selecting a profile is valid run construction."""
    cfg = _live_cfg()
    cfg["articulations"][0]["name"] = "humanoid"
    cfg["articulations"][0]["params"] = {"profile": "g1"}
    cfg["controls"][0]["articulation"] = "humanoid"

    spec = Run(cfg).spec

    assert spec.articulations[0].name == "humanoid"
    assert spec.articulations[0].params == {"profile": "g1"}
