"""``SimulationBuilder.from_config`` loads a run config through one parser."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest
import yaml

from unigym.errors import ConfigError
from unigym.sim import SimulationBuilder
from tests._support.sim_fixtures import write_sim_pack

_CONFIG = {
    "run_name": "roundtrip",
    "seed": 7,
    "env": {
        "backend": "mujoco",
        "num_envs": 1,
        "decimation": 4,
        "sim_dt": 0.005,
    },
    "world_base": {"mjcf": "world.xml"},
    "collect": {"steps": 10},
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
    "cameras": [{"name": "top", "rig": {"type": "world"}}],
}


def _forbidden_discovery(*args: object, **kwargs: object) -> NoReturn:
    """Fail the test if the ambient discovery path is touched."""
    raise AssertionError("SimulationBuilder.from_config must not discover")


def test_from_config_without_pack_set_refuses_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pack-needing config names the explicit facade seam, never discovery."""
    cfg = {
        "run_name": "no-pack",
        "seed": 0,
        "env": {
            "backend": "mujoco",
            "num_envs": 1,
            "decimation": 4,
            "sim_dt": 0.005,
        },
        "collect": {"steps": 1},
        "scene_preset": "missing.preset",
    }
    path = tmp_path / "no-pack.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr("unigym.pack_api.discover", _forbidden_discovery)

    with pytest.raises(ConfigError, match="pack_set=.*pack_runtime="):
        SimulationBuilder.from_config(path)


def test_from_config_uses_the_explicit_pack_set(tmp_path: Path) -> None:
    """A pack-referencing config is loaded through the caller's explicit pack set."""
    pack_set, ref = write_sim_pack(tmp_path / "pack")
    cfg = dict(_CONFIG)
    cfg["articulations"] = [{"name": "body", "params": {"profile": ref.name}}]
    cfg["action_spec"] = [
        {
            "name": "body",
            "articulation": "body",
            "joints": ["shoulder", "elbow"],
            "target": "position",
            "role": "body",
        }
    ]
    cfg["policies"] = [{"adapter": "manual", "groups": [["body", "body"]]}]
    path = tmp_path / "pack.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    builder = SimulationBuilder.from_config(path, pack_set=pack_set)

    assert builder.pack_runtime.pack_set == pack_set


def test_from_config_rejects_the_removed_episode_clock_key(tmp_path: Path) -> None:
    """A config that still declares env.episode_length_s fails at the spec."""
    cfg = dict(_CONFIG)
    cfg["env"] = {**cfg["env"], "episode_length_s": 20.0}
    path = tmp_path / "clock.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="env.episode_length_s was removed"):
        SimulationBuilder.from_config(path)
