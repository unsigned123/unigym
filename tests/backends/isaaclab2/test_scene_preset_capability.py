"""IsaacLab rejects scene-preset/scene-provider requests before env construction."""

from __future__ import annotations

from argparse import Namespace

import pytest

from unigym.errors import ConfigError
from unigym.run.spec import build_spec


def _cfg(scene_preset: object) -> dict:
    return {
        "run_name": "isaaclab_scene",
        "seed": 1,
        "scene_preset": scene_preset,
        "env": {"backend": "isaaclab2", "num_envs": 1},
        "collect": {"steps": 10},
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


def test_scene_provider_fails_before_env_construction() -> None:
    with pytest.raises(ConfigError, match="no scene-provider consumer"):
        build_spec(
            _cfg({"provider": "acme.table_top", "params": {}}),
            Namespace(config="run.yaml"),
        )


def test_world_preset_without_a_compatible_format_fails_at_resolution() -> None:
    with pytest.raises(ConfigError, match="no compatible world format"):
        build_spec(
            _cfg({"provider": "acme.table_top", "world": {"mjcf": "scene.xml"}}),
            Namespace(config="run.yaml"),
        )


def test_env_kwargs_rejects_internal_scene_routing_keys() -> None:
    from unigym.backends.isaaclab2.build import env_kwargs

    with pytest.raises(ValueError, match="scene_provider"):
        env_kwargs(
            {"params": {"scene_provider": "acme.table_top"}},
            None,
            None,
            spec=None,
        )
