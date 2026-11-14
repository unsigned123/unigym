"""IsaacLab consumes asset scene providers but refuses content and world_base."""

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


def test_scene_provider_is_accepted_for_isaaclab2() -> None:
    """A provider-only preset routes the scene keys and no static shapes."""
    spec = build_spec(
        _cfg({"provider": "acme.table_top", "params": {}}),
        Namespace(config="run.yaml"),
    )

    assert spec.backend_params["scene_provider"] == "acme.table_top"
    assert spec.static_shapes == ()


def test_world_preset_without_a_compatible_format_fails_at_resolution() -> None:
    with pytest.raises(ConfigError, match="no compatible world format"):
        build_spec(
            _cfg({"provider": "acme.table_top", "world": {"mjcf": "scene.xml"}}),
            Namespace(config="run.yaml"),
        )


def test_content_fill_is_still_refused_by_isaaclab2() -> None:
    """The content fill keeps failing: this backend declares no router."""
    cfg = _cfg({"provider": "acme.table_top", "params": {}})
    cfg["objects"] = {"filler": "acme.fill"}

    with pytest.raises(ConfigError, match="content spec is not supported"):
        build_spec(cfg, Namespace(config="run.yaml"))


def test_world_base_is_still_refused_by_isaaclab2() -> None:
    """The backend declares no world_base consumer."""
    cfg = _cfg(None)
    cfg["world_base"] = {"usd": "table.usd"}

    with pytest.raises(ConfigError, match="does not consume a world base"):
        build_spec(cfg, Namespace(config="run.yaml"))


def test_env_kwargs_forwards_internal_scene_routing_keys() -> None:
    """The backend boundary forwards the routed provider keys, not a reject."""
    from unigym.backends.isaaclab2.build import env_kwargs

    result = env_kwargs(
        {"params": {"scene_provider": "acme.table_top"}},
        None,
        None,
        spec=None,
    )

    assert result["scene_provider"] == "acme.table_top"
