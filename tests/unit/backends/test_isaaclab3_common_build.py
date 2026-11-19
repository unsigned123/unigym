"""Shared IsaacLab 3 builder hooks: kwargs forwarding, terrain, events stub."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from unigym.backends.shared.isaaclab3_common.build import (
    build_terrain,
    env_kwargs,
)

_EXPECTED_KEYS = {
    "terrain",
    "static_shapes",
    "articulations",
    "controls",
    "profiles",
    "robot_realizations",
    "backend_options",
    "scene_provider",
    "scene_provider_params",
    "scene_provider_preset",
    "artifact_dir",
}


@dataclass
class _RunSpec:
    terrain: object | None = None
    static_shapes: object | None = None
    articulations: object | None = None


def test_env_kwargs_forwards_backend_payload():
    """The builder kwargs carry the native run payload verbatim."""
    terrain = object()
    static_shapes = [object()]
    articulations = [object()]
    controls = object()
    profiles = object()
    robot_realizations = object()
    backend_options = {"headless": True}
    spec = _RunSpec(terrain=terrain, static_shapes=static_shapes, articulations=articulations)

    result = env_kwargs(
        {"ignored": True},
        "artifacts",
        "world",
        spec,
        controls=controls,
        profiles=profiles,
        robot_realizations=robot_realizations,
        backend_options=backend_options,
    )

    assert result["terrain"] is terrain
    assert result["static_shapes"] is static_shapes
    assert result["articulations"] is articulations
    assert result["controls"] is controls
    assert result["profiles"] is profiles
    assert result["robot_realizations"] is robot_realizations
    assert result["backend_options"] is backend_options


def test_env_kwargs_forwards_scene_provider_keys_and_artifact_dir():
    """The routed scene-provider keys and the artifact dir reach the env."""
    params = {
        "scene_provider": "acme.table_top",
        "scene_provider_params": {"rows": 2},
        "scene_provider_preset": "tabletop",
    }

    result = env_kwargs({"params": params}, "artifacts-out", None, _RunSpec())

    assert result["scene_provider"] == "acme.table_top"
    assert result["scene_provider_params"] == {"rows": 2}
    assert result["scene_provider_preset"] == "tabletop"
    assert result["artifact_dir"] == "artifacts-out"


def test_env_kwargs_defaults_scene_provider_keys_to_none():
    """A run without a scene provider forwards explicit ``None`` values."""
    result = env_kwargs({"env": 1}, None, object(), _RunSpec())

    assert result["scene_provider"] is None
    assert result["scene_provider_params"] is None
    assert result["scene_provider_preset"] is None
    assert result["artifact_dir"] is None


def test_env_kwargs_does_not_leak_builder_only_arguments():
    """env_cfg and world_base are consumed, not forwarded."""
    result = env_kwargs({"env": 1}, "artifacts", object(), _RunSpec())

    assert set(result) == _EXPECTED_KEYS
    assert "world_base" not in result
    assert "backend_params" not in result


def test_env_kwargs_rejects_content_with_the_existing_teaching_error():
    """A content fill has no consumer on this backend."""
    with pytest.raises(ValueError, match="env.params.content has no consumer"):
        env_kwargs({"params": {"content": {"filler": "any"}}}, None, None, _RunSpec())


def test_env_kwargs_accepts_and_ignores_actuation_plans():
    """The common run-layer plan kwarg is accepted but not forwarded in D4."""
    plans = {"arm": (object(),)}

    result = env_kwargs({}, None, None, _RunSpec(), actuation_plans=plans)

    assert "actuation_plans" not in result
    assert set(result) == _EXPECTED_KEYS


def test_build_terrain_without_any_terrain_raises_teaching_error():
    """Neither native nor yaml terrain is a fail-fast configuration error."""
    with pytest.raises(ValueError, match="declares no terrain.*terrain:.*World.terrain"):
        build_terrain({}, num_envs=1, env_spacing=2.0)


def test_build_terrain_native_spec_terrain_wins_over_yaml():
    """A declared native terrain IR short-circuits the yaml preset."""
    assert build_terrain({"terrain": {"name": "flat"}}, 1, 2.0, spec_terrain=object()) is None


def test_build_terrain_forwards_yaml_section_to_shared_builder(monkeypatch):
    """A yaml ``terrain`` section is handed to the shared terrains builder."""
    import unigym.backends.shared.isaaclab3_common.terrains as terrains

    calls = {}

    def _record_build_terrain(section, num_envs=1, env_spacing=4.0, classes=None):
        calls["section"] = section
        calls["num_envs"] = num_envs
        calls["env_spacing"] = env_spacing
        return {"sentinel": True}

    monkeypatch.setattr(terrains, "build_terrain", _record_build_terrain)
    section = {"name": "flat", "params": {}}

    result = build_terrain({"terrain": section}, num_envs=7, env_spacing=4.5)

    assert result == {"sentinel": True}
    assert calls == {"section": section, "num_envs": 7, "env_spacing": 4.5}
