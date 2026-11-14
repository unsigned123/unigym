"""IsaacLab2 builder kwargs forwarding, artifact dir and terrain errors."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from unigym.backends.isaaclab2.build import build_terrain, env_kwargs

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


def test_env_kwargs_rejects_content_with_the_existing_teaching_error():
    """A content fill has no consumer on this backend."""
    with pytest.raises(ValueError, match="env.params.content has no consumer"):
        env_kwargs({"params": {"content": {"filler": "any"}}}, None, None, _RunSpec())


def test_build_terrain_without_any_terrain_raises_teaching_error():
    """Neither native nor yaml terrain is a fail-fast configuration error."""
    with pytest.raises(ValueError, match="declares no terrain.*terrain:.*World.terrain"):
        build_terrain({}, num_envs=1, env_spacing=2.0)


def test_env_kwargs_accepts_and_ignores_actuation_plans():
    """The common run-layer plan kwarg is accepted but not forwarded in D4."""
    plans = {"arm": (object(),)}

    result = env_kwargs({}, None, None, _RunSpec(), actuation_plans=plans)

    assert "actuation_plans" not in result
    assert set(result) == _EXPECTED_KEYS
