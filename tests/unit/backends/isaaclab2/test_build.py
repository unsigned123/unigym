"""IsaacLab2 builder kwargs forwarding and terrain error boundary."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from unigym.backends.isaaclab2.build import build_terrain, env_kwargs


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


def test_env_kwargs_does_not_leak_builder_only_arguments():
    """env_cfg, artifacts_dir and world_base are consumed, not forwarded."""
    result = env_kwargs({"env": 1}, "artifacts", object(), _RunSpec())

    assert set(result) == {
        "terrain",
        "static_shapes",
        "articulations",
        "controls",
        "profiles",
        "robot_realizations",
        "backend_options",
    }


def test_build_terrain_without_any_terrain_raises_teaching_error():
    """Neither native nor yaml terrain is a fail-fast configuration error."""
    with pytest.raises(ValueError, match="declares no terrain.*terrain:.*World.terrain"):
        build_terrain({}, num_envs=1, env_spacing=2.0)


def test_env_kwargs_accepts_and_ignores_actuation_plans():
    """The common run-layer plan kwarg is accepted but not forwarded in D4."""
    plans = {"arm": (object(),)}

    result = env_kwargs({}, None, None, _RunSpec(), actuation_plans=plans)

    assert "actuation_plans" not in result
    assert set(result) == {
        "terrain",
        "static_shapes",
        "articulations",
        "controls",
        "profiles",
        "robot_realizations",
        "backend_options",
    }
