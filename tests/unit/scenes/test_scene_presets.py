"""Unit contract for scene-preset merge and backend selection."""

from __future__ import annotations

import pytest

from unigym.errors import ConfigError
from unigym.scene_presets import BackendBinding, resolve_scene_plan


def _binding(name: str = "mujoco") -> BackendBinding:
    return BackendBinding(
        name=name,
        asset_formats=("mjcf",),
        backend_param_keys=frozenset({"base", "initializer"}),
        known_backends=("mujoco", "mjwarp", "isaaclab2"),
    )


def test_absent_scene_preset_returns_none() -> None:
    assert resolve_scene_plan({"env": {}}, runtime=None, binding=_binding()) is None


def test_run_overrides_merge_keywise_over_the_inline_preset() -> None:
    cfg = {
        "scene_preset": {"provider": "acme.tabletop", "params": {"style": 1, "seed": 4}},
        "world_base": {"mjcf": "run.xml"},
        "objects": {"types": [{"name": "run_box"}]},
        "env": {
            "params": {
                "initializer": {"type": "run_initializer", "params": {"seed": 9}},
            }
        },
    }

    plan = resolve_scene_plan(cfg, runtime=None, binding=_binding())

    assert plan is not None
    assert plan.provider == "acme.tabletop"
    assert dict(plan.provider_params) == {"style": 1, "seed": 4}
    assert plan.world_path == "run.xml"
    assert plan.props == {"types": [{"name": "run_box"}]}
    assert dict(plan.backend_params) == {
        "initializer": {"type": "run_initializer", "params": {"seed": 9}}
    }


def test_run_initializer_partial_params_merge_over_preset_defaults() -> None:
    """A partial run override keeps the preset type and every other param."""
    cfg = {
        "scene_preset": {
            "provider": "acme.tabletop",
            "backend_params": {
                "mujoco": {
                    "initializer": {
                        "type": "preset_initializer",
                        "params": {"seed": 1, "mode": "preset"},
                    },
                    "base": {"joint": "fixed"},
                }
            },
        },
        "env": {"params": {"initializer": {"params": {"seed": 9}}}},
    }

    plan = resolve_scene_plan(cfg, runtime=None, binding=_binding())

    assert plan is not None
    assert plan.backend_params["initializer"] == {
        "type": "preset_initializer",
        "params": {"seed": 9, "mode": "preset"},
    }
    assert plan.backend_params["base"] == {"joint": "fixed"}


def test_backend_block_is_selected_by_canonical_name() -> None:
    cfg = {
        "scene_preset": {
            "provider": "acme.tabletop",
            "backend_params": {
                "mujoco": {"initializer": {"type": "m", "params": {}}},
                "mjwarp": {"initializer": {"type": "w", "params": {}}},
            },
        }
    }

    plan = resolve_scene_plan(cfg, runtime=None, binding=_binding("mjwarp"))

    assert plan is not None
    assert plan.backend_params["initializer"] == {"type": "w", "params": {}}


def test_content_fill_requires_a_filler_and_provider() -> None:
    with pytest.raises(ConfigError, match="objects.filler"):
        resolve_scene_plan(
            {"scene_preset": {"provider": "acme.tabletop", "objects": {"sets": {"cup": 1}}}},
            runtime=None,
            binding=_binding(),
        )


def test_raw_run_objects_normalize_once_in_resolution() -> None:
    cfg = {
        "scene_preset": {"provider": "acme.tabletop"},
        "objects": {"types": [{"name": "box"}], "filler": "f", "sets": {"cup": 1}},
    }

    plan = resolve_scene_plan(cfg, runtime=None, binding=_binding())

    assert plan is not None
    assert plan.props == {"types": [{"name": "box"}]}
    assert plan.content == {"filler": "f", "sets": {"cup": 1}}


def test_world_format_selects_the_backend_preferred_declared_format() -> None:
    """A urdf/usd backend selects its preferred format from a format-neutral world."""
    binding = BackendBinding(
        name="isaaclab2",
        asset_formats=("usd", "urdf"),
        backend_param_keys=frozenset({"base", "initializer"}),
        known_backends=("mujoco", "mjwarp", "isaaclab2"),
    )

    urdf_only = resolve_scene_plan(
        {"scene_preset": {"provider": "acme.tabletop", "world": {"urdf": "w.urdf"}}},
        runtime=None,
        binding=binding,
    )
    both = resolve_scene_plan(
        {
            "scene_preset": {
                "provider": "acme.tabletop",
                "world": {"urdf": "w.urdf", "usd": "w.usd"},
            }
        },
        runtime=None,
        binding=binding,
    )

    assert urdf_only is not None
    assert (urdf_only.world_format, urdf_only.world_path) == ("urdf", "w.urdf")
    assert both is not None
    assert (both.world_format, both.world_path) == ("usd", "w.usd")


def test_unknown_declared_backend_names_the_known_set() -> None:
    cfg = {
        "scene_preset": {
            "provider": "acme.tabletop",
            "backend_params": {"nope": {"base": {"joint": "fixed"}}},
        }
    }

    with pytest.raises(ConfigError, match="unknown backend"):
        resolve_scene_plan(cfg, runtime=None, binding=_binding())
