"""Conformance: the same preset resolves per backend with no family fallback."""

from __future__ import annotations

from unigym.scene_presets import BackendBinding, apply_scene_preset

BASE_CFG = {
    "scene_preset": {
        "provider": "acme.table_top",
        "params": {"style": 1},
        "world": {"mjcf": "world.xml"},
        "backend_params": {
            "mujoco": {"initializer": {"type": "mujoco_init", "params": {}}},
            "mjwarp": {"initializer": {"type": "mjwarp_init", "params": {}}},
        },
    },
    "env": {"backend": "mujoco"},
}


def _binding(name: str, keys: frozenset[str] | None = None) -> BackendBinding:
    return BackendBinding(
        name=name,
        asset_formats=("mjcf",),
        backend_param_keys=keys or frozenset({"base", "initializer"}),
        known_backends=("mujoco", "mjwarp", "isaaclab2"),
    )


def test_same_preset_selects_each_backend_block() -> None:
    mujoco = apply_scene_preset(BASE_CFG, binding=_binding("mujoco"))
    mjwarp = apply_scene_preset(BASE_CFG, binding=_binding("mjwarp"))

    assert mujoco["env"]["params"]["initializer"]["type"] == "mujoco_init"
    assert mjwarp["env"]["params"]["initializer"]["type"] == "mjwarp_init"


def test_absent_backend_block_is_empty_not_a_family_fallback() -> None:
    cfg = {
        "scene_preset": {
            "provider": "acme.table_top",
            "backend_params": {"mujoco": {"initializer": {"type": "mujoco_init", "params": {}}}},
        }
    }

    mjwarp = apply_scene_preset(cfg, binding=_binding("mjwarp"))

    assert "initializer" not in mjwarp["env"]["params"]
