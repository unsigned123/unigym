"""The removed scene vocabulary fails cleanly and names its replacement.

Every name below was removed by the ScenePreset/SceneProvider cutover with no
compatibility window.  A residual alias, dual-read or silent fallback would
turn one of these cases green without a teaching error, so each surface is
checked through its public boundary.
"""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path

import pytest

import unigym.pack_api as pack_api
import unigym.packs as packs
import unigym.scene_presets as scene_presets
import unigym.scenes as scenes
import unigym.world as world
from unigym.backends.registry import BackendInfo
from unigym.errors import ConfigError
from unigym.pack_api import discover, load_manifest_v1
from unigym.packs import PackError
from unigym.scene_presets import (
    ScenePresetError,
    load_scene_preset,
    reject_legacy_scene_keys,
)

_REMOVED_PROVIDER_NAMES = (
    "SceneContext",
    "register_scene",
    "registered_scenes",
    "build_scene_definition",
)
_REMOVED_CONTENT_NAMES = (
    "ContentProvider",
    "register_content_provider",
    "content_provider",
    "registered_content_providers",
    "PROVIDER_WHENS",
)
_REMOVED_WORLD_FACTORIES = ("from_pack", "from_generator")

_PRESET = """\
schema: unigym.scene_preset/1
name: tabletop
world:
  mjcf: world.xml
"""


def _runtime(tmp_path: Path, preset_text: str):
    root = tmp_path / "acme_scene"
    (root / "scene_presets").mkdir(parents=True)
    (root / "unigym.pack.yaml").write_text(
        "schema: unigym.pack/1\n"
        "name: acme_scene\n"
        "version: 1.0.0\n"
        "api: 1\n"
        "trust: content\n"
        "provides:\n"
        "  scene_presets:\n"
        "    - id: acme_scene.tabletop\n"
        "      resource: scene_presets/tabletop.yaml\n",
        encoding="utf-8",
    )
    (root / "scene_presets" / "tabletop.yaml").write_text(preset_text, encoding="utf-8")
    (root / "world.xml").write_text(
        '<mujoco model="tabletop"><worldbody/></mujoco>\n', encoding="utf-8"
    )
    return discover([root])


def test_removed_scene_modules_are_not_importable() -> None:
    """The deleted scene modules raise ModuleNotFoundError."""
    for module in ("unigym.scene_packs", "unigym.scenes.registry"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)


@pytest.mark.parametrize("name", _REMOVED_PROVIDER_NAMES + _REMOVED_CONTENT_NAMES)
def test_removed_scene_names_are_absent_from_the_public_facades(name: str) -> None:
    """No public facade re-exports a removed scene or content name."""
    for facade in (pack_api, packs, scene_presets, scenes):
        assert not hasattr(facade, name), f"{facade.__name__}.{name}"
        assert name not in getattr(facade, "__all__", ()), f"{facade.__name__}.__all__"


@pytest.mark.parametrize("name", _REMOVED_WORLD_FACTORIES)
def test_removed_world_factories_are_absent(name: str) -> None:
    """``World.from_pack`` / ``World.from_generator`` have no alias."""
    assert not hasattr(world.World, name)
    assert not hasattr(world, name)


def test_removed_manifest_face_names_both_replacements(tmp_path: Path) -> None:
    """``provides.scenes`` fails with the data and code replacement hints."""
    root = tmp_path / "acme"
    root.mkdir()
    (root / "unigym.pack.yaml").write_text(
        "schema: unigym.pack/1\n"
        "name: acme\n"
        "version: 1.0.0\n"
        "api: 1\n"
        "trust: content\n"
        "provides:\n"
        "  scenes: []\n",
        encoding="utf-8",
    )

    with pytest.raises(PackError) as info:
        load_manifest_v1(root)

    message = str(info.value)
    assert "unknown pack face 'scenes'" in message
    assert "scene_presets" in message
    assert "providers" in message
    assert "kind: scene" in message


@pytest.mark.parametrize("key", ["scene", "scene_pack", "scene_bundle", "scenes"])
def test_removed_top_level_run_keys_name_scene_preset(key: str) -> None:
    """A removed top-level run key fails with ``ConfigError`` naming the replacement."""
    with pytest.raises(ConfigError, match="scene_preset"):
        reject_legacy_scene_keys({key: "acme.tabletop"})


@pytest.mark.parametrize("key", ["mj_scene", "mj_scene_params"])
def test_removed_env_params_keys_name_scene_preset(key: str) -> None:
    """``env.params.mj_scene*`` fails with ``ConfigError`` naming the replacement."""
    with pytest.raises(ConfigError, match="scene_preset"):
        reject_legacy_scene_keys({"env": {"params": {key: {"seed": 1}}}})


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ("backend:\n  mujoco: {}\n", "backend_params"),
        ("initializer: {type: tabletop_initializer}\n", "backend_params.<backend>.initializer"),
        ("scene: tabletop\n", "'provider:'"),
        ("backend_params:\n  mujoco:\n    mj_scene: tabletop\n", "'provider:'"),
        ("backend_params:\n  mujoco:\n    mj_scene_params: {seed: 1}\n", "'provider:'"),
        ("backend_params:\n  mujoco:\n    mj_model: world.xml\n", "top-level 'world:'"),
        ("objects:\n  provider: tabletop_filler\n", "objects.filler"),
    ],
    ids=[
        "top-level-backend",
        "top-level-initializer",
        "top-level-scene",
        "backend-mj-scene",
        "backend-mj-scene-params",
        "backend-mj-model",
        "objects-provider",
    ],
)
def test_removed_preset_document_keys_name_the_replacements(
    tmp_path: Path, extra: str, match: str
) -> None:
    """Removed preset keys fail at load with a ``ScenePresetError`` hint."""
    runtime = _runtime(tmp_path, _PRESET + extra)

    with pytest.raises(ScenePresetError, match=match):
        load_scene_preset("acme_scene.tabletop", runtime)


def test_removed_backend_trait_is_replaced_not_aliased() -> None:
    """``BackendInfo.mj_scene_consumer`` is gone; the provider trait owns it."""
    fields = {field.name for field in dataclasses.fields(BackendInfo)}

    assert "mj_scene_consumer" not in fields
    assert "scene_provider_consumer" in fields
