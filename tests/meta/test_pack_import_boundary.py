"""Pack import boundary: construction stays engine-free and legacy names stay gone."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import unigym.pack_api as pack_api
import unigym.packs as packs
import unigym.scene_presets as scene_presets
import unigym.scenes as scenes
import unigym.world as world

_HEAVY_ROOTS = frozenset({"numpy", "scipy", "torch", "mujoco", "isaaclab", "omni", "warp"})
_PACK_PREFIXES = ("unigym_test_pack",)
_HEAVY_SCENE_SUBMODULES = (
    "unigym.actuators",
    "unigym.scenes.articulation",
    "unigym.scenes.pipeline",
    "unigym.scenes.terrain",
)
_LEGACY_NAMES = (
    "SceneContext",
    "register_scene",
    "registered_scenes",
    "build_scene_definition",
    "ContentProvider",
    "register_content_provider",
    "content_provider",
    "registered_content_providers",
    "PROVIDER_WHENS",
)


@pytest.mark.subprocess
def test_scene_import_chain_stays_lazy() -> None:
    """The pack API chain never imports the heavy scene/actuator submodules.

    Regression: ``packs`` -> ``packset`` imports
    ``unigym.scenes.provider_params``, which previously loaded
    ``unigym.scenes/__init__.py`` eagerly, dragging ``articulation`` ->
    ``actuators`` -> numpy into a ``--no-deps`` wheel venv.
    """
    code = (
        "import sys\n"
        "import unigym.pack_api\n"
        "import unigym.packs\n"
        "import unigym.packs.packset\n"
        "import unigym.scene_presets\n"
        "import unigym.scenes.provider_params\n"
        "import unigym.scenes.providers\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in "
        f"{sorted(_HEAVY_ROOTS)!r})\n"
        "assert not heavy, heavy\n"
        "chain = sorted(m for m in sys.modules if m.startswith("
        f"{_HEAVY_SCENE_SUBMODULES!r}))\n"
        "assert not chain, chain\n"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


@pytest.mark.subprocess
def test_pack_object_construction_imports_no_engine_or_pack_code(tmp_path: Path) -> None:
    """Minting preset handles never imports an engine or the pack's Python."""
    root = tmp_path / "acme"
    (root / "scene_presets").mkdir(parents=True)
    (root / "unigym.pack.yaml").write_text(
        "schema: unigym.pack/1\n"
        "name: acme\n"
        "version: 1.0.0\n"
        "api: 1\n"
        "trust: content\n"
        "provides:\n"
        "  scene_presets:\n"
        "    - id: acme.tabletop\n"
        "      resource: scene_presets/tabletop.yaml\n",
        encoding="utf-8",
    )
    (root / "scene_presets" / "tabletop.yaml").write_text(
        "schema: unigym.scene_preset/1\nname: tabletop\n", encoding="utf-8"
    )
    code = (
        "import sys\n"
        "import unigym.packs as packs\n"
        "packs.require('acme').pack('acme').scene_preset('acme.tabletop')\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in "
        f"{sorted(_HEAVY_ROOTS)!r})\n"
        "assert not heavy, heavy\n"
        "pack_mods = sorted(m for m in sys.modules if m.startswith("
        f"{_PACK_PREFIXES!r}) or m == 'unigym_packs')\n"
        "assert not pack_mods, pack_mods\n"
    )
    env = dict(os.environ, UNIGYM_PACKS=str(root), UNIGYM_DEPS_ROOT="")

    result = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=tmp_path, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", _LEGACY_NAMES)
def test_public_facades_advertise_no_legacy_scene_names(name: str) -> None:
    """The repository's public facades expose no removed scene vocabulary."""
    for facade in (pack_api, packs, scene_presets, scenes):
        assert name not in getattr(facade, "__all__", ()), f"{facade.__name__}.__all__"
        assert not hasattr(facade, name), f"{facade.__name__}.{name}"


@pytest.mark.parametrize("name", ["from_pack", "from_generator"])
def test_world_has_no_legacy_factory(name: str) -> None:
    """``World`` keeps only the preset/provider factories."""
    assert not hasattr(world.World, name)
