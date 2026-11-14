"""Install/import smoke for the pack API and scene-preset surface.

The public pack contract must import without an engine and without executing
pack Python, so the heavy-module check runs in a fresh interpreter rather than
the already-warm pytest process.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_HEAVY_ROOTS = frozenset({"numpy", "scipy", "torch", "mujoco", "isaaclab", "omni", "warp"})
_PACK_PREFIXES = ("unigym_test_pack",)


def test_pack_api_public_names_import() -> None:
    """Every advertised pack-API name resolves from the public facades."""
    import unigym.pack_api as pack_api
    import unigym.packs as packs
    import unigym.scene_presets as scene_presets
    from unigym.world import World

    for name in pack_api.__all__:
        assert getattr(pack_api, name) is not None, name
    for name in packs.__all__:
        assert getattr(packs, name) is not None, name
    for name in ("ScenePreset", "ScenePresetError", "load_scene_preset", "apply_scene_preset"):
        assert getattr(scene_presets, name) is not None, name

    assert callable(pack_api.register_scene_provider)
    assert callable(pack_api.register_scene_content_filler)
    assert callable(pack_api.build_scene_from_provider)
    assert callable(packs.require)
    assert callable(packs.installed)
    assert callable(World.from_preset)
    assert callable(World.from_provider)


@pytest.mark.subprocess
@pytest.mark.parametrize(
    "module",
    ["unigym.pack_api", "unigym.packs", "unigym.scene_presets"],
)
def test_pack_api_import_loads_no_engine_or_pack_code(module: str) -> None:
    """A fresh interpreter imports the module without engine or pack modules."""
    code = (
        "import importlib, sys\n"
        "importlib.import_module(sys.argv[1])\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in "
        f"{sorted(_HEAVY_ROOTS)!r})\n"
        "assert not heavy, heavy\n"
        "pack_mods = sorted(m for m in sys.modules if m.startswith("
        f"{_PACK_PREFIXES!r}) or m == 'unigym_packs')\n"
        "assert not pack_mods, pack_mods\n"
    )

    result = subprocess.run([sys.executable, "-c", code, module], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
