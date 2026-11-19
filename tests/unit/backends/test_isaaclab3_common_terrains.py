"""Shared IsaacLab 3 terrain presets build offline through fake cfg tables.

The presets never import isaaclab at module scope; every test injects its own
plain cfg classes.  The material key is the IsaacLab 3 ``RigidBodyMaterialBaseCfg``
(the old ``RigidBodyMaterialCfg`` name is gone).
"""

from __future__ import annotations

import importlib.util
import sys
import warnings

import numpy as np
import pytest

from unigym.backends.shared.isaaclab3_common.terrains import (
    TERRAIN_REGISTRY,
    FlatTerrain,
    GeneratorTerrain,
    NativeTerrainPreset,
    StepTerrain,
    build_terrain,
)
from unigym.backends.shared.isaaclab3_common.terrains.custom import _compat
from unigym.scenes.terrain import Terrain

_ISAACLAB_INSTALLED = importlib.util.find_spec("isaaclab") is not None


class _Cfg:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class _AssetBaseCfg(_Cfg):
    pass


class _GroundPlaneCfg(_Cfg):
    pass


class _RigidBodyMaterialBaseCfg(_Cfg):
    pass


class _PreviewSurfaceCfg(_Cfg):
    pass


class _TerrainImporterCfg(_Cfg):
    pass


class _TerrainGeneratorCfg(_Cfg):
    pass


class _MeshPyramidStairsTerrainCfg(_Cfg):
    pass


class _MeshPlaneTerrainCfg(_Cfg):
    pass


class _HfRandomUniformTerrainCfg(_Cfg):
    pass


class _HfTerrainBaseCfg(_Cfg):
    pass


_CLASSES = {
    "AssetBaseCfg": _AssetBaseCfg,
    "GroundPlaneCfg": _GroundPlaneCfg,
    "RigidBodyMaterialBaseCfg": _RigidBodyMaterialBaseCfg,
    "PreviewSurfaceCfg": _PreviewSurfaceCfg,
    "TerrainImporterCfg": _TerrainImporterCfg,
    "TerrainGeneratorCfg": _TerrainGeneratorCfg,
    "MeshPyramidStairsTerrainCfg": _MeshPyramidStairsTerrainCfg,
    "MeshPlaneTerrainCfg": _MeshPlaneTerrainCfg,
    "HfRandomUniformTerrainCfg": _HfRandomUniformTerrainCfg,
    "HfTerrainBaseCfg": _HfTerrainBaseCfg,
}


def _material(importer: _TerrainImporterCfg) -> _RigidBodyMaterialBaseCfg:
    material = importer.kwargs["physics_material"]
    assert isinstance(material, _RigidBodyMaterialBaseCfg)
    return material


def test_flat_terrain_builds_ground_plane_without_isaaclab():
    """FlatTerrain maps to AssetBaseCfg + GroundPlaneCfg with a base material."""
    fields = FlatTerrain(friction=0.7, color=(0.1, 0.2, 0.3), visible=False).build(
        3, 4.0, classes=_CLASSES
    )

    assert set(fields) == {"ground"}
    asset = fields["ground"]
    assert isinstance(asset, _AssetBaseCfg)
    assert asset.kwargs["prim_path"] == "/World/ground"
    spawn = asset.kwargs["spawn"]
    assert isinstance(spawn, _GroundPlaneCfg)
    assert spawn.kwargs["color"] == (0.1, 0.2, 0.3)
    assert spawn.kwargs["visible"] is False
    material = spawn.kwargs["physics_material"]
    assert isinstance(material, _RigidBodyMaterialBaseCfg)
    assert material.kwargs == {
        "static_friction": 0.7,
        "dynamic_friction": 0.7,
        "restitution": 0.0,
    }


def test_step_terrain_builds_generator_importer_with_base_material():
    """StepTerrain derives the tile side and wires the stair sub-terrain."""
    fields = StepTerrain(
        step_width=0.25,
        step_height=0.05,
        num_steps=4,
        platform_width=1.0,
        border_width=0.4,
        friction=0.9,
        color=(0.2, 0.3, 0.4),
    ).build(2, 7.0, classes=_CLASSES)

    assert set(fields) == {"terrain"}
    importer = fields["terrain"]
    assert isinstance(importer, _TerrainImporterCfg)
    assert importer.kwargs["prim_path"] == "/World/ground"
    assert importer.kwargs["num_envs"] == 2
    assert importer.kwargs["terrain_type"] == "generator"
    assert importer.kwargs["use_terrain_origins"] is True
    assert importer.kwargs["max_init_terrain_level"] == 0
    assert importer.kwargs["env_spacing"] == 7.0
    assert importer.kwargs["debug_vis"] is False
    assert isinstance(importer.kwargs["visual_material"], _PreviewSurfaceCfg)
    assert _material(importer).kwargs == {
        "static_friction": 0.9,
        "dynamic_friction": 0.9,
        "restitution": 0.0,
    }

    generator = importer.kwargs["terrain_generator"]
    assert isinstance(generator, _TerrainGeneratorCfg)
    side = (4 - 1) * 2 * 0.25 + 1.0
    assert generator.kwargs["size"] == (side, side)
    assert generator.kwargs["border_width"] == 0.4
    assert generator.kwargs["num_rows"] == 1
    assert generator.kwargs["num_cols"] == 2
    assert generator.kwargs["curriculum"] is False
    assert generator.kwargs["use_cache"] is False

    steps = generator.kwargs["sub_terrains"]["steps"]
    assert isinstance(steps, _MeshPyramidStairsTerrainCfg)
    assert steps.kwargs["proportion"] == 1.0
    assert steps.kwargs["step_height_range"] == (0.05, 0.05)
    assert steps.kwargs["step_width"] == 0.25
    assert steps.kwargs["platform_width"] == 1.0
    assert steps.kwargs["border_width"] == 0.0


def test_generator_terrain_resolves_builtin_sub_terrain_types():
    """GeneratorTerrain maps registry type strings onto the fake cfg classes."""
    preset = GeneratorTerrain(
        {
            "num_rows": 2,
            "num_cols": 3,
            "size": [6.0, 6.0],
            "friction": 0.6,
            "sub_terrains": {
                "flat": {"type": "mesh_plane", "proportion": 0.25},
                "rough": {
                    "type": "hf_random_uniform",
                    "proportion": 0.75,
                    "params": {"noise_range": [0.02, 0.05]},
                },
            },
        }
    )

    importer = preset.build(2, 5.0, classes=_CLASSES)["terrain"]

    assert isinstance(importer, _TerrainImporterCfg)
    assert _material(importer).kwargs == {
        "static_friction": 0.6,
        "dynamic_friction": 0.6,
        "restitution": 0.0,
    }
    generator = importer.kwargs["terrain_generator"]
    assert isinstance(generator, _TerrainGeneratorCfg)
    assert generator.kwargs["num_rows"] == 2
    assert generator.kwargs["num_cols"] == 3
    assert generator.kwargs["size"] == (6.0, 6.0)
    assert generator.kwargs["horizontal_scale"] == 0.1
    assert generator.kwargs["vertical_scale"] == 0.005
    assert generator.kwargs["slope_threshold"] == 0.75

    flat = generator.kwargs["sub_terrains"]["flat"]
    assert isinstance(flat, _MeshPlaneTerrainCfg)
    assert flat.kwargs == {"proportion": 0.25}
    rough = generator.kwargs["sub_terrains"]["rough"]
    assert isinstance(rough, _HfRandomUniformTerrainCfg)
    assert rough.kwargs == {"proportion": 0.75, "noise_range": [0.02, 0.05]}


def test_generator_terrain_resolves_a_custom_height_field_function():
    """A ``custom:`` entry wraps the decorated height-field producer."""
    path = "unigym.backends.shared.isaaclab3_common.terrains.custom.gap.gap_terrain"
    preset = GeneratorTerrain(
        {
            "sub_terrains": {
                "gap": {
                    "type": f"custom:{path}",
                    "proportion": 0.4,
                    "params": {"gap_width": 0.6, "gap_depth": -1.0},
                },
            },
        }
    )

    importer = preset.build(1, 4.0, classes=_CLASSES)["terrain"]

    entry = importer.kwargs["terrain_generator"].kwargs["sub_terrains"]["gap"]
    assert isinstance(entry, _HfTerrainBaseCfg)
    assert entry.kwargs == {"proportion": 0.4}
    assert entry.gap_width == 0.6
    assert entry.gap_depth == -1.0
    from unigym.backends.shared.isaaclab3_common.terrains.custom.gap import gap_terrain

    assert entry.function is gap_terrain


def test_generator_terrain_rejects_an_unknown_sub_terrain_type():
    """An unknown type string fails with the closed registry named."""
    preset = GeneratorTerrain({"sub_terrains": {"moon": {"type": "hf_moon"}}})

    with pytest.raises(ValueError, match="unknown sub-terrain type 'hf_moon'"):
        preset.build(1, 4.0, classes=_CLASSES)


def test_native_terrain_preset_samples_the_neutral_height_field():
    """NativeTerrainPreset resamples a neutral Terrain onto the il3 grid."""
    heights = np.zeros((5, 5), dtype=np.float32)
    heights[2, 2] = 0.05
    terrain = Terrain(size=(4.0, 4.0), resolution=(5, 5), heights=heights)
    preset = NativeTerrainPreset(terrain, friction=0.7, vertical_scale=0.001)

    full, origin = preset.height_field()
    assert full.dtype == np.int16
    assert full.shape == (5, 5)
    assert origin == (2.0, 2.0, pytest.approx(0.05, abs=1e-9))

    fields = preset.build(2, 5.0, classes=_CLASSES)

    assert set(fields) == {"terrain"}
    importer = fields["terrain"]
    assert isinstance(importer, _TerrainImporterCfg)
    assert _material(importer).kwargs == {
        "static_friction": 0.7,
        "dynamic_friction": 0.7,
        "restitution": 0.0,
    }
    generator = importer.kwargs["terrain_generator"]
    assert isinstance(generator, _TerrainGeneratorCfg)
    assert generator.kwargs["size"] == (4.0, 4.0)
    assert generator.kwargs["num_rows"] == 1
    assert generator.kwargs["num_cols"] == 2
    assert generator.kwargs["horizontal_scale"] == 1.0
    assert generator.kwargs["vertical_scale"] == 0.001
    native = generator.kwargs["sub_terrains"]["native"]
    assert isinstance(native, _HfTerrainBaseCfg)
    assert native.kwargs == {"proportion": 1.0}


def test_native_terrain_preset_refuses_a_too_coarse_resolution():
    """The decorator reserves a border, so a 2x2 IR has no interior."""
    terrain = Terrain(size=(4.0, 4.0), resolution=(2, 2))
    preset = NativeTerrainPreset(terrain)

    with pytest.raises(ValueError, match="too coarse"):
        preset.build(1, 4.0, classes=_CLASSES)


def test_terrain_cfg_tables_require_rigid_body_material_base_cfg():
    """The material key is IsaacLab 3's ``RigidBodyMaterialBaseCfg``."""
    table = {key: value for key, value in _CLASSES.items() if key != "RigidBodyMaterialBaseCfg"}
    assert "RigidBodyMaterialCfg" not in table

    with pytest.raises(KeyError) as excinfo:
        FlatTerrain().build(1, 4.0, classes=table)

    assert "RigidBodyMaterialBaseCfg" in str(excinfo.value)


def test_terrains_package_exports_and_factory_dispatch():
    """The package re-exports the presets and dispatches the canonical names."""
    assert set(TERRAIN_REGISTRY) == {"flat", "steps", "native", "generator"}
    assert TERRAIN_REGISTRY["flat"] is FlatTerrain
    assert TERRAIN_REGISTRY["steps"] is StepTerrain
    assert TERRAIN_REGISTRY["native"] is NativeTerrainPreset
    assert TERRAIN_REGISTRY["generator"] is GeneratorTerrain

    fields = build_terrain({"name": "flat", "params": {"friction": 0.5}}, classes=_CLASSES)

    assert set(fields) == {"ground"}

    with pytest.raises(ValueError, match="unknown terrain 'moon'"):
        build_terrain({"name": "moon"}, classes=_CLASSES)


@pytest.mark.skipif(_ISAACLAB_INSTALLED, reason="requires the d0 base environment without isaaclab")
def test_height_field_decorator_warns_once_and_keeps_the_identity(monkeypatch):
    """Without isaaclab the first call warns once and returns plain height fields."""
    monkeypatch.setattr(_compat, "_warned_identity_fallback", False)

    @_compat.height_field_to_mesh
    def generator(difficulty, cfg):
        return "field"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert generator(0.0, None) == "field"
        assert generator(0.0, None) == "field"

    fallback = [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
        and "identity fallback" in str(warning.message)
    ]
    assert len(fallback) == 1


@pytest.fixture
def clean_isaaclab_modules():
    """Drop and restore any ``isaaclab`` modules a fake install imports."""
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "isaaclab" or name.startswith("isaaclab.")
    }
    for name in list(sys.modules):
        if name == "isaaclab" or name.startswith("isaaclab."):
            del sys.modules[name]
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "isaaclab" or name.startswith("isaaclab."):
                del sys.modules[name]
        sys.modules.update(saved)


def test_broken_isaaclab_install_raises_and_is_not_cached(
    tmp_path, monkeypatch, clean_isaaclab_modules
):
    """An installed-but-broken isaaclab raises ``ImportError`` on every call."""
    package = tmp_path / "isaaclab"
    height_field = package / "terrains" / "height_field"
    height_field.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "terrains" / "__init__.py").write_text("", encoding="utf-8")
    (height_field / "__init__.py").write_text("", encoding="utf-8")
    (height_field / "utils.py").write_text(
        "raise ImportError('fake broken isaaclab decorator')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(_compat, "_warned_identity_fallback", False)

    @_compat.height_field_to_mesh
    def generator(difficulty, cfg):
        return "field"

    assert _compat._isaaclab_installed() is True
    with pytest.raises(ImportError, match="refusing to fall back"):
        generator(0.0, None)
    with pytest.raises(ImportError, match="refusing to fall back"):
        generator(0.0, None)
