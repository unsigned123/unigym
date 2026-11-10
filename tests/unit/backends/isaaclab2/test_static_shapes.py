"""Static World primitives map to collision-only IsaacLab scene fields."""

from __future__ import annotations

import pytest

from unigym.backends.isaaclab2.static_shapes import build_static_scene_fields
from unigym.contacts.base import ENV_PREFIX_SENTINEL


class _Cfg:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class _AssetBaseCfg(_Cfg):
    class InitialStateCfg(_Cfg):
        pass


class _CuboidCfg(_Cfg):
    pass


class _SphereCfg(_Cfg):
    pass


class _CylinderCfg(_Cfg):
    pass


class _CapsuleCfg(_Cfg):
    pass


class _GroundPlaneCfg(_Cfg):
    pass


class _CollisionPropertiesCfg(_Cfg):
    pass


class _PreviewSurfaceCfg(_Cfg):
    pass


_CLASSES = {
    "AssetBaseCfg": _AssetBaseCfg,
    "CuboidCfg": _CuboidCfg,
    "SphereCfg": _SphereCfg,
    "CylinderCfg": _CylinderCfg,
    "CapsuleCfg": _CapsuleCfg,
    "GroundPlaneCfg": _GroundPlaneCfg,
    "CollisionPropertiesCfg": _CollisionPropertiesCfg,
    "PreviewSurfaceCfg": _PreviewSurfaceCfg,
}


def _declaration(
    *, name="shape", kind="box", params=None, pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)
):
    return {
        "name": name,
        "kind": kind,
        "params": dict(params or {}),
        "pos": pos,
        "quat": quat,
    }


def _spawn_for(**declaration_kwargs):
    fields = build_static_scene_fields([_declaration(**declaration_kwargs)], classes=_CLASSES)
    name = declaration_kwargs.get("name", "shape")
    return fields[f"static_{name}"]


@pytest.mark.parametrize(
    ("kind", "params", "expected_cls", "expected"),
    [
        ("box", {"size": [1.0, 2.0, 3.0]}, _CuboidCfg, {"size": (1.0, 2.0, 3.0)}),
        ("sphere", {"radius": 0.5}, _SphereCfg, {"radius": 0.5}),
        (
            "cylinder",
            {"radius": 0.25, "half_height": 0.75},
            _CylinderCfg,
            {"radius": 0.25, "height": 1.5},
        ),
        (
            "capsule",
            {"radius": 0.25, "half_height": 0.75},
            _CapsuleCfg,
            {"radius": 0.25, "height": 1.5},
        ),
    ],
    ids=["box", "sphere", "cylinder", "capsule"],
)
def test_static_primitive_kinds_map_to_injected_cfg(kind, params, expected_cls, expected):
    """Each neutral primitive kind spawns through its injected cfg class."""
    asset = _spawn_for(kind=kind, params=params)
    spawn = asset.kwargs["spawn"]

    assert isinstance(asset, _AssetBaseCfg)
    assert type(spawn) is expected_cls
    for key, value in expected.items():
        assert spawn.kwargs[key] == value


def test_plane_maps_to_ground_plane_with_doubled_visual_extent():
    """A plane stays a GroundPlaneCfg and doubles size into the visual extent."""
    spawn = _spawn_for(kind="plane", params={"size": [1.5, 0.5]}).kwargs["spawn"]

    assert type(spawn) is _GroundPlaneCfg
    assert spawn.kwargs["size"] == (3.0, 1.0)


def test_generated_spawn_is_collision_only_without_rigid_body_props():
    """The spawner carries collision and visual cfg but no rigid-body cfg."""
    asset = _spawn_for(
        name="crate",
        kind="box",
        params={"size": [1.0, 1.0, 1.0]},
        pos=(1.0, 2.0, 3.0),
        quat=(0.0, 0.0, 0.0, 1.0),
    )
    spawn = asset.kwargs["spawn"]

    assert isinstance(spawn.kwargs["collision_props"], _CollisionPropertiesCfg)
    assert spawn.kwargs["collision_props"].kwargs == {"collision_enabled": True}
    assert isinstance(spawn.kwargs["visual_material"], _PreviewSurfaceCfg)
    assert "rigid_props" not in spawn.kwargs
    assert asset.kwargs["prim_path"] == f"{ENV_PREFIX_SENTINEL}/Static_crate"
    assert asset.kwargs["init_state"].kwargs == {
        "pos": (1.0, 2.0, 3.0),
        "rot": (0.0, 0.0, 0.0, 1.0),
    }


def test_duplicate_declaration_names_fail_fast():
    """Two declarations with one name collide on the scene field."""
    declarations = [
        _declaration(name="crate", kind="box", params={"size": [1.0, 1.0, 1.0]}),
        _declaration(name="crate", kind="sphere", params={"radius": 0.5}),
    ]

    with pytest.raises(ValueError, match="duplicate scene field 'static_crate'"):
        build_static_scene_fields(declarations, classes=_CLASSES)


def test_unknown_primitive_kind_fails_fast_with_known_kinds():
    """An unsupported kind is rejected with the closed vocabulary named."""
    with pytest.raises(ValueError) as excinfo:
        build_static_scene_fields([_declaration(kind="wedge")], classes=_CLASSES)

    message = str(excinfo.value)
    assert "kind 'wedge' has no isaaclab static mapping" in message
    assert "known: ['box', 'capsule', 'cylinder', 'plane', 'sphere']" in message
