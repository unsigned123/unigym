"""Static World primitives map to collision-only IsaacLab scene fields."""

from __future__ import annotations

import pytest

from unigym.backends.isaaclab2.static_shapes import (
    build_static_field,
    build_static_payload_fields,
    build_static_scene_fields,
)
from unigym.contacts.base import ENV_PREFIX_SENTINEL
from unigym.scenes.primitives import BoxShape, ShapePayload


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


def test_build_static_field_extracts_the_prim_override():
    """The extracted builder assigns the explicit prim name."""
    field = build_static_field(
        "crate",
        "box",
        {"size": [1.0, 2.0, 3.0]},
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0, 1.0),
        classes=_CLASSES,
        prim_path=f"{ENV_PREFIX_SENTINEL}/Static/custom",
    )

    assert isinstance(field, _AssetBaseCfg)
    assert field.kwargs["prim_path"] == f"{ENV_PREFIX_SENTINEL}/Static/custom"
    assert field.kwargs["init_state"].kwargs == {
        "pos": (1.0, 2.0, 3.0),
        "rot": (0.0, 0.0, 0.0, 1.0),
    }


def test_build_static_payload_fields_single_shape_keeps_world_add_naming():
    """A single-shape payload keeps static_<id> and the shape's own pose."""
    payload = ShapePayload(
        (
            BoxShape(
                name="crate_geom",
                size=(1.0, 2.0, 3.0),
                material="wood",
                pos=(1.0, 2.0, 3.0),
                quat_wxyz=(0.0, 0.0, 0.0, 1.0),
                layer="collision",
            ),
        )
    )

    fields = build_static_payload_fields(payload, instance_id="crate", classes=_CLASSES)

    assert set(fields) == {"static_crate"}
    field = fields["static_crate"]
    assert field.kwargs["prim_path"] == f"{ENV_PREFIX_SENTINEL}/Static_crate"
    spawn = field.kwargs["spawn"]
    assert isinstance(spawn, _CuboidCfg)
    assert spawn.kwargs["size"] == (1.0, 2.0, 3.0)
    assert field.kwargs["init_state"].kwargs == {
        "pos": (1.0, 2.0, 3.0),
        "rot": (0.0, 0.0, 0.0, 1.0),
    }


def test_build_static_payload_fields_multi_shape_uses_parent_prim_and_own_poses():
    """A multi-shape payload gets static_<id>_<shape> under Static_<id>."""
    payload = ShapePayload(
        (
            BoxShape(
                name="floor",
                size=(2.0, 2.0, 0.1),
                material="wood",
                pos=(0.0, 0.0, -0.05),
                layer="collision",
            ),
            BoxShape(
                name="wall",
                size=(0.1, 2.0, 2.0),
                material="wood",
                pos=(1.05, 0.0, 1.0),
                layer="collision",
            ),
        )
    )

    fields = build_static_payload_fields(payload, instance_id="room", classes=_CLASSES)

    assert set(fields) == {"static_room_floor", "static_room_wall"}
    floor = fields["static_room_floor"]
    assert floor.kwargs["prim_path"] == f"{ENV_PREFIX_SENTINEL}/Static_room/floor"
    assert floor.kwargs["init_state"].kwargs == {
        "pos": (0.0, 0.0, -0.05),
        "rot": (1.0, 0.0, 0.0, 0.0),
    }
    wall = fields["static_room_wall"]
    assert wall.kwargs["prim_path"] == f"{ENV_PREFIX_SENTINEL}/Static_room/wall"
    assert wall.kwargs["init_state"].kwargs["pos"] == (1.05, 0.0, 1.0)


def test_build_static_payload_fields_duplicate_shape_names_fail_fast():
    """Two payload shapes with one name collide on the generated field."""
    shape = BoxShape(
        name="floor",
        size=(1.0, 1.0, 1.0),
        material="wood",
        pos=(0.0, 0.0, 0.0),
        layer="collision",
    )
    payload = ShapePayload((shape, shape))

    with pytest.raises(ValueError, match="duplicate scene field 'static_room_floor'"):
        build_static_payload_fields(payload, instance_id="room", classes=_CLASSES)


def test_build_static_payload_fields_empty_payload_fails_fast():
    """An empty payload cannot silently produce zero scene fields."""
    with pytest.raises(ValueError, match="has no shapes"):
        build_static_payload_fields(ShapePayload(()), instance_id="room", classes=_CLASSES)
