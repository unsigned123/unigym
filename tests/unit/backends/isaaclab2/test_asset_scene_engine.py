"""The IsaacLab asset-scene engine composes and attaches static USD fields."""

from __future__ import annotations

import json
from dataclasses import fields as dataclass_fields

import pytest

from unigym.assets.asset_ref import AssetRef, AssetRefError
from unigym.assets.plan import AssetPlan
from unigym.backends.isaaclab2.asset_scene.engine import (
    AssetFieldSpec,
    AssetSceneEngine,
)
from unigym.backends.isaaclab2.asset_scene.inspect import UsdAuthoring
from unigym.contacts.base import ENV_PREFIX_SENTINEL
from unigym.packs.runtime import AssetContext, discover
from unigym.scenes.artifacts import ASSET_SCENE_DIGEST_VERSION, asset_instance_rows
from unigym.scenes.engine import SceneEngine, scene_engine
from unigym.scenes.scene_ir import AssetInstance, Instance, Pose, Scene
from unigym.scenes.terrain import Terrain


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


class _UsdFileCfg(_Cfg):
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
    "UsdFileCfg": _UsdFileCfg,
    "CollisionPropertiesCfg": _CollisionPropertiesCfg,
    "PreviewSurfaceCfg": _PreviewSurfaceCfg,
}


class _SceneCfg:
    pass


def _asset_plan() -> AssetPlan:
    return AssetPlan(AssetContext(discover([])))


def _usd_file(tmp_path, name: str = "cup.usd"):
    path = tmp_path / name
    path.write_text("#usda 1.0\n", encoding="utf-8")
    return path


def _static_inspector(path: str) -> UsdAuthoring:
    return UsdAuthoring()


def _engine(inspector=_static_inspector) -> AssetSceneEngine:
    return AssetSceneEngine(classes=_CLASSES, inspect_usd_authoring=inspector)


def _asset_scene(
    path,
    *,
    asset_id: str = "cup",
    kind: str = "static_rigid",
    collision: bool = True,
    pose: Pose | None = None,
    asset: AssetRef | None = None,
) -> Scene:
    return Scene(
        name="bench",
        asset_instances=(
            AssetInstance(
                id=asset_id,
                asset=AssetRef(usd=str(path)) if asset is None else asset,
                pose=Pose(xyz=(0.1, 0.2, 0.3), quat_wxyz=(0.0, 0.0, 0.0, 1.0))
                if pose is None
                else pose,
                source="scene.yaml:1",
                kind=kind,
                collision=collision,
            ),
        ),
    )


def test_asset_field_spec_keeps_the_plan_fields():
    """AssetFieldSpec carries exactly the declared field contract."""
    names = {entry.name for entry in dataclass_fields(AssetFieldSpec)}

    assert names == {
        "field",
        "prim_path",
        "cfg_kind",
        "kind",
        "format",
        "asset",
        "resolved_path",
        "pos",
        "rot",
        "collision",
        "source",
    }


def test_compose_static_usd_builds_field_and_neutral_rows(tmp_path):
    """Compose resolves a static USD and keeps rows absolute-path free."""
    path = _usd_file(tmp_path)
    scene = _asset_scene(path)
    composed = _engine().compose(scene, _asset_plan())

    assert composed.scene_name == "bench"
    assert composed.scene is scene
    (spec,) = composed.fields
    assert spec.field == "asset_bench_cup"
    assert spec.prim_path == f"{ENV_PREFIX_SENTINEL}/Scene_bench/cup"
    assert spec.cfg_kind == "static_usd"
    assert spec.kind == "static_rigid"
    assert spec.format == "usd"
    assert spec.asset == AssetRef(usd=str(path))
    assert spec.resolved_path == str(path)
    assert spec.pos == (0.1, 0.2, 0.3)
    assert spec.rot == (0.0, 0.0, 0.0, 1.0)
    assert spec.collision is True
    assert spec.source == "scene.yaml:1"
    assert composed.rows == asset_instance_rows(scene, {"cup": "usd"})
    assert composed.rows[0]["pose"]["xyz"][0] == (0.1).hex()


@pytest.mark.parametrize(
    ("collision", "expected"),
    [(True, {"collision_enabled": True}), (False, {"collision_enabled": False})],
    ids=["collision-on", "collision-off"],
)
def test_attach_static_usd_builds_asset_base_cfg_without_rigid_props(tmp_path, collision, expected):
    """Attach builds AssetBaseCfg + UsdFileCfg with the collision toggle."""
    path = _usd_file(tmp_path)
    scene = _asset_scene(path, collision=collision)
    engine = _engine()
    composed = engine.compose(scene, _asset_plan())
    base = _SceneCfg()

    attached = engine.attach(base, composed, frame="world")

    assert attached == {}
    cfg = base.asset_bench_cup
    assert isinstance(cfg, _AssetBaseCfg)
    assert cfg.kwargs["prim_path"] == f"{ENV_PREFIX_SENTINEL}/Scene_bench/cup"
    spawn = cfg.kwargs["spawn"]
    assert isinstance(spawn, _UsdFileCfg)
    assert spawn.kwargs["usd_path"] == str(path)
    assert "rigid_props" not in spawn.kwargs
    collision_props = spawn.kwargs["collision_props"]
    assert isinstance(collision_props, _CollisionPropertiesCfg)
    assert collision_props.kwargs == expected
    init_state = cfg.kwargs["init_state"]
    assert isinstance(init_state, _AssetBaseCfg.InitialStateCfg)
    assert init_state.kwargs == {
        "pos": (0.1, 0.2, 0.3),
        "rot": (0.0, 0.0, 0.0, 1.0),
    }


def test_attach_inspects_the_resolved_usd_path(tmp_path):
    """The static check receives the resolved absolute path."""
    path = _usd_file(tmp_path)
    calls = []

    def inspector(value):
        calls.append(value)
        return UsdAuthoring()

    engine = _engine(inspector)
    composed = engine.compose(_asset_scene(path), _asset_plan())
    engine.attach(_SceneCfg(), composed, frame="world")

    assert calls == [str(path)]


def test_describe_is_deterministic_and_has_no_cfg_names_or_resolved_path(tmp_path):
    """Describe carries neutral rows and a declared field slice only."""
    path = _usd_file(tmp_path)
    scene = _asset_scene(path)
    engine = _engine()
    first = engine.describe(engine.compose(scene, _asset_plan()))
    second = engine.describe(engine.compose(scene, _asset_plan()))

    assert first == second
    assert first["digest_version"] == ASSET_SCENE_DIGEST_VERSION
    assert first["asset_scene"]["digest_version"] == ASSET_SCENE_DIGEST_VERSION
    assert first["asset_scene"]["instances"] == [
        dict(row) for row in engine.compose(scene, _asset_plan()).rows
    ]
    assert first["asset_scene"]["instances"][0]["asset"] == {"usd": str(path)}
    assert first["ext"]["scene"]["fields"] == [
        {
            "field": "asset_bench_cup",
            "prim_path": f"{ENV_PREFIX_SENTINEL}/Scene_bench/cup",
            "kind": "static_rigid",
            "format": "usd",
        }
    ]
    text = json.dumps(first, sort_keys=True)
    assert "resolved_path" not in text
    assert "AssetBaseCfg" not in text
    assert "UsdFileCfg" not in text


def test_describe_keeps_an_authored_absolute_ref_verbatim(tmp_path):
    """The digest carries declared refs verbatim; use pack refs for a portable digest."""
    path = _usd_file(tmp_path)
    engine = _engine()
    described = engine.describe(engine.compose(_asset_scene(path), _asset_plan()))

    (row,) = described["asset_scene"]["instances"]
    assert row["asset"] == {"usd": str(path)}
    serialized = json.dumps(described, sort_keys=True)
    assert "resolved_path" not in serialized
    assert str(path) in serialized


def test_to_xml_is_canonical_json_and_may_carry_the_absolute_path(tmp_path):
    """The artifact text is canonical JSON and includes the resolved path."""
    path = _usd_file(tmp_path)
    engine = _engine()
    composed = engine.compose(_asset_scene(path), _asset_plan())

    text = engine.to_xml(composed)
    payload = json.loads(text)

    assert text == json.dumps(payload, sort_keys=True, indent=2)
    (field,) = payload["fields"]
    assert field["field"] == "asset_bench_cup"
    assert field["prim_path"] == f"{ENV_PREFIX_SENTINEL}/Scene_bench/cup"
    assert field["kind"] == "static_rigid"
    assert field["format"] == "usd"
    assert field["asset"] == {"usd": str(path)}
    assert field["resolved_path"] == str(path)
    assert field["pose"] == {
        "pos": [0.1, 0.2, 0.3],
        "rot": [0.0, 0.0, 0.0, 1.0],
    }
    assert field["collision"] is True


@pytest.mark.parametrize(
    ("authored", "needle"),
    [
        (UsdAuthoring(rigid_body_paths=("/World/Cup",)), "RigidBodyAPI"),
        (
            UsdAuthoring(articulation_root_paths=("/World/Cup",)),
            "ArticulationRootAPI",
        ),
    ],
    ids=["rigid-body", "articulation-root"],
)
def test_attach_refuses_a_static_usd_that_authors_dynamic_schemas(tmp_path, authored, needle):
    """A USD authored with a rigid body or articulation root is refused."""
    path = _usd_file(tmp_path)

    def inspector(value):
        assert value == str(path)
        return authored

    engine = _engine(inspector)
    composed = engine.compose(_asset_scene(path), _asset_plan())
    base = _SceneCfg()

    with pytest.raises(ValueError, match=needle):
        engine.attach(base, composed, frame="world")

    assert not hasattr(base, "asset_bench_cup")


def test_compose_refuses_static_urdf(tmp_path):
    """A static URDF has no IsaacLab static field form."""
    path = tmp_path / "cup.urdf"
    path.write_text("<robot name='cup'/>", encoding="utf-8")
    scene = _asset_scene(path, asset=AssetRef(urdf=str(path)))

    with pytest.raises(ValueError, match="static_rigid URDF"):
        _engine().compose(scene, _asset_plan())


def test_compose_refuses_dynamic_assets_as_c5(tmp_path):
    """Dynamic file-backed assets are refused by the static-only scaffold."""
    path = _usd_file(tmp_path)
    scene = _asset_scene(path, kind="dynamic_rigid")

    with pytest.raises(
        ValueError,
        match="dynamic file-backed assets are not supported by the static asset-scene scaffold yet",
    ):
        _engine().compose(scene, _asset_plan())


def test_compose_rejects_mjcf_only_asset_from_format_selection(tmp_path):
    """An MJCF-only AssetRef fails in format selection before any field."""
    path = tmp_path / "cup.xml"
    path.write_text("<mujoco/>", encoding="utf-8")
    scene = _asset_scene(path, asset=AssetRef(mjcf=str(path)))

    with pytest.raises(AssetRefError, match="no compatible format"):
        _engine().compose(scene, _asset_plan())


def test_compose_refuses_scene_terrain():
    """The run terrain owns the ground; an asset scene declares none."""
    scene = Scene(name="bench", terrain=Terrain(size=(1.0, 1.0)))

    with pytest.raises(ValueError, match="terrain is not supported"):
        _engine().compose(scene, _asset_plan())


def test_compose_refuses_scene_engine_hints():
    """Engine hints are the MuJoCo-only vocabulary."""
    scene = Scene(name="bench", engine_hints={"gravity": [0.0, 0.0, -9.81]})

    with pytest.raises(ValueError, match="engine_hints are not supported"):
        _engine().compose(scene, _asset_plan())


def test_compose_refuses_a_non_primitive_instance():
    """An unregistered generator has no IsaacLab file-backed realization."""
    scene = Scene(
        name="bench",
        instances=(
            Instance(
                id="cup",
                generator="custom_prop",
                params={},
                pose=Pose(),
                source="scene.yaml:1",
            ),
        ),
    )

    with pytest.raises(ValueError, match="has no IsaacLab file-backed realization"):
        _engine().compose(scene, _asset_plan())


def test_attach_rejects_an_existing_scene_field(tmp_path):
    """A field that already exists on the base fails before any assignment."""
    path = _usd_file(tmp_path)
    engine = _engine()
    composed = engine.compose(_asset_scene(path), _asset_plan())
    base = _SceneCfg()
    base.asset_bench_cup = object()

    with pytest.raises(ValueError, match="collides with an existing scene entry"):
        engine.attach(base, composed, frame="world")


def test_attach_builds_single_shape_primitive_with_world_add_naming():
    """A primitive instance keeps static_<id> / Static_<id> and its pose."""
    scene = Scene(
        name="bench",
        instances=(
            Instance(
                id="crate",
                generator="box",
                params={"size": (1.0, 2.0, 3.0), "material": "wood"},
                pose=Pose(xyz=(1.0, 2.0, 3.0)),
                source="scene.yaml:1",
            ),
        ),
    )
    engine = _engine()
    composed = engine.compose(scene, _asset_plan())
    base = _SceneCfg()

    engine.attach(base, composed, frame="world")

    cfg = base.static_crate
    assert isinstance(cfg, _AssetBaseCfg)
    assert cfg.kwargs["prim_path"] == f"{ENV_PREFIX_SENTINEL}/Static_crate"
    spawn = cfg.kwargs["spawn"]
    assert isinstance(spawn, _CuboidCfg)
    assert spawn.kwargs["size"] == (1.0, 2.0, 3.0)
    assert cfg.kwargs["init_state"].kwargs == {
        "pos": (1.0, 2.0, 3.0),
        "rot": (1.0, 0.0, 0.0, 0.0),
    }


def test_attach_builds_multi_shape_primitive_fields_under_a_parent_prim():
    """A room shell maps every slab to static_<id>_<shape>/Static_<id>/<shape>."""
    scene = Scene(
        name="bench",
        instances=(
            Instance(
                id="room",
                generator="room_shell",
                params={
                    "width": 2.0,
                    "depth": 3.0,
                    "height": 2.5,
                    "wall_material": "wood",
                    "floor_material": "wood",
                },
                pose=Pose(),
                source="scene.yaml:1",
            ),
        ),
    )
    engine = _engine()
    composed = engine.compose(scene, _asset_plan())
    base = _SceneCfg()

    engine.attach(base, composed, frame="world")

    expected = {
        "static_room_room_floor",
        "static_room_room_ceiling",
        "static_room_room_wall_xp",
        "static_room_room_wall_xm",
        "static_room_room_wall_yp",
        "static_room_room_wall_ym",
    }
    assert {name for name in vars(base)} == expected
    floor = base.static_room_room_floor
    assert floor.kwargs["prim_path"] == f"{ENV_PREFIX_SENTINEL}/Static_room/room_floor"
    assert isinstance(floor.kwargs["spawn"], _CuboidCfg)
    assert floor.kwargs["init_state"].kwargs == {
        "pos": (0.0, 0.0, -0.05),
        "rot": (1.0, 0.0, 0.0, 0.0),
    }
    wall = base.static_room_room_wall_xp
    assert wall.kwargs["prim_path"] == f"{ENV_PREFIX_SENTINEL}/Static_room/room_wall_xp"


def test_scene_engine_registry_returns_the_isaaclab2_engine():
    """The engine self-registers under the isaaclab2 name."""
    engine = scene_engine("isaaclab2")

    assert isinstance(engine, SceneEngine)
    assert isinstance(engine, AssetSceneEngine)
    assert engine.name == "isaaclab2"
