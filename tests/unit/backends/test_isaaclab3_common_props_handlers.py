"""Shared IsaacLab 3 prop registry, spawn material key and world-pose reads."""

from __future__ import annotations

import torch
import pytest

from unigym.backends.shared.isaaclab3_common.props import IsaacLabPropBackend
from unigym.backends.shared.isaaclab3_common.props.handlers import (
    BallProp,
    BoxProp,
    GenericProp,
    PrimitiveProp,
    PropArticulation,
    UsdArticulationProp,
)
from unigym.props.base import PropSpec


class _Cfg:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class _RigidObjectCfg(_Cfg):
    class InitialStateCfg(_Cfg):
        pass


class _CylinderCfg(_Cfg):
    pass


class _CapsuleCfg(_Cfg):
    pass


class _Material(_Cfg):
    pass


def _classes() -> dict:
    """Cfg-class table keyed exactly like ``spawn._resolve`` returns it."""
    return {
        "RigidObjectCfg": _RigidObjectCfg,
        "CylinderCfg": _CylinderCfg,
        "CapsuleCfg": _CapsuleCfg,
        "CuboidCfg": _Cfg,
        "SphereCfg": _Cfg,
        "UrdfFileCfg": _Cfg,
        "UsdFileCfg": _Cfg,
        "MassPropertiesCfg": _Cfg,
        "RigidBodyPropertiesCfg": _Cfg,
        "CollisionPropertiesCfg": _Cfg,
        "PreviewSurfaceCfg": _Cfg,
        "RigidBodyMaterialBaseCfg": _Material,
    }


def test_registry_maps_the_neutral_types_to_shared_handlers():
    """The shared registry owns every declared type without an engine import."""
    registry = IsaacLabPropBackend().handler_registry

    assert registry["sphere"] is BallProp
    assert registry["rigid_object"] is GenericProp
    assert registry["usd"] is GenericProp
    assert registry["articulation"] is PropArticulation
    assert registry["usd_articulation"] is UsdArticulationProp
    assert registry["container"] is BoxProp
    assert registry["cylinder"] is PrimitiveProp
    assert registry["capsule"] is PrimitiveProp


def test_handler_for_and_build_handler_use_the_registry():
    """A cylinder spec resolves to the shared primitive handler."""
    backend = IsaacLabPropBackend()
    spec = PropSpec(name="peg", type="cylinder", radius=0.03, half_height=0.05)

    assert backend.handler_for(spec) is PrimitiveProp
    assert isinstance(backend.build_handler(spec, num_envs=1, device="cpu"), PrimitiveProp)


def test_primitive_spawn_requires_the_rigid_body_material_base_key():
    """The IsaacLab 3 material cfg name is ``RigidBodyMaterialBaseCfg``."""
    spec = PropSpec(
        name="peg",
        type="cylinder",
        radius=0.03,
        half_height=0.05,
        spawn={"friction": 1.0},
    )
    handler = PrimitiveProp(spec, num_envs=1, device="cpu")
    classes = _classes()
    classes.pop("RigidBodyMaterialBaseCfg")

    with pytest.raises(KeyError, match="RigidBodyMaterialBaseCfg"):
        handler.setup_scene(classes=classes)


def test_primitive_spawn_uses_the_shared_material_cfg_for_friction():
    """The resolved material cfg is built through the shared key."""
    spec = PropSpec(
        name="peg",
        type="cylinder",
        radius=0.03,
        half_height=0.05,
        spawn={"friction": 1.0},
    )
    handler = PrimitiveProp(spec, num_envs=1, device="cpu")

    _name, cfg = handler.setup_scene(classes=_classes())

    spawn = cfg.kwargs["spawn"]
    assert isinstance(spawn, _CylinderCfg)
    material = spawn.kwargs["physics_material"]
    assert isinstance(material, _Material)
    assert material.kwargs["static_friction"] == 1.0
    assert material.kwargs["dynamic_friction"] == 1.0


def _contact_sensor_prop(kind: str) -> PropSpec:
    if kind == "box":
        return PropSpec(
            name="crate",
            type="box",
            size=[0.1, 0.1, 0.1],
            aabb={"center": [0.0, 0.0, 0.05], "half_extents": [0.05, 0.05, 0.05]},
        )
    if kind == "sphere":
        return PropSpec(name="ball", type="sphere", radius=0.05)
    if kind == "cylinder":
        return PropSpec(name="peg", type="cylinder", radius=0.03, half_height=0.05)
    if kind == "rigid_object":
        return PropSpec(name="crate", type="rigid_object", urdf_path="/tmp/crate.urdf")
    if kind == "usd":
        return PropSpec(name="crate", type="usd", usd_path="/tmp/crate.usda")
    raise AssertionError(f"unknown prop kind {kind!r}")


@pytest.mark.parametrize("kind", ("box", "sphere", "cylinder", "rigid_object", "usd"))
def test_il3_prop_contact_sensor_flag_goes_to_the_spawner_not_the_asset_cfg(kind):
    """IL3 ``RigidObjectCfg`` has no flag; the spawner cfg owns it."""
    spec = _contact_sensor_prop(kind)
    setattr(spec, "_contact_sensor", True)
    handler = IsaacLabPropBackend().build_handler(spec, num_envs=1, device="cpu")

    _name, cfg = handler.setup_scene(classes=_classes())

    assert "activate_contact_sensors" not in cfg.kwargs
    assert cfg.kwargs["spawn"].kwargs["activate_contact_sensors"] is True


class _AssetData:
    def __init__(self, pos: torch.Tensor, quat: torch.Tensor):
        self.root_pos_w = pos
        self.root_quat_w = quat


class _Asset:
    def __init__(self):
        self.data = _AssetData(
            torch.tensor([[0.1, 0.2, 0.3]]),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        )


class _Scene:
    def __init__(self, items: dict):
        self._items = dict(items)

    def __getitem__(self, name: str):
        return self._items[name]


class _Env:
    device = "cpu"

    def __init__(self, items: dict):
        self.scene = _Scene(items)


def _ball() -> BallProp:
    spec = PropSpec(name="ball", type="sphere", radius=0.05)
    return BallProp(spec, num_envs=1, device="cpu")


def test_ball_get_world_pose_returns_tensors_after_bind():
    """A bound handler returns cloned (N,3) pos and wxyz (N,4) quat tensors."""
    handler = _ball()
    handler.bind_env(_Env({"prop_ball": _Asset()}))

    pose = handler.get_world_pose(_Env({}))

    assert pose is not None
    pos, quat = pose
    assert isinstance(pos, torch.Tensor)
    assert isinstance(quat, torch.Tensor)
    assert pos.shape == (1, 3)
    assert quat.shape == (1, 4)
    assert quat[0, 0].item() == 1.0


def test_ball_get_world_pose_is_none_before_bind():
    """An unbound handler keeps the honest null pose."""
    assert _ball().get_world_pose(_Env({})) is None
