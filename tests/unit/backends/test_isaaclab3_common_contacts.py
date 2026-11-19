"""Shared IsaacLab 3 contact refs: prim paths, activation, F alignment."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from unigym.backends.shared.isaaclab3_common.contacts import IsaacLabContactBackend
from unigym.contacts.manager import build_contact_manager


class _Cfg:
    def __init__(self, **kwargs):
        self.kw = kwargs


class _InitialStateCfg(_Cfg):
    pass


class _RigidObjectCfg(_Cfg):
    InitialStateCfg = _InitialStateCfg


class _SensorCfg(_Cfg):
    pass


@pytest.fixture
def fake_isaaclab(monkeypatch):
    package = types.ModuleType("isaaclab")
    sensors = types.ModuleType("isaaclab.sensors")
    sensors.ContactSensorCfg = _SensorCfg
    package.sensors = sensors
    monkeypatch.setitem(sys.modules, "isaaclab", package)
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", sensors)
    return sensors


def _manager(groups, *, props=None, articulated=None, names=("tiny",)):
    manager = build_contact_manager(
        groups,
        backend="isaaclab3_physx",
        articulation_names=names,
        prop_names=props,
        articulated_props=articulated,
    )
    assert manager is not None
    assert manager.backend.backend_key == "isaaclab3_physx"
    return manager


def test_implicit_single_articulation_robot_path(fake_isaaclab):
    manager = _manager(
        [
            {
                "name": "feet",
                "sensor_refs": [{"owner": "robot", "body": "left_foot"}],
                "frame": "world",
            }
        ]
    )
    fields = manager.setup_scene()
    assert fields["contact_feet_s0"].kw["prim_path"] == "{ENV_REGEX_NS}/Robot/left_foot"


def test_nested_robot_scope_path(fake_isaaclab):
    manager = _manager(
        [
            {
                "name": "finger",
                "sensor_refs": [
                    {"owner": "robot", "articulation": "g1", "body": "left_hand/finger_1"}
                ],
                "frame": "world",
            }
        ],
        names=("g1",),
    )
    assert (
        manager.setup_scene()["contact_finger_s0"].kw["prim_path"]
        == "{ENV_REGEX_NS}/Robot/left_hand/finger_1"
    )


def test_robot_refs_use_declaration_order_and_articulation_leaf(fake_isaaclab):
    manager = _manager(
        [
            {
                "name": "hands",
                "sensor_refs": [
                    {"owner": "robot", "articulation": "left_hand", "body": "left_.*"},
                    {"owner": "robot", "articulation": "right_hand", "body": "right_.*"},
                ],
                "frame": "world",
            }
        ],
        names=("cartpole", "left_hand", "right_hand"),
    )
    fields = manager.setup_scene()
    assert fields["contact_hands_s0"].kw["prim_path"] == "{ENV_REGEX_NS}/Robot_left_hand/(left_.*)"
    assert (
        fields["contact_hands_s1"].kw["prim_path"] == "{ENV_REGEX_NS}/Robot_right_hand/(right_.*)"
    )


def test_body_set_expands_to_a_union_path(fake_isaaclab):
    manager = build_contact_manager(
        [
            {
                "name": "grasp",
                "sensor_refs": [{"owner": "robot", "articulation": "g1", "body_set": "grasp"}],
                "frame": "world",
            }
        ],
        backend="isaaclab3_physx",
        articulation_names=("g1",),
        articulation_body_sets={"g1": {"grasp": ("index_0", "thumb_2")}},
    )
    assert manager is not None
    assert (
        manager.setup_scene()["contact_grasp_s0"].kw["prim_path"]
        == "{ENV_REGEX_NS}/Robot/(index_0|thumb_2)"
    )


def test_unknown_body_set_is_a_teaching_error(fake_isaaclab):
    manager = _manager(
        [
            {
                "name": "grasp",
                "sensor_refs": [{"owner": "robot", "body_set": "ghost"}],
                "frame": "world",
            }
        ],
        names=("g1",),
    )
    with pytest.raises(ValueError, match="body_set 'ghost' is not declared"):
        manager.setup_scene()


def test_object_sensor_and_robot_filter_paths(fake_isaaclab):
    manager = _manager(
        [
            {
                "name": "crate_tip",
                "sensor_refs": [{"owner": "object", "name": "crate"}],
                "filter_refs": [{"owner": "robot", "articulation": "tiny", "body": "tip"}],
                "frame": "world",
            }
        ],
        props=["crate"],
    )
    cfg = manager.setup_scene()["contact_crate_tip_s0"]
    assert cfg.kw["prim_path"] == "{ENV_REGEX_NS}/Prop_crate"
    assert cfg.kw["filter_prim_paths_expr"] == ["{ENV_REGEX_NS}/Robot/tip"]


def test_object_body_defaults_to_the_scope_root(fake_isaaclab):
    manager = _manager(
        [
            {
                "name": "crate",
                "sensor_refs": [{"owner": "object", "name": "crate"}],
                "frame": "world",
            }
        ],
        props=["crate"],
    )
    assert manager.setup_scene()["contact_crate_s0"].kw["prim_path"] == "{ENV_REGEX_NS}/Prop_crate"


def test_cross_articulation_filter_paths(fake_isaaclab):
    manager = _manager(
        [
            {
                "name": "cross",
                "sensor_refs": [{"owner": "robot", "articulation": "g1", "body": "tip"}],
                "filter_refs": [{"owner": "robot", "articulation": "cartpole", "body": "pole"}],
                "frame": "world",
            }
        ],
        names=("cartpole", "g1"),
    )
    cfg = manager.setup_scene()["contact_cross_s0"]
    assert cfg.kw["prim_path"] == "{ENV_REGEX_NS}/Robot_g1/tip"
    assert cfg.kw["filter_prim_paths_expr"] == ["{ENV_REGEX_NS}/Robot/pole"]


def test_single_articulation_body_set_omits_the_owner(fake_isaaclab):
    manager = build_contact_manager(
        [
            {
                "name": "grasp",
                "sensor_refs": [{"owner": "robot", "body_set": "grasp"}],
                "frame": "world",
            }
        ],
        backend="isaaclab3_physx",
        articulation_names=("tiny",),
        articulation_body_sets={"tiny": {"grasp": ("tip", "base")}},
    )
    assert manager is not None
    assert (
        manager.setup_scene()["contact_grasp_s0"].kw["prim_path"]
        == "{ENV_REGEX_NS}/Robot/(tip|base)"
    )


def test_unknown_object_ref_fails_at_setup(fake_isaaclab):
    manager = _manager(
        [
            {
                "name": "ghost",
                "sensor_refs": [{"owner": "object", "name": "ghost"}],
                "frame": "world",
            }
        ],
        props=["crate"],
    )
    with pytest.raises(ValueError, match="undeclared object prop 'ghost'") as err:
        manager.setup_scene()
    assert "crate" in str(err.value)


def test_unknown_object_ref_has_no_synthesized_inventory_row():
    manager = _manager(
        [
            {
                "name": "ghost",
                "sensor_refs": [{"owner": "object", "name": "ghost"}],
                "frame": "world",
            }
        ],
        props=["crate"],
    )
    with pytest.raises(ValueError, match="unknown object scope") as err:
        manager.bind_env(_env())
    assert "object/crate/crate" in str(err.value)


@pytest.mark.parametrize(
    "ref",
    [
        {"owner": "support", "name": "table"},
        {"owner": "environment", "body": "world"},
        {"owner": "environment", "path": "{ENV_REGEX_NS}/World/ground"},
    ],
)
def test_static_support_and_environment_refs_are_refused(fake_isaaclab, ref):
    manager = _manager(
        [
            {
                "name": "stat",
                "sensor_refs": [{"owner": "object", "name": "crate"}],
                "filter_refs": [ref],
                "frame": "world",
            }
        ],
        props=["crate"],
    )
    with pytest.raises(NotImplementedError, match="static support/environment"):
        manager.setup_scene()


def test_non_root_fixture_link_ref_is_refused(fake_isaaclab):
    manager = _manager(
        [
            {
                "name": "mw",
                "sensor_refs": [
                    {"owner": "fixture", "name": "microwave", "body": "microwave_main"}
                ],
                "frame": "world",
            }
        ],
        props=["microwave"],
        articulated={"microwave"},
    )
    with pytest.raises(NotImplementedError, match="non-root articulated prop"):
        manager.setup_scene()


def test_unknown_articulation_ref_is_rejected(fake_isaaclab):
    with pytest.raises(ValueError, match="not declared by the run"):
        _manager(
            [
                {
                    "name": "feet",
                    "sensor_refs": [{"owner": "robot", "articulation": "ghost", "body": "foot"}],
                    "frame": "world",
                }
            ],
            names=("cartpole", "g1"),
        )


class _Proxy:
    """IsaacLab 3 ``ProxyArray`` stand-in exposing only ``.torch``."""

    def __init__(self, value: torch.Tensor):
        self.torch = value


class _Data:
    def __init__(self, bodies, forces=None, matrix=None):
        self.body_names = list(bodies)
        self.net_normal_forces_w = None if forces is None else _Proxy(forces)
        self.normal_force_matrix_w = None if matrix is None else _Proxy(matrix)


class _Sensor:
    def __init__(self, bodies, forces=None, matrix=None):
        self.data = _Data(bodies, forces, matrix)
        self.cfg = types.SimpleNamespace(prim_path="?")

    @property
    def body_names(self):
        return self.data.body_names


def _env(bodies=("tip", "palm")):
    robot = types.SimpleNamespace(data=types.SimpleNamespace(body_names=list(bodies)))
    return types.SimpleNamespace(scene={"robot": robot})


def _attach(backend, spec, sensors, perm, filters=()):
    backend._sensors[spec.name] = {"mode": spec.mode, "sensors": sensors}
    backend._perm[spec.name] = perm
    backend._filters[spec.name] = list(filters)


def test_bind_resolves_object_sensor_and_robot_filter_axis():
    manager = _manager(
        [
            {
                "name": "crate_tip",
                "sensor_refs": [{"owner": "object", "name": "crate"}],
                "filter_refs": [{"owner": "robot", "body": "tip"}],
                "frame": "world",
            }
        ],
        props=["crate"],
    )
    env = _env()
    env.scene["contact_crate_tip_s0"] = _Sensor(
        ["crate"], matrix=torch.tensor([[[[1.0, 0.0, 0.0]]]])
    )
    manager.bind_env(env)

    assert manager.backend.group_axes(manager.specs[0]) == (1, 1)
    out = manager.backend.read_raw(manager.specs[0])
    assert tuple(out.shape) == (1, 1, 1, 3)


def test_f_axis_drift_asserts_instead_of_silently_mismatching():
    manager = _manager(
        [
            {
                "name": "crate_tip",
                "sensor_refs": [{"owner": "object", "name": "crate"}],
                "filter_refs": [{"owner": "robot", "body": "tip"}],
                "frame": "world",
            }
        ],
        props=["crate"],
    )
    env = _env()
    env.scene["contact_crate_tip_s0"] = _Sensor(["crate"], matrix=torch.zeros(1, 1, 2, 3))
    manager.bind_env(env)

    with pytest.raises(AssertionError, match="resolved filter refs"):
        manager.backend.read_raw(manager.specs[0])


def test_multi_body_filter_ref_is_refused_at_bind():
    manager = _manager(
        [
            {
                "name": "crate_hand",
                "sensor_refs": [{"owner": "object", "name": "crate"}],
                "filter_refs": [{"owner": "robot", "body": ".*"}],
                "frame": "world",
            }
        ],
        props=["crate"],
    )
    env = _env()
    env.scene["contact_crate_hand_s0"] = _Sensor(["crate"])

    with pytest.raises(ValueError, match="exactly one body"):
        manager.bind_env(env)


def test_frame_body_remains_unsupported():
    manager = _manager(
        [
            {
                "name": "hand",
                "sensor_refs": [{"owner": "robot", "body": "tip"}],
                "frame": "body",
            }
        ]
    )
    with pytest.raises(NotImplementedError, match="frame='body'"):
        manager.bind_env(_env())


def test_general_read_concatenates_refs_and_permutes_body_axis():
    from unigym.contacts.base import ContactGroupSpec

    spec = ContactGroupSpec(
        name="hands",
        sensor_refs=[
            {"owner": "robot", "body": "left_.*"},
            {"owner": "robot", "body": "right_tip"},
        ],
        frame="world",
    )
    backend = IsaacLabContactBackend([spec])
    left = _Sensor(
        ["left_a", "left_b"],
        forces=torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
    )
    right = _Sensor(
        ["right_x", "right_tip"],
        forces=torch.tensor([[[8.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]),
    )
    _attach(backend, spec, [left, right], [[0, 1], [1]])

    out = backend.read_raw(spec)

    assert out.shape == (1, 3, 3)
    assert torch.equal(out[0, :, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_pairwise_read_concatenates_entries_and_clones():
    from unigym.contacts.base import ContactGroupSpec

    spec = ContactGroupSpec(
        name="feet_props",
        sensor_refs=[
            {"owner": "robot", "body": "left_foot"},
            {"owner": "robot", "body": "right_foot"},
        ],
        filter_refs=[{"owner": "robot", "body": "ball"}],
        frame="world",
    )
    backend = IsaacLabContactBackend([spec])
    left = _Sensor(["left_foot"], matrix=torch.tensor([[[[1.0, 0.0, 0.0]]]]))
    right = _Sensor(["right_foot"], matrix=torch.tensor([[[[2.0, 0.0, 0.0]]]]))
    _attach(backend, spec, [left, right], [[0], [0]], filters=[object()])

    out = backend.read_raw(spec)

    assert out.shape == (1, 2, 1, 3)
    assert torch.equal(out[0, :, 0, 0], torch.tensor([1.0, 2.0]))
    out.fill_(-1.0)
    assert torch.equal(
        backend.read_raw(spec)[0, :, 0, 0],
        torch.tensor([1.0, 2.0]),
    )


def test_shared_backend_key_is_empty_and_caller_settable():
    default = IsaacLabContactBackend([])
    canonical = IsaacLabContactBackend([], backend_key="isaaclab3_newton")
    assert default.backend_key == ""
    assert canonical.backend_key == "isaaclab3_newton"


def test_prop_activation_flag_flows_to_the_rigid_object_cfg():
    from unigym.backends.shared.isaaclab3_common.props.handlers.generic import GenericProp
    from unigym.props.base import PropSpec

    spec = PropSpec(name="crate", type="rigid_object", urdf_path="/tmp/crate.urdf")
    handler = GenericProp(spec, num_envs=1, device="cpu")
    classes = {
        "RigidObjectCfg": _RigidObjectCfg,
        "UrdfFileCfg": _Cfg,
    }
    _, cfg = handler.setup_scene(classes=classes)
    assert cfg.kw["activate_contact_sensors"] is False

    spec._contact_sensor = True
    _, cfg = handler.setup_scene(classes=classes)
    assert cfg.kw["activate_contact_sensors"] is True
