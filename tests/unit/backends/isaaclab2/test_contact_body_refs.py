"""IsaacLab2 contact refs: prim paths, prop activation, F alignment, refusal."""

from __future__ import annotations

import types

import pytest

from tests._support.backends import contact_factory
from tests._support.fakes.isaaclab import _Cfg, _RigidObjectCfg
from unigym.contacts.manager import build_contact_manager


def _manager(groups, *, props=None, articulated=None):
    manager = build_contact_manager(
        groups,
        backend="isaaclab2",
        articulation_names=("tiny",),
        prop_names=props,
        articulated_props=articulated,
        contact_factory=contact_factory("isaaclab2"),
    )
    assert manager is not None
    return manager


def test_object_sensor_and_robot_filter_prim_paths(fake_isaaclab):
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

    fields = manager.setup_scene()

    cfg = fields["contact_crate_tip_s0"]
    assert cfg.kw["prim_path"] == "{ENV_REGEX_NS}/Prop_crate"
    assert cfg.kw["filter_prim_paths_expr"] == ["{ENV_REGEX_NS}/Robot/tip"]


def test_general_robot_ref_builds_a_union_path(fake_isaaclab):
    manager = _manager(
        [
            {
                "name": "hand",
                "sensor_refs": [
                    {"owner": "robot", "articulation": "tiny", "body": "left_.*"},
                    {"owner": "robot", "articulation": "tiny", "body": "right_.*"},
                ],
                "frame": "world",
            }
        ]
    )

    fields = manager.setup_scene()

    assert fields["contact_hand_s0"].kw["prim_path"] == "{ENV_REGEX_NS}/Robot/(left_.*)"
    assert fields["contact_hand_s1"].kw["prim_path"] == "{ENV_REGEX_NS}/Robot/(right_.*)"


def test_single_articulation_body_set_omits_the_owner(fake_isaaclab):
    manager = build_contact_manager(
        [
            {
                "name": "grasp",
                "sensor_refs": [{"owner": "robot", "body_set": "grasp"}],
                "frame": "world",
            }
        ],
        backend="isaaclab2",
        articulation_names=("tiny",),
        articulation_body_sets={"tiny": {"grasp": ("tip", "base")}},
        contact_factory=contact_factory("isaaclab2"),
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


def test_static_support_and_environment_refs_are_refused(fake_isaaclab):
    for ref in (
        {"owner": "support", "name": "table"},
        {"owner": "environment", "body": "world"},
        {"owner": "environment", "path": "{ENV_REGEX_NS}/World/ground"},
    ):
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


def test_prop_manager_activates_contact_sensors_on_referenced_props():
    from unigym.backends.isaaclab2.props.handlers.generic import GenericProp
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


def test_require_contact_sensors_sets_the_internal_flag_and_validates_names():
    from tests._support.backends import prop_faces
    from unigym.props.base import PropManager

    prop_backend, mujoco_prop_backend = prop_faces("mujoco")
    manager = PropManager(
        {"types": [{"name": "crate", "type": "rigid_object", "urdf_path": "/tmp/crate.urdf"}]},
        num_envs=1,
        env_spacing=1.0,
        backend="mujoco",
        prop_backend=prop_backend,
        mujoco_prop_backend=mujoco_prop_backend,
    )
    assert manager.require_contact_sensors(["crate"]) == ("crate",)
    assert getattr(manager.specs[0], "_contact_sensor", False) is True
    with pytest.raises(ValueError, match="unknown props"):
        manager.require_contact_sensors(["ghost"])


class _Data:
    def __init__(self, bodies, forces=None, matrix=None):
        self.body_names = list(bodies)
        self.net_forces_w = forces
        self.force_matrix_w = matrix


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


def test_bind_resolves_object_sensor_and_robot_filter_axis():
    import torch

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
    import torch

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


def test_frame_body_remains_unsupported_on_isaaclab2():
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
