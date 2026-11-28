"""IsaacLab2 contact refs: prim paths, prop activation, F alignment, refusal."""

from __future__ import annotations

import types

import pytest

from tests._support.backends import contact_factory
from tests._support.fakes.isaaclab import _Cfg, _RigidObjectCfg
from unigym.contacts.manager import build_contact_manager

_DEFAULT_CONTACT_ASSETS = {"tiny": "/tmp/tiny.usd"}


def _manager(groups, *, props=None, articulated=None, contact_assets=_DEFAULT_CONTACT_ASSETS):
    manager = build_contact_manager(
        groups,
        backend="isaaclab2",
        articulation_names=("tiny",),
        prop_names=props,
        articulated_props=articulated,
        contact_assets=contact_assets,
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


def test_single_articulation_body_set_omits_the_owner(fake_isaaclab, monkeypatch):
    from pathlib import Path

    from unigym.backends.isaaclab2 import contacts as contacts_module
    from unigym.backends.isaaclab2.body_inventory import RigidBody, RobotBodyInventory

    inventory = RobotBodyInventory(
        Path("/tmp/tiny.usd"),
        (
            RigidBody("tip", "tip", True, None),
            RigidBody("base", "base", True, None),
        ),
    )
    monkeypatch.setattr(contacts_module, "robot_body_inventory", lambda path: inventory)

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
        contact_assets={"tiny": "/tmp/tiny.usd"},
        contact_factory=contact_factory("isaaclab2"),
    )
    assert manager is not None
    assert (
        manager.setup_scene()["contact_grasp_s0_p0"].kw["prim_path"]
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


@pytest.mark.parametrize(
    ("group", "props"),
    [
        (
            {
                "name": "single",
                "sensor_refs": [{"owner": "robot", "articulation": "tiny", "body": "tip"}],
                "frame": "world",
            },
            None,
        ),
        (
            {
                "name": "wildcard",
                "sensor_refs": [{"owner": "robot", "articulation": "tiny", "body": "*"}],
                "frame": "world",
            },
            None,
        ),
        (
            {
                "name": "pairwise",
                "sensor_refs": [{"owner": "object", "name": "crate"}],
                "filter_refs": [{"owner": "robot", "articulation": "tiny", "body": "tip"}],
                "frame": "world",
            },
            ["crate"],
        ),
    ],
)
def test_every_robot_ref_requires_contact_assets_before_scene(group, props):
    manager = _manager([group], props=props, contact_assets=None)

    with pytest.raises(ValueError, match="contact_assets has no entry"):
        manager.setup_scene()


def test_contact_assets_for_robot_must_be_a_usd_asset(fake_isaaclab):
    manager = _manager(
        [
            {
                "name": "all",
                "sensor_refs": [{"owner": "robot", "articulation": "tiny", "body": "*"}],
                "frame": "world",
            }
        ],
        contact_assets={"tiny": "/tmp/tiny.urdf"},
    )

    with pytest.raises(ValueError, match="USD-only"):
        manager.setup_scene()


def test_general_wildcard_fields_are_parent_scope_parts(fake_isaaclab, monkeypatch):
    from pathlib import Path

    from unigym.backends.isaaclab2 import contacts as contacts_module
    from unigym.backends.isaaclab2.body_inventory import RigidBody, RobotBodyInventory

    inventory = RobotBodyInventory(
        Path("/tmp/tiny.usd"),
        (
            RigidBody("pelvis", "pelvis", True, None),
            RigidBody("left_hand/index", "index", True, None),
            RigidBody("right_hand/index", "index", True, None),
        ),
    )
    monkeypatch.setattr(contacts_module, "robot_body_inventory", lambda path: inventory)
    manager = _manager(
        [
            {
                "name": "all",
                "sensor_refs": [{"owner": "robot", "articulation": "tiny", "body": "*"}],
                "frame": "world",
            }
        ]
    )

    fields = manager.setup_scene()

    assert set(fields) == {
        "contact_all_s0_p0",
        "contact_all_s0_p1",
        "contact_all_s0_p2",
    }
    assert [fields[key].kw["prim_path"] for key in sorted(fields)] == [
        "{ENV_REGEX_NS}/Robot/(.*)",
        "{ENV_REGEX_NS}/Robot/left_hand/(.*)",
        "{ENV_REGEX_NS}/Robot/right_hand/(.*)",
    ]


def _general_backend(fields):
    from unigym.backends.isaaclab2.contacts import IsaacLabContactBackend
    from unigym.contacts.base import ContactGroupSpec

    spec = ContactGroupSpec.from_yaml(
        {
            "name": "all",
            "sensor_refs": [{"owner": "robot", "articulation": "tiny", "body": "*"}],
            "frame": "world",
        }
    )
    backend = IsaacLabContactBackend([spec], articulation_names=("tiny",))
    backend._fields[spec.name] = fields
    return backend, spec


def test_general_sensor_permutation_maps_into_live_body_order():
    import torch

    backend, spec = _general_backend([["contact_all_s0", "contact_all_s1"]])
    env = _env(("a", "b", "c", "d"))
    env.scene["contact_all_s0"] = _Sensor(
        ["c", "a"], forces=torch.tensor([[[30.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    )
    env.scene["contact_all_s1"] = _Sensor(
        ["d", "b"], forces=torch.tensor([[[40.0, 0.0, 0.0], [20.0, 0.0, 0.0]]])
    )

    backend.bind_env(env)

    assert backend._perm["all"] == [[1, 3, 0, 2]]
    out = backend.read_raw(spec)
    assert out[0, :, 0].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_general_sensor_missing_live_body_fails():
    backend, _ = _general_backend([["contact_all_s0"]])
    env = _env(("a", "b", "c"))
    env.scene["contact_all_s0"] = _Sensor(["a", "b"])

    with pytest.raises(ValueError, match="does not cover the measured live robot body set"):
        backend.bind_env(env)


def test_general_sensor_extra_body_fails():
    backend, _ = _general_backend([["contact_all_s0"]])
    env = _env(("a", "b"))
    env.scene["contact_all_s0"] = _Sensor(["a", "b", "ghost"])

    with pytest.raises(ValueError, match="unexpected in sensors"):
        backend.bind_env(env)
