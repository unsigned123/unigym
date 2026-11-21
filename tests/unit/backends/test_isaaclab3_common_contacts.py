"""Shared IsaacLab 3 contact refs: prim paths, activation, F alignment."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from unigym.backends.shared.isaaclab3_common.contacts import (
    MANIFOLD_MAX_CONTACT_DATA_COUNT_PER_PRIM,
    IsaacLabContactBackend,
)
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


def test_prop_activation_flag_flows_to_the_spawner_cfg():
    from unigym.backends.shared.isaaclab3_common.props.handlers.generic import GenericProp
    from unigym.props.base import PropSpec

    spec = PropSpec(name="crate", type="rigid_object", urdf_path="/tmp/crate.urdf")
    handler = GenericProp(spec, num_envs=1, device="cpu")
    classes = {
        "RigidObjectCfg": _RigidObjectCfg,
        "UrdfFileCfg": _Cfg,
    }
    _, cfg = handler.setup_scene(classes=classes)
    assert "activate_contact_sensors" not in cfg.kw
    assert cfg.kw["spawn"].kw["activate_contact_sensors"] is False

    spec._contact_sensor = True
    _, cfg = handler.setup_scene(classes=classes)
    assert "activate_contact_sensors" not in cfg.kw
    assert cfg.kw["spawn"].kw["activate_contact_sensors"] is True


class _ContactView:
    """Stand-in for the PhysX ``RigidContactView.get_contact_data`` face."""

    def __init__(self, buffers):
        self._buffers = buffers
        self.dt = None

    def get_contact_data(self, *, dt):
        self.dt = dt
        return self._buffers


class _ManifoldSensor:
    """A bound sensor exposing the public contact-view read face."""

    def __init__(self, bodies, buffers):
        self._bodies = list(bodies)
        self.contact_view = _ContactView(buffers)

    @property
    def body_names(self):
        return self._bodies


def _filter_row(body):
    from unigym.contacts.inventory import ContactBodyAddress

    return ContactBodyAddress(
        owner="robot",
        scope="robot",
        body=body,
        native=body,
        prim_path=None,
        collision=True,
        contact_capable=True,
        key=f"robot/robot/{body}",
    )


def _manifold_spec():
    from unigym.contacts.base import ContactGroupSpec
    from unigym.contacts.refs import ContactBodyRef

    return ContactGroupSpec(
        name="grip",
        sensor_refs=[ContactBodyRef(owner="robot", body="tip")],
        filter_refs=[ContactBodyRef(owner="robot", body="ball")],
        frame="world",
        manifold=True,
    )


def test_manifold_group_tracks_contact_points_with_a_declared_capacity(fake_isaaclab):
    """Opting in sets the engine contact-point tracking and an explicit cap."""
    manager = _manager(
        [
            {
                "name": "grip",
                "sensor_refs": [{"owner": "robot", "body": "tip"}],
                "filter_refs": [{"owner": "robot", "body": "ball"}],
                "frame": "world",
                "manifold": True,
            }
        ]
    )

    cfg = manager.setup_scene()["contact_grip_s0"]

    assert cfg.kw["track_contact_points"] is True
    assert cfg.kw["max_contact_data_count_per_prim"] == MANIFOLD_MAX_CONTACT_DATA_COUNT_PER_PRIM


def test_force_only_group_does_not_enable_contact_point_tracking(fake_isaaclab):
    """A group without the flag keeps the force-only sensor configuration."""
    manager = _manager(
        [
            {
                "name": "grip",
                "sensor_refs": [{"owner": "robot", "body": "tip"}],
                "filter_refs": [{"owner": "robot", "body": "ball"}],
                "frame": "world",
            }
        ]
    )

    cfg = manager.setup_scene()["contact_grip_s0"]

    assert "track_contact_points" not in cfg.kw
    assert "max_contact_data_count_per_prim" not in cfg.kw


def test_manifold_read_maps_filtered_engine_rows_to_body_named_points():
    """Two contacts in env 0 become two points; env 1 has none.

    The engine force scalar is taken as a magnitude: a negative engine scalar
    does not leak a fabricated sign into the neutral record.
    """
    spec = _manifold_spec()
    backend = IsaacLabContactBackend([spec])
    forces = torch.tensor([[-2.0], [4.0], [0.0], [0.0]], dtype=torch.float32)
    points = torch.tensor(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0]],
        dtype=torch.float32,
    )
    normals = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    separations = torch.tensor([[-1e-3], [-2e-3], [0.0], [0.0]], dtype=torch.float32)
    counts = torch.tensor([[2], [0]], dtype=torch.int32)
    starts = torch.tensor([[0], [2]], dtype=torch.int32)
    sensor = _ManifoldSensor(["tip"], (forces, points, normals, separations, counts, starts))
    _attach(backend, spec, [sensor], [[0]], filters=[_filter_row("ball")])

    out = backend.read_manifold(spec, torch.tensor([0, 1], dtype=torch.int64), dt=0.005)

    assert sensor.contact_view.dt == 0.005
    assert len(out) == 2
    first, second = out
    assert (first.env_id, second.env_id) == (0, 0)
    assert (first.body1, first.body2) == ("ball", "tip")
    assert (first.contact_id, second.contact_id) == (0, 1)
    assert first.normal_force_n == pytest.approx(2.0)
    assert second.normal_force_n == pytest.approx(4.0)
    assert first.separation_m == pytest.approx(-1e-3)
    assert first.pos_w.dtype == torch.float64
    assert first.pos_w.tolist() == pytest.approx([0.0, 0.0, 1.0])

    points[0, 0] = 123.0
    assert first.pos_w[0].item() == pytest.approx(0.0)


def test_manifold_read_respects_the_selected_env_subset():
    """A selected env with no contacts yields an empty point tuple."""
    spec = _manifold_spec()
    backend = IsaacLabContactBackend([spec])
    sensor = _ManifoldSensor(
        ["tip"],
        (
            torch.zeros((2, 1)),
            torch.zeros((2, 3)),
            torch.zeros((2, 3)),
            torch.zeros((2, 1)),
            torch.tensor([[1], [0]], dtype=torch.int32),
            torch.tensor([[0], [1]], dtype=torch.int32),
        ),
    )
    _attach(backend, spec, [sensor], [[0]], filters=[_filter_row("ball")])

    out = backend.read_manifold(spec, torch.tensor([1], dtype=torch.int64), dt=0.005)

    assert out == ()


def test_manifold_read_is_env_major_across_multiple_sensors():
    """Env is the outer axis: env0 sensor/filter order precedes env1."""
    from unigym.contacts.base import ContactGroupSpec
    from unigym.contacts.refs import ContactBodyRef

    spec = ContactGroupSpec(
        name="both_feet",
        sensor_refs=[
            ContactBodyRef(owner="robot", body="left_foot"),
            ContactBodyRef(owner="robot", body="right_foot"),
        ],
        filter_refs=[ContactBodyRef(owner="robot", body="ball")],
        frame="world",
        manifold=True,
    )
    backend = IsaacLabContactBackend([spec])

    def sensor(body: str, env0_force: float, env1_force: float) -> _ManifoldSensor:
        return _ManifoldSensor(
            [body],
            (
                torch.tensor([[env0_force], [env1_force]], dtype=torch.float32),
                torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=torch.float32),
                torch.zeros((2, 3)),
                torch.tensor([[-1e-3], [-2e-3]], dtype=torch.float32),
                torch.tensor([[1], [1]], dtype=torch.int32),
                torch.tensor([[0], [1]], dtype=torch.int32),
            ),
        )

    _attach(
        backend,
        spec,
        [sensor("left_foot", 1.0, 3.0), sensor("right_foot", 2.0, 4.0)],
        [[0], [0]],
        filters=[_filter_row("ball")],
    )

    out = backend.read_manifold(spec, torch.tensor([0, 1], dtype=torch.int64), dt=0.005)

    assert [(point.env_id, point.body2, point.normal_force_n) for point in out] == [
        (0, "left_foot", pytest.approx(1.0)),
        (0, "right_foot", pytest.approx(2.0)),
        (1, "left_foot", pytest.approx(3.0)),
        (1, "right_foot", pytest.approx(4.0)),
    ]
    assert [point.contact_id for point in out] == [0, 1, 0, 1]
