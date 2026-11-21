"""IsaacLab runtime-query behavior over a fake articulation surface.

The unit layer must not import an engine, so these tests build the same
``model_set`` / ``articulations`` / ``robots`` / profile surface the live
IsaacLab envs expose.  They prove the shared model+pose math, the IL3
``xyzw`` boundary conversion, the robot-profile site composition, the
capability labels and the unsupported-operation gates.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unigym.backends.isaaclab2.query import (
    ISAACLAB2_QUERY_CAPABILITIES,
    Isaaclab2Query,
)
from unigym.backends.shared.isaaclab3_common.query import (
    ISAACLAB3_NEWTON_QUERY_CAPABILITIES,
    ISAACLAB3_PHYSX_QUERY_CAPABILITIES,
    Isaaclab3Query,
)
from unigym.model import Actuator, Body, Joint, ModelIR, ModelSet, Table
from unigym.query import (
    ComRef,
    ContactPoint,
    EntityRef,
    Query,
    QueryCapabilityError,
    QueryReferenceError,
)
from unigym.query.errors import QueryStaleError
from unigym.robots.base import BodyBinding, CanonicalBodyId, RobotProfile
from unigym.run.instance import RunInstance
from unigym.sites import SiteSpec

_QUAT_WXYZ = torch.tensor(
    [
        [
            [0.7071067811865476, 0.0, 0.0, 0.7071067811865476],
            [1.0, 0.0, 0.0, 0.0],
        ],
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
    ],
    dtype=torch.float64,
)


def _ir() -> ModelIR:
    return ModelIR(
        name="robot",
        backend="isaaclab2",
        compiled=True,
        bodies=Table(
            [
                Body(name="base", ipos=(0.0, 0.0, 0.1)),
                Body(name="link", ipos=(0.0, 0.0, 0.05)),
            ]
        ),
        joints=Table(
            [
                Joint(name="j1", type="hinge", body="link"),
                Joint(name="j2", type="hinge", body="link"),
            ]
        ),
        actuators=Table([Actuator(name="a1", target="j1", trntype="joint", actuator_id=0)]),
    )


def _profile() -> RobotProfile:
    return RobotProfile(
        name="robot",
        joint_sets={"body": ("j1", "j2")},
        default_joint_pos={"j1": 0.0, "j2": 0.0},
        assets=None,
        extra_action_sets={},
        base_body=CanonicalBodyId("base"),
        foot_bodies=(),
        body_sets={},
        body_bindings={
            CanonicalBodyId("base"): BodyBinding(
                canonical=CanonicalBodyId("base"),
                names={"usd": "base"},
            )
        },
        sites={"ee": SiteSpec(name="ee", body="base", pos=(0.5, 0.0, 0.2))},
    )


def _env(backend: str) -> SimpleNamespace:
    quat_order = "wxyz" if backend == "isaaclab2" else "xyzw"
    quats = _QUAT_WXYZ if quat_order == "wxyz" else _QUAT_WXYZ[..., [1, 2, 3, 0]]
    data = SimpleNamespace(
        body_names=["base", "link"],
        body_pos_w=torch.tensor(
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
                [[10.0, 0.0, 1.0], [10.0, 0.0, 2.0]],
            ],
            dtype=torch.float32,
        ),
        body_quat_w=quats,
        joint_names=["j1", "j2"],
        joint_pos=torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
        joint_vel=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        root_pos_w=torch.tensor([[0.0, 0.0, 1.0], [5.0, 0.0, 1.0]], dtype=torch.float32),
        root_quat_w=quats[:, 0],
    )
    asset = SimpleNamespace(
        data=data,
        body_names=["base", "link"],
        joint_names=["j1", "j2"],
    )
    art = SimpleNamespace(
        asset=asset,
        root="free",
        joint_names=("j1", "j2"),
        base_body="base",
        profile=_profile(),
    )
    ir = _ir()
    values = {
        "num_envs": 2,
        "model_set": ModelSet({"robot": ir}, merged=ir),
        "articulations": {"robot": art},
        "robots": {"robot": asset},
        "prop_manager": None,
    }
    if backend != "isaaclab2":
        values["backend_name"] = backend
    return SimpleNamespace(**values)


def _query(backend: str):
    env = _env(backend)
    if backend == "isaaclab2":
        return Isaaclab2Query(env), env
    return Isaaclab3Query(env), env


def _env_with_joint_types(joint_types: dict[str, str]) -> SimpleNamespace:
    """Build the fake env with an IR/data joint table of the given types."""
    env = _env("isaaclab2")
    base = env.model_set["robot"]
    names = tuple(joint_types)
    ir = ModelIR(
        name="robot",
        backend="isaaclab2",
        compiled=True,
        bodies=base.bodies,
        joints=Table(
            [Joint(name=name, type=kind, body="link") for name, kind in joint_types.items()]
        ),
        actuators=base.actuators,
    )
    env.model_set = ModelSet({"robot": ir}, merged=ir)
    env.articulations["robot"].joint_names = names
    data = env.robots["robot"].data
    data.joint_names = list(names)
    data.joint_pos = torch.zeros((env.num_envs, len(names)), dtype=torch.float32)
    data.joint_vel = torch.zeros_like(data.joint_pos)
    return env


@pytest.mark.parametrize("backend", ("isaaclab2", "isaaclab3_physx", "isaaclab3_newton"))
def test_capabilities_name_the_backend_and_stay_p0b2_honest(backend: str) -> None:
    """Only model facts and live pose are supported; P1 groups are unsupported."""
    caps = {
        "isaaclab2": ISAACLAB2_QUERY_CAPABILITIES,
        "isaaclab3_physx": ISAACLAB3_PHYSX_QUERY_CAPABILITIES,
        "isaaclab3_newton": ISAACLAB3_NEWTON_QUERY_CAPABILITIES,
    }[backend]

    assert caps.backend == backend
    assert isinstance(_query(backend)[0], Query)
    assert caps.supports("model.ir") == "native"
    assert caps.supports("model.body") == "native"
    assert caps.supports("model.joint") == "native"
    assert caps.supports("model.actuator") == "native"
    assert caps.supports("model.geom") == "unsupported"
    assert caps.supports("model.site") == "unsupported"
    assert caps.supports("state.body_pose") == "native"
    assert caps.supports("state.root_pose") == "native"
    assert caps.supports("state.site_pose") == "partial"
    assert caps.supports("state.com_pose") == "partial"
    assert caps.supports("scratch.copy") == "unsupported"
    assert caps.supports("jacobian.body") == "unsupported"
    assert caps.supports("geometry.distance") == "unsupported"
    assert caps.supports("contacts.manifold") == "unsupported"
    assert caps.supports("actuators.state") == "unsupported"


def test_model_facts_come_from_the_published_ir() -> None:
    """Body/joint/actuator rows resolve from the ModelIR; geom/site are absent."""
    query, env = _query("isaaclab2")

    assert query.model.ir() is env.model_set.merged
    assert query.model.ir("robot") is env.model_set["robot"]
    assert query.model.body("base").name == "base"
    assert query.model.joint("j1").body == "link"
    assert query.model.actuator("a1").actuator_id == 0
    with pytest.raises(QueryCapabilityError, match="model.geom"):
        query.model.geom("base_geom")
    with pytest.raises(QueryCapabilityError, match="model.site"):
        query.model.site("ee")
    with pytest.raises(QueryReferenceError, match="bogus"):
        query.model.body("bogus")


@pytest.mark.parametrize("backend", ("isaaclab2", "isaaclab3_physx", "isaaclab3_newton"))
def test_live_body_root_and_joint_state_matches_the_articulation_data(backend: str) -> None:
    """Body/root/joint reads are detached CPU copies of the live buffers."""
    query, env = _query(backend)
    data = env.robots["robot"].data

    pose = query.state.body_pose("base")
    assert tuple(pose.env_ids.tolist()) == (0, 1)
    torch.testing.assert_close(pose.pos_w, data.body_pos_w[:, 0].to(torch.float64))
    torch.testing.assert_close(pose.quat_wxyz, _QUAT_WXYZ[:, 0])

    one = query.state.body_pose("link", env=1)
    assert tuple(one.env_ids.tolist()) == (1,)
    torch.testing.assert_close(one.pos_w, data.body_pos_w[1:, 1].to(torch.float64))

    root = query.state.root_pose("robot")
    torch.testing.assert_close(root.pos_w, data.root_pos_w.to(torch.float64))
    torch.testing.assert_close(root.quat_wxyz, _QUAT_WXYZ[:, 0])

    positions = query.state.joint_position("robot")
    velocities = query.state.joint_velocity("robot", joint_names=("j2",))
    torch.testing.assert_close(positions, data.joint_pos.to(torch.float64))
    torch.testing.assert_close(velocities, data.joint_vel[:, 1:].to(torch.float64))

    pose.pos_w[0, 0] = 123.0
    assert float(data.body_pos_w[0, 0, 0]) != 123.0


def test_com_pose_uses_the_ir_com_offset_and_identity_orientation() -> None:
    """A body COM is the body frame plus its published local COM offset."""
    query, _ = _query("isaaclab2")

    com = query.state.com_pose(ComRef("robot", body="base"))
    torch.testing.assert_close(
        com.pos_w,
        torch.tensor([[0.0, 0.0, 1.1], [10.0, 0.0, 1.1]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        com.quat_wxyz,
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
    )
    with pytest.raises(QueryCapabilityError, match="subtree"):
        query.state.com_pose(ComRef("robot", body="base", subtree=True))


@pytest.mark.parametrize("backend", ("isaaclab2", "isaaclab3_physx", "isaaclab3_newton"))
def test_site_pose_composes_the_robot_profile_body_and_local_offset(backend: str) -> None:
    """A robot-profile site is body pose composed with its local offset."""
    query, env = _query(backend)
    assert not hasattr(env, "_profiles")

    pose = query.state.site_pose("ee")
    torch.testing.assert_close(
        pose.pos_w,
        torch.tensor([[0.0, 0.5, 1.2], [10.5, 0.0, 1.2]], dtype=torch.float64),
    )
    torch.testing.assert_close(pose.quat_wxyz, _QUAT_WXYZ[:, 0])


@pytest.mark.parametrize("joint_type", ("ball", "fixed"))
def test_non_1dof_joint_reads_raise_capability_error(joint_type: str) -> None:
    """A ball/fixed IR joint is never returned as a live 1-DOF column."""
    env = _env_with_joint_types({"j1": "hinge", "odd": joint_type})
    query = Isaaclab2Query(env)

    with pytest.raises(QueryCapabilityError, match=f"odd.*{joint_type}"):
        query.state.joint_position("robot", joint_names=("odd",))
    with pytest.raises(QueryCapabilityError, match=f"odd.*{joint_type}"):
        query.state.joint_velocity("robot", joint_names=("odd",))
    with pytest.raises(QueryCapabilityError, match=f"odd.*{joint_type}"):
        query.state.joint_position("robot")


def _contact_point() -> ContactPoint:
    return ContactPoint(
        env_id=0,
        group="foot_floor",
        body1="base",
        body2="world",
        geom1=None,
        geom2=None,
        pos_w=torch.zeros(3, dtype=torch.float64),
        normal_w=None,
        frame_w=None,
        separation_m=0.0,
        normal_force_n=1.0,
        friction_force_w=None,
        contact_id=0,
    )


def test_unsupported_subfaces_raise_typed_capability_errors() -> None:
    """Scratch, Jacobian, geometry, contacts and actuators are P1."""
    query, _ = _query("isaaclab2")
    point = _contact_point()

    with pytest.raises(QueryCapabilityError, match="scratch"):
        query.scratch()
    with pytest.raises(QueryCapabilityError, match="jacobian"):
        query.kinematics.jacobian(EntityRef("body", "base"))
    with pytest.raises(QueryCapabilityError, match="geometry.distance"):
        query.geometry.distance(EntityRef("geom", "a"), EntityRef("geom", "b"))
    with pytest.raises(QueryCapabilityError, match="geometry.aabb"):
        query.geometry.aabb(EntityRef("body", "base"))
    with pytest.raises(QueryCapabilityError, match="contacts.manifold"):
        query.contacts.manifold("foot_floor")
    with pytest.raises(QueryCapabilityError, match="contacts.jacobian"):
        query.contacts.jacobian(point, "robot")
    with pytest.raises(QueryCapabilityError, match="contacts.wrench_to_joint"):
        query.contacts.wrench_to_joint(point, "robot")
    with pytest.raises(QueryCapabilityError, match="actuators.facts"):
        query.actuators.rows()
    with pytest.raises(QueryCapabilityError, match="actuators.facts"):
        query.actuators.facts("a1")
    with pytest.raises(QueryCapabilityError, match="actuators.state"):
        query.actuators.state("a1")


def test_run_query_reuses_the_isaaclab_handle() -> None:
    """run.query returns the exact IsaacLab env.query object."""
    query, _ = _query("isaaclab2")
    instance = RunInstance(
        spec=SimpleNamespace(backend="isaaclab2"),
        profile=None,
        adapter=None,
        sampler=None,
        term_group=None,
        env=SimpleNamespace(query=query),
        bridge=None,
    )

    assert instance.query is query
    assert instance.query is instance.query


def test_close_invalidates_the_query_handle() -> None:
    """Every subface rejects a closed query with QueryStaleError."""
    query, _ = _query("isaaclab2")
    query.close()

    with pytest.raises(QueryStaleError, match="stale"):
        query.state.body_pose("base")
    with pytest.raises(QueryStaleError, match="stale"):
        query.model.ir()


def test_isaaclab3_query_rejects_an_unknown_backend_identity() -> None:
    """The IL3 query requires a concrete PhysX/Newton env identity."""
    env = _env("isaaclab2")
    env.backend_name = "isaaclab3"
    with pytest.raises(TypeError, match="backend_name"):
        Isaaclab3Query(env)
