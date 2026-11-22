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
from unigym.backends.shared.isaaclab3_common.contacts import ManifoldContact
from unigym.backends.shared.isaaclab3_common.query import (
    ISAACLAB3_NEWTON_QUERY_CAPABILITIES,
    ISAACLAB3_PHYSX_QUERY_CAPABILITIES,
    Isaaclab3Query,
)
from unigym.contacts.base import ContactGroupSpec
from unigym.core.quat import to_xyzw
from unigym.model import Actuator, Body, Joint, ModelIR, ModelSet, Table, UsdExt
from unigym.query import (
    ComRef,
    ContactPoint,
    EntityRef,
    ObjectRef,
    PointRef,
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
        actuators=Table(
            [
                Actuator(name="a1", target="j1", trntype="joint", kind="robot", actuator_id=0),
                Actuator(name="a2", kind="other"),
            ]
        ),
        ext={
            "usd": UsdExt(
                attrs={
                    "actuator/a1": {"joint_names": ["j1"]},
                    "actuator/a2": {"class": "IdealPDActuator"},
                }
            )
        },
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
        root_lin_vel_w=torch.zeros((2, 3), dtype=torch.float32),
        root_ang_vel_w=torch.zeros((2, 3), dtype=torch.float32),
    )
    asset = SimpleNamespace(
        data=data,
        body_names=["base", "link"],
        backend_body_names=["base", "link"],
        joint_names=["j1", "j2"],
    )
    art = SimpleNamespace(
        asset=asset,
        root="free",
        joint_names=("j1", "j2"),
        base_body="base",
        base_body_id=CanonicalBodyId("base"),
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


def _transaction_env():
    """Build an IL3 fake env with the live state-write face the scratch needs."""
    query, env = _query("isaaclab3_physx")
    data = env.robots["robot"].data
    env.device = "cpu"
    calls: list[str] = []

    def set_root_state(
        env_ids,
        *,
        name: str,
        pos_w: torch.Tensor,
        quat_wxyz: torch.Tensor,
        lin_vel_w: torch.Tensor,
        ang_vel_w: torch.Tensor,
    ) -> torch.Tensor:
        del name
        ids = torch.as_tensor(env_ids, dtype=torch.int64)
        data.root_pos_w[ids] = pos_w.to(data.root_pos_w.dtype)
        data.root_quat_w[ids] = torch.as_tensor(to_xyzw(quat_wxyz)).to(data.root_quat_w.dtype)
        data.root_lin_vel_w[ids] = lin_vel_w.to(data.root_lin_vel_w.dtype)
        data.root_ang_vel_w[ids] = ang_vel_w.to(data.root_ang_vel_w.dtype)
        calls.append("set_root_state")
        return ids

    def set_joint_state(
        env_ids,
        *,
        articulation: str,
        joint_names: list[str],
        position: torch.Tensor,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del articulation
        ids = torch.as_tensor(env_ids, dtype=torch.int64)
        columns = [list(data.joint_names).index(name) for name in joint_names]
        data.joint_pos[ids[:, None], columns] = position.to(torch.float32)
        if velocity is not None:
            data.joint_vel[ids[:, None], columns] = velocity.to(torch.float32)
        calls.append("set_joint_state")
        return ids

    env.set_root_state = set_root_state
    env.set_joint_state = set_joint_state
    env.scene = SimpleNamespace(write_data_to_sim=lambda: calls.append("write_data_to_sim"))
    env.sim = SimpleNamespace(forward=lambda: calls.append("forward"))
    return query, env, data, calls


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


def _native_jacobians(fixed: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic native link/COM Jacobians for the fake articulation.

    Each Jacobian body row carries a distinct offset so a body-row mix-up is
    observable, not just a layout-shape mismatch.
    """
    num_envs = 2
    base_columns = 0 if fixed else 6
    nv = base_columns + 2
    jacobi_bodies = 1 if fixed else 2
    link = torch.zeros((num_envs, jacobi_bodies, 6, nv), dtype=torch.float32)
    for body in range(jacobi_bodies):
        for column in range(nv):
            link[:, body, 0, column] = 1.0 + column + 100.0 * body
            link[:, body, 1, column] = 2.0 + column + 100.0 * body
            link[:, body, 2, column] = 3.0 + column + 100.0 * body
            link[:, body, 5, column] = 1.0 + 0.1 * column
    com = link.clone()
    com[:, :, 0, :] += 10.0
    com[:, :, 1, :] += 20.0
    com[:, :, 2, :] += 30.0
    return link, com


_BODY_POS = {
    "base": (0.0, 0.0, 1.0),
    "base_link": (0.0, 0.0, 1.0),
    "link": (0.0, 0.0, 2.0),
}


def _jacobian_query(
    fixed: bool = False,
    *,
    order: tuple[str, ...] = ("base", "link"),
    base_body: str = "base",
    base_body_id: str | None = "base",
    backend_body_names: tuple[str, ...] | None = ("base", "link"),
):
    """Build an IL3 fake query whose live data carries native Jacobian rows.

    ``base_body`` is the backend-native USD binding and ``base_body_id`` is the
    canonical id; they differ when the robot profile maps a canonical base body
    onto a differently named USD body.  ``backend_body_names=None`` models a
    live asset that does not publish its backend body order.
    """
    query, env = _query("isaaclab3_physx")
    data = env.robots["robot"].data
    art = env.articulations["robot"]
    art.base_body = base_body
    art.base_body_id = CanonicalBodyId(base_body_id) if base_body_id else None
    env.robots["robot"].backend_body_names = (
        list(backend_body_names) if backend_body_names is not None else None
    )
    if tuple(order) != ("base", "link"):
        data.body_names = list(order)
        env.robots["robot"].body_names = list(order)
        data.body_pos_w = torch.tensor(
            [
                [_BODY_POS[name] for name in order],
                [
                    (_BODY_POS[name][0] + 10.0, _BODY_POS[name][1], _BODY_POS[name][2])
                    for name in order
                ],
            ],
            dtype=torch.float32,
        )
        data.body_quat_w = torch.zeros((2, len(order), 4), dtype=torch.float32)
        data.body_quat_w[..., 3] = 1.0
    link, com = _native_jacobians(fixed)
    data.body_link_jacobian_w = link
    data.body_com_jacobian_w = com
    return query, data, link, com


@pytest.mark.parametrize("backend", ("isaaclab2", "isaaclab3_physx", "isaaclab3_newton"))
def test_capabilities_name_the_backend_and_stay_honest(backend: str) -> None:
    """Model facts/pose are native; IL3 adds Jacobians, IL2 has no Jacobian API."""
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
    assert caps.supports("model.geom_contact") == "unsupported"
    assert caps.supports("state.body_pose") == "native"
    assert caps.supports("state.root_pose") == "native"
    assert caps.supports("state.site_pose") == "partial"
    assert caps.supports("state.com_pose") == "partial"
    if backend == "isaaclab2":
        assert caps.scratch.mode == "copy"
        assert caps.scratch.isolation == "bit_exact"
        assert caps.supports("scratch.copy") == "unsupported"
    else:
        assert caps.scratch.mode == "transaction"
        assert caps.scratch.isolation == "restored"
        assert caps.supports("scratch.copy") == "synthetic"
        assert caps.supports("scratch.live_untouched") == "partial"
        assert caps.supports("scratch.set_root_state") == "synthetic"
        assert caps.supports("scratch.set_joint_state") == "synthetic"
        assert caps.supports("scratch.forward") == "synthetic"
        assert caps.supports("scratch.body_pose") == "synthetic"
        assert caps.supports("scratch.site_pose") == "synthetic"
        assert caps.supports("scratch.com_pose") == "synthetic"
        assert caps.supports("scratch.jacobian") == "synthetic"
        assert caps.supports("scratch.geometry") == "unsupported"
    assert caps.supports("scratch.contacts") == "unsupported"
    if backend == "isaaclab2":
        assert caps.supports("jacobian.body") == "unsupported"
        assert caps.supports("jacobian.site") == "unsupported"
        assert caps.supports("jacobian.com") == "unsupported"
        assert caps.supports("jacobian.point") == "unsupported"
    else:
        assert caps.supports("jacobian.body") == "native"
        assert caps.supports("jacobian.site") == "synthetic"
        assert caps.supports("jacobian.com") == "partial"
        assert caps.supports("jacobian.point") == "synthetic"
    assert caps.supports("jacobian.object") == "unsupported"
    assert caps.supports("geometry.distance") == "unsupported"
    assert (
        caps.supports("contacts.manifold")
        == {
            "isaaclab2": "unsupported",
            "isaaclab3_physx": "partial",
            "isaaclab3_newton": "unsupported",
        }[backend]
    )
    assert caps.supports("contacts.scan") == "unsupported"
    assert caps.supports("contacts.jacobian") == "unsupported"
    assert caps.supports("contacts.wrench_to_joint") == "unsupported"
    assert caps.supports("actuators.facts") == "partial"
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
def test_geom_contact_is_typed_unsupported(backend: str) -> None:
    """Every IsaacLab identity rejects geom contact parameters with a typed error."""
    query, _ = _query(backend)

    with pytest.raises(QueryCapabilityError) as caught:
        query.model.geom_contact("base_geom")

    assert caught.value.op == "model.geom_contact"
    assert caught.value.support == "unsupported"
    assert caught.value.detail.strip()


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


def test_il3_body_jacobian_reads_the_native_link_row_with_a_base_first_layout() -> None:
    """A body Jacobian is the native link row with the free base DoFs first."""
    query, _, link, _ = _jacobian_query(fixed=False)

    jac = query.kinematics.jacobian(EntityRef("body", "link"))
    expected = link[:, 1].to(torch.float64)

    torch.testing.assert_close(jac.transl, expected[:, 0:3])
    torch.testing.assert_close(jac.rot, expected[:, 3:6])
    assert jac.frame == "world"
    assert jac.reference == "body_origin"
    assert jac.layout.articulation == "robot"
    assert jac.layout.base_names == ("base",)
    assert jac.layout.base_dofs == (
        "base.px",
        "base.py",
        "base.pz",
        "base.rx",
        "base.ry",
        "base.rz",
    )
    assert jac.layout.joint_names == ("j1", "j2")
    assert jac.layout.nv == 8
    assert jac.layout.order == jac.layout.base_dofs + jac.layout.joint_names


def test_il3_fixed_base_jacobian_skips_the_fixed_root_row() -> None:
    """A fixed base has no free columns and drops the root body Jacobian row."""
    query, _, link, _ = _jacobian_query(fixed=True)

    jac = query.kinematics.jacobian(EntityRef("body", "link"))
    torch.testing.assert_close(jac.transl, link[:, 0, 0:3].to(torch.float64))
    assert jac.layout.base_dofs == ()
    assert jac.layout.base_names == ()
    assert jac.layout.joint_names == ("j1", "j2")
    assert jac.layout.nv == 2

    with pytest.raises(QueryCapabilityError, match="fixed root"):
        query.kinematics.jacobian(EntityRef("body", "base"))


def test_il3_com_jacobian_reads_the_native_com_row_and_rejects_subtree() -> None:
    """A body COM uses the native COM Jacobian; a subtree COM has no engine row."""
    query, _, link, com = _jacobian_query(fixed=False)

    jac = query.kinematics.jacobian(ComRef("robot", body="link"))
    torch.testing.assert_close(jac.transl, com[:, 1, 0:3].to(torch.float64))
    torch.testing.assert_close(jac.rot, com[:, 1, 3:6].to(torch.float64))
    assert not torch.equal(jac.transl, link[:, 1, 0:3].to(torch.float64))

    with pytest.raises(QueryCapabilityError, match="subtree"):
        query.kinematics.jacobian(ComRef("robot", subtree=True))
    with pytest.raises(QueryCapabilityError, match="subtree"):
        query.kinematics.jacobian(ComRef("robot", body="link", subtree=True))


def test_il3_default_com_uses_the_declared_base_body_not_the_live_row_zero() -> None:
    """A floating base labels base DoFs and defaults the COM from the declared base."""
    query, data, _, com = _jacobian_query(fixed=False, order=("link", "base"), base_body="base")

    jac = query.kinematics.jacobian(ComRef("robot"))

    assert data.body_names[0] == "link"
    assert jac.layout.base_names == ("base",)
    assert jac.layout.base_dofs == (
        "base.px",
        "base.py",
        "base.pz",
        "base.rx",
        "base.ry",
        "base.rz",
    )
    assert jac.layout.nv == 8
    assert jac.layout.order == jac.layout.base_dofs + jac.layout.joint_names
    torch.testing.assert_close(jac.transl, com[:, 1, 0:3].to(torch.float64))
    torch.testing.assert_close(jac.rot, com[:, 1, 3:6].to(torch.float64))
    assert not torch.equal(jac.transl, com[:, 0, 0:3].to(torch.float64))


def test_il3_fixed_base_default_com_is_the_fixed_root_with_no_row() -> None:
    """A fixed root at public index 1 is the declared base, not the live row 0."""
    query, data, _, com = _jacobian_query(fixed=True, order=("link", "base"), base_body="base")

    assert data.body_names[0] == "link"
    with pytest.raises(QueryCapabilityError, match="fixed root"):
        query.kinematics.jacobian(ComRef("robot"))

    jac = query.kinematics.jacobian(ComRef("robot", body="link"))
    assert jac.layout.base_dofs == ()
    assert jac.layout.base_names == ()
    assert jac.layout.nv == 2
    torch.testing.assert_close(jac.transl, com[:, 0, 0:3].to(torch.float64))


def test_il3_floating_jacobian_requires_a_declared_base_body() -> None:
    """An undeclared floating base fails instead of labelling an arbitrary body."""
    query, _, _, _ = _jacobian_query(fixed=False, base_body="", base_body_id=None)

    with pytest.raises(QueryCapabilityError, match="free root unresolved"):
        query.kinematics.jacobian(EntityRef("body", "link"))
    with pytest.raises(QueryReferenceError, match="no base body is declared"):
        query.kinematics.jacobian(ComRef("robot"))


def test_il3_base_labels_use_the_canonical_base_id_not_the_usd_binding() -> None:
    """Layout labels use the canonical base id while the COM row uses the USD body."""
    query, data, _, com = _jacobian_query(
        fixed=False,
        order=("link", "base_link"),
        base_body="base_link",
        base_body_id="pelvis",
    )

    jac = query.kinematics.jacobian(ComRef("robot"))

    assert tuple(data.body_names) == ("link", "base_link")
    assert jac.layout.base_names == ("pelvis",)
    assert jac.layout.base_dofs == (
        "pelvis.px",
        "pelvis.py",
        "pelvis.pz",
        "pelvis.rx",
        "pelvis.ry",
        "pelvis.rz",
    )
    torch.testing.assert_close(jac.transl, com[:, 1, 0:3].to(torch.float64))


def test_il3_fixed_root_falls_back_to_declared_base_when_backend_names_missing() -> None:
    """A missing backend body order resolves the root from the declared base binding."""
    query, _, link, _ = _jacobian_query(
        fixed=True,
        order=("link", "base"),
        base_body="base",
        backend_body_names=None,
    )

    torch.testing.assert_close(
        query.kinematics.jacobian(EntityRef("body", "link")).transl,
        link[:, 0, 0:3].to(torch.float64),
    )
    with pytest.raises(QueryCapabilityError, match="fixed root"):
        query.kinematics.jacobian(EntityRef("body", "base"))


def test_il3_fixed_root_without_backend_names_or_declared_base_fails_loud() -> None:
    """A fixed root with no resolvable public index never silently maps to row 0."""
    query, _, _, _ = _jacobian_query(
        fixed=True,
        order=("link", "base"),
        base_body="",
        base_body_id=None,
        backend_body_names=None,
    )

    with pytest.raises(QueryCapabilityError, match="unresolved"):
        query.kinematics.jacobian(EntityRef("body", "link"))


def test_il3_fixed_root_backend_name_mismatch_fails_loud() -> None:
    """A backend root absent from the public body order is an error, not row 0."""
    query, _, _, _ = _jacobian_query(
        fixed=True,
        order=("link", "base"),
        base_body="base",
        backend_body_names=("ghost", "link"),
    )

    with pytest.raises(QueryCapabilityError, match="not in the public body order"):
        query.kinematics.jacobian(EntityRef("body", "link"))


def test_il3_body_rows_follow_the_public_body_order() -> None:
    """Floating rows use the public body index; fixed rows omit the resolved root."""
    floating, _, link, _ = _jacobian_query(fixed=False, order=("link", "base"), base_body="base")
    torch.testing.assert_close(
        floating.kinematics.jacobian(EntityRef("body", "link")).transl,
        link[:, 0, 0:3].to(torch.float64),
    )
    torch.testing.assert_close(
        floating.kinematics.jacobian(EntityRef("body", "base")).transl,
        link[:, 1, 0:3].to(torch.float64),
    )

    fixed, _, fixed_link, _ = _jacobian_query(fixed=True, order=("link", "base"), base_body="base")
    torch.testing.assert_close(
        fixed.kinematics.jacobian(EntityRef("body", "link")).transl,
        fixed_link[:, 0, 0:3].to(torch.float64),
    )
    with pytest.raises(QueryCapabilityError, match="fixed root"):
        fixed.kinematics.jacobian(EntityRef("body", "base"))


def test_il3_site_jacobian_composes_the_body_row_with_the_local_offset() -> None:
    """A site row shifts the link linear block by ``omega x r`` for its body offset."""
    query, _, link, _ = _jacobian_query(fixed=False)

    jac = query.kinematics.jacobian(EntityRef("site", "ee"))
    r = torch.tensor([[0.0, 0.5, 0.2], [0.5, 0.0, 0.2]], dtype=torch.float64)
    base = link[:, 0].to(torch.float64)
    expected = base[:, 0:3] + torch.cross(
        base[:, 3:6], r[:, :, None].expand(-1, -1, base.shape[-1]), dim=1
    )

    torch.testing.assert_close(jac.transl, expected)
    torch.testing.assert_close(jac.rot, base[:, 3:6])
    assert jac.layout.joint_names == ("j1", "j2")


def test_il3_point_jacobian_composes_the_body_row_with_the_world_point() -> None:
    """A world-point row shifts the link linear block by the world offset."""
    query, data, link, _ = _jacobian_query(fixed=False)

    point = torch.tensor([1.0, 0.0, 2.0], dtype=torch.float64)
    jac = query.kinematics.jacobian(PointRef(EntityRef("body", "link"), point))
    offset = point - data.body_pos_w[:, 1].to(torch.float64)
    base = link[:, 1].to(torch.float64)
    expected = base[:, 0:3] + torch.cross(
        base[:, 3:6], offset[:, :, None].expand(-1, -1, base.shape[-1]), dim=1
    )

    torch.testing.assert_close(jac.transl, expected)
    torch.testing.assert_close(jac.rot, base[:, 3:6])


def test_il3_jacobian_rotational_false_zeroes_the_angular_block() -> None:
    """``rotational=False`` keeps the translational block and zeroes the angular one."""
    query, _, link, _ = _jacobian_query(fixed=False)

    jac = query.kinematics.jacobian(EntityRef("body", "link"), rotational=False)

    torch.testing.assert_close(jac.transl, link[:, 1, 0:3].to(torch.float64))
    assert not bool(torch.count_nonzero(jac.rot))


def test_il3_jacobian_rejects_unsupported_frame_reference_and_geom() -> None:
    """Only world-frame body-origin Jacobians exist; a geom target has no frame."""
    query, _, _, _ = _jacobian_query(fixed=False)

    with pytest.raises(QueryCapabilityError, match="frame"):
        query.kinematics.jacobian(EntityRef("body", "link"), frame="local")
    with pytest.raises(QueryCapabilityError, match="reference"):
        query.kinematics.jacobian(EntityRef("body", "link"), reference="com")
    with pytest.raises(QueryCapabilityError, match="jacobian.geom"):
        query.kinematics.jacobian(EntityRef("geom", "link_geom"))


def test_il3_scratch_writes_joints_and_restores_on_close() -> None:
    """A transaction writes the live buffers, forwards, and restores on close."""
    query, _, data, calls = _transaction_env()
    original_pos = data.joint_pos.clone()
    original_vel = data.joint_vel.clone()

    with query.scratch(0) as scratch:
        scratch.set_joint_state(
            "robot",
            ("j1",),
            torch.tensor([0.75]),
            velocity=torch.tensor([2.5]),
        )
        scratch.forward()
        assert float(data.joint_pos[0, 0]) == pytest.approx(0.75)
        assert float(data.joint_vel[0, 0]) == pytest.approx(2.5)
        assert tuple(scratch.body_pose("base").pos_w.shape) == (1, 3)
        assert calls[-2:] == ["write_data_to_sim", "forward"]

    assert calls[-2:] == ["write_data_to_sim", "forward"]
    torch.testing.assert_close(data.joint_pos, original_pos)
    torch.testing.assert_close(data.joint_vel, original_vel)


def test_il3_scratch_writes_and_restores_a_free_root() -> None:
    """A free root write round-trips through capture and restore."""
    query, env, data, calls = _transaction_env()
    original_pos = data.root_pos_w.clone()
    original_quat = data.root_quat_w.clone()
    original_lin = data.root_lin_vel_w.clone()
    original_ang = data.root_ang_vel_w.clone()

    pos = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    quat = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    with query.scratch(0) as scratch:
        scratch.set_root_state(
            "robot",
            pos_w=pos,
            quat_wxyz=quat,
            lin_vel_w=torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64),
            ang_vel_w=torch.tensor([0.0, 0.0, 1.5], dtype=torch.float64),
        )
        torch.testing.assert_close(data.root_pos_w[0], pos.to(torch.float32))
        torch.testing.assert_close(
            data.root_quat_w[0], torch.as_tensor(to_xyzw(quat)).to(data.root_quat_w.dtype)
        )
        assert float(data.root_lin_vel_w[0, 0]) == pytest.approx(0.5)
        assert float(data.root_ang_vel_w[0, 2]) == pytest.approx(1.5)
        assert env.articulations["robot"].root == "free"

    torch.testing.assert_close(data.root_pos_w, original_pos)
    torch.testing.assert_close(data.root_quat_w, original_quat)
    torch.testing.assert_close(data.root_lin_vel_w, original_lin)
    torch.testing.assert_close(data.root_ang_vel_w, original_ang)
    assert calls.count("set_root_state") == 2


def test_il3_scratch_restores_after_a_body_exception() -> None:
    """A body exception still restores the captured state and is preserved."""
    query, _, data, calls = _transaction_env()
    original_pos = data.joint_pos.clone()

    with pytest.raises(RuntimeError, match="boom"):
        with query.scratch(0) as scratch:
            scratch.set_joint_state("robot", ("j2",), torch.tensor([-1.0]))
            scratch.forward()
            raise RuntimeError("boom")

    torch.testing.assert_close(data.joint_pos, original_pos)
    assert calls[-2:] == ["write_data_to_sim", "forward"]


def test_il3_scratch_restore_failure_is_attached_to_the_body_exception() -> None:
    """A failing restore never hides the body exception; it is attached as a note."""
    query, env, _, _ = _transaction_env()
    original_writer = env.set_joint_state

    def failing(*args: object, **kwargs: object) -> None:
        raise RuntimeError("restore boom")

    with pytest.raises(ValueError, match="body boom") as caught:
        with query.scratch(0) as scratch:
            scratch.set_joint_state("robot", ("j1",), torch.tensor([0.2]))
            env.set_joint_state = failing
            raise ValueError("body boom")

    notes = getattr(caught.value, "__notes__", [])
    assert any("restore boom" in note for note in notes)
    assert any("ValueError" in note for note in notes)

    env.set_joint_state = original_writer
    with query.scratch(0):
        pass


@pytest.mark.parametrize("fixed_source", ("declared_root", "asset_is_fixed_base"))
def test_il3_scratch_fixed_root_is_rejected_and_exit_skips_root_restore(
    fixed_source: str,
) -> None:
    """A fixed root is read-only and never enters the restore-root set."""
    query, env, data, calls = _transaction_env()
    original = data.joint_pos.clone()
    if fixed_source == "declared_root":
        env.articulations["robot"].root = "fixed"
    else:
        env.robots["robot"].is_fixed_base = True

    with query.scratch(0) as scratch:
        with pytest.raises(QueryCapabilityError, match="fixed root"):
            scratch.set_root_state(
                "robot",
                pos_w=torch.zeros(3, dtype=torch.float64),
                quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            )
        scratch.set_joint_state("robot", ("j1",), torch.tensor([0.4]))
        scratch.forward()
        assert float(data.joint_pos[0, 0]) == pytest.approx(0.4)

    assert "set_root_state" not in calls
    torch.testing.assert_close(data.joint_pos, original)


def test_il3_scratch_geometry_is_unsupported() -> None:
    """A transaction scratch carries no geometry copy."""
    query, _, _, _ = _transaction_env()

    with query.scratch(0) as scratch:
        with pytest.raises(QueryCapabilityError, match="scratch.geometry"):
            scratch.geometry().distance(EntityRef("body", "base"), EntityRef("body", "link"))
        with pytest.raises(QueryCapabilityError, match="scratch.geometry"):
            scratch.geometry().aabb(EntityRef("body", "base"))


def test_il3_scratch_contacts_are_typed_unsupported() -> None:
    """A transaction scratch has no contact copy; scratch.contacts is typed unsupported."""
    query, _, _, _ = _transaction_env()

    with query.scratch(0) as scratch:
        with pytest.raises(QueryCapabilityError) as caught:
            scratch.contacts()

    assert caught.value.op == "scratch.contacts"
    assert caught.value.support == "unsupported"
    assert caught.value.detail.strip()


def test_il3_scratch_is_exclusive_per_env_and_stale_after_close() -> None:
    """One env holds one transaction; different envs may open concurrently."""
    query, _, _, _ = _transaction_env()

    with query.scratch(0) as scratch:
        with pytest.raises(QueryCapabilityError, match="already has an active scratch"):
            with query.scratch(0):
                pass
        with query.scratch(1) as other:
            assert tuple(other.body_pose("base").env_ids.tolist()) == (1,)

    with pytest.raises(QueryStaleError, match="scratch world is closed"):
        scratch.body_pose("base")


def test_il3_scratch_rejects_an_unknown_joint_name() -> None:
    """Scratch joint writes resolve explicit names against the live joint order."""
    query, _, _, _ = _transaction_env()

    with query.scratch(0) as scratch:
        with pytest.raises(QueryReferenceError, match="bogus"):
            scratch.set_joint_state("robot", ("bogus",), torch.zeros(1))


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


@pytest.mark.parametrize("backend", ("isaaclab2", "isaaclab3_physx", "isaaclab3_newton"))
def test_state_twist_operations_are_typed_unsupported(backend: str) -> None:
    """Direct object/root/COM twist calls raise the typed capability error."""
    query, _ = _query(backend)
    caps = query.capabilities
    for op in (
        "state.root_state",
        "state.com_state",
        "state.object_state",
        "state.object_com_state",
    ):
        assert caps.supports(op) == "unsupported"
        with pytest.raises(QueryCapabilityError, match="typed rigid-object") as caught:
            caps.require(op)
        assert caught.value.op == op

    with pytest.raises(QueryCapabilityError, match="state.root_state"):
        query.state.root_state("robot")
    with pytest.raises(QueryCapabilityError, match="state.com_state"):
        query.state.com_state(ComRef("robot", body="base"))
    with pytest.raises(QueryCapabilityError, match="state.object_state"):
        query.state.object_state(ObjectRef("crate"))
    with pytest.raises(QueryCapabilityError, match="state.object_com_state"):
        query.state.object_com_state(ObjectRef("crate"))


@pytest.mark.parametrize("backend", ("isaaclab2", "isaaclab3_physx", "isaaclab3_newton"))
def test_object_jacobian_is_typed_unsupported(backend: str) -> None:
    """Direct object_jacobian calls raise a typed capability error, not AttributeError."""
    query, _ = _query(backend)

    assert query.capabilities.supports("jacobian.object") == "unsupported"
    with pytest.raises(QueryCapabilityError) as caught:
        query.kinematics.object_jacobian(ObjectRef("crate"))
    assert caught.value.op == "jacobian.object"
    assert caught.value.backend == backend


def test_unsupported_subfaces_raise_typed_capability_errors() -> None:
    """Scratch, Jacobian, geometry and contacts are P1; actuator state is too."""
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
    with pytest.raises(QueryCapabilityError, match="contacts.scan"):
        query.contacts.scan()
    with pytest.raises(QueryCapabilityError, match="contacts.jacobian"):
        query.contacts.jacobian(point, "robot")
    with pytest.raises(QueryCapabilityError, match="contacts.wrench_to_joint"):
        query.contacts.wrench_to_joint(point, "robot")
    with pytest.raises(QueryCapabilityError, match="actuators.state"):
        query.actuators.state("a1")


@pytest.mark.parametrize("backend", ("isaaclab2", "isaaclab3_physx", "isaaclab3_newton"))
def test_actuator_facts_come_from_the_ir_and_usd_attrs(backend: str) -> None:
    """Actuator groups publish facts-only rows with their declared joint names."""
    query, _ = _query(backend)

    rows = query.actuators.rows()

    assert tuple(facts.name for facts in rows) == ("a1", "a2")
    facts = query.actuators.facts("a1")
    assert facts.name == "a1"
    assert facts.target == "j1"
    assert facts.kind == "robot"
    assert facts.slot is None
    assert facts.slot_kind == "isaaclab_group"
    assert facts.ctrlrange is None
    assert facts.gear is None
    assert facts.gainprm is None
    assert facts.biasprm is None
    assert facts.joint_names == ("j1",)
    assert query.actuators.facts("a1", articulation="robot").joint_names == ("j1",)


@pytest.mark.parametrize("backend", ("isaaclab2", "isaaclab3_physx", "isaaclab3_newton"))
def test_actuator_facts_without_usd_joint_names_stay_empty(backend: str) -> None:
    """An actuator group with no published joint list reports an empty tuple."""
    query, _ = _query(backend)

    facts = query.actuators.facts("a2")

    assert facts.target is None
    assert facts.joint_names == ()


def test_unknown_actuator_names_the_available_rows() -> None:
    """An unknown group is a reference error naming the published actuators."""
    query, _ = _query("isaaclab2")

    with pytest.raises(QueryReferenceError, match=r"actuators\.facts.*'bogus'.*a1"):
        query.actuators.facts("bogus")


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


def _manifold_group(*, manifold: bool = True) -> ContactGroupSpec:
    from unigym.contacts.refs import ContactBodyRef

    return ContactGroupSpec(
        name="grip",
        sensor_refs=[ContactBodyRef(owner="robot", body="base")],
        filter_refs=[ContactBodyRef(owner="robot", body="link")],
        frame="world",
        manifold=manifold,
    )


def _manifold_record(env_id: int, contact_id: int) -> ManifoldContact:
    return ManifoldContact(
        env_id=env_id,
        body1="link",
        body2="base",
        pos_w=torch.tensor([0.0, 0.0, 1.0 + env_id], dtype=torch.float64),
        separation_m=-0.001,
        normal_force_n=3.0,
        contact_id=contact_id,
    )


class _ManifoldBackend:
    """A bound contact backend double for the group-resolution tests."""

    def __init__(self, records):
        self.inventory = object()
        self.records = tuple(records)
        self.calls = []

    def read_manifold(self, spec, env_ids, *, dt):
        selected = {int(value) for value in env_ids.tolist()}
        self.calls.append((spec.name, tuple(env_ids.tolist()), dt))
        return tuple(record for record in self.records if record.env_id in selected)


def _manifold_env(specs, records):
    query, env = _query("isaaclab3_physx")
    backend = _ManifoldBackend(records)
    env.contact_manager = SimpleNamespace(specs=list(specs), backend=backend)
    env.sim = SimpleNamespace(get_physics_dt=lambda: 0.005)
    return query, env, backend


def test_physx_manifold_reads_body_named_points_and_reports_its_inputs() -> None:
    """Only provable fields are mapped; normal/geom/frame/friction stay None."""
    spec = _manifold_group()
    query, _, backend = _manifold_env([spec], [_manifold_record(0, 0), _manifold_record(1, 1)])

    manifold = query.contacts.manifold("grip")

    assert tuple(manifold.env_ids.tolist()) == (0, 1)
    assert manifold.group == "grip"
    first, second = manifold.points
    assert (first.body1, first.body2) == ("link", "base")
    assert (second.body1, second.body2) == ("link", "base")
    assert (first.geom1, first.geom2) == (None, None)
    assert first.normal_w is None
    assert second.normal_w is None
    assert first.frame_w is None
    assert first.friction_force_w is None
    assert first.normal_force_n == pytest.approx(3.0)
    assert first.separation_m == pytest.approx(-0.001)
    torch.testing.assert_close(first.pos_w, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(second.pos_w, torch.tensor([0.0, 0.0, 2.0], dtype=torch.float64))
    assert backend.calls == [("grip", (0, 1), 0.005)]


def test_physx_manifold_env_selection_can_return_an_empty_manifold() -> None:
    """A selected env with no contacts is an empty tuple, not an error."""
    spec = _manifold_group()
    query, _, backend = _manifold_env([spec], [_manifold_record(0, 0)])

    manifold = query.contacts.manifold("grip", env=1)

    assert tuple(manifold.env_ids.tolist()) == (1,)
    assert manifold.points == ()
    assert backend.calls == [("grip", (1,), 0.005)]


def test_physx_manifold_requires_the_group_opt_in() -> None:
    """A group without manifold: true is a typed capability error."""
    query, _, _ = _manifold_env([_manifold_group(manifold=False)], [])

    with pytest.raises(QueryCapabilityError, match="manifold: true"):
        query.contacts.manifold("grip")


def test_physx_manifold_unknown_group_is_a_reference_error() -> None:
    """An unknown group names the declared groups instead of scanning."""
    query, _, _ = _manifold_env([_manifold_group()], [])

    with pytest.raises(QueryReferenceError, match="ghost"):
        query.contacts.manifold("ghost")


def test_physx_manifold_without_a_contact_manager_is_a_capability_error() -> None:
    """A run without contacts: groups has no manifold source."""
    query, _ = _query("isaaclab3_physx")

    with pytest.raises(QueryCapabilityError, match="no contacts"):
        query.contacts.manifold("grip")


@pytest.mark.parametrize("backend", ("isaaclab2", "isaaclab3_newton"))
def test_contact_manifold_stays_unsupported_on_il2_and_newton(backend: str) -> None:
    """IL2 and Newton declare no contact manifold."""
    query, _ = _query(backend)

    with pytest.raises(QueryCapabilityError, match="contacts.manifold"):
        query.contacts.manifold("grip")


def test_physx_contact_jacobian_and_wrench_stay_typed_unsupported() -> None:
    """An IL3 PhysX manifold does not turn the other contact ops into AttributeError."""
    query, _ = _query("isaaclab3_physx")
    point = _contact_point()

    with pytest.raises(QueryCapabilityError, match="contacts.jacobian"):
        query.contacts.jacobian(point, "robot")
    with pytest.raises(QueryCapabilityError, match="contacts.wrench_to_joint"):
        query.contacts.wrench_to_joint(point, "robot")


@pytest.mark.parametrize("backend", ("isaaclab2", "isaaclab3_physx", "isaaclab3_newton"))
def test_contact_scan_is_typed_unsupported(backend: str) -> None:
    """No IsaacLab identity offers an unfiltered all-contact scan."""
    query, _ = _query(backend)

    assert query.capabilities.supports("contacts.scan") == "unsupported"
    with pytest.raises(QueryCapabilityError) as caught:
        query.contacts.scan()

    assert caught.value.op == "contacts.scan"
    assert caught.value.support == "unsupported"
    assert caught.value.backend == backend
    assert "all-contact scan" in caught.value.detail
