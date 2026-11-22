"""MuJoCo runtime-query behavior against the real tiny arm fixture."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from unigym.backends.mujoco.query import MujocoQuery
from unigym.core import quat as quat_math
from unigym.model import GeomContact
from unigym.query import (
    ComRef,
    EntityRef,
    PointRef,
    Query,
    QueryCapabilityError,
    QueryReferenceError,
    QueryStaleError,
)
from unigym.run.instance import RunInstance


def _query(env) -> MujocoQuery:
    return MujocoQuery(env)


def _with_site(source: Path, tmp_path: Path) -> Path:
    """Copy the tiny MJCF with one site on the tip body."""
    text = source.read_text(encoding="utf-8")
    marker = '<geom name="tip_geom"'
    replacement = '<site name="ee" pos="0 0 -0.2" size="0.01"/>\n          ' + marker
    target = tmp_path / "tiny_arm_site.xml"
    target.write_text(text.replace(marker, replacement, 1), encoding="utf-8")
    return target


def _with_ball_joint(source: Path, tmp_path: Path) -> Path:
    """Copy the tiny MJCF with one unactuated ball joint on a child body."""
    text = source.read_text(encoding="utf-8")
    marker = '<geom name="tip_geom"'
    replacement = (
        '<body name="wrist" pos="0 0 -0.2">'
        '<joint name="wrist_ball" type="ball"/>'
        '<geom name="wrist_geom" type="sphere" size="0.03" mass="0.1"/>'
        "</body>\n          " + marker
    )
    target = tmp_path / "tiny_arm_ball.xml"
    target.write_text(text.replace(marker, replacement, 1), encoding="utf-8")
    return target


def test_model_facts_come_from_the_compiled_ir(tiny_env) -> None:
    """Body/geom/joint/actuator facts resolve to the compiled ids."""
    query = _query(tiny_env)
    model = tiny_env.mj_model

    ir = query.model.ir()
    assert ir.name == "tiny_arm"
    assert query.model.ir("tiny").name == ir.name
    assert query.model.body("base").body_id == model.body("base").id
    assert query.model.geom("base_geom").geom_id == model.geom("base_geom").id
    assert query.model.joint("shoulder").dof_adr == model.joint("shoulder").dofadr
    assert query.model.actuator("elbow_motor").actuator_id == model.actuator("elbow_motor").id


def test_unknown_body_names_the_available_bodies(tiny_env) -> None:
    """A missing name is a reference error, never a silent miss."""
    with pytest.raises(QueryReferenceError, match=r"bogus.*base"):
        _query(tiny_env).model.body("bogus")


def test_geom_contact_matches_the_compiled_model(tiny_env) -> None:
    """The public geom contact row equals the compiled IR and MjModel values."""
    query = _query(tiny_env)
    contact = query.model.geom_contact("base_geom")
    ir = query.model.ir()
    assert ir.mj is not None
    assert contact == ir.mj.geom_contact["base_geom"]
    assert isinstance(contact, GeomContact)

    model = tiny_env.mj_model
    geom_id = model.geom("base_geom").id
    assert contact.friction == tuple(float(value) for value in model.geom_friction[geom_id])
    assert contact.solref == tuple(float(value) for value in model.geom_solref[geom_id])
    assert contact.solimp == tuple(float(value) for value in model.geom_solimp[geom_id])
    assert contact.margin == float(model.geom_margin[geom_id])
    assert contact.gap == float(model.geom_gap[geom_id])
    assert contact.priority == int(model.geom_priority[geom_id])
    assert contact.condim == int(model.geom_condim[geom_id])


def test_unknown_geom_contact_names_the_available_geoms(tiny_env) -> None:
    """An unknown geom is a reference error naming the available geoms."""
    with pytest.raises(QueryReferenceError, match=r"bogus.*base_geom"):
        _query(tiny_env).model.geom_contact("bogus")


def test_body_root_and_com_poses_match_mjdata(tiny_env) -> None:
    """Live poses are detached copies of the current MjData."""
    query = _query(tiny_env)
    data = tiny_env.mj_data
    body_id = tiny_env.mj_model.body("base").id

    pose = query.state.body_pose("base")
    np.testing.assert_allclose(pose.pos_w[0].numpy(), data.xpos[body_id])
    np.testing.assert_allclose(pose.quat_wxyz[0].numpy(), data.xquat[body_id])

    root = query.state.root_pose("tiny")
    adr = tiny_env.articulations["tiny"].free_qpos_adr
    np.testing.assert_allclose(root.pos_w[0].numpy(), data.qpos[adr : adr + 3])
    np.testing.assert_allclose(root.quat_wxyz[0].numpy(), data.qpos[adr + 3 : adr + 7])

    com = query.state.com_pose(ComRef("tiny", body="base", subtree=True))
    np.testing.assert_allclose(com.pos_w[0].numpy(), data.subtree_com[body_id])
    body_com = query.state.com_pose(ComRef("tiny", body="base"))
    np.testing.assert_allclose(body_com.pos_w[0].numpy(), data.xipos[body_id])


def test_state_results_are_detached_copies(tiny_env) -> None:
    """Mutating a result never mutates the live MjData."""
    query = _query(tiny_env)
    body_id = tiny_env.mj_model.body("base").id
    original = float(tiny_env.mj_data.xpos[body_id][0])

    pose = query.state.body_pose("base")
    pose.pos_w[0, 0] = 123.0

    assert float(tiny_env.mj_data.xpos[body_id][0]) == original


def test_site_pose_matches_site_xmat(tiny_env_factory, tiny_mjcf, tmp_path) -> None:
    """A site pose is read from site_xpos/site_xmat."""
    env = tiny_env_factory(mjcf=_with_site(tiny_mjcf, tmp_path))
    site_id = env.mj_model.site("ee").id
    pose = _query(env).state.site_pose("ee")

    np.testing.assert_allclose(pose.pos_w[0].numpy(), env.mj_data.site_xpos[site_id])
    expected = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(expected, env.mj_data.site_xmat[site_id])
    np.testing.assert_allclose(pose.quat_wxyz[0].numpy(), expected)


def test_joint_position_and_velocity_follow_the_asset_order(tiny_env) -> None:
    """Joint reads use the articulation's declared joint order and addresses."""
    tiny_env.set_joint_state(
        None,
        articulation="tiny",
        joint_names=("shoulder", "elbow"),
        position=torch.tensor([[0.3, 0.4]], dtype=torch.float32),
        velocity=torch.tensor([[0.5, 0.6]], dtype=torch.float32),
    )
    tiny_env.settle(None)
    query = _query(tiny_env)
    art = tiny_env.articulations["tiny"]

    positions = query.state.joint_position("tiny")
    velocities = query.state.joint_velocity("tiny")
    np.testing.assert_allclose(positions[0].numpy(), tiny_env.mj_data.qpos[art.jadr_art])
    np.testing.assert_allclose(velocities[0].numpy(), tiny_env.mj_data.qvel[art.jdr_art])

    elbow = query.state.joint_position("tiny", joint_names=("elbow",))
    assert tuple(elbow.shape) == (1, 1)
    np.testing.assert_allclose(elbow[0].numpy(), tiny_env.mj_data.qpos[art.jadr_art[1]])


def test_scratch_forward_moves_only_the_copy(tiny_env) -> None:
    """Scratch writes/forward never touch the live MjData."""
    query = _query(tiny_env)
    live_qpos = tiny_env.mj_data.qpos.copy()

    with query.scratch(0) as scratch:
        scratch.set_root_state(
            "tiny",
            pos_w=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        )
        scratch.set_joint_state(
            "tiny",
            ("shoulder", "elbow"),
            torch.tensor([0.1, 0.2], dtype=torch.float64),
        )
        scratch.forward()
        moved = scratch.body_pose("base")
        np.testing.assert_allclose(moved.pos_w[0].numpy(), [1.0, 2.0, 3.0], atol=1e-9)
        jacobian = scratch.jacobian(EntityRef("body", "base"))
        assert tuple(jacobian.transl.shape) == (1, 3, 8)

    np.testing.assert_allclose(tiny_env.mj_data.qpos, live_qpos)
    with pytest.raises(QueryStaleError, match="scratch"):
        scratch.body_pose("base")


def test_scratch_is_exclusive_per_env(tiny_env) -> None:
    """A second active scratch for the same env fails loudly."""
    query = _query(tiny_env)
    with query.scratch(0):
        with pytest.raises(QueryCapabilityError, match="active scratch"):
            with query.scratch(0):
                pass


def test_scratch_bound_geometry_is_stale_after_close(tiny_env) -> None:
    """A geometry subface captured from a scratch dies with the scratch."""
    query = _query(tiny_env)
    a = EntityRef("geom", "base_geom")
    b = EntityRef("geom", "tip_geom")

    with query.scratch(0) as scratch:
        geometry = scratch.geometry()
        assert geometry.distance(a, b).capped.tolist() == [False]

    with pytest.raises(QueryStaleError, match="scratch"):
        geometry.distance(a, b)


def test_scratch_close_releases_the_env_exclusivity(tiny_env) -> None:
    """An explicitly closed scratch no longer blocks the same env."""
    query = _query(tiny_env)

    with query.scratch(0) as first:
        first.close()
        with query.scratch(0) as second:
            assert second is not first


def test_scratch_enter_after_query_close_is_stale(tiny_env) -> None:
    """Entering a scratch context after the query closed is a stale read."""
    query = _query(tiny_env)
    context = query.scratch(0)
    query.close()

    with pytest.raises(QueryStaleError, match="stale"):
        context.__enter__()


def test_geometry_distance_and_cap_flag(tiny_env) -> None:
    """mj_geomDistance feeds distance, closest points and the cap flag."""
    query = _query(tiny_env)
    a = EntityRef("geom", "base_geom")
    b = EntityRef("geom", "tip_geom")

    uncapped = query.geometry.distance(a, b, max_distance_m=10.0)
    assert uncapped.capped.tolist() == [False]
    assert uncapped.mesh_exact is False
    assert float(uncapped.signed_distance_m[0]) > 0.0
    assert tuple(uncapped.normal_w.shape) == (1, 3)

    capped = query.geometry.distance(a, b, max_distance_m=0.1)
    assert capped.capped.tolist() == [True]
    assert float(capped.signed_distance_m[0]) == pytest.approx(0.1)


def test_geometry_distance_rejects_env_and_scratch_together(tiny_env) -> None:
    """A distance query has exactly one data source."""
    query = _query(tiny_env)
    a = EntityRef("geom", "base_geom")
    b = EntityRef("geom", "tip_geom")

    with query.scratch(0) as scratch:
        with pytest.raises(ValueError, match="exactly one of env or scratch"):
            query.geometry.distance(a, b, env=0, scratch=scratch)


def test_aabb_matches_the_box_geometry(tiny_env) -> None:
    """A AABB is the world-aligned box of the named body's geometry."""
    query = _query(tiny_env)
    geom_id = tiny_env.mj_model.geom("base_geom").id
    center = np.asarray(tiny_env.mj_data.geom_xpos[geom_id], dtype=np.float64)
    rotation = np.asarray(tiny_env.mj_data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    half = np.abs(rotation) @ np.array([0.08, 0.08, 0.08], dtype=np.float64)

    box = query.geometry.aabb(EntityRef("body", "base"))

    np.testing.assert_allclose(box.min_w[0].numpy(), center - half)
    np.testing.assert_allclose(box.max_w[0].numpy(), center + half)


def test_aabb_covers_a_capsule_geometry(tiny_env) -> None:
    """Capsule half length and radius both contribute to the AABB."""
    query = _query(tiny_env)
    geom_id = tiny_env.mj_model.geom("link_geom").id
    center = np.asarray(tiny_env.mj_data.geom_xpos[geom_id], dtype=np.float64)
    rotation = np.asarray(tiny_env.mj_data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    size = np.asarray(tiny_env.mj_model.geom_size[geom_id], dtype=np.float64)
    half = np.abs(rotation) @ np.array([size[0], size[0], size[1] + size[0]], dtype=np.float64)

    box = query.geometry.aabb(EntityRef("geom", "link_geom"))

    np.testing.assert_allclose(box.min_w[0].numpy(), center - half)
    np.testing.assert_allclose(box.max_w[0].numpy(), center + half)


def test_body_jacobian_matches_mj_jacbody(tiny_env) -> None:
    """The body Jacobian/layout agree with the MuJoCo call."""
    query = _query(tiny_env)
    model, data = tiny_env.mj_model, tiny_env.mj_data
    body_id = model.body("base").id
    expected_p = np.zeros((3, model.nv), dtype=np.float64)
    expected_r = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, expected_p, expected_r, body_id)

    jacobian = query.kinematics.jacobian(EntityRef("body", "base"))

    assert jacobian.layout.articulation == "tiny"
    assert jacobian.layout.nv == model.nv
    assert jacobian.layout.base_dofs[0] == "base.px"
    assert jacobian.layout.order[-2:] == ("shoulder", "elbow")
    np.testing.assert_allclose(jacobian.transl[0].numpy(), expected_p)
    np.testing.assert_allclose(jacobian.rot[0].numpy(), expected_r)


def test_jacobian_rejects_an_unrepresented_ball_joint(
    tiny_env_factory, tiny_mjcf, tmp_path
) -> None:
    """A compiled ball joint outside the neutral joint table is not native."""
    env = tiny_env_factory(mjcf=_with_ball_joint(tiny_mjcf, tmp_path))
    query = _query(env)

    with pytest.raises(QueryCapabilityError, match=r"wrist_ball.*ball"):
        query.kinematics.jacobian(EntityRef("body", "base"))


def test_free_root_with_unresolved_base_body_fails_loudly(tiny_env_factory) -> None:
    """A declared free root without a resolved base body is not shortened."""
    env = tiny_env_factory(base_body=None)
    query = _query(env)

    with pytest.raises(QueryCapabilityError, match="free root"):
        query.kinematics.jacobian(EntityRef("body", "base"))
    with pytest.raises(QueryCapabilityError, match="free root"):
        query.state.root_pose("tiny")
    with pytest.raises(QueryCapabilityError, match="free root"):
        with query.scratch(0) as scratch:
            scratch.set_root_state(
                "tiny",
                pos_w=torch.zeros(3, dtype=torch.float64),
                quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            )


def test_resolved_free_root_still_produces_the_full_layout(tiny_env) -> None:
    """A normally resolved free root keeps the 6 base DOFs in the layout."""
    query = _query(tiny_env)

    jacobian = query.kinematics.jacobian(EntityRef("body", "base"))
    root = query.state.root_pose("tiny")

    assert jacobian.layout.nv == tiny_env.mj_model.nv
    assert len(jacobian.layout.base_dofs) == 6
    assert tuple(root.pos_w.shape) == (1, 3)


def test_root_state_reads_free_joint_qvel_in_world_frame(tiny_env) -> None:
    """A free-base root twist is qpos/qvel with local angular converted to world."""
    query = _query(tiny_env)
    model, data = tiny_env.mj_model, tiny_env.mj_data
    art = tiny_env.articulations["tiny"]
    qadr = int(art.free_qpos_adr)
    dadr = int(art.free_dof_adr)
    quat = np.array([0.7071067811865476, 0.0, 0.7071067811865476, 0.0])
    position = np.array([0.4, -0.2, 0.9])
    linear = np.array([0.1, -0.2, 0.3])
    angular_local = np.array([0.4, -0.5, 0.6])

    data.qpos[qadr : qadr + 3] = position
    data.qpos[qadr + 3 : qadr + 7] = quat
    data.qvel[dadr : dadr + 3] = linear
    data.qvel[dadr + 3 : dadr + 6] = angular_local
    data.qvel[np.asarray(art.jdr_art, dtype=np.int64)] = [1.1, -1.2]
    mujoco.mj_forward(model, data)

    state = query.state.root_state("tiny")

    np.testing.assert_allclose(state.pos_w[0].numpy(), position)
    np.testing.assert_allclose(state.quat_wxyz[0].numpy(), quat)
    np.testing.assert_allclose(state.lin_vel_w[0].numpy(), linear)
    expected = quat_math.to_rotmat(quat) @ angular_local
    np.testing.assert_allclose(state.ang_vel_w[0].numpy(), expected)
    assert not np.allclose(state.ang_vel_w[0].numpy(), angular_local)
    assert tuple(state.ang_vel_w.shape) == (1, 3)
    assert not np.allclose(state.ang_vel_w[0].numpy()[0], 1.1)


def test_com_state_uses_the_body_and_subtree_com_jacobians(tiny_env) -> None:
    """COM velocity is mj_jacBodyCom/mj_jacSubtreeCom @ qvel at the COM point."""
    query = _query(tiny_env)
    model, data = tiny_env.mj_model, tiny_env.mj_data
    body_id = model.body("base").id
    data.qvel[:] = np.linspace(-0.4, 0.6, model.nv)
    mujoco.mj_forward(model, data)

    body = query.state.com_state(ComRef("tiny", body="base"))
    subtree = query.state.com_state(ComRef("tiny", body="base", subtree=True))

    expected_body = np.zeros((3, model.nv), dtype=np.float64)
    expected_rot = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBodyCom(model, data, expected_body, expected_rot, body_id)
    np.testing.assert_allclose(body.pos_w[0].numpy(), data.xipos[body_id])
    np.testing.assert_allclose(body.lin_vel_w[0].numpy(), expected_body @ data.qvel)

    expected_subtree = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSubtreeCom(model, data, expected_subtree, body_id)
    np.testing.assert_allclose(subtree.pos_w[0].numpy(), data.subtree_com[body_id])
    np.testing.assert_allclose(subtree.lin_vel_w[0].numpy(), expected_subtree @ data.qvel)


def test_fixed_asset_root_state_matches_root_pose_error_and_fixed_com_is_zero(
    tiny_env_factory, tiny_mjcf, tmp_path
) -> None:
    """A free-declared fixed asset matches root_pose; a fixed body COM is engine zero."""
    fixed = tmp_path / "tiny_arm_fixed.xml"
    fixed.write_text(
        tiny_mjcf.read_text(encoding="utf-8").replace('<freejoint name="root"/>', ""),
        encoding="utf-8",
    )
    env = tiny_env_factory(mjcf=fixed)
    query = _query(env)

    with pytest.raises(QueryCapabilityError, match="free root") as state_error:
        query.state.root_state("tiny")
    with pytest.raises(QueryCapabilityError, match="free root") as pose_error:
        query.state.root_pose("tiny")
    assert state_error.value.op == "state.root_state"
    assert pose_error.value.op == "state.root_pose"

    com = query.state.com_state(ComRef("tiny", body="base"))
    np.testing.assert_allclose(com.lin_vel_w[0].numpy(), np.zeros(3))


def test_empty_env_selection_returns_zero_rows(tiny_env) -> None:
    """An empty sequence and an empty int64 tensor both select zero envs."""
    query = _query(tiny_env)
    a = EntityRef("geom", "base_geom")
    b = EntityRef("geom", "tip_geom")

    pose = query.state.body_pose("base", env=[])
    assert pose.env_ids.dtype == torch.int64
    assert tuple(pose.env_ids.shape) == (0,)
    assert tuple(pose.pos_w.shape) == (0, 3)

    empty = query.state.body_pose("base", env=torch.empty(0, dtype=torch.int64))
    assert tuple(empty.pos_w.shape) == (0, 3)
    assert tuple(query.state.body_pose("base", env=()).pos_w.shape) == (0, 3)

    distance = query.geometry.distance(a, b, env=[])
    assert tuple(distance.signed_distance_m.shape) == (0,)
    assert tuple(distance.capped.shape) == (0,)


def test_site_com_and_point_jacobians_match_mujoco(tiny_env_factory, tiny_mjcf, tmp_path) -> None:
    """Site, subtree-COM and world-point Jacobians use the matching mj_jac*."""
    env = tiny_env_factory(mjcf=_with_site(tiny_mjcf, tmp_path))
    query = _query(env)
    model, data = env.mj_model, env.mj_data
    site_id = model.site("ee").id
    expected_p = np.zeros((3, model.nv), dtype=np.float64)
    expected_r = np.zeros((3, model.nv), dtype=np.float64)

    mujoco.mj_jacSite(model, data, expected_p, expected_r, site_id)
    site = query.kinematics.jacobian(EntityRef("site", "ee"))
    np.testing.assert_allclose(site.transl[0].numpy(), expected_p)

    base_id = model.body("base").id
    expected_p.fill(0.0)
    mujoco.mj_jacSubtreeCom(model, data, expected_p, base_id)
    with pytest.raises(QueryCapabilityError, match="subtree COM"):
        query.kinematics.jacobian(ComRef("tiny", body="base", subtree=True))
    com = query.kinematics.jacobian(ComRef("tiny", body="base", subtree=True), rotational=False)
    np.testing.assert_allclose(com.transl[0].numpy(), expected_p)
    np.testing.assert_allclose(com.rot[0].numpy(), np.zeros((3, model.nv)))

    expected_p.fill(0.0)
    expected_r.fill(0.0)
    mujoco.mj_jacBodyCom(model, data, expected_p, expected_r, base_id)
    body_com = query.kinematics.jacobian(ComRef("tiny", body="base"))
    np.testing.assert_allclose(body_com.transl[0].numpy(), expected_p)
    np.testing.assert_allclose(body_com.rot[0].numpy(), expected_r)

    tip_id = model.body("tip").id
    point = np.asarray(data.xpos[tip_id], dtype=np.float64)
    expected_p.fill(0.0)
    expected_r.fill(0.0)
    mujoco.mj_jac(model, data, expected_p, expected_r, point, tip_id)
    result = query.kinematics.jacobian(
        PointRef(EntityRef("body", "tip"), torch.tensor(point, dtype=torch.float64))
    )
    np.testing.assert_allclose(result.transl[0].numpy(), expected_p)
    np.testing.assert_allclose(result.rot[0].numpy(), expected_r)


def test_contacts_without_a_declared_group_are_capability_gated(tiny_env) -> None:
    """A run with no contacts: groups cannot answer a manifold read."""
    query = _query(tiny_env)

    with pytest.raises(QueryCapabilityError, match=r"contacts\.manifold.*no contacts"):
        query.contacts.manifold("anything")


def test_require_accepts_every_implemented_op(tiny_env) -> None:
    """require() accepts every declared MuJoCo query operation."""
    query = _query(tiny_env)

    query.require(
        "model.ir",
        "state.body_pose",
        "geometry.distance",
        "actuators.facts",
        "actuators.state",
    )


def test_query_after_close_is_stale(tiny_env) -> None:
    """A closed query handle rejects every state read."""
    query = _query(tiny_env)
    query.close()

    with pytest.raises(QueryStaleError, match="stale"):
        query.state.body_pose("base")


def test_run_query_is_lazy_reused_and_closed_with_the_run(tiny_env) -> None:
    """RunInstance.query uses the registry factory and reuses one object."""
    instance = RunInstance(
        spec=SimpleNamespace(backend="mujoco"),
        profile=None,
        adapter=None,
        sampler=None,
        term_group=None,
        env=tiny_env,
        bridge=None,
    )

    first = instance.query
    assert isinstance(first, Query)
    assert first is instance.query
    assert tiny_env.query is first
    assert first.backend == "mujoco"
    assert "engine thread" in first.capabilities.notes["engine_thread"]

    instance.close()
    with pytest.raises(QueryStaleError):
        instance.query


def test_env_query_after_close_is_stale(tiny_env) -> None:
    """The optional backend face is a stale-query error after close."""
    query = tiny_env.query
    tiny_env.close()

    with pytest.raises(QueryStaleError, match="closed"):
        tiny_env.query
    with pytest.raises(QueryStaleError, match="stale"):
        query.state.body_pose("base")


def _two_arm_query(tmp_path, tiny_mjcf):
    from tests.backends.mujoco.test_env import _two_articulation_env

    env = _two_articulation_env(tmp_path, tiny_mjcf)
    return env, _query(env)


def test_geom_contact_is_articulation_scoped_without_a_merged_ir(tmp_path, tiny_mjcf) -> None:
    """A per-articulation IR answers the read; the ambiguous whole-scene read fails."""
    env, query = _two_arm_query(tmp_path, tiny_mjcf)
    try:
        assert env.model_set.merged is None
        left = query.model.geom_contact("left_base_geom", articulation="left")
        left_ir = query.model.ir("left")
        assert left_ir.mj is not None
        assert left == left_ir.mj.geom_contact["left_base_geom"]

        with pytest.raises(QueryReferenceError, match="articulation"):
            query.model.geom_contact("left_base_geom")
        with pytest.raises(QueryReferenceError, match="geom") as caught:
            query.model.geom_contact("right_base_geom", articulation="left")
        assert "right_base_geom" not in caught.value.available
    finally:
        env.close()


def test_jacobian_uses_the_entity_reference_articulation(tmp_path, tiny_mjcf) -> None:
    """EntityRef.articulation selects both the fill and the layout."""
    env, query = _two_arm_query(tmp_path, tiny_mjcf)
    try:
        left = query.kinematics.jacobian(EntityRef("body", "base", articulation="left"))
        right = query.kinematics.jacobian(EntityRef("body", "base", articulation="right"))
        assert left.layout.articulation == "left"
        assert right.layout.articulation == "right"

        by_kwarg = query.kinematics.jacobian(EntityRef("body", "base"), articulation="right")
        assert by_kwarg.layout.articulation == "right"
        np.testing.assert_allclose(by_kwarg.transl[0].numpy(), right.transl[0].numpy())

        art = env.articulations["right"]
        body_id = query.model.body("base", articulation="right").body_id
        expected_p = np.zeros((3, env.mj_model.nv), dtype=np.float64)
        expected_r = np.zeros((3, env.mj_model.nv), dtype=np.float64)
        mujoco.mj_jacBody(env.mj_model, env.mj_data, expected_p, expected_r, body_id)
        columns = list(range(int(art.free_dof_adr), int(art.free_dof_adr) + 6))
        columns.extend(int(value) for value in np.asarray(art.jdr_art).reshape(-1))
        np.testing.assert_allclose(right.transl[0].numpy(), expected_p[:, columns])
        np.testing.assert_allclose(right.rot[0].numpy(), expected_r[:, columns])
    finally:
        env.close()


def test_jacobian_requires_articulation_for_an_ambiguous_entity(tmp_path, tiny_mjcf) -> None:
    """A shared neutral name is ambiguous without an explicit articulation."""
    env, query = _two_arm_query(tmp_path, tiny_mjcf)
    try:
        with pytest.raises(QueryReferenceError, match="ambiguous"):
            query.kinematics.jacobian(EntityRef("body", "base"))
    finally:
        env.close()


@pytest.mark.parametrize(
    "target",
    [
        EntityRef("body", "base", articulation="left"),
        ComRef("left", body="base"),
        PointRef(
            EntityRef("body", "base", articulation="left"),
            torch.zeros(3, dtype=torch.float64),
        ),
    ],
)
def test_jacobian_rejects_conflicting_articulation(tmp_path, tiny_mjcf, target) -> None:
    """A reference articulation and the articulation kwarg must agree."""
    env, query = _two_arm_query(tmp_path, tiny_mjcf)
    try:
        with pytest.raises(QueryReferenceError, match="conflict"):
            query.kinematics.jacobian(target, articulation="right")
    finally:
        env.close()


def test_jacobian_com_and_point_reference_articulations(tmp_path, tiny_mjcf) -> None:
    """ComRef/PointRef articulation also selects their layout."""
    env, query = _two_arm_query(tmp_path, tiny_mjcf)
    try:
        com = query.kinematics.jacobian(ComRef("left", body="base", subtree=True), rotational=False)
        assert com.layout.articulation == "left"

        point = query.kinematics.jacobian(
            PointRef(
                EntityRef("body", "base", articulation="left"),
                torch.zeros(3, dtype=torch.float64),
            )
        )
        assert point.layout.articulation == "left"
    finally:
        env.close()
