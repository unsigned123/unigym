"""mjWarp runtime-query behavior over the real tiny warp fixture.

The query reads the per-world CPU ``MjData`` mirror; these tests compare every
read against that mirror and prove the facade shares one cached instance with
``run.query``.
"""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from unigym.backends.mjwarp.query import MJWARP_QUERY_CAPABILITIES, MjwarpQuery
from unigym.query import (
    ComRef,
    EntityRef,
    Query,
    QueryCapabilityError,
    QueryStaleError,
)
from unigym.run.instance import RunInstance


def _query(env) -> MjwarpQuery:
    return MjwarpQuery(env)


def test_capability_object_is_mirror_honest() -> None:
    """mjWarp is native only for model/state; CPU mirror math is synthetic."""
    caps = MJWARP_QUERY_CAPABILITIES
    assert caps.backend == "mjwarp"
    assert caps.supports("model.ir") == "native"
    assert caps.supports("state.body_pose") == "native"
    assert caps.supports("state.joint_position") == "native"
    assert caps.supports("scratch.copy") == "synthetic"
    assert caps.supports("scratch.forward") == "synthetic"
    assert caps.supports("jacobian.body") == "synthetic"
    assert caps.supports("geometry.distance") == "synthetic"
    assert caps.supports("jacobian.com") == "partial"
    assert caps.supports("contacts.manifold") == "unsupported"
    assert caps.supports("actuators.state") == "unsupported"
    assert "mirror" in caps.notes


def test_env_query_is_cached_mjwarp_and_matches_run_query(tiny_env) -> None:
    """``env.query`` and ``run.query`` are the same cached mjWarp object."""
    instance = RunInstance(
        spec=SimpleNamespace(backend="mjwarp"),
        profile=None,
        adapter=None,
        sampler=None,
        term_group=None,
        env=tiny_env,
        bridge=None,
    )

    first = instance.query
    assert isinstance(first, MjwarpQuery)
    assert isinstance(first, Query)
    assert first is instance.query
    assert tiny_env.query is first
    assert first.backend == "mjwarp"
    assert "mirror" in first.capabilities.notes["engine_thread"]


def test_model_facts_come_from_the_compiled_ir(tiny_env) -> None:
    """Body/geom/joint/actuator facts resolve to the compiled ids."""
    query = _query(tiny_env)
    model = tiny_env.mj_model

    ir = query.model.ir()
    assert ir.name == "tiny_arm"
    assert query.model.body("base").body_id == model.body("base").id
    assert query.model.geom("base_geom").geom_id == model.geom("base_geom").id
    assert query.model.joint("shoulder").dof_adr == model.joint("shoulder").dofadr
    assert query.model.actuator("elbow_motor").actuator_id == model.actuator("elbow_motor").id


def test_live_state_reads_the_cpu_mirror(tiny_env) -> None:
    """Body/root/COM poses are detached reads of the CPU mirror."""
    query = _query(tiny_env)
    data = tiny_env.mj_datas[0]
    body_id = tiny_env.mj_model.body("base").id

    pose = query.state.body_pose("base")
    np.testing.assert_allclose(pose.pos_w[0].numpy(), data.xpos[body_id])
    np.testing.assert_allclose(pose.quat_wxyz[0].numpy(), data.xquat[body_id])

    art = tiny_env.articulations["tiny"]
    adr = art.free_qpos_adr
    root = query.state.root_pose("tiny")
    np.testing.assert_allclose(root.pos_w[0].numpy(), data.qpos[adr : adr + 3])
    np.testing.assert_allclose(root.quat_wxyz[0].numpy(), data.qpos[adr + 3 : adr + 7])

    com = query.state.com_pose(ComRef("tiny", body="base", subtree=True))
    np.testing.assert_allclose(com.pos_w[0].numpy(), data.subtree_com[body_id])

    pose.pos_w[0, 0] = 123.0
    assert float(tiny_env.mj_datas[0].xpos[body_id][0]) != 123.0


def test_joint_reads_follow_the_compiled_addresses(tiny_env) -> None:
    """Joint position/velocity read the mirror at the compiled addresses."""
    query = _query(tiny_env)
    data = tiny_env.mj_datas[0]
    art = tiny_env.articulations["tiny"]

    positions = query.state.joint_position("tiny")
    velocities = query.state.joint_velocity("tiny")
    np.testing.assert_allclose(positions[0].numpy(), data.qpos[art.jadr_art])
    np.testing.assert_allclose(velocities[0].numpy(), data.qvel[art.jdr_art])


def test_scratch_is_a_cpu_copy_and_never_touches_the_live_mirror(tiny_env) -> None:
    """Scratch writes/forward stay on the private MjData copy."""
    query = _query(tiny_env)
    live_qpos = tiny_env.mj_datas[0].qpos.copy()

    with query.scratch(0) as scratch:
        scratch.set_root_state(
            "tiny",
            pos_w=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        )
        scratch.forward()
        moved = scratch.body_pose("base")
        np.testing.assert_allclose(moved.pos_w[0].numpy(), [1.0, 2.0, 3.0], atol=1e-9)

    np.testing.assert_allclose(tiny_env.mj_datas[0].qpos, live_qpos)
    with pytest.raises(QueryStaleError, match="scratch"):
        scratch.body_pose("base")


def test_scratch_is_exclusive_per_env(tiny_env) -> None:
    """A second active scratch for the same env fails loudly."""
    query = _query(tiny_env)
    with query.scratch(0):
        with pytest.raises(QueryCapabilityError, match="active scratch"):
            with query.scratch(0):
                pass


def test_geometry_distance_and_aabb_use_the_mirror(tiny_env) -> None:
    """Geometry math matches the CPU call on the mirror."""
    query = _query(tiny_env)
    model = tiny_env.mj_model
    data = tiny_env.mj_datas[0]
    a = EntityRef("geom", "base_geom")
    b = EntityRef("geom", "tip_geom")

    expected = np.zeros(6, dtype=np.float64)
    raw = float(
        mujoco.mj_geomDistance(
            model,
            data,
            model.geom("base_geom").id,
            model.geom("tip_geom").id,
            10.0,
            expected,
        )
    )
    result = query.geometry.distance(a, b, max_distance_m=10.0)
    assert result.capped.tolist() == [False]
    assert float(result.signed_distance_m[0]) == pytest.approx(raw)

    geom_id = model.geom("base_geom").id
    center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    half = np.abs(rotation) @ np.array([0.08, 0.08, 0.08], dtype=np.float64)
    box = query.geometry.aabb(EntityRef("body", "base"))
    np.testing.assert_allclose(box.min_w[0].numpy(), center - half)
    np.testing.assert_allclose(box.max_w[0].numpy(), center + half)


def test_body_jacobian_matches_cpu_mujoco_on_the_mirror(tiny_env) -> None:
    """The mjWarp Jacobian uses the same CPU math on the mirror."""
    query = _query(tiny_env)
    model, data = tiny_env.mj_model, tiny_env.mj_datas[0]
    body_id = model.body("base").id
    expected_p = np.zeros((3, model.nv), dtype=np.float64)
    expected_r = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, expected_p, expected_r, body_id)

    jacobian = query.kinematics.jacobian(EntityRef("body", "base"))

    assert jacobian.layout.articulation == "tiny"
    assert jacobian.layout.nv == model.nv
    np.testing.assert_allclose(jacobian.transl[0].numpy(), expected_p)
    np.testing.assert_allclose(jacobian.rot[0].numpy(), expected_r)


def test_contacts_and_actuators_are_capability_gated(tiny_env) -> None:
    """The shared subfaces raise the mjWarp capability error until P1."""
    query = _query(tiny_env)
    geom = EntityRef("geom", "base_geom")

    with pytest.raises(QueryCapabilityError, match=r"mjwarp.*contacts.manifold"):
        query.contacts.manifold(geom)
    with pytest.raises(QueryCapabilityError, match=r"mjwarp.*actuators.facts"):
        query.actuators.facts("shoulder_motor")


def test_query_close_is_stale(tiny_env) -> None:
    """A closed query handle rejects further reads with the mjWarp label."""
    query = tiny_env.query
    query.close()

    with pytest.raises(QueryStaleError, match="mjwarp"):
        query.state.body_pose("base")


def test_env_close_invalidates_the_cached_query(tiny_env) -> None:
    """Closing the env closes the cached query and the env face."""
    env_query = tiny_env.query
    tiny_env.close()

    with pytest.raises(QueryStaleError, match="closed"):
        tiny_env.query
    with pytest.raises(QueryStaleError, match="stale"):
        env_query.state.body_pose("base")
