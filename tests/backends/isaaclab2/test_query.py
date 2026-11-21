"""Real IsaacLab 2 runtime-query behavior on the Cartpole lane fixture."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unigym.backends.isaaclab2.query import Isaaclab2Query
from unigym.query import EntityRef, Query, QueryCapabilityError
from unigym.query.errors import QueryStaleError
from unigym.run.instance import RunInstance

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]

ARTICULATION = "cartpole"


def test_env_query_is_cached_and_matches_run_query(cartpole_env) -> None:
    """env.query is one cached Isaaclab2Query reused by run.query."""
    query = cartpole_env.query

    assert isinstance(query, Isaaclab2Query)
    assert isinstance(query, Query)
    assert query is cartpole_env.query
    assert query.backend == "isaaclab2"

    instance = RunInstance(
        spec=SimpleNamespace(backend="isaaclab2"),
        profile=None,
        adapter=None,
        sampler=None,
        term_group=None,
        env=cartpole_env,
        bridge=None,
    )
    assert instance.query is query


def test_model_facts_and_live_body_state(cartpole_env) -> None:
    """The compiled IR facts and live body poses are exposed by name."""
    query = cartpole_env.query

    ir = query.model.ir()
    assert query.model.ir(ARTICULATION) is cartpole_env.model_set[ARTICULATION]
    assert query.model.body(ir.bodies.names()[0]).name == ir.bodies.names()[0]
    assert query.model.joint(ir.joints.names()[0]).name == ir.joints.names()[0]

    pose = query.state.body_pose(ir.bodies.names()[0])
    assert tuple(pose.pos_w.shape) == (cartpole_env.num_envs, 3)
    assert tuple(pose.quat_wxyz.shape) == (cartpole_env.num_envs, 4)

    positions = query.state.joint_position(ARTICULATION)
    assert tuple(positions.shape) == (cartpole_env.num_envs, len(ir.joints))


def test_geom_site_model_facts_and_p1_subfaces_are_unsupported(cartpole_env) -> None:
    """P0B-2 exposes no geom/site model facts and no scratch/geometry/Jacobian."""
    query = cartpole_env.query

    with pytest.raises(QueryCapabilityError, match="model.geom"):
        query.model.geom("bogus")
    with pytest.raises(QueryCapabilityError, match="model.site"):
        query.model.site("bogus")
    with pytest.raises(QueryCapabilityError, match="scratch"):
        query.scratch()
    with pytest.raises(QueryCapabilityError, match="jacobian"):
        query.kinematics.jacobian(EntityRef("body", "bogus"))


def test_close_invalidates_env_query(cartpole_env) -> None:
    """Closing the env closes the cached query handle."""
    query = cartpole_env.query
    cartpole_env.close()

    with pytest.raises(QueryStaleError, match="stale"):
        query.state.body_pose("bogus")
