"""MuJoCo actuator facts and per-env runtime reads against the real fixture."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from unigym.backends.mujoco.query import MUJOCO_QUERY_CAPABILITIES, MujocoQuery
from unigym.query import QueryReferenceError, QueryStaleError


def _query(env) -> MujocoQuery:
    return MujocoQuery(env)


def test_actuator_facts_come_from_the_compiled_row(tiny_env) -> None:
    """Facts resolve the compiled slot and every declared parameter tuple."""
    query = _query(tiny_env)
    model = tiny_env.mj_model
    row = query.model.actuator("shoulder_motor")
    slot = model.actuator("shoulder_motor").id

    facts = query.actuators.facts("shoulder_motor")

    assert facts.name == "shoulder_motor"
    assert facts.target == row.target == "shoulder"
    assert facts.kind == row.kind
    assert facts.slot == slot
    assert facts.slot_kind == "mujoco_actuator"
    np.testing.assert_allclose(facts.ctrlrange, model.actuator_ctrlrange[slot])
    np.testing.assert_allclose(facts.gear, model.actuator_gear[slot])
    np.testing.assert_allclose(facts.gainprm, model.actuator_gainprm[slot])
    np.testing.assert_allclose(facts.biasprm, model.actuator_biasprm[slot])


def test_actuator_rows_preserve_the_model_ir_order(tiny_env) -> None:
    """Rows are the compiled table order, never a re-sorted view."""
    query = _query(tiny_env)

    merged = tuple(facts.name for facts in query.actuators.rows())
    scoped = tuple(facts.name for facts in query.actuators.rows("tiny"))

    assert merged == tuple(row.name for row in query.model.ir().actuators)
    assert merged == ("shoulder_motor", "elbow_motor")
    assert scoped == merged


def test_actuator_state_reads_the_live_channels(tiny_env) -> None:
    """ctrl/force/length/velocity match the current MjData at the slot."""
    query = _query(tiny_env)
    model, data = tiny_env.mj_model, tiny_env.mj_data
    slot = model.actuator("shoulder_motor").id
    data.ctrl[slot] = 0.3
    data.ctrl[model.actuator("elbow_motor").id] = -0.2
    mujoco.mj_forward(model, data)

    state = query.actuators.state("shoulder_motor")

    assert state.env_ids.tolist() == [0]
    assert state.ctrl is not None
    assert state.force is not None
    assert state.length is not None
    assert state.velocity is not None
    assert float(state.ctrl[0]) == pytest.approx(float(data.ctrl[slot]))
    assert float(state.force[0]) == pytest.approx(float(data.actuator_force[slot]))
    assert float(state.length[0]) == pytest.approx(float(data.actuator_length[slot]))
    assert float(state.velocity[0]) == pytest.approx(float(data.actuator_velocity[slot]))


def test_actuator_state_selects_envs_in_the_requested_order(tiny_env_factory) -> None:
    """Per-env selection preserves order and reads each env's own data."""
    env = tiny_env_factory(num_envs=2)
    query = _query(env)
    slot = env.mj_model.actuator("shoulder_motor").id
    for index, value in enumerate((0.25, -0.5)):
        env.mj_datas[index].ctrl[slot] = value
        mujoco.mj_forward(env.mj_model, env.mj_datas[index])

    state = query.actuators.state("shoulder_motor", env=[1, 0])

    assert state.env_ids.tolist() == [1, 0]
    assert state.ctrl is not None
    assert state.ctrl.tolist() == pytest.approx([-0.5, 0.25])


def test_actuator_state_empty_selection_is_zero_rows(tiny_env) -> None:
    """An empty env selection returns zero-row channels, never all envs."""
    state = _query(tiny_env).actuators.state("shoulder_motor", env=[])

    assert state.env_ids.numel() == 0
    assert state.ctrl is not None
    assert state.ctrl.shape == (0,)


def test_actuator_state_is_a_detached_copy(tiny_env) -> None:
    """Mutating a result never mutates the live control buffer."""
    query = _query(tiny_env)
    slot = tiny_env.mj_model.actuator("shoulder_motor").id
    original = float(tiny_env.mj_data.ctrl[slot])

    state = query.actuators.state("shoulder_motor")
    assert state.ctrl is not None
    state.ctrl[0] = 99.0

    assert float(tiny_env.mj_data.ctrl[slot]) == original


def test_unknown_actuator_names_the_available_rows(tiny_env) -> None:
    """An unknown neutral name is a reference error, never a zero row."""
    query = _query(tiny_env)

    with pytest.raises(
        QueryReferenceError,
        match=r"actuators\.facts.*'bogus'.*shoulder_motor",
    ):
        query.actuators.facts("bogus")
    with pytest.raises(QueryReferenceError, match=r"actuators\.state.*'bogus'"):
        query.actuators.state("bogus")


def test_actuator_reads_are_stale_after_close(tiny_env) -> None:
    """A closed query handle rejects actuator reads."""
    query = _query(tiny_env)
    query.close()

    with pytest.raises(QueryStaleError, match="stale"):
        query.actuators.rows()


def test_actuator_reads_require_articulation_without_a_merged_ir(tmp_path, tiny_mjcf) -> None:
    """An unmerged multi-articulation run cannot answer an ambiguous actuator read."""
    from tests.backends.mujoco.test_query import _two_arm_query

    env, query = _two_arm_query(tmp_path, tiny_mjcf)
    try:
        with pytest.raises(QueryReferenceError, match="articulation"):
            query.actuators.rows()
        with pytest.raises(QueryReferenceError, match="articulation"):
            query.actuators.facts("shoulder_motor")
        with pytest.raises(QueryReferenceError, match="articulation"):
            query.actuators.state("shoulder_motor")

        scoped = [facts.name for facts in query.actuators.rows("left")]
        assert scoped == ["shoulder_motor", "elbow_motor"]
    finally:
        env.close()


def test_actuator_capabilities_are_native() -> None:
    """MuJoCo answers actuator facts and runtime natively."""
    caps = MUJOCO_QUERY_CAPABILITIES

    assert caps.supports("actuators.facts") == "native"
    assert caps.supports("actuators.state") == "native"
