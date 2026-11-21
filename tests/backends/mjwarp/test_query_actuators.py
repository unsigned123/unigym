"""mjWarp actuator facts and CPU-mirror runtime reads over the real fixture."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from unigym.backends.mjwarp.query import MJWARP_QUERY_CAPABILITIES, MjwarpQuery
from unigym.query import QueryReferenceError


def _query(env) -> MjwarpQuery:
    return MjwarpQuery(env)


def test_actuator_facts_match_the_cpu_mirror_model(tiny_env) -> None:
    """Facts resolve the compiled slot and the mirror model's declared values."""
    query = _query(tiny_env)
    model = tiny_env.mj_model
    slot = model.actuator("elbow_motor").id

    facts = query.actuators.facts("elbow_motor")

    assert facts.name == "elbow_motor"
    assert facts.slot == slot
    assert facts.slot_kind == "mujoco_actuator"
    np.testing.assert_allclose(facts.ctrlrange, model.actuator_ctrlrange[slot])
    np.testing.assert_allclose(facts.gear, model.actuator_gear[slot])


def test_actuator_rows_preserve_the_mirror_model_order(tiny_env) -> None:
    """Rows are the compiled mirror table order for one articulation."""
    query = _query(tiny_env)

    names = tuple(facts.name for facts in query.actuators.rows("tiny"))

    assert names == tuple(row.name for row in query.model.ir("tiny").actuators)
    assert names == ("shoulder_motor", "elbow_motor")


def test_actuator_state_reads_the_cpu_mirror(tiny_env) -> None:
    """Runtime channels are the mirror's data, not a warp device read."""
    query = _query(tiny_env)
    model, data = tiny_env.mj_model, tiny_env.mj_datas[0]
    slot = model.actuator("elbow_motor").id
    data.ctrl[slot] = -0.2
    mujoco.mj_forward(model, data)

    state = query.actuators.state("elbow_motor")

    assert state.env_ids.tolist() == [0]
    assert state.ctrl is not None
    assert state.force is not None
    assert state.length is not None
    assert state.velocity is not None
    assert float(state.ctrl[0]) == pytest.approx(float(data.ctrl[slot]))
    assert float(state.force[0]) == pytest.approx(float(data.actuator_force[slot]))
    assert float(state.length[0]) == pytest.approx(float(data.actuator_length[slot]))
    assert float(state.velocity[0]) == pytest.approx(float(data.actuator_velocity[slot]))


def test_actuator_state_is_a_copy_and_leaves_the_mirror_untouched(tiny_env) -> None:
    """Mutating a result never mutates the CPU mirror or the warp device."""
    query = _query(tiny_env)
    slot = tiny_env.mj_model.actuator("shoulder_motor").id
    data = tiny_env.mj_datas[0]
    before = (
        np.array(data.ctrl, copy=True),
        np.array(data.actuator_force, copy=True),
        np.array(data.actuator_length, copy=True),
        np.array(data.actuator_velocity, copy=True),
    )

    state = query.actuators.state("shoulder_motor")
    assert state.ctrl is not None
    state.ctrl[0] = 99.0

    np.testing.assert_array_equal(data.ctrl, before[0])
    np.testing.assert_array_equal(data.actuator_force, before[1])
    np.testing.assert_array_equal(data.actuator_length, before[2])
    np.testing.assert_array_equal(data.actuator_velocity, before[3])
    assert float(data.ctrl[slot]) != 99.0


def test_unknown_actuator_is_a_reference_error(tiny_env) -> None:
    """An unknown name names the available actuator rows."""
    with pytest.raises(QueryReferenceError, match=r"actuators\.facts.*'bogus'"):
        _query(tiny_env).actuators.facts("bogus")


def test_mjwarp_actuator_capabilities_are_synthetic() -> None:
    """Actuator facts/state are CPU-mirror math, labelled synthetic."""
    caps = MJWARP_QUERY_CAPABILITIES

    assert caps.supports("actuators.facts") == "synthetic"
    assert caps.supports("actuators.state") == "synthetic"
