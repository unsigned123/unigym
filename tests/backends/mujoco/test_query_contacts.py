"""MuJoCo contact-manifold query behavior over a declared contact group."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from unigym.backends.mujoco.query import MujocoQuery
from unigym.contacts.manager import build_contact_manager
from unigym.query import QueryReferenceError, QueryStaleError
from unigym.runtime.actions import NamedAction

PROP_CRATE = """    <body name="Prop_crate" pos="0.3 0 0.05">
      <freejoint/>
      <geom name="crate_geom" type="box" size="0.05 0.05 0.05" mass="0.5"/>
    </body>"""


def _mjcf(tmp_path: Path) -> Path:
    from tests.backends.mujoco.conftest import tiny_mjcf_source

    source = tiny_mjcf_source().replace("  </worldbody>", PROP_CRATE + "\n  </worldbody>", 1)
    path = tmp_path / "query_contacts.xml"
    path.write_text(source, encoding="utf-8")
    return path


def _manager(groups=None):
    manager = build_contact_manager(
        groups
        or [
            {
                "name": "crate_floor",
                "sensor_refs": [{"owner": "object", "name": "crate"}],
                "filter_refs": [{"owner": "environment", "body": "world"}],
                "frame": "world",
            }
        ],
        backend="mujoco",
        articulation_names=("tiny",),
        prop_names=["crate"],
    )
    assert manager is not None
    return manager


def _step(env, steps: int = 5) -> None:
    action = NamedAction.zeros(env.num_envs, "tiny", ("elbow", "shoulder"), "position")
    for _ in range(steps):
        env.step((action,))


def _env(tiny_env_factory, tmp_path, manager, num_envs: int = 1):
    return tiny_env_factory(mjcf=_mjcf(tmp_path), contact_manager=manager, num_envs=num_envs)


def test_manifold_reads_the_declared_group_contact(tiny_env_factory, tmp_path) -> None:
    """Active crate/world contacts are returned with neutral body/geom names."""
    manager = _manager()
    env = _env(tiny_env_factory, tmp_path, manager)
    _step(env)

    manifold = MujocoQuery(env).contacts.manifold("crate_floor")

    assert manifold.group == "crate_floor"
    assert manifold.env_ids.tolist() == [0]
    assert manifold.points
    point = manifold.points[0]
    assert {point.body1, point.body2} == {"Prop_crate", "world"}
    assert {point.geom1, point.geom2} == {"crate_geom", "floor"}
    assert point.env_id == 0
    assert point.normal_w is not None
    assert point.frame_w is not None
    assert point.friction_force_w is not None
    assert tuple(point.pos_w.shape) == (3,)
    assert tuple(point.normal_w.shape) == (3,)
    assert tuple(point.frame_w.shape) == (3, 3)
    assert tuple(point.friction_force_w.shape) == (3,)
    assert point.normal_force_n > 0.0
    assert point.separation_m <= 1e-3


def test_manifold_force_frame_and_sign_match_the_contact_list(tiny_env_factory, tmp_path) -> None:
    """Force is the geom2-body force in the contact frame, rotated to world."""
    manager = _manager()
    env = _env(tiny_env_factory, tmp_path, manager)
    _step(env)
    point = MujocoQuery(env).contacts.manifold("crate_floor").points[0]

    data = env.mj_data
    contact_id = point.contact_id
    expected = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(env.mj_model, data, contact_id, expected)
    frame = np.asarray(data.contact.frame)[contact_id].reshape(3, 3)
    assert point.frame_w is not None
    assert point.normal_w is not None
    assert point.friction_force_w is not None

    np.testing.assert_allclose(point.pos_w.numpy(), data.contact.pos[contact_id])
    np.testing.assert_allclose(point.frame_w.numpy(), frame)
    np.testing.assert_allclose(point.normal_w.numpy(), frame[0])
    assert point.normal_force_n == pytest.approx(float(expected[0]))
    assert point.separation_m == pytest.approx(float(data.contact.dist[contact_id]))
    tangential = np.array([0.0, expected[1], expected[2]], dtype=np.float64)
    np.testing.assert_allclose(point.friction_force_w.numpy(), frame.T @ tangential)


def test_manifold_never_scans_contacts_outside_the_group(tiny_env_factory, tmp_path) -> None:
    """A contact group with no active pair returns an empty manifold."""
    manager = _manager(
        [
            {
                "name": "crate_floor",
                "sensor_refs": [{"owner": "object", "name": "crate"}],
                "filter_refs": [{"owner": "environment", "body": "world"}],
                "frame": "world",
            },
            {
                "name": "tip_floor",
                "sensor_refs": [{"owner": "robot", "articulation": "tiny", "body": "tip"}],
                "filter_refs": [{"owner": "environment", "body": "world"}],
                "frame": "world",
            },
        ]
    )
    env = _env(tiny_env_factory, tmp_path, manager)
    _step(env)
    query = MujocoQuery(env)

    assert query.contacts.manifold("crate_floor").points
    empty = query.contacts.manifold("tip_floor")
    assert empty.points == ()
    assert empty.group == "tip_floor"


def test_manifold_selects_one_env(tiny_env_factory, tmp_path) -> None:
    """The env selection limits both the contact read and the output ids."""
    manager = _manager()
    env = _env(tiny_env_factory, tmp_path, manager, num_envs=2)
    _step(env)

    manifold = MujocoQuery(env).contacts.manifold("crate_floor", env=1)

    assert manifold.env_ids.tolist() == [1]
    assert manifold.points
    assert {point.env_id for point in manifold.points} == {1}


def test_unknown_group_names_the_declared_groups(tiny_env_factory, tmp_path) -> None:
    """An unknown group is a reference error, never an empty manifold."""
    env = _env(tiny_env_factory, tmp_path, _manager())
    _step(env)

    with pytest.raises(QueryReferenceError, match=r"ghost.*crate_floor"):
        MujocoQuery(env).contacts.manifold("ghost")


def test_manifold_after_query_close_is_stale(tiny_env_factory, tmp_path) -> None:
    """A closed query handle rejects contact reads."""
    env = _env(tiny_env_factory, tmp_path, _manager())
    _step(env)
    query = MujocoQuery(env)
    query.close()

    with pytest.raises(QueryStaleError, match="stale"):
        query.contacts.manifold("crate_floor")


def test_contact_jacobian_and_wrench_are_native(tiny_env_factory, tmp_path) -> None:
    """P1A-2 declares the contact constraint rows and wrench as native."""
    env = _env(tiny_env_factory, tmp_path, _manager())
    query = MujocoQuery(env)

    assert query.capabilities.supports("contacts.manifold") == "native"
    assert query.capabilities.supports("contacts.jacobian") == "native"
    assert query.capabilities.supports("contacts.wrench_to_joint") == "native"
    query.require("contacts.jacobian", "contacts.wrench_to_joint")


def test_manifold_result_is_a_detached_copy(tiny_env_factory, tmp_path) -> None:
    """Mutating a returned point never mutates the live MjData."""
    env = _env(tiny_env_factory, tmp_path, _manager())
    _step(env)
    point = MujocoQuery(env).contacts.manifold("crate_floor").points[0]
    contact_id = point.contact_id
    original = float(env.mj_data.contact.pos[contact_id][0])

    point.pos_w[0] = 123.0

    assert float(env.mj_data.contact.pos[contact_id][0]) == original
