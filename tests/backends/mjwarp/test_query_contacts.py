"""mjWarp contact-manifold query behavior over the CPU MjData mirror."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from unigym.backends.mjwarp.query import MjwarpQuery
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
        backend="mjwarp",
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


def test_manifold_reads_the_cpu_mirror_contact(tiny_env_factory, tmp_path) -> None:
    """Active crate/world contacts come from the CPU contact mirror."""
    manager = _manager()
    env = _env(tiny_env_factory, tmp_path, manager)
    _step(env)

    manifold = MjwarpQuery(env).contacts.manifold("crate_floor")

    assert manifold.group == "crate_floor"
    assert manifold.env_ids.tolist() == [0]
    assert manifold.points
    point = manifold.points[0]
    assert {point.body1, point.body2} == {"Prop_crate", "world"}
    assert {point.geom1, point.geom2} == {"crate_geom", "floor"}
    assert point.env_id == 0
    assert point.frame_w is not None
    assert tuple(point.frame_w.shape) == (3, 3)
    assert point.normal_force_n > 0.0


def test_manifold_force_matches_the_mirror_contact_list(tiny_env_factory, tmp_path) -> None:
    """The mirror contact force and frame are the same CPU MuJoCo math."""
    manager = _manager()
    env = _env(tiny_env_factory, tmp_path, manager)
    _step(env)
    point = MjwarpQuery(env).contacts.manifold("crate_floor").points[0]

    data = env.mj_datas[0]
    contact_id = point.contact_id
    expected = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(env.mj_model, data, contact_id, expected)
    frame = np.asarray(data.contact.frame)[contact_id].reshape(3, 3)
    assert point.frame_w is not None
    assert point.normal_w is not None

    np.testing.assert_allclose(point.pos_w.numpy(), data.contact.pos[contact_id])
    np.testing.assert_allclose(point.frame_w.numpy(), frame)
    np.testing.assert_allclose(point.normal_w.numpy(), frame[0])
    assert point.normal_force_n == pytest.approx(float(expected[0]))


def test_manifold_never_scans_contacts_outside_the_group(tiny_env_factory, tmp_path) -> None:
    """A group with no active pair stays empty while another group has points."""
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
    query = MjwarpQuery(env)

    assert query.contacts.manifold("crate_floor").points
    assert query.contacts.manifold("tip_floor").points == ()


def test_manifold_result_is_a_detached_mirror_copy(tiny_env_factory, tmp_path) -> None:
    """Mutating a returned point never mutates the live CPU mirror."""
    env = _env(tiny_env_factory, tmp_path, _manager())
    _step(env)
    point = MjwarpQuery(env).contacts.manifold("crate_floor").points[0]
    contact_id = point.contact_id
    original = float(env.mj_datas[0].contact.pos[contact_id][0])

    point.pos_w[0] = 123.0

    assert float(env.mj_datas[0].contact.pos[contact_id][0]) == original


def test_unknown_group_names_the_declared_groups(tiny_env_factory, tmp_path) -> None:
    """An unknown group is a reference error, never an empty manifold."""
    env = _env(tiny_env_factory, tmp_path, _manager())
    _step(env)

    with pytest.raises(QueryReferenceError, match=r"ghost.*crate_floor"):
        MjwarpQuery(env).contacts.manifold("ghost")


def test_manifold_after_query_close_is_stale(tiny_env_factory, tmp_path) -> None:
    """A closed query handle rejects contact reads."""
    env = _env(tiny_env_factory, tmp_path, _manager())
    _step(env)
    query = MjwarpQuery(env)
    query.close()

    with pytest.raises(QueryStaleError, match="stale"):
        query.contacts.manifold("crate_floor")


def test_contact_jacobian_and_wrench_are_synthetic(tiny_env_factory, tmp_path) -> None:
    """The mjWarp mirror declares the contact rows/wrench as synthetic CPU math."""
    env = _env(tiny_env_factory, tmp_path, _manager())
    query = MjwarpQuery(env)

    assert query.capabilities.supports("contacts.manifold") == "synthetic"
    assert query.capabilities.supports("contacts.jacobian") == "synthetic"
    assert query.capabilities.supports("contacts.wrench_to_joint") == "synthetic"
    query.require("contacts.jacobian", "contacts.wrench_to_joint")
