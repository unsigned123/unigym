"""mjWarp contact-manifold query behavior over the CPU MjData mirror."""

from __future__ import annotations

from unigym.backends.compose import backend_runtime

from pathlib import Path

import mujoco
import numpy as np
import pytest

from unigym.backends.mjwarp.query import MjwarpQuery
from unigym.contacts.manager import build_contact_manager
from unigym.props import build_prop_manager
from unigym.query import ObjectRef, QueryReferenceError, QueryStaleError
from unigym.runtime.actions import NamedAction

PROP_CRATE = """    <body name="Prop_crate" pos="0.3 0 0.05">
      <freejoint/>
      <geom name="crate_geom" type="box" size="0.05 0.05 0.05" mass="0.5"/>
    </body>"""

PROP_DECOY = """    <body name="Prop_decoy" pos="-0.3 0 0.05">
      <freejoint/>
      <geom name="decoy_geom" type="box" size="0.05 0.05 0.05" mass="0.5"/>
    </body>"""


def _mjcf(tmp_path: Path) -> Path:
    from tests.backends.mujoco.conftest import tiny_mjcf_source

    source = tiny_mjcf_source().replace("  </worldbody>", PROP_CRATE + "\n  </worldbody>", 1)
    path = tmp_path / "query_contacts.xml"
    path.write_text(source, encoding="utf-8")
    return path


def _mjcf_with_decoy(tmp_path: Path) -> Path:
    from tests.backends.mujoco.conftest import tiny_mjcf_source

    source = tiny_mjcf_source().replace(
        "  </worldbody>", PROP_CRATE + "\n" + PROP_DECOY + "\n  </worldbody>", 1
    )
    path = tmp_path / "query_contacts_scan.xml"
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
        contact_factory=backend_runtime("mjwarp").bindings.contact_backend,
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


def _prop_manager():
    """Declare a free crate prop so the scratch can move it through the public face."""
    return build_prop_manager(
        {
            "types": [
                {
                    "name": "crate",
                    "type": "box",
                    "size": [0.1, 0.1, 0.1],
                    "aabb": {
                        "center": [0.0, 0.0, 0.05],
                        "half_extents": [0.05, 0.05, 0.05],
                    },
                    "placement": {
                        "mode": "fixed",
                        "x_range": [0.5, 0.5],
                        "y_range": [0.0, 0.0],
                        "z_range": [0.3, 0.3],
                    },
                }
            ]
        },
        num_envs=1,
        env_spacing=2.0,
        device="cpu",
        backend="mjwarp",
        prop_backend=backend_runtime("mjwarp").bindings.prop_backend,
        mujoco_prop_backend=backend_runtime("mujoco").bindings.prop_backend,
    )


def _scratch_env(tiny_env_factory):
    """Build a one-env crate scene and settle the crate onto the floor."""
    env = tiny_env_factory(prop_manager=_prop_manager(), contact_manager=_manager())
    _step(env, 60)
    return env


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


def test_scratch_contacts_parity_on_the_cpu_mirror(tiny_env_factory) -> None:
    """Scratch contact math runs on the CPU mirror copy and leaves the live mirror untouched."""
    env = _scratch_env(tiny_env_factory)
    query = MjwarpQuery(env)
    live_ncon = int(env.mj_datas[0].ncon)
    live_contacts = np.asarray(env.mj_datas[0].contact.pos[:live_ncon]).copy()
    live_pose = query.state.object_state(ObjectRef("crate"))
    live_pos = live_pose.pos_w[0].clone()
    live_quat = live_pose.quat_wxyz[0].clone()
    away = live_pos.clone()
    away[2] += 1.0

    with query.scratch(0) as scratch:
        assert scratch.contacts().manifold("crate_floor").points

        scratch.set_root_state("crate", pos_w=away, quat_wxyz=live_quat)
        scratch.forward()
        assert scratch.contacts().manifold("crate_floor").points == ()
        assert int(env.mj_datas[0].ncon) == live_ncon
        np.testing.assert_allclose(
            np.asarray(env.mj_datas[0].contact.pos[:live_ncon]),
            live_contacts,
        )

        scratch.set_root_state("crate", pos_w=live_pos, quat_wxyz=live_quat)
        scratch.forward()
        assert scratch.contacts().manifold("crate_floor").points


def test_scan_reads_cpu_mirror_contacts_without_a_declared_group(
    tiny_env_factory, tmp_path
) -> None:
    """A scene with no contacts: section still scans its CPU mirror contacts."""
    env = tiny_env_factory(mjcf=_mjcf(tmp_path), contact_manager=None)
    _step(env)
    query = MjwarpQuery(env)

    manifold = query.contacts.scan()

    assert manifold.group == "*"
    assert manifold.env_ids.tolist() == [0]
    assert manifold.points
    assert all(point.group == "*" for point in manifold.points)
    bodies = {name for point in manifold.points for name in (point.body1, point.body2)}
    assert {"Prop_crate", "world"} <= bodies
    assert query.capabilities.supports("contacts.scan") == "synthetic"


def test_scan_returns_grouped_and_ungrouped_mirror_contacts(tiny_env_factory, tmp_path) -> None:
    """The mirror scan includes contacts outside the declared contact group."""
    env = tiny_env_factory(mjcf=_mjcf_with_decoy(tmp_path), contact_manager=_manager())
    _step(env)
    query = MjwarpQuery(env)

    grouped = query.contacts.manifold("crate_floor")
    scanned = query.contacts.scan()

    assert grouped.points
    assert not any("Prop_decoy" in (point.body1, point.body2) for point in grouped.points)
    assert any("Prop_decoy" in (point.body1, point.body2) for point in scanned.points)
    assert any("Prop_crate" in (point.body1, point.body2) for point in scanned.points)
    assert len(scanned.points) > len(grouped.points)
    assert all(point.group == "*" for point in scanned.points)


def test_scan_preserves_env_order_and_empty_selection(tiny_env_factory, tmp_path) -> None:
    """The mirror scan follows the requested env order and supports empty selection."""
    env = _env(tiny_env_factory, tmp_path, _manager(), num_envs=2)
    _step(env)
    query = MjwarpQuery(env)

    manifold = query.contacts.scan(env=[1, 0])
    order = {env_id: index for index, env_id in enumerate([1, 0])}
    point_order = [order[point.env_id] for point in manifold.points]

    assert manifold.env_ids.tolist() == [1, 0]
    assert manifold.points
    assert point_order == sorted(point_order)

    empty = query.contacts.scan(env=[])
    assert empty.group == "*"
    assert empty.env_ids.numel() == 0
    assert empty.points == ()


def test_scratch_scan_follows_writes_and_leaves_the_live_mirror_untouched(
    tiny_env_factory,
) -> None:
    """A scratch scan tracks scratch mirror contacts while the live mirror stays put."""
    env = _scratch_env(tiny_env_factory)
    query = MjwarpQuery(env)
    live_ncon = int(env.mj_datas[0].ncon)
    live_contacts = np.asarray(env.mj_datas[0].contact.pos[:live_ncon]).copy()
    live_pose = query.state.object_state(ObjectRef("crate"))
    live_pos = live_pose.pos_w[0].clone()
    live_quat = live_pose.quat_wxyz[0].clone()
    away = live_pos.clone()
    away[2] += 1.0

    with query.scratch(0) as scratch:
        before = scratch.contacts().scan()
        assert before.group == "*"
        assert before.points

        scratch.set_root_state("crate", pos_w=away, quat_wxyz=live_quat)
        scratch.forward()
        after = scratch.contacts().scan()
        assert not any("Prop_crate" in (point.body1, point.body2) for point in after.points)
        assert int(env.mj_datas[0].ncon) == live_ncon
        np.testing.assert_allclose(
            np.asarray(env.mj_datas[0].contact.pos[:live_ncon]),
            live_contacts,
        )

        scratch.set_root_state("crate", pos_w=live_pos, quat_wxyz=live_quat)
        scratch.forward()
        restored = scratch.contacts().scan()
        assert any("Prop_crate" in (point.body1, point.body2) for point in restored.points)
