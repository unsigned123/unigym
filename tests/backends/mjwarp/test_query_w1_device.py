"""mjWarp runtime-query surfaces on the real W1 CUDA warp device.

The query still reads the per-world CPU ``MjData`` mirror, but the env builds
its warp model/state on ``cuda:0``.  These tests prove the new model/state,
all-pair geometry and full contact-scan surfaces answer on that device path and
that a scratch copy never mutates the live mirror.
"""

from __future__ import annotations

import numpy as np
import pytest

from unigym.backends.compose import backend_runtime
from unigym.props import build_prop_manager as _library_build_prop_manager
from unigym.query import EntityRef, ContactManifold, DistanceScan, ObjectRef
from unigym.runtime.actions import NamedAction

CRATE_GEOM = "Prop_crate_geom0"
CRATE_JOINT = "Prop_crate_free"


def build_prop_manager():
    """Declare the free crate box settled onto the floor for this lane."""
    return _library_build_prop_manager(
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


def _settled_env(tiny_env_factory):
    """Build a one-env CUDA crate scene and settle the crate onto the floor."""
    env = tiny_env_factory(device="cuda:0", prop_manager=build_prop_manager())
    action = NamedAction.zeros(env.num_envs, "tiny", ("elbow", "shoulder"), "position")
    for _ in range(60):
        env.step((action,))
    return env


def _pair_distances(scan: DistanceScan) -> dict[tuple[str, str], float]:
    return {(pair.geom_a, pair.geom_b): float(pair.signed_distance_m[0]) for pair in scan.pairs}


def _contact_signature(manifold: ContactManifold) -> tuple:
    return tuple(
        (
            point.env_id,
            point.contact_id,
            point.body1,
            point.body2,
            point.geom1,
            point.geom2,
            tuple(float(value) for value in point.pos_w.numpy()),
        )
        for point in manifold.points
    )


def _assert_pair_distances_equal(
    actual: dict[tuple[str, str], float],
    expected: dict[tuple[str, str], float],
) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, abs=1e-9), key


@pytest.mark.gpu
def test_real_cuda_device_answers_the_new_query_surfaces(tiny_env_factory) -> None:
    """The real cuda:0 warp env answers model, state, all-pair and scan reads."""
    env = _settled_env(tiny_env_factory)
    query = env.query
    model = env.mj_model
    data = env.mj_datas[0]

    assert env.warp_device == "cuda:0"

    for name in ("base_geom", CRATE_GEOM):
        contact = query.model.geom_contact(name)
        geom_id = model.geom(name).id
        assert contact.name == name
        assert contact.friction == pytest.approx(
            tuple(float(value) for value in model.geom_friction[geom_id])
        )
        assert contact.condim == int(model.geom_condim[geom_id])

    body_id = model.body("base").id
    pose = query.state.body_pose("base")
    np.testing.assert_allclose(pose.pos_w[0].numpy(), data.xpos[body_id])
    np.testing.assert_allclose(pose.quat_wxyz[0].numpy(), data.xquat[body_id])

    crate = query.state.object_state(ObjectRef("crate"))
    qpos_adr = query.model.joint(CRATE_JOINT).qpos_adr
    assert qpos_adr is not None
    assert crate.env_ids.tolist() == [0]
    np.testing.assert_allclose(crate.pos_w[0].numpy(), data.qpos[qpos_adr : qpos_adr + 3])
    np.testing.assert_allclose(crate.quat_wxyz[0].numpy(), data.qpos[qpos_adr + 3 : qpos_adr + 7])

    scan = query.geometry.all_pairs(max_distance_m=2.0)
    assert scan.env_ids.tolist() == [0]
    pairs = _pair_distances(scan)
    crate_floor = ("floor", CRATE_GEOM)
    assert crate_floor in pairs
    direct = query.geometry.distance(
        EntityRef("geom", "floor"),
        EntityRef("geom", CRATE_GEOM),
        max_distance_m=2.0,
    )
    assert pairs[crate_floor] == pytest.approx(float(direct.signed_distance_m[0]))

    manifold = query.contacts.scan()
    assert manifold.group == "*"
    assert manifold.env_ids.tolist() == [0]
    assert manifold.points
    assert any(CRATE_GEOM in (point.geom1, point.geom2) for point in manifold.points)
    assert all(point.group == "*" for point in manifold.points)


@pytest.mark.gpu
def test_real_cuda_scratch_requires_forward_and_leaves_live_mirror_untouched(
    tiny_env_factory,
) -> None:
    """A scratch write needs an explicit forward and never reaches the live mirror."""
    env = _settled_env(tiny_env_factory)
    query = env.query

    live_scan = query.contacts.scan()
    live_signature = _contact_signature(live_scan)
    assert any(CRATE_GEOM in (point.geom1, point.geom2) for point in live_scan.points)

    live_state = query.state.object_state(ObjectRef("crate"))
    live_pos = live_state.pos_w[0].clone()
    live_quat = live_state.quat_wxyz[0].clone()
    live_pairs = _pair_distances(query.geometry.all_pairs(max_distance_m=2.0))
    crate_floor = ("floor", CRATE_GEOM)

    away = live_pos.clone()
    away[2] += 1.0

    with query.scratch(0) as scratch:
        assert _contact_signature(scratch.contacts().scan()) == live_signature
        _assert_pair_distances_equal(
            _pair_distances(scratch.geometry().all_pairs(max_distance_m=2.0)),
            live_pairs,
        )

        scratch.set_root_state("crate", pos_w=away, quat_wxyz=live_quat)
        assert _contact_signature(scratch.contacts().scan()) == live_signature
        _assert_pair_distances_equal(
            _pair_distances(scratch.geometry().all_pairs(max_distance_m=2.0)),
            live_pairs,
        )

        scratch.forward()
        moved_scan = scratch.contacts().scan()
        moved_pairs = _pair_distances(scratch.geometry().all_pairs(max_distance_m=2.0))

    assert not any(CRATE_GEOM in (point.geom1, point.geom2) for point in moved_scan.points)
    assert moved_pairs[crate_floor] == pytest.approx(live_pairs[crate_floor] + 1.0, abs=1e-6)

    after_state = query.state.object_state(ObjectRef("crate"))
    np.testing.assert_allclose(after_state.pos_w[0].numpy(), live_pos.numpy())
    np.testing.assert_allclose(after_state.quat_wxyz[0].numpy(), live_quat.numpy())
    assert _contact_signature(query.contacts.scan()) == live_signature
    _assert_pair_distances_equal(
        _pair_distances(query.geometry.all_pairs(max_distance_m=2.0)),
        live_pairs,
    )
