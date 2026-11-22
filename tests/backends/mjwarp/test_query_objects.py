"""mjWarp runtime-query object/root/COM state parity over the CPU mirror."""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from unigym.backends.mjwarp.query import MJWARP_QUERY_CAPABILITIES, MjwarpQuery
from unigym.core import quat as quat_math
from unigym.props import build_prop_manager
from unigym.query import ComRef, ObjectRef

_QUAT_90_Y = np.array([0.7071067811865476, 0.0, 0.7071067811865476, 0.0])


def _query(env) -> MjwarpQuery:
    return MjwarpQuery(env)


def _prop_manager(num_envs: int = 1):
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
        num_envs=num_envs,
        env_spacing=2.0,
        device="cpu",
        backend="mjwarp",
    )


def test_object_state_capabilities_are_native_mirror_reads() -> None:
    """P2-A state reads keep the existing native CPU-mirror state convention."""
    caps = MJWARP_QUERY_CAPABILITIES

    assert caps.supports("state.root_state") == "native"
    assert caps.supports("state.com_state") == "native"
    assert caps.supports("state.object_state") == "native"
    assert caps.supports("state.object_com_state") == "native"
    assert caps.supports("jacobian.object") == "unsupported"


def test_object_state_and_com_parity_on_the_cpu_mirror(tiny_env_factory) -> None:
    """Object state/twist and COM velocity match the mirror qpos/qvel and jacBodyCom."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    pose = [0.4, 0.1, 0.5]
    ang = [0.0, 0.0, 0.4]
    env.set_root_state(
        None,
        name="crate",
        pos_w=torch.tensor([pose]),
        quat_wxyz=torch.from_numpy(_QUAT_90_Y).unsqueeze(0),
        lin_vel_w=torch.tensor([[0.1, 0.2, 0.3]]),
        ang_vel_w=torch.tensor([ang]),
    )
    env.settle(None)
    query = _query(env)
    data = env.mj_datas[0]
    body_id, qadr, vadr = env.prop_backend.free_addresses("crate")

    state = query.state.object_state(ObjectRef("crate"))
    com = query.state.object_com_state(ObjectRef("crate"))

    np.testing.assert_allclose(state.pos_w[0].numpy(), data.qpos[qadr : qadr + 3])
    np.testing.assert_allclose(state.quat_wxyz[0].numpy(), data.qpos[qadr + 3 : qadr + 7])
    mirror_quat = data.qpos[qadr + 3 : qadr + 7]
    expected_ang = quat_math.to_rotmat(mirror_quat) @ data.qvel[vadr + 3 : vadr + 6]
    np.testing.assert_allclose(state.ang_vel_w[0].numpy(), expected_ang, atol=1e-12)
    np.testing.assert_allclose(state.ang_vel_w[0].numpy(), ang, atol=1e-6)

    np.testing.assert_allclose(com.pos_w[0].numpy(), data.xipos[body_id])
    jacp = np.zeros((3, env.mj_model.nv), dtype=np.float64)
    jacr = np.zeros((3, env.mj_model.nv), dtype=np.float64)
    mujoco.mj_jacBodyCom(env.mj_model, data, jacp, jacr, body_id)
    np.testing.assert_allclose(com.lin_vel_w[0].numpy(), jacp @ data.qvel, atol=1e-12)


def test_root_and_com_state_read_the_cpu_mirror(tiny_env_factory) -> None:
    """Free-base root and articulation COM twist are mirror reads, not device reads."""
    env = tiny_env_factory()
    query = _query(env)
    data = env.mj_datas[0]
    art = env.articulations["tiny"]
    qadr = int(art.free_qpos_adr)
    dadr = int(art.free_dof_adr)
    data.qpos[qadr : qadr + 3] = [0.2, -0.1, 0.9]
    data.qpos[qadr + 3 : qadr + 7] = _QUAT_90_Y
    data.qvel[dadr : dadr + 3] = [0.1, 0.2, 0.3]
    data.qvel[dadr + 3 : dadr + 6] = [0.4, -0.5, 0.6]
    mujoco.mj_forward(env.mj_model, data)

    root = query.state.root_state("tiny")
    body_id = env.mj_model.body("base").id
    body = query.state.com_state(ComRef("tiny", body="base"))

    np.testing.assert_allclose(root.pos_w[0].numpy(), data.qpos[qadr : qadr + 3])
    expected = quat_math.to_rotmat(_QUAT_90_Y) @ data.qvel[dadr + 3 : dadr + 6]
    np.testing.assert_allclose(root.ang_vel_w[0].numpy(), expected)
    np.testing.assert_allclose(body.pos_w[0].numpy(), data.xipos[body_id])
    jacp = np.zeros((3, env.mj_model.nv), dtype=np.float64)
    jacr = np.zeros((3, env.mj_model.nv), dtype=np.float64)
    mujoco.mj_jacBodyCom(env.mj_model, data, jacp, jacr, body_id)
    np.testing.assert_allclose(body.lin_vel_w[0].numpy(), jacp @ data.qvel)


def test_mirror_object_reads_are_detached_and_do_not_mutate_live_data(tiny_env_factory) -> None:
    """Object reads detach and leave the mirror qpos/qvel/xipos unchanged."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    query = _query(env)
    data = env.mj_datas[0]
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    xipos = data.xipos.copy()

    state = query.state.object_state(ObjectRef("crate"))
    state.pos_w[0, 0] = 123.0
    query.state.object_com_state(ObjectRef("crate"))

    np.testing.assert_array_equal(data.qpos, qpos)
    np.testing.assert_array_equal(data.qvel, qvel)
    np.testing.assert_array_equal(data.xipos, xipos)
