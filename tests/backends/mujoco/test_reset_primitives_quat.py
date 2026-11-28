"""MuJoCo reset chain: a non-yaw quaternion survives reset + one step."""

from __future__ import annotations

from unigym.backends.compose import backend_runtime

import numpy as np
import pytest
import torch

from unigym.core import quat as quat_math
from unigym.initializers import INITIALIZER_REGISTRY, InitializerBase, register
from unigym.props import build_prop_manager
from unigym.runtime.actions import NamedAction

_INIT_TYPE = "test_quat_write_entry"
_QUAT = [0.7071067811865476, 0.0, 0.7071067811865476, 0.0]  # +90 deg about y


class _QuatWriteInitializer(InitializerBase):
    """A test-only pack face that writes a full quaternion."""

    name = _INIT_TYPE

    def validate(self) -> None:
        self.normalized = dict(self.params)

    def resolve(self) -> None:
        pass

    @property
    def places_props(self) -> bool:
        return True

    def apply(self) -> dict:
        params = self.params
        self.primitives.write_freejoint(
            params["prop"],
            params["pos"],
            quat_wxyz=params["quat_wxyz"],
            lin_vel_w=params.get("lin_vel_w"),
            ang_vel_w=params.get("ang_vel_w"),
        )
        return {params["prop"]: {"pos": list(params["pos"])}}


@pytest.fixture
def _quat_initializer():
    registered = _INIT_TYPE in INITIALIZER_REGISTRY
    if not registered:
        register(_QuatWriteInitializer)
    try:
        yield
    finally:
        if not registered:
            INITIALIZER_REGISTRY.pop(_INIT_TYPE, None)


def _prop_manager():
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
        backend="mujoco",
        prop_backend=backend_runtime("mujoco").bindings.prop_backend,
        mujoco_prop_backend=backend_runtime("mujoco").bindings.prop_backend,
    )


def _action(env):
    control = env.controls[("tiny", "tiny")]
    values = torch.zeros(env.num_envs, len(control.joint_names), dtype=torch.float32)
    return (NamedAction(control.articulation, values, tuple(control.joint_names), control.target),)


def test_full_quat_survives_the_mujoco_reset_chain_and_one_step(
    tiny_env_factory, _quat_initializer
) -> None:
    lin = [0.1, -0.2, 0.3]
    ang = [0.0, 0.0, 0.4]
    env = tiny_env_factory(
        prop_manager=_prop_manager(),
        params={
            "initializer": {
                "type": _INIT_TYPE,
                "params": {
                    "prop": "crate",
                    "pos": [0.4, 0.1, 0.5],
                    "quat_wxyz": _QUAT,
                    "lin_vel_w": lin,
                    "ang_vel_w": ang,
                },
            }
        },
    )
    _body, qadr, vadr = env.prop_backend.free_addresses("crate")

    env.reset_envs(torch.tensor([0], dtype=torch.int64))

    data = env.mj_datas[0]
    assert np.allclose(data.qpos[qadr : qadr + 3], [0.4, 0.1, 0.5], atol=1e-12)
    assert np.allclose(data.qpos[qadr + 3 : qadr + 7], _QUAT, atol=1e-12)
    assert np.allclose(data.qvel[vadr : vadr + 3], lin, atol=1e-12)
    expected_body = quat_math.to_rotmat(np.asarray(_QUAT)).T @ np.asarray(ang)
    assert np.allclose(data.qvel[vadr + 3 : vadr + 6], expected_body, atol=1e-12)

    env.step(_action(env))

    data = env.mj_datas[0]
    quat = data.qpos[qadr + 3 : qadr + 7]
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-9)
    assert abs(float(np.dot(quat, _QUAT))) > 0.999
