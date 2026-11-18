"""MJWarp ResetPrimitives robot writes: mirror + device readback."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unigym.initializers import INITIALIZER_REGISTRY, InitializerBase, register
from unigym.runtime.actions import NamedAction

ARTICULATION = "tiny"
_INIT_TYPE = "test_robot_state_write_warp"

pytestmark = pytest.mark.sim


class _RobotStateInitializer(InitializerBase):
    """A test-only pack face that exercises the robot write primitives."""

    name = _INIT_TYPE

    def validate(self) -> None:
        self.normalized = dict(self.params)

    def resolve(self) -> None:
        pass

    def apply(self) -> dict:
        params = dict(self.params)
        if "root" in params:
            self.primitives.set_root_state(**params["root"])
        if "joints" in params:
            self.primitives.set_joint_state(**params["joints"])
        if params.get("raise_after_write"):
            raise ValueError("initializer apply blew up")
        return {}


@pytest.fixture
def _robot_initializer():
    registered = _INIT_TYPE in INITIALIZER_REGISTRY
    if not registered:
        register(_RobotStateInitializer)
    try:
        yield
    finally:
        if not registered:
            INITIALIZER_REGISTRY.pop(_INIT_TYPE, None)


def _params() -> dict:
    return {"initializer": {"type": _INIT_TYPE, "params": {}}}


def _prime(env, *, root=None, joints=None, raise_after_write: bool = False) -> None:
    """Install the next reset's write plan on the constructed initializer."""
    plan = {}
    if root is not None:
        plan["root"] = root
    if joints is not None:
        plan["joints"] = joints
    if raise_after_write:
        plan["raise_after_write"] = True
    env._initializer.params.clear()
    env._initializer.params.update(plan)


def _root(env):
    art = env.articulations[ARTICULATION]
    assert art.free_qpos_adr is not None
    assert art.free_dof_adr is not None
    return art.free_qpos_adr, art.free_dof_adr


def _device_qpos(env, e: int) -> np.ndarray:
    return np.asarray(env._wdata.qpos.numpy()[e], dtype=np.float64)


def _device_qvel(env, e: int) -> np.ndarray:
    return np.asarray(env._wdata.qvel.numpy()[e], dtype=np.float64)


def _action(env):
    control = env.controls[(ARTICULATION, ARTICULATION)]
    names = tuple(control.joint_names)
    values = torch.zeros(env.num_envs, len(names), dtype=torch.float32)
    return (NamedAction(control.articulation, values, names, control.target),)


def test_root_write_is_device_visible_after_reset_and_step(tiny_env_factory, _robot_initializer):
    """A reset root write lands on the mirror and the warp world; a step keeps it."""
    env = tiny_env_factory(params=_params())
    _prime(
        env,
        root={
            "name": ARTICULATION,
            "pos_w": [0.4, -0.3, 1.1],
            "quat_wxyz": [0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
            "lin_vel_w": [0.1, 0.0, 0.0],
        },
    )
    qadr, vadr = _root(env)

    env.reset_envs(torch.tensor([0], dtype=torch.int64))

    mirror = env.mj_datas[0]
    device = _device_qpos(env, 0)
    assert mirror.qpos[qadr : qadr + 3].tolist() == pytest.approx([0.4, -0.3, 1.1], abs=1e-9)
    assert device[qadr : qadr + 3].tolist() == pytest.approx([0.4, -0.3, 1.1], abs=1e-5)
    assert device[qadr + 3 : qadr + 7].tolist() == pytest.approx(
        [0.7071067811865476, 0.7071067811865476, 0.0, 0.0], abs=1e-6
    )
    assert _device_qvel(env, 0)[vadr : vadr + 3].tolist() == pytest.approx(
        [0.1, 0.0, 0.0], abs=1e-6
    )

    env.step(_action(env))

    mirror = env.mj_datas[0]
    device = _device_qpos(env, 0)
    assert mirror.qpos[qadr : qadr + 3].tolist() == pytest.approx([0.4, -0.3, 1.1], abs=0.02)
    assert device[qadr : qadr + 3].tolist() == pytest.approx(
        mirror.qpos[qadr : qadr + 3].tolist(), abs=1e-5
    )


def test_root_write_only_touches_the_active_env(tiny_env_factory, _robot_initializer):
    """A reset of env 1 writes env 1's root; env 0 stays at spawn."""
    env = tiny_env_factory(num_envs=2, params=_params())
    _prime(
        env,
        root={
            "name": ARTICULATION,
            "pos_w": [0.3, -0.4, 1.2],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    )
    qadr, _ = _root(env)
    spawn_mirror = env.mj_datas[0].qpos[qadr : qadr + 3].copy()
    spawn_device = _device_qpos(env, 0)[qadr : qadr + 3].copy()

    env.reset_envs(torch.tensor([1], dtype=torch.int64))

    assert _device_qpos(env, 1)[qadr : qadr + 3].tolist() == pytest.approx(
        [0.3, -0.4, 1.2], abs=1e-5
    )
    assert env.mj_datas[1].qpos[qadr : qadr + 3].tolist() == pytest.approx(
        [0.3, -0.4, 1.2], abs=1e-9
    )
    assert env.mj_datas[0].qpos[qadr : qadr + 3].tolist() == pytest.approx(spawn_mirror.tolist())
    assert _device_qpos(env, 0)[qadr : qadr + 3].tolist() == pytest.approx(spawn_device.tolist())
    assert env._active_reset_env_id is None


def test_joint_write_subset_reaches_mirror_and_device(tiny_env_factory, _robot_initializer):
    """The named joint row lands on the mirror and the warp world."""
    env = tiny_env_factory(params=_params())
    _prime(env, joints={"articulation": ARTICULATION, "joint_names": ["elbow"], "position": [0.35]})
    art = env.articulations[ARTICULATION]
    elbow = list(art.joint_names).index("elbow")
    shoulder = list(art.joint_names).index("shoulder")
    qadr, dadr = art.jadr_art[elbow], art.jdr_art[elbow]

    env.reset_envs(torch.tensor([0], dtype=torch.int64))

    mirror = env.mj_datas[0]
    device = _device_qpos(env, 0)
    assert mirror.qpos[qadr] == pytest.approx(0.35)
    assert device[qadr] == pytest.approx(0.35, abs=1e-6)
    assert mirror.qpos[art.jadr_art[shoulder]] == pytest.approx(0.0)
    assert _device_qvel(env, 0)[dadr] == pytest.approx(0.0, abs=1e-6)


def test_exception_path_restores_the_active_reset_env(tiny_env_factory, _robot_initializer):
    """A raising apply still restores the neutral active id on the warp backend."""
    env = tiny_env_factory(params=_params())
    _prime(
        env,
        joints={"articulation": ARTICULATION, "joint_names": ["elbow"], "position": [0.1]},
        raise_after_write=True,
    )
    data0 = env.mj_datas[0]

    with pytest.raises(ValueError, match="initializer apply blew up"):
        env.reset_envs(torch.tensor([0], dtype=torch.int64))

    assert env._active_reset_env_id is None
    assert env._active_data is data0
