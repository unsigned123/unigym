"""ResetPrimitives robot writes: root/joint state on the active reset env."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unigym.core import quat as quat_math
from unigym.initializers import INITIALIZER_REGISTRY, InitializerBase, register
from unigym.props import build_prop_manager

ARTICULATION = "tiny"
_INIT_TYPE = "test_robot_state_write"
_QUAT_90_X = [0.7071067811865476, 0.7071067811865476, 0.0, 0.0]


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


def _params(*, root=None, joints=None, raise_after_write: bool = False) -> dict:
    params: dict = {}
    if root is not None:
        params["root"] = root
    if joints is not None:
        params["joints"] = joints
    if raise_after_write:
        params["raise_after_write"] = True
    return {"initializer": {"type": _INIT_TYPE, "params": params}}


def _root_state(env):
    art = env.articulations[ARTICULATION]
    assert art.free_qpos_adr is not None
    assert art.free_dof_adr is not None
    return art.free_qpos_adr, art.free_dof_adr


def _prime(env, *, root=None, joints=None, raise_after_write: bool = False) -> None:
    """Install the next reset's write plan on the constructed initializer."""
    env._initializer.params.clear()
    env._initializer.params.update(
        _params(root=root, joints=joints, raise_after_write=raise_after_write)["initializer"][
            "params"
        ]
    )


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
        backend="mujoco",
    )


def test_root_write_hits_the_active_env_only(tiny_env_factory, _robot_initializer):
    """A reset of env 1 writes env 1's root; env 0 keeps its spawn state."""
    env = tiny_env_factory(
        num_envs=2,
        params=_params(
            root={
                "name": ARTICULATION,
                "pos_w": [0.3, -0.4, 1.1],
                "quat_wxyz": _QUAT_90_X,
                "lin_vel_w": [0.1, -0.2, 0.3],
                "ang_vel_w": [0.0, 0.0, 0.4],
            }
        ),
    )
    qadr, vadr = _root_state(env)
    spawn_qpos = env.mj_datas[0].qpos[qadr : qadr + 7].copy()
    spawn_qvel = env.mj_datas[0].qvel[vadr : vadr + 6].copy()

    env.reset_envs(torch.tensor([1], dtype=torch.int64))

    data = env.mj_datas[1]
    assert data.qpos[qadr : qadr + 3].tolist() == pytest.approx([0.3, -0.4, 1.1], abs=1e-9)
    assert data.qpos[qadr + 3 : qadr + 7].tolist() == pytest.approx(_QUAT_90_X, abs=1e-9)
    assert data.qvel[vadr : vadr + 3].tolist() == pytest.approx([0.1, -0.2, 0.3], abs=1e-9)
    expected_ang = quat_math.to_rotmat(np.asarray(_QUAT_90_X)).T @ np.array([0.0, 0.0, 0.4])
    assert data.qvel[vadr + 3 : vadr + 6].tolist() == pytest.approx(expected_ang.tolist(), abs=1e-9)

    untouched = env.mj_datas[0]
    assert untouched.qpos[qadr : qadr + 7].tolist() == pytest.approx(spawn_qpos.tolist())
    assert untouched.qvel[vadr : vadr + 6].tolist() == pytest.approx(spawn_qvel.tolist())
    assert env._active_reset_env_id is None


def test_root_write_normalizes_the_quaternion(tiny_env_factory, _robot_initializer):
    """A non-unit wxyz quaternion is normalized by the state-write face."""
    env = tiny_env_factory(
        params=_params(
            root={
                "name": ARTICULATION,
                "pos_w": [0.0, 0.0, 0.9],
                "quat_wxyz": [2.0, 0.0, 0.0, 0.0],
            }
        ),
    )
    qadr, _ = _root_state(env)

    env.reset_envs(torch.tensor([0], dtype=torch.int64))

    data = env.mj_datas[0]
    assert data.qpos[qadr + 3 : qadr + 7].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert data.qvel.tolist() == pytest.approx(np.zeros_like(data.qvel).tolist())


def test_root_write_on_a_free_rigid_prop(tiny_env_factory, _robot_initializer):
    """The same root write addresses a free prop by name."""
    env = tiny_env_factory(
        prop_manager=_prop_manager(),
        params=_params(
            root={
                "name": "crate",
                "pos_w": [0.4, 0.1, 0.5],
                "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                "lin_vel_w": [0.1, 0.0, 0.0],
            }
        ),
    )
    _body, qadr, vadr = env.prop_backend.free_addresses("crate")

    env.reset_envs(torch.tensor([0], dtype=torch.int64))

    data = env.mj_datas[0]
    assert data.qpos[qadr : qadr + 3].tolist() == pytest.approx([0.4, 0.1, 0.5], abs=1e-9)
    assert data.qpos[qadr + 3 : qadr + 7].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert data.qvel[vadr : vadr + 3].tolist() == pytest.approx([0.1, 0.0, 0.0], abs=1e-9)


def test_fixed_root_fails_with_the_state_write_error(
    tiny_env_factory, tiny_mjcf, tmp_path, _robot_initializer
):
    """A welded base root has no freejoint: the honest state-write error wins."""
    fixed = tmp_path / "tiny_arm_fixed.xml"
    fixed.write_text(
        tiny_mjcf.read_text(encoding="utf-8").replace('<freejoint name="root"/>', ""),
        encoding="utf-8",
    )
    env = tiny_env_factory(mjcf=fixed, params=_params())
    _prime(
        env,
        root={
            "name": ARTICULATION,
            "pos_w": [0.0, 0.0, 0.9],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    )

    with pytest.raises(ValueError, match="has a fixed root"):
        env.reset_envs(torch.tensor([0], dtype=torch.int64))

    assert env._active_reset_env_id is None


def test_joint_write_resolves_a_subset_and_keeps_velocity_when_none(
    tiny_env_factory, _robot_initializer
):
    """Only the named joint row moves; ``velocity=None`` leaves qvel alone."""
    env = tiny_env_factory(
        params=_params(
            joints={"articulation": ARTICULATION, "joint_names": ["elbow"], "position": [0.25]}
        )
    )
    art = env.articulations[ARTICULATION]
    elbow = list(art.joint_names).index("elbow")
    shoulder = list(art.joint_names).index("shoulder")
    elbow_qadr, elbow_dadr = art.jadr_art[elbow], art.jdr_art[elbow]
    shoulder_qadr = art.jadr_art[shoulder]

    env.reset_envs(torch.tensor([0], dtype=torch.int64))

    data = env.mj_datas[0]
    assert data.qpos[elbow_qadr] == pytest.approx(0.25)
    assert data.qpos[shoulder_qadr] == pytest.approx(0.0)
    assert data.qvel[elbow_dadr] == pytest.approx(0.0)


def test_joint_write_supplied_velocity_and_multiple_names(tiny_env_factory, _robot_initializer):
    """Explicit velocities land on every selected joint row."""
    env = tiny_env_factory(
        params=_params(
            joints={
                "articulation": ARTICULATION,
                "joint_names": ["shoulder", "elbow"],
                "position": [0.1, -0.2],
                "velocity": [0.5, -0.6],
            }
        )
    )
    art = env.articulations[ARTICULATION]
    names = list(art.joint_names)
    qadr = [art.jadr_art[names.index(n)] for n in ("shoulder", "elbow")]
    dadr = [art.jdr_art[names.index(n)] for n in ("shoulder", "elbow")]

    env.reset_envs(torch.tensor([0], dtype=torch.int64))

    data = env.mj_datas[0]
    assert [data.qpos[a] for a in qadr] == pytest.approx([0.1, -0.2])
    assert [data.qvel[a] for a in dadr] == pytest.approx([0.5, -0.6])


@pytest.mark.parametrize(
    ("joints", "match"),
    [
        (
            {"articulation": ARTICULATION, "joint_names": ["wrist"], "position": [0.0]},
            r"\['wrist'\] are not asset joints",
        ),
        (
            {"articulation": ARTICULATION, "joint_names": ["elbow", "elbow"], "position": [0, 0]},
            "duplicate names",
        ),
        (
            {"articulation": ARTICULATION, "joint_names": [], "position": []},
            "must be non-empty",
        ),
        (
            {"articulation": ARTICULATION, "joint_names": ["elbow"], "position": [0.0, 0.1]},
            r"position must have shape \(1,\)",
        ),
        (
            {
                "articulation": ARTICULATION,
                "joint_names": ["elbow"],
                "position": [0.0],
                "velocity": [0.0, 0.1],
            },
            r"velocity must have shape \(1,\)",
        ),
        (
            {"articulation": "nope", "joint_names": ["elbow"], "position": [0.0]},
            "unknown articulation",
        ),
    ],
)
def test_joint_write_validation_fails_in_apply(tiny_env_factory, _robot_initializer, joints, match):
    """Name/shape errors fail the reset with a teaching message."""
    env = tiny_env_factory(params=_params())
    _prime(env, joints=joints)

    with pytest.raises(ValueError, match=match):
        env.reset_envs(torch.tensor([0], dtype=torch.int64))

    assert env._active_reset_env_id is None


def test_joint_names_must_not_be_a_bare_string(tiny_env_factory, _robot_initializer):
    """A bare string is a teaching error, not a one-joint shorthand."""
    env = tiny_env_factory(params=_params())
    _prime(env, joints={"articulation": ARTICULATION, "joint_names": "elbow", "position": [0.0]})

    with pytest.raises(TypeError, match="sequence of joint names"):
        env.reset_envs(torch.tensor([0], dtype=torch.int64))


def test_bad_root_shape_fails_with_the_primitive_name(tiny_env_factory, _robot_initializer):
    """Root validation errors name the primitive and the field."""
    env = tiny_env_factory(params=_params())
    _prime(
        env,
        root={"name": ARTICULATION, "pos_w": [0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]},
    )

    with pytest.raises(ValueError, match=r"ResetPrimitives\.set_root_state: pos_w must have shape"):
        env.reset_envs(torch.tensor([0], dtype=torch.int64))


def test_exception_path_restores_the_active_reset_env(tiny_env_factory, _robot_initializer):
    """A raising apply still restores the neutral active id and data alias."""
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


def test_robot_writes_outside_a_reset_fail(tiny_env_factory, _robot_initializer):
    """The primitives are inert without an active reset env."""
    env = tiny_env_factory(params=_params())
    primitives = env._initializer.primitives

    with pytest.raises(RuntimeError, match="no active reset env"):
        primitives.set_root_state(ARTICULATION, [0.0, 0.0, 0.9], [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="no active reset env"):
        primitives.set_joint_state(ARTICULATION, ["elbow"], [0.0])
