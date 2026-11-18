"""Real MuJoCo state-write contract: world frames, wxyz, settle and reset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unigym.backends.mujoco.env import _one_dof_joint_id
from unigym.props import build_prop_manager

ARTICULATION = "tiny"


def _quat_90_x() -> torch.Tensor:
    """Unit quaternion for +90 degrees about world x (wxyz)."""
    return torch.tensor([[0.7071067811865476, 0.7071067811865476, 0.0, 0.0]])


def _root(env):
    art = env.articulations[ARTICULATION]
    assert art.free_qpos_adr is not None
    assert art.free_dof_adr is not None
    return art.free_qpos_adr, art.free_dof_adr


def test_set_root_state_normalizes_quat_and_writes_world_pose(tiny_env):
    """A non-unit wxyz quat is normalized; pose and linear qvel are world."""
    env = tiny_env
    qadr, vadr = _root(env)

    returned = env.set_root_state(
        None,
        name=ARTICULATION,
        pos_w=torch.tensor([[1.0, 2.0, 0.9]]),
        quat_wxyz=torch.tensor([[2.0, 0.0, 0.0, 0.0]]),
        lin_vel_w=torch.tensor([[0.1, -0.2, 0.3]]),
        ang_vel_w=torch.zeros(1, 3),
    )

    assert returned.dtype == torch.int64
    assert returned.tolist() == [0]
    data = env.mj_datas[0]
    assert data.qpos[qadr : qadr + 3].tolist() == pytest.approx([1.0, 2.0, 0.9])
    assert data.qpos[qadr + 3 : qadr + 7].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert data.qvel[vadr : vadr + 3].tolist() == pytest.approx([0.1, -0.2, 0.3])
    assert float(np.linalg.norm(data.qpos[qadr + 3 : qadr + 7])) == pytest.approx(1.0)


def test_set_root_state_selected_env_ids_leaves_other_rows_untouched(tiny_env_factory):
    """Only the selected env's root state changes; the other keeps its spawn."""
    env = tiny_env_factory(num_envs=2)
    qadr, _ = _root(env)
    before = env.mj_datas[0].qpos[qadr : qadr + 7].copy()

    returned = env.set_root_state(
        [1],
        name=ARTICULATION,
        pos_w=torch.tensor([[0.3, -0.4, 1.1]]),
        quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        lin_vel_w=torch.zeros(1, 3),
        ang_vel_w=torch.zeros(1, 3),
    )

    assert returned.tolist() == [1]
    assert env.mj_datas[0].qpos[qadr : qadr + 7].tolist() == pytest.approx(before.tolist())
    assert env.mj_datas[1].qpos[qadr : qadr + 3].tolist() == pytest.approx([0.3, -0.4, 1.1])


def test_world_angular_velocity_is_written_as_body_local_qvel(tiny_env):
    """qvel angular is body-local: qvel = R.T @ world_angular."""
    env = tiny_env
    qadr, vadr = _root(env)
    quat = _quat_90_x()
    world_ang = torch.tensor([[0.0, 0.0, 1.0]])

    env.set_root_state(
        None,
        name=ARTICULATION,
        pos_w=torch.tensor([[0.0, 0.0, 0.8]]),
        quat_wxyz=quat,
        lin_vel_w=torch.zeros(1, 3),
        ang_vel_w=world_ang,
    )

    data = env.mj_datas[0]
    rot = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert data.qvel[vadr + 3 : vadr + 6].tolist() == pytest.approx(
        (rot.T @ world_ang[0].numpy()).tolist(), abs=1e-6
    )
    assert (rot @ data.qvel[vadr + 3 : vadr + 6]).tolist() == pytest.approx(
        [0.0, 0.0, 1.0], abs=1e-6
    )


def test_set_joint_state_resolves_names_and_writes_a_subset(tiny_env):
    """Only the named joint row moves; velocity=None leaves qvel unchanged."""
    env = tiny_env
    art = env.articulations[ARTICULATION]
    names = list(art.joint_names)
    elbow = names.index("elbow")
    qadr, dadr = art.jadr_art[elbow], art.jdr_art[elbow]
    before_vel = env.mj_datas[0].qvel[dadr]

    returned = env.set_joint_state(
        None,
        articulation=ARTICULATION,
        joint_names=("elbow",),
        position=torch.tensor([[0.25]]),
    )

    assert returned.tolist() == [0]
    assert env.mj_datas[0].qpos[qadr] == pytest.approx(0.25)
    assert env.mj_datas[0].qvel[dadr] == pytest.approx(before_vel)

    env.set_joint_state(
        None,
        articulation=ARTICULATION,
        joint_names=("elbow",),
        position=torch.tensor([[0.1]]),
        velocity=torch.tensor([[0.5]]),
    )
    assert env.mj_datas[0].qpos[qadr] == pytest.approx(0.1)
    assert env.mj_datas[0].qvel[dadr] == pytest.approx(0.5)


def test_set_joint_state_rejects_an_unknown_joint_name(tiny_env):
    """Name resolution is explicit: an unknown joint is a teaching error."""
    with pytest.raises(ValueError, match=r"\['wrist'\] are not asset joints"):
        tiny_env.set_joint_state(
            None,
            articulation=ARTICULATION,
            joint_names=("wrist",),
            position=torch.zeros(1, 1),
        )


def test_set_root_state_rejects_an_unknown_name(tiny_env):
    """Unknown root names list the declared articulations and props."""
    with pytest.raises(ValueError, match=r"unknown body 'nope'.*declared articulations"):
        tiny_env.set_root_state(
            None,
            name="nope",
            pos_w=torch.zeros(1, 3),
            quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            lin_vel_w=torch.zeros(1, 3),
            ang_vel_w=torch.zeros(1, 3),
        )


def test_set_root_state_rejects_a_fixed_articulation_root(tiny_env_factory, tiny_mjcf, tmp_path):
    """A root without a freejoint uses set_joint_state, not set_root_state."""
    fixed = tmp_path / "tiny_arm_fixed.xml"
    fixed.write_text(
        tiny_mjcf.read_text(encoding="utf-8").replace('<freejoint name="root"/>', ""),
        encoding="utf-8",
    )
    env = tiny_env_factory(mjcf=fixed)
    assert env.articulations[ARTICULATION].free_qpos_adr is None

    with pytest.raises(ValueError, match="has a fixed root"):
        env.set_root_state(
            None,
            name=ARTICULATION,
            pos_w=torch.zeros(1, 3),
            quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            lin_vel_w=torch.zeros(1, 3),
            ang_vel_w=torch.zeros(1, 3),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "pos_w",
            torch.zeros(1, 2),
            r"pos_w shape \(1, 2\) != required \(1, 3\)",
        ),
        (
            "quat_wxyz",
            torch.zeros(1, 4),
            r"quat_wxyz contains a zero-norm quaternion",
        ),
        (
            "ang_vel_w",
            torch.tensor([[0.0, float("nan"), 0.0]]),
            r"ang_vel_w contains non-finite values",
        ),
    ],
)
def test_set_root_state_rejects_bad_tensors(tiny_env, field, value, match):
    """Shape, non-finite and zero-quat inputs fail at the boundary."""
    kwargs = {
        "pos_w": torch.zeros(1, 3),
        "quat_wxyz": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "lin_vel_w": torch.zeros(1, 3),
        "ang_vel_w": torch.zeros(1, 3),
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=match):
        tiny_env.set_root_state(None, name=ARTICULATION, **kwargs)


def test_state_write_requires_tensors_on_the_env_device(tiny_env):
    """A CPU write face rejects a device-mismatched tensor, not coerce it."""
    if tiny_env.device != "cpu":
        pytest.skip("the MuJoCo backend is cpu-only by construction")
    with pytest.raises(ValueError, match="expected cpu"):
        tiny_env.set_root_state(
            None,
            name=ARTICULATION,
            pos_w=torch.zeros(1, 3, device="meta"),
            quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="meta"),
            lin_vel_w=torch.zeros(1, 3, device="meta"),
            ang_vel_w=torch.zeros(1, 3, device="meta"),
        )


def test_non_one_dof_joints_are_rejected():
    """The joint-type gate rejects a compiled ball joint."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="b">'
        '<joint name="ball" type="ball"/>'
        '<geom type="sphere" size="0.1"/>'
        "</body></worldbody></mujoco>"
    )
    with pytest.raises(ValueError, match="only 1-DOF joints"):
        _one_dof_joint_id(model, "ball", "ball", ARTICULATION)


def test_settle_refreshes_observations_and_angular_derivations(tiny_env_factory):
    """settle runs mj_forward and republishes world/base angular cache values."""
    env = tiny_env_factory()
    env.set_root_state(
        None,
        name=ARTICULATION,
        pos_w=torch.tensor([[0.2, -0.1, 0.9]]),
        quat_wxyz=_quat_90_x(),
        lin_vel_w=torch.zeros(1, 3),
        ang_vel_w=torch.tensor([[0.0, 0.0, 1.0]]),
    )

    returned = env.settle(None)

    assert returned.tolist() == [0]
    assert env.record()["articulation/tiny/base_pos_w"][0].tolist() == pytest.approx(
        [0.2, -0.1, 0.9]
    )
    assert env.record()["articulation/tiny/base_quat_w"][0].tolist() == pytest.approx(
        _quat_90_x()[0].tolist()
    )
    # body-local angular qvel for a world +z angular velocity under Rx(90).
    assert env.record()["articulation/tiny/base_ang_vel_b"][0].tolist() == pytest.approx(
        [0.0, 1.0, 0.0], abs=1e-6
    )
    state = env._read_state()
    assert state["ang_vel_b"].tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-6)
    assert state["ang_vel_w"].tolist() == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)


def test_reset_overwrites_written_state(tiny_env):
    """A later reset restores spawn pose/quat/velocity, the documented lifecycle."""
    env = tiny_env
    qadr, vadr = _root(env)
    env.set_root_state(
        None,
        name=ARTICULATION,
        pos_w=torch.tensor([[1.0, 2.0, 1.2]]),
        quat_wxyz=_quat_90_x(),
        lin_vel_w=torch.ones(1, 3),
        ang_vel_w=torch.ones(1, 3),
    )
    env.settle(None)

    env.reset_envs(torch.tensor([0], dtype=torch.int64))

    data = env.mj_datas[0]
    assert data.qpos[qadr : qadr + 3].tolist() == pytest.approx([0.0, 0.0, 0.8])
    assert data.qpos[qadr + 3 : qadr + 7].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert data.qvel[vadr : vadr + 6].tolist() == pytest.approx([0.0] * 6)


def test_reset_writes_a_full_spawn_quaternion_to_root_qpos(tiny_env_factory):
    """A non-yaw spawn quat reaches root qpos wxyz (normalized once, no yaw)."""
    env = tiny_env_factory(spawn={"pos": [0.2, -0.1, 0.9], "quat_wxyz": [1.0, 0.0, 1.0, 0.0]})
    qadr, _ = _root(env)
    expected = [2**-0.5, 0.0, 2**-0.5, 0.0]

    env.reset_envs(torch.tensor([0], dtype=torch.int64))

    data = env.mj_datas[0]
    assert data.qpos[qadr : qadr + 3].tolist() == pytest.approx([0.2, -0.1, 0.9])
    assert data.qpos[qadr + 3 : qadr + 7].tolist() == pytest.approx(expected)
    assert float(np.linalg.norm(data.qpos[qadr + 3 : qadr + 7])) == pytest.approx(1.0)


def test_free_rigid_prop_root_state_roundtrip(tiny_env_factory):
    """A free rigid prop writes pose/quat/qvel through its freejoint address."""
    manager = build_prop_manager(
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
    )
    assert manager is not None
    env = tiny_env_factory(prop_manager=manager)
    body_id, qadr, vadr = env.prop_backend.free_addresses("crate")
    pose = [0.4, 0.1, 0.5]
    quat = _quat_90_x()[0]
    lin = [0.1, 0.2, 0.3]
    ang = [0.0, 0.0, 0.4]

    returned = env.set_root_state(
        None,
        name="crate",
        pos_w=torch.tensor([pose]),
        quat_wxyz=quat.unsqueeze(0),
        lin_vel_w=torch.tensor([lin]),
        ang_vel_w=torch.tensor([ang]),
    )
    env.settle(None)

    assert returned.tolist() == [0]
    data = env.mj_datas[0]
    assert data.qpos[qadr : qadr + 3].tolist() == pytest.approx(pose)
    assert data.qpos[qadr + 3 : qadr + 7].tolist() == pytest.approx(quat.tolist())
    assert data.qvel[vadr : vadr + 3].tolist() == pytest.approx(lin)
    assert data.qvel[vadr + 3 : vadr + 6].tolist() == pytest.approx([0.0, 0.4, 0.0], abs=1e-6)
    assert data.xpos[body_id].tolist() == pytest.approx(pose)
    assert data.xquat[body_id].tolist() == pytest.approx(quat.tolist())
