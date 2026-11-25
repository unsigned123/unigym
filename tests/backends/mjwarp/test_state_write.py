"""Real MuJoCo Warp state-write contract: mirror writes, device pushes, settle."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unigym.backends.mjwarp.env import BACKEND_CAPABILITIES

ARTICULATION = "tiny"

pytestmark = pytest.mark.sim


def _quat_90_x() -> torch.Tensor:
    """Unit quaternion for +90 degrees about world x (wxyz)."""
    return torch.tensor([[0.7071067811865476, 0.7071067811865476, 0.0, 0.0]])


def _quat_90_z() -> torch.Tensor:
    """Unit quaternion for +90 degrees about world z (wxyz)."""
    return torch.tensor([[0.7071067811865476, 0.0, 0.0, 0.7071067811865476]])


def _root(env):
    art = env.articulations[ARTICULATION]
    assert art.free_qpos_adr is not None
    assert art.free_dof_adr is not None
    return art.free_qpos_adr, art.free_dof_adr


def _action(env, values):
    control = env.controls[(ARTICULATION, ARTICULATION)]
    names = tuple(control.joint_names)
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(env.num_envs, len(names))
    from unigym.runtime.actions import NamedAction

    return (NamedAction(control.articulation, tensor, names, control.target),)


def _device_qpos(env, e: int) -> np.ndarray:
    return np.asarray(env._wdata.qpos.numpy()[e], dtype=np.float64)


def _device_qvel(env, e: int) -> np.ndarray:
    return np.asarray(env._wdata.qvel.numpy()[e], dtype=np.float64)


def _device_warmstart(env, e: int) -> np.ndarray:
    return np.asarray(env._wdata.qacc_warmstart.numpy()[e], dtype=np.float64)


def test_mjwarp_declares_state_write_capability() -> None:
    """The trait bit and the env capability bit are declared together."""
    from unigym.backends.registry import MJWARP_TRAITS, Trait

    assert Trait.STATE_WRITE in MJWARP_TRAITS
    assert Trait.STATE_WRITE in BACKEND_CAPABILITIES.traits()


def test_set_root_state_normalizes_quat_and_pushes_to_device(tiny_env):
    """A non-unit wxyz quat is normalized; mirror and device get the state."""
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

    dev_qpos = _device_qpos(env, 0)
    dev_qvel = _device_qvel(env, 0)
    assert dev_qpos[qadr : qadr + 3].tolist() == pytest.approx([1.0, 2.0, 0.9], abs=1e-5)
    assert dev_qpos[qadr + 3 : qadr + 7].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert dev_qvel[vadr : vadr + 3].tolist() == pytest.approx([0.1, -0.2, 0.3], abs=1e-6)


def test_world_angular_velocity_is_written_as_body_local_qvel(tiny_env):
    """qvel angular is body-local: qvel = R.T @ world_angular, mirror + device."""
    env = tiny_env
    qadr, vadr = _root(env)

    env.set_root_state(
        None,
        name=ARTICULATION,
        pos_w=torch.tensor([[0.0, 0.0, 0.9]]),
        quat_wxyz=_quat_90_x(),
        lin_vel_w=torch.zeros(1, 3),
        ang_vel_w=torch.tensor([[0.0, 0.0, 1.0]]),
    )

    rot = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert env.mj_datas[0].qvel[vadr + 3 : vadr + 6].tolist() == pytest.approx(
        (rot.T @ np.array([0.0, 0.0, 1.0])).tolist(), abs=1e-6
    )
    assert _device_qvel(env, 0)[vadr + 3 : vadr + 6].tolist() == pytest.approx(
        [0.0, 1.0, 0.0], abs=1e-6
    )


def test_set_root_state_next_step_keeps_the_written_root(tiny_env):
    """The write is device-visible: the next warp step starts from it, not spawn."""
    env = tiny_env
    qadr, _ = _root(env)
    env.set_root_state(
        None,
        name=ARTICULATION,
        pos_w=torch.tensor([[0.4, -0.3, 1.1]]),
        quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        lin_vel_w=torch.zeros(1, 3),
        ang_vel_w=torch.zeros(1, 3),
    )

    env.step(_action(env, [[0.0, 0.0]]))

    data = env.mj_datas[0]
    assert data.qpos[qadr : qadr + 3].tolist() == pytest.approx([0.4, -0.3, 1.1], abs=0.02)
    assert _device_qpos(env, 0)[qadr : qadr + 3].tolist() == pytest.approx(
        data.qpos[qadr : qadr + 3].tolist(), abs=1e-5
    )


def test_reset_writes_a_full_spawn_quaternion_to_mirror_and_device(tiny_env_factory):
    """A non-yaw spawn quat reaches root qpos wxyz on the mirror and the device."""
    env = tiny_env_factory(spawn={"pos": [0.2, -0.1, 0.9], "quat_wxyz": [1.0, 0.0, 1.0, 0.0]})
    qadr, _ = _root(env)
    expected = [2**-0.5, 0.0, 2**-0.5, 0.0]

    env.reset_envs(torch.tensor([0], dtype=torch.int64))

    data = env.mj_datas[0]
    assert data.qpos[qadr : qadr + 3].tolist() == pytest.approx([0.2, -0.1, 0.9])
    assert data.qpos[qadr + 3 : qadr + 7].tolist() == pytest.approx(expected)
    device = _device_qpos(env, 0)
    assert device[qadr : qadr + 3].tolist() == pytest.approx([0.2, -0.1, 0.9], abs=1e-6)
    assert device[qadr + 3 : qadr + 7].tolist() == pytest.approx(expected, abs=1e-6)


def test_set_root_state_clears_warmstart_and_settle_keeps_it_cleared(tiny_env):
    """A write drops the pre-write warm start on mirror + device; settle keeps zero."""
    env = tiny_env
    for _ in range(3):
        env.step(_action(env, [[0.0, 0.0]]))

    mirror_before = np.asarray(env.mj_datas[0].qacc_warmstart, dtype=np.float64)
    device_before = _device_warmstart(env, 0)
    assert float(np.linalg.norm(mirror_before)) > 0.0
    assert float(np.linalg.norm(device_before)) > 0.0

    env.set_root_state(
        None,
        name=ARTICULATION,
        pos_w=torch.tensor([[0.4, -0.3, 1.0]]),
        quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        lin_vel_w=torch.zeros(1, 3),
        ang_vel_w=torch.zeros(1, 3),
    )

    assert not env.mj_datas[0].qacc_warmstart.any()
    assert not _device_warmstart(env, 0).any()

    env.settle(None)

    assert not env.mj_datas[0].qacc_warmstart.any()
    assert not _device_warmstart(env, 0).any()


def test_set_joint_state_outside_reset_clears_the_touched_env_warmstart(tiny_env):
    """Outside a reset a joint write drops the touched env's warm start."""
    env = tiny_env
    for _ in range(3):
        env.step(_action(env, [[0.0, 0.0]]))

    assert float(np.linalg.norm(env.mj_datas[0].qacc_warmstart)) > 0.0
    assert float(np.linalg.norm(_device_warmstart(env, 0))) > 0.0

    env.set_joint_state(
        None,
        articulation=ARTICULATION,
        joint_names=("elbow",),
        position=torch.tensor([[0.25]]),
    )

    assert not env.mj_datas[0].qacc_warmstart.any()
    assert not _device_warmstart(env, 0).any()


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
    """The torch-cache device is cpu; a mismatched tensor is not coerced."""
    assert tiny_env.device == "cpu"
    with pytest.raises(ValueError, match="expected cpu"):
        tiny_env.set_root_state(
            None,
            name=ARTICULATION,
            pos_w=torch.zeros(1, 3, device="meta"),
            quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="meta"),
            lin_vel_w=torch.zeros(1, 3, device="meta"),
            ang_vel_w=torch.zeros(1, 3, device="meta"),
        )


def test_set_joint_state_resolves_names_and_writes_a_subset(tiny_env):
    """Only the named joint row moves; velocity=None leaves qvel unchanged."""
    env = tiny_env
    art = env.articulations[ARTICULATION]
    names = list(art.joint_names)
    elbow = names.index("elbow")
    shoulder = names.index("shoulder")
    qadr, dadr = art.jadr_art[elbow], art.jdr_art[elbow]
    sqadr = art.jadr_art[shoulder]
    before_shoulder = env.mj_datas[0].qpos[sqadr]
    before_vel = env.mj_datas[0].qvel[dadr]

    returned = env.set_joint_state(
        None,
        articulation=ARTICULATION,
        joint_names=("elbow",),
        position=torch.tensor([[0.25]]),
    )

    assert returned.tolist() == [0]
    assert env.mj_datas[0].qpos[qadr] == pytest.approx(0.25)
    assert env.mj_datas[0].qpos[sqadr] == pytest.approx(before_shoulder)
    assert env.mj_datas[0].qvel[dadr] == pytest.approx(before_vel)
    assert _device_qpos(env, 0)[qadr] == pytest.approx(0.25, abs=1e-6)

    env.set_joint_state(
        None,
        articulation=ARTICULATION,
        joint_names=("elbow",),
        position=torch.tensor([[0.1]]),
        velocity=torch.tensor([[0.5]]),
    )
    assert env.mj_datas[0].qpos[qadr] == pytest.approx(0.1)
    assert env.mj_datas[0].qvel[dadr] == pytest.approx(0.5)
    assert _device_qvel(env, 0)[dadr] == pytest.approx(0.5, abs=1e-6)

    env.set_joint_state(
        None,
        articulation=ARTICULATION,
        joint_names=("elbow",),
        position=torch.tensor([[0.2]]),
    )
    assert env.mj_datas[0].qpos[qadr] == pytest.approx(0.2)
    assert env.mj_datas[0].qvel[dadr] == pytest.approx(0.5)


def test_set_joint_state_rejects_unknown_names_and_bad_selections(tiny_env):
    """Name resolution is explicit; duplicate/empty/unknown names fail."""
    with pytest.raises(ValueError, match=r"\['wrist'\] are not asset joints"):
        tiny_env.set_joint_state(
            None,
            articulation=ARTICULATION,
            joint_names=("wrist",),
            position=torch.zeros(1, 1),
        )
    with pytest.raises(ValueError, match="unknown articulation"):
        tiny_env.set_joint_state(
            None,
            articulation="nope",
            joint_names=("elbow",),
            position=torch.zeros(1, 1),
        )
    with pytest.raises(ValueError, match="duplicate names"):
        tiny_env.set_joint_state(
            None,
            articulation=ARTICULATION,
            joint_names=("elbow", "elbow"),
            position=torch.zeros(1, 2),
        )


def test_settle_refreshes_obs_contacts_and_keeps_episode_bookkeeping(tiny_env_factory, tiny_mjcf):
    """settle runs mjw.forward + mirror sync without touching the step counter."""
    from unigym.contacts.manager import build_contact_manager

    manager = build_contact_manager(
        [
            {
                "name": "robot",
                "sensor_refs": [
                    {"owner": "robot", "articulation": "tiny", "body": "base"},
                    {"owner": "robot", "articulation": "tiny", "body": "link"},
                    {"owner": "robot", "articulation": "tiny", "body": "tip"},
                ],
                "frame": "world",
            }
        ],
        backend="mjwarp",
        articulation_names=(ARTICULATION,),
    )
    env = tiny_env_factory(contact_manager=manager)
    before_episode = env._episode_id.clone()
    before_clock = env._time_in_episode.clone()
    before_ctrl = env.mj_datas[0].ctrl.copy()
    assert manager.read()["contact/robot"].count_nonzero() == 0

    env.set_root_state(
        None,
        name=ARTICULATION,
        pos_w=torch.tensor([[0.2, -0.1, 0.55]]),
        quat_wxyz=_quat_90_z(),
        lin_vel_w=torch.zeros(1, 3),
        ang_vel_w=torch.tensor([[0.0, 0.0, 1.0]]),
    )

    returned = env.settle(None)

    assert returned.tolist() == [0]
    assert env.record()["articulation/tiny/base_pos_w"][0].tolist() == pytest.approx(
        [0.2, -0.1, 0.55], abs=1e-5
    )
    assert env.record()["articulation/tiny/base_quat_w"][0].tolist() == pytest.approx(
        _quat_90_z()[0].tolist(), abs=1e-6
    )
    assert env.record()["articulation/tiny/base_ang_vel_b"][0].tolist() == pytest.approx(
        [0.0, 0.0, 1.0], abs=1e-5
    )
    assert manager.read()["contact/robot"].count_nonzero() > 0
    assert env._episode_id.tolist() == before_episode.tolist()
    assert env._time_in_episode.tolist() == before_clock.tolist()
    assert env.mj_datas[0].ctrl.tolist() == before_ctrl.tolist()


def test_subset_write_at_n2_changes_only_the_selected_env(tiny_env_factory):
    """The unselected mirror/device world keeps its state across write, settle, step."""
    env = tiny_env_factory(num_envs=2)
    qadr, _ = _root(env)
    before0 = env.mj_datas[0].qpos[qadr : qadr + 7].copy()
    before0_dev = _device_qpos(env, 0)[qadr : qadr + 7].copy()

    returned = env.set_root_state(
        [1],
        name=ARTICULATION,
        pos_w=torch.tensor([[0.3, -0.4, 1.1]]),
        quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        lin_vel_w=torch.zeros(1, 3),
        ang_vel_w=torch.zeros(1, 3),
    )

    assert returned.tolist() == [1]
    assert env.mj_datas[0].qpos[qadr : qadr + 7].tolist() == pytest.approx(before0.tolist())
    assert _device_qpos(env, 0)[qadr : qadr + 7].tolist() == pytest.approx(
        before0_dev.tolist(), abs=1e-6
    )
    assert env.mj_datas[1].qpos[qadr : qadr + 3].tolist() == pytest.approx([0.3, -0.4, 1.1])
    assert _device_qpos(env, 1)[qadr : qadr + 3].tolist() == pytest.approx(
        [0.3, -0.4, 1.1], abs=1e-5
    )

    env.settle([1])
    assert env.mj_datas[0].qpos[qadr : qadr + 7].tolist() == pytest.approx(before0.tolist())

    env.step(_action(env, [[0.0, 0.0], [0.0, 0.0]]))
    assert env.mj_datas[0].qpos[qadr : qadr + 3].tolist() == pytest.approx(
        [0.0, 0.0, 0.8], abs=0.03
    )
    assert env.mj_datas[1].qpos[qadr : qadr + 3].tolist() == pytest.approx(
        [0.3, -0.4, 1.1], abs=0.03
    )


def test_repeated_env_ids_preserve_order_and_last_write_wins(tiny_env_factory):
    """Repeated ids are repeated writes; the returned ids keep the input order."""
    env = tiny_env_factory(num_envs=2)
    art = env.articulations[ARTICULATION]
    elbow = list(art.joint_names).index("elbow")
    qadr = art.jadr_art[elbow]

    returned = env.set_joint_state(
        [1, 1],
        articulation=ARTICULATION,
        joint_names=("elbow",),
        position=torch.tensor([[0.15], [0.35]]),
    )

    assert returned.tolist() == [1, 1]
    assert env.mj_datas[1].qpos[qadr] == pytest.approx(0.35)
    assert _device_qpos(env, 1)[qadr] == pytest.approx(0.35, abs=1e-6)


def test_free_rigid_prop_root_state_roundtrip(tiny_env_factory):
    """A free rigid prop writes pose/quat/qvel through its freejoint address."""
    from unigym.props import build_prop_manager

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
        backend="mjwarp",
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
    assert data.xpos[body_id].tolist() == pytest.approx(pose, abs=1e-5)
    assert data.xquat[body_id].tolist() == pytest.approx(quat.tolist(), abs=1e-5)
    assert _device_qpos(env, 0)[qadr : qadr + 3].tolist() == pytest.approx(pose, abs=1e-5)
