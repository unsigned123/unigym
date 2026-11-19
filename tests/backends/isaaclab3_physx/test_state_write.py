"""Free-root state writes and partial reset of a PhysX ``box`` prop."""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]

CRATE = "crate"
SPAWN_XYZ = (0.5, 0.0, 0.3)
SET_XYZ = (0.2, -0.1, 0.6)
# ``set_root_state`` writes the link origin; ``get_world_pose`` returns the
# box logical bottom-face origin, so the 90-degree x rotation offsets y.
SET_LOGICAL_XYZ = (0.2, -0.05, 0.6)
QUAT_90_X = (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
LIN_VEL = (0.1, 0.2, 0.3)
ANG_VEL = (0.0, 0.0, 0.4)


def _identity_wxyz(num_envs: int, device: str) -> torch.Tensor:
    """The repo-internal ``wxyz`` identity quaternion for each env row."""
    quat = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
    quat[:, 0] = 1.0
    return quat


def test_set_root_state_then_partial_reset_roundtrips_on_a_prop(env_factory, crate_prop_manager):
    """``set_root_state`` writes one prop row; partial reset restores its spawn."""
    manager = crate_prop_manager(num_envs=2)
    assert manager is not None
    env = env_factory(2, prop_manager=manager)

    spawn_pos = env.scene.env_origins + torch.tensor(
        SPAWN_XYZ, dtype=torch.float32, device=env.device
    )
    identity = _identity_wxyz(env.num_envs, env.device)
    before_pos, before_quat = manager.get_world_poses(env)[CRATE]
    torch.testing.assert_close(before_pos, spawn_pos, atol=1e-4, rtol=0)
    torch.testing.assert_close(before_quat, identity, atol=1e-5, rtol=0)

    returned = env.set_root_state(
        [1],
        name=CRATE,
        pos_w=torch.tensor([SET_XYZ], dtype=torch.float32, device=env.device),
        quat_wxyz=torch.tensor([QUAT_90_X], dtype=torch.float32, device=env.device),
        lin_vel_w=torch.tensor([LIN_VEL], dtype=torch.float32, device=env.device),
        ang_vel_w=torch.tensor([ANG_VEL], dtype=torch.float32, device=env.device),
    )
    assert returned.dtype == torch.int64
    assert returned.tolist() == [1]

    env.settle([1])
    pos, quat = manager.get_world_poses(env)[CRATE]

    # The written link origin is the box centre; the logical pose subtracts
    # the rotated bottom-face offset.
    torch.testing.assert_close(
        pos[1],
        torch.tensor(SET_LOGICAL_XYZ, dtype=torch.float32, device=env.device),
        atol=1e-4,
        rtol=0,
    )
    asset = manager.handler(CRATE).asset
    assert asset is not None
    torch.testing.assert_close(
        asset.data.root_lin_vel_w.torch[1],
        torch.tensor(LIN_VEL, dtype=torch.float32, device=env.device),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        asset.data.root_ang_vel_w.torch[1],
        torch.tensor(ANG_VEL, dtype=torch.float32, device=env.device),
        atol=1e-5,
        rtol=0,
    )

    # The read-back quaternion is wxyz, so w and x carry the 90-degree rotation.
    torch.testing.assert_close(
        quat[1],
        torch.tensor(QUAT_90_X, dtype=torch.float32, device=env.device),
        atol=1e-5,
        rtol=0,
    )
    # An xyzw read would put w=0.707 in slot 3; wxyz leaves slot 3 at ~zero.
    assert quat[1, 3].abs().item() < 1e-5
    # Env 0 keeps its spawn pose while only env 1 was written.
    torch.testing.assert_close(pos[0], spawn_pos[0], atol=1e-4, rtol=0)
    torch.testing.assert_close(quat[0], identity[0], atol=1e-5, rtol=0)
    torch.testing.assert_close(
        asset.data.root_lin_vel_w.torch[0],
        torch.zeros(3, dtype=torch.float32, device=env.device),
        atol=1e-5,
        rtol=0,
    )

    env.reset_envs([1])
    env.settle([1])
    pos, quat = manager.get_world_poses(env)[CRATE]
    torch.testing.assert_close(pos[1], spawn_pos[1], atol=1e-4, rtol=0)
    torch.testing.assert_close(quat[1], identity[1], atol=1e-5, rtol=0)
    # Env 0 is unchanged by resetting the env 1 subset.
    torch.testing.assert_close(pos[0], spawn_pos[0], atol=1e-4, rtol=0)
    torch.testing.assert_close(quat[0], identity[0], atol=1e-5, rtol=0)
