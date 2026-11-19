"""A real ``box`` prop spawns and reads its world pose on PhysX."""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]

CRATE = "crate"
SPAWN_XYZ = (0.5, 0.0, 0.3)


def _identity_wxyz(num_envs: int, device: str) -> torch.Tensor:
    """The repo-internal ``wxyz`` identity quaternion for each env row."""
    quat = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
    quat[:, 0] = 1.0
    return quat


def test_rigid_box_prop_spawns_and_reads_its_world_pose(env_factory, crate_prop_manager):
    """A declared ``box`` prop is a scene asset with an env-relative spawn pose."""
    manager = crate_prop_manager(num_envs=2)
    assert manager is not None
    assert manager.names == (CRATE,)

    env = env_factory(2, prop_manager=manager)
    assert f"prop_{CRATE}" in env.scene.keys()

    poses = manager.get_world_poses(env)
    assert CRATE in poses
    pos, quat = poses[CRATE]
    assert tuple(pos.shape) == (2, 3)
    assert tuple(quat.shape) == (2, 4)

    # Placement is env-relative: world pose = env origin + declared spawn.
    spawn = torch.tensor(SPAWN_XYZ, dtype=torch.float32, device=env.device)
    expected = env.scene.env_origins + spawn
    torch.testing.assert_close(pos, expected, atol=1e-4, rtol=0)

    # The manager exposes wxyz: identity here, so the xyz part is ~zero.
    torch.testing.assert_close(
        quat,
        _identity_wxyz(env.num_envs, env.device),
        atol=1e-5,
        rtol=0,
    )
    assert bool((quat[:, 1:].abs() < 1e-4).all())
