"""The neutral ``Director`` write paths on the real PhysX lane.

Every op goes through the public ``Director.handle`` face; prop poses are read
back through the prop manager and velocities through the spawned asset data.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]

CRATE = "crate"
BOX_HALF_HEIGHT = 0.05
DIRECTOR_CFG = {
    "gripper": {
        "name": "vgrip",
        "pos": [0.5, 0.0, 0.5],
        "quat": [1.0, 0.0, 0.0, 0.0],
    }
}
IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)


def _director(env):
    """Mount the neutral Director with this backend's injected ops face."""
    from unigym.backends.compose import backend_runtime
    from unigym.director import Director

    return Director(
        env,
        cfg=DIRECTOR_CFG,
        director_ops=backend_runtime("isaaclab3_physx").bindings.director_ops,
    )


def _assert_logical_pose(poses, expected_xyz, expected_wxyz, *, device):
    """The manager exposes the box logical (bottom-face) pose in ``wxyz``."""
    pos, quat = poses[CRATE]
    torch.testing.assert_close(
        pos[0],
        torch.tensor(expected_xyz, dtype=torch.float32, device=device),
        atol=1e-4,
        rtol=0,
    )
    torch.testing.assert_close(
        quat[0],
        torch.tensor(expected_wxyz, dtype=torch.float32, device=device),
        atol=1e-5,
        rtol=0,
    )


def test_teleport_writes_a_rigid_prop_pose(env_factory, crate_prop_manager):
    """``teleport`` writes the root link pose visible through the prop pose."""
    manager = crate_prop_manager(num_envs=1)
    env = env_factory(1, prop_manager=manager, director_cfg=DIRECTOR_CFG)
    director = _director(env)

    root_xyz = (0.2, -0.1, 0.6)
    result = director.handle(
        1,
        {
            "op": "teleport",
            "name": CRATE,
            "pos": list(root_xyz),
            "quat": list(IDENTITY_WXYZ),
        },
    )
    assert result["teleported"] == CRATE

    env.settle([0])
    # The write targets the link origin; the manager reads the logical pose.
    expected = (root_xyz[0], root_xyz[1], root_xyz[2] - BOX_HALF_HEIGHT)
    _assert_logical_pose(manager.get_world_poses(env), expected, IDENTITY_WXYZ, device=env.device)


def test_setvel_writes_world_linear_and_angular_velocity(env_factory, crate_prop_manager):
    """``setvel`` writes the root velocity visible through the asset data."""
    manager = crate_prop_manager(num_envs=1)
    env = env_factory(1, prop_manager=manager, director_cfg=DIRECTOR_CFG)
    director = _director(env)

    director.handle(
        1,
        {
            "op": "teleport",
            "name": CRATE,
            "pos": [0.2, -0.1, 0.6],
            "quat": list(IDENTITY_WXYZ),
        },
    )
    lin = (0.3, -0.2, 0.0)
    ang = (0.0, 0.0, 0.5)
    director.handle(2, {"op": "setvel", "name": CRATE, "lin": list(lin), "ang": list(ang)})

    env.settle([0])
    asset = manager.handler(CRATE).asset
    assert asset is not None
    torch.testing.assert_close(
        asset.data.root_lin_vel_w.torch[0],
        torch.tensor(lin, dtype=torch.float32, device=env.device),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        asset.data.root_ang_vel_w.torch[0],
        torch.tensor(ang, dtype=torch.float32, device=env.device),
        atol=1e-5,
        rtol=0,
    )


def test_follow_locks_the_gripper_offset_and_release_clears_it(env_factory, crate_prop_manager):
    """``follow`` re-applies the locked offset from the moved kinematic hand."""
    manager = crate_prop_manager(num_envs=1)
    env = env_factory(1, prop_manager=manager, director_cfg=DIRECTOR_CFG)
    director = _director(env)

    director.handle(
        1,
        {
            "op": "teleport",
            "name": CRATE,
            "pos": [0.4, 0.0, 0.6],
            "quat": list(IDENTITY_WXYZ),
        },
    )
    director.handle(
        2,
        {
            "op": "gripper",
            "action": "move",
            "pose": {"pos": [0.5, 0.0, 0.6], "quat": list(IDENTITY_WXYZ)},
        },
    )
    env.settle([0])
    before, _ = manager.get_world_poses(env)[CRATE]

    director.handle(3, {"op": "follow", "obj": CRATE, "host": "gripper"})
    assert director.status_doc()["following"] == {CRATE: "gripper"}

    delta = (0.15, 0.1, 0.0)
    director.handle(
        4,
        {
            "op": "gripper",
            "action": "move",
            "pose": {"pos": [0.65, 0.1, 0.6]},
        },
    )
    env.settle([0])
    director.on_step(0, {})
    env.settle([0])

    after, _ = manager.get_world_poses(env)[CRATE]
    torch.testing.assert_close(
        after[0] - before[0],
        torch.tensor(delta, dtype=torch.float32, device=env.device),
        atol=1e-4,
        rtol=0,
    )

    result = director.handle(5, {"op": "release", "name": CRATE})
    assert result == {"released": CRATE, "was": True}
    assert director.status_doc()["following"] == {}


def test_pin_holds_the_captured_pose_across_on_step_and_release_clears_it(
    env_factory, crate_prop_manager
):
    """``pin`` restores the captured pose after an external write; release clears."""
    manager = crate_prop_manager(num_envs=1)
    env = env_factory(1, prop_manager=manager, director_cfg=DIRECTOR_CFG)
    director = _director(env)

    root_xyz = (0.2, -0.1, 0.6)
    director.handle(
        1,
        {
            "op": "teleport",
            "name": CRATE,
            "pos": list(root_xyz),
            "quat": list(IDENTITY_WXYZ),
        },
    )
    env.settle([0])
    captured, captured_quat = manager.get_world_poses(env)[CRATE]
    captured = captured.clone()
    captured_quat = captured_quat.clone()

    director.handle(2, {"op": "pin", "name": CRATE})
    assert director.status_doc()["pinned"] == [CRATE]

    # Move the prop away through the public face; the sink hook must restore it.
    director.handle(
        3,
        {
            "op": "teleport",
            "name": CRATE,
            "pos": [0.6, 0.3, 0.9],
            "quat": list(IDENTITY_WXYZ),
        },
    )
    env.settle([0])
    director.on_step(0, {})
    env.settle([0])

    held, held_quat = manager.get_world_poses(env)[CRATE]
    torch.testing.assert_close(held[0], captured[0], atol=1e-4, rtol=0)
    torch.testing.assert_close(held_quat[0], captured_quat[0], atol=1e-5, rtol=0)

    result = director.handle(4, {"op": "release", "name": CRATE})
    assert result == {"released": CRATE, "was": True}
    assert director.status_doc()["pinned"] == []


def test_gripper_where_and_move_roundtrip_through_the_scene_asset(env_factory, crate_prop_manager):
    """``gripper where`` reads the kinematic scene asset after a ``move``."""
    manager = crate_prop_manager(num_envs=1)
    env = env_factory(1, prop_manager=manager, director_cfg=DIRECTOR_CFG)
    director = _director(env)

    env.settle([0])
    initial = director.handle(1, {"op": "gripper", "action": "where"})
    assert initial["gripper"] == "vgrip"
    torch.testing.assert_close(
        torch.tensor(initial["pos"], dtype=torch.float32),
        torch.tensor([0.5, 0.0, 0.5], dtype=torch.float32),
        atol=1e-4,
        rtol=0,
    )
    torch.testing.assert_close(
        torch.tensor(initial["quat"], dtype=torch.float32),
        torch.tensor(IDENTITY_WXYZ, dtype=torch.float32),
        atol=1e-5,
        rtol=0,
    )

    target_pos = (0.7, -0.2, 0.8)
    half_turn_z = (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
    moved = director.handle(
        2,
        {
            "op": "gripper",
            "action": "move",
            "pose": {"pos": list(target_pos), "quat": list(half_turn_z)},
        },
    )
    assert moved["gripper"] == "vgrip"

    env.settle([0])
    readback = director.handle(3, {"op": "gripper", "action": "where"})
    torch.testing.assert_close(
        torch.tensor(readback["pos"], dtype=torch.float32),
        torch.tensor(target_pos, dtype=torch.float32),
        atol=1e-4,
        rtol=0,
    )
    torch.testing.assert_close(
        torch.tensor(readback["quat"], dtype=torch.float32),
        torch.tensor(half_turn_z, dtype=torch.float32),
        atol=1e-5,
        rtol=0,
    )


def test_physx_declares_the_lane_proven_director_traits():
    """Only the traits with passing PhysX lane coverage are declared."""
    from unigym.capabilities.traits import ISAACLAB3_PHYSX_TRAITS, Trait

    proven = {
        Trait.DIRECTOR_TELEPORT,
        Trait.DIRECTOR_SETVEL,
        Trait.DIRECTOR_ATTACH_SLAVE,
        Trait.DIRECTOR_GRIPPER,
    }
    assert proven <= ISAACLAB3_PHYSX_TRAITS
    assert Trait.DIRECTOR_ATTACH_WELD not in ISAACLAB3_PHYSX_TRAITS
