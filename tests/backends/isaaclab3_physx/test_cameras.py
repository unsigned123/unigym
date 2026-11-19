"""World and body camera rigs plus recorder capture on the real PhysX lane."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]

ARTICULATION = "cartpole"
JOINT_ORDER = ("slider_to_cart", "cart_to_pole")

CAMERAS = [
    {
        "name": "cam",
        "rig": {
            "type": "world",
            "pos": [2.0, 0.0, 2.5],
            "lookat": [0.0, 0.0, 2.0],
        },
        "types": ["rgb", "depth"],
        "depth_type": "distance_to_camera",
        "count": "global",
        "width": 64,
        "height": 36,
    },
    {
        "name": "eye",
        "rig": {"type": "body", "attach": "pole"},
        "types": ["rgb"],
        "width": 32,
        "height": 18,
    },
    {
        "name": "side",
        "rig": {
            "type": "world",
            "pos": [2.0, 0.0, 2.5],
            "lookat": [0.0, 0.0, 2.0],
            "target_env": 1,
        },
        "instance": "per_env",
        "env_poses": [
            {"pos": [2.0, 0.0, 2.5], "quat": [1.0, 0.0, 0.0, 0.0]},
            {"pos": [-2.0, 0.0, 2.5], "quat": [1.0, 0.0, 0.0, 0.0]},
        ],
        "types": ["rgb"],
        "width": 32,
        "height": 18,
    },
]


def _manager():
    from unigym.cameras.manager import build_camera_manager

    return build_camera_manager(
        CAMERAS,
        obs_cameras=("cam", "eye", "side"),
        enable_cameras=True,
        backend="isaaclab3_physx",
    )


def _step(env) -> None:
    from unigym.runtime.actions import NamedAction

    env.step(
        (
            NamedAction.zeros(
                env.num_envs,
                ARTICULATION,
                JOINT_ORDER,
                "position",
                device=env.device,
            ),
        )
    )


def test_world_and_body_rigs_serve_rgb_and_depth(env_factory):
    """A global world rig serves RGB + metric depth; a body rig serves RGB."""
    import torch

    manager = _manager()
    env = env_factory(2, camera_manager=manager)
    assert manager.names == ["cam", "eye", "side"]

    _step(env)
    frames = manager.tick_and_read()

    rgb = frames["cam/cam/rgb"]
    assert tuple(rgb.shape) == (2, 36, 64, 3)
    assert rgb.dtype == torch.uint8
    assert bool(rgb.any())

    depth = frames["cam/cam/depth"]
    assert tuple(depth.shape) == (2, 1, 36, 64)
    assert depth.dtype == torch.float32
    finite = depth[torch.isfinite(depth)]
    assert finite.numel() > 0
    assert float(finite.min()) >= 0.1
    assert float(finite.max()) <= 20.0
    assert torch.unique(finite).numel() > 1

    eye = frames["cam/eye/rgb"]
    assert tuple(eye.shape) == (2, 18, 32, 3)
    assert eye.dtype == torch.uint8
    assert bool(eye.any())

    side = frames["cam/side/rgb"]
    assert tuple(side.shape) == (2, 18, 32, 3)
    assert side.dtype == torch.uint8
    assert bool(side.any())


def test_per_env_world_camera_keeps_its_own_state(env_factory):
    """The per-instance world rig reads the env 1 declared pose, not env 0's."""
    import torch

    manager = _manager()
    env_factory(2, camera_manager=manager)

    pose = manager.get_world_pose("side")
    torch.testing.assert_close(
        torch.tensor(pose["pos"], dtype=torch.float32),
        torch.tensor([-2.0, 0.0, 2.5], dtype=torch.float32),
        atol=1e-4,
        rtol=0,
    )


def test_recorder_captures_and_writes_the_declared_camera(env_factory, tmp_path):
    """``build_recorder`` binds the rig, captures RGB and writes an MP4."""
    from unigym.recorder import build_recorder

    manager = _manager()
    env = env_factory(2, camera_manager=manager)
    recorder = build_recorder(
        {"enabled": True, "camera_ref": "cam", "record_env": 0},
        2,
        manager,
    )
    assert recorder.camera_ref == "cam"

    _step(env)
    manager.tick_and_read()
    frame = recorder.capture_frame()

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (36, 64, 3)
    assert frame.dtype == np.uint8
    assert bool(frame.any())

    video = tmp_path / "cam.mp4"
    written = recorder.write_video([frame, frame], str(video), fps=5)
    assert written == str(video)
    assert video.is_file()
    assert video.stat().st_size > 0


def test_physx_declares_only_the_lane_proven_camera_traits():
    """The camera rigs and MP4 recorder are proven; depth aliases are deferred."""
    from unigym.backends.isaaclab3_physx.env import Isaaclab3PhysxEnv
    from unigym.backends.registry import ISAACLAB3_PHYSX_TRAITS, Trait

    proven = {
        Trait.CAMERAS_RGB,
        Trait.CAMERAS_DEPTH_RIG,
        Trait.CAMERAS_GLOBAL,
        Trait.CAMERAS_PER_ENV,
        Trait.CAMERAS_PER_ENV_STATE,
        Trait.RECORDER_VIDEO,
    }
    declared = {
        trait
        for trait in ISAACLAB3_PHYSX_TRAITS
        if trait.name.startswith(("CAMERAS_", "DEPTH_OBS_", "RECORDER_"))
    }
    assert declared == proven
    assert Trait.CAMERAS_BATCH_RENDER not in ISAACLAB3_PHYSX_TRAITS
    assert Trait.CAMERAS_PER_ENV_INTRINSICS not in ISAACLAB3_PHYSX_TRAITS
    assert Trait.DEPTH_OBS_ALIASES not in ISAACLAB3_PHYSX_TRAITS

    capabilities = Isaaclab3PhysxEnv.BACKEND_CAPABILITIES
    assert capabilities.cameras is True
    assert capabilities.depth_camera is False
    assert capabilities.recorder is True
    assert capabilities.traits() <= ISAACLAB3_PHYSX_TRAITS
