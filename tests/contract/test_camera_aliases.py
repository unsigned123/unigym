"""Depth obs aliases require an explicitly declared ``depth`` camera."""

from __future__ import annotations

import pytest
import torch

from unigym.adapters.base import ImageObsRequest, RawObsRequest
from unigym.cameras.base import DEPTH_ALIAS_KEYS, CameraSpec
from unigym.cameras.manager import build_camera_manager


def _depth_entry() -> dict:
    """The classic depth-encoder geometry declared as an explicit camera."""
    return {
        "name": "depth",
        "rig": {
            "type": "body",
            "attach": "pelvis",
            "pos": [0.33, 0.0, 0.043],
            "pitch_down_deg": 5.0,
        },
        "types": ["depth"],
        "width": 106,
        "height": 60,
        "hfov_deg": 87.0,
        "far_clip": 20.0,
        "depth_type": "distance_to_camera",
        "interval": 1,
    }


def _alias_requests() -> list:
    return [
        RawObsRequest("depth_image", 1, "camera"),
        ImageObsRequest("depth_image_raw", (1, 60, 106), torch.float32, "camera", "depth"),
    ]


def test_camera_spec_has_no_implicit_depth_camera_factory():
    """The removed implicit depth shim is not a public CameraSpec surface."""
    assert not hasattr(CameraSpec, "depth_body_default")


def test_depth_alias_keys_survive_the_shim_removal():
    """DEPTH_ALIAS_KEYS stays the adapter-facing alias vocabulary."""
    assert DEPTH_ALIAS_KEYS == ("depth_image", "depth_image_raw")


@pytest.mark.parametrize(
    "obs_request",
    [
        RawObsRequest("depth_image", 1, "camera"),
        RawObsRequest("depth_image_raw", 1, "camera"),
        ImageObsRequest("depth_image", (1, 58, 87), torch.float32, "camera", "depth"),
        ImageObsRequest("depth_image_raw", (1, 60, 106), torch.float32, "camera", "depth"),
    ],
)
def test_depth_alias_request_without_a_depth_camera_fails(obs_request):
    """A depth alias request fails when no ``depth`` camera is declared."""
    with pytest.raises(ValueError, match=r"declares no 'depth' camera"):
        build_camera_manager(
            None,
            [obs_request],
            backend="mujoco",
            num_envs=1,
        )


def test_explicit_depth_camera_serves_both_alias_keys():
    """A declared ``depth`` camera maps both aliases onto its depth stream."""
    manager = build_camera_manager(
        [_depth_entry()],
        _alias_requests(),
        backend="mujoco",
        num_envs=1,
    )

    assert manager is not None
    assert manager.obs_keys() == {"cam/depth/depth", "depth_image", "depth_image_raw"}
    raw = manager.frame_request("depth_image_raw")
    assert raw is not None
    assert raw.shape == (1, 60, 106)
    preprocessed = manager.frame_request("depth_image")
    assert preprocessed is not None
    assert preprocessed.shape == (1, 58, 87)


def test_explicit_depth_camera_without_the_depth_stream_fails():
    """The alias target must actually render the depth stream."""
    entry = _depth_entry()
    entry["types"] = ["rgb"]

    with pytest.raises(ValueError, match="must render the depth stream"):
        build_camera_manager(
            [entry],
            [RawObsRequest("depth_image_raw", 1, "camera")],
            backend="mujoco",
            num_envs=1,
        )
