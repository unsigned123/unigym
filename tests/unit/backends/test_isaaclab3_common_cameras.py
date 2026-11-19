"""Engine-free camera helpers: frame normalization and state-table pose writes."""

from __future__ import annotations

import torch
import pytest

from unigym.backends.shared.isaaclab3_common.cameras.cfg import (
    push_state_table_poses,
    read_frames,
)
from unigym.cameras.base import CameraSpec, depth_key, rgb_key
from unigym.cameras.state import CameraStateTable


class _Proxy:
    """IsaacLab 3 ``ProxyArray`` stand-in exposing the ``.torch`` payload."""

    def __init__(self, value: torch.Tensor):
        self.torch = value


class _SensorData:
    def __init__(self, output: dict):
        self.output = dict(output)


class _Sensor:
    def __init__(self, output: dict):
        self.data = _SensorData(output)
        self.calls = []

    def set_world_poses(self, positions=None, orientations=None, env_ids=None, convention=None):
        self.calls.append(
            {
                "positions": positions,
                "orientations": orientations,
                "env_ids": env_ids,
                "convention": convention,
            }
        )


def test_read_frames_normalizes_rgb_and_depth_layouts():
    """RGBA becomes (N,H,W,3) uint8 and (N,H,W) depth becomes (N,1,H,W) float32."""
    spec = CameraSpec(name="cam", types=("rgb", "depth"), width=3, height=2)
    rgb = torch.arange(1 * 2 * 3 * 4, dtype=torch.uint8).reshape(1, 2, 3, 4)
    depth = torch.ones(1, 2, 3)
    sensor = _Sensor(
        {"rgb": _Proxy(rgb), "distance_to_camera": _Proxy(depth)},
    )

    frames = read_frames(sensor, spec, num_envs=1)

    assert frames[rgb_key("cam")].shape == (1, 2, 3, 3)
    assert frames[rgb_key("cam")].dtype == torch.uint8
    assert torch.equal(frames[rgb_key("cam")], rgb[..., :3])
    assert frames[depth_key("cam")].shape == (1, 1, 2, 3)
    assert frames[depth_key("cam")].dtype == torch.float32


def test_read_frames_clones_the_annotator_buffer():
    """The returned frame owns its storage (annotator buffers are recycled)."""
    spec = CameraSpec(name="cam", types=("rgb",), width=1, height=1)
    rgb = torch.tensor([[[[1, 2, 3]]]], dtype=torch.uint8)
    sensor = _Sensor({"rgb": _Proxy(rgb)})

    frame = read_frames(sensor, spec, num_envs=1)[rgb_key("cam")]
    rgb.fill_(0)

    assert frame.tolist() == [[[[1, 2, 3]]]]


def test_read_frames_normalizes_the_declared_depth_background():
    """A finite no-hit sentinel becomes the manager's inf background."""
    spec = CameraSpec(name="cam", types=("depth",), width=1, height=1)
    depth = torch.tensor([[[0.0]]])
    sensor = _Sensor({"distance_to_camera": _Proxy(depth)})

    frame = read_frames(sensor, spec, num_envs=1, depth_background=0.0)[depth_key("cam")]

    assert torch.isinf(frame).all()


def test_push_state_table_poses_writes_a_full_range_without_env_ids():
    """A full-range write keeps the historical env_ids=None call shape."""
    table = CameraStateTable()
    table.declare("cam", "per_instance")
    table.set("cam", 0, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0])
    table.set("cam", 1, [0.0, 0.0, 2.0], [1.0, 0.0, 0.0, 0.0])
    sensor = _Sensor({})

    ids = push_state_table_poses(sensor, table, "cam", [0, 1], num_envs=2)

    assert ids == [0, 1]
    call = sensor.calls[0]
    assert call["env_ids"] is None
    assert call["convention"] == "world"
    assert torch.equal(call["positions"], torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]))


def test_push_state_table_poses_selects_env_ids():
    """A subset write addresses exactly those prims."""
    table = CameraStateTable()
    table.declare("cam", "per_instance")
    table.set("cam", 1, [0.0, 0.0, 2.0], [1.0, 0.0, 0.0, 0.0])
    sensor = _Sensor({})

    ids = push_state_table_poses(sensor, table, "cam", [1], num_envs=2, force_env_ids=True)

    call = sensor.calls[0]
    assert ids == [1]
    assert call["env_ids"].tolist() == [1]
    assert call["orientations"].shape == (1, 4)


def test_push_state_table_poses_rejects_a_missing_row():
    """An env without a state row fails with the camera and env named."""
    table = CameraStateTable()
    table.declare("cam", "per_instance")
    sensor = _Sensor({})

    with pytest.raises(KeyError, match="no state row"):
        push_state_table_poses(sensor, table, "cam", [0], num_envs=1)
