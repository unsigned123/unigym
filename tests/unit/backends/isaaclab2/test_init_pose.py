"""init_pose yaml validation and deterministic torch sampling semantics."""

from __future__ import annotations

import math

import pytest
import torch

from unigym.backends.isaaclab2.init_pose import build_init_pose_cfg, sample_init_pose


def test_build_init_pose_cfg_absent_or_empty_returns_none():
    """An absent or empty init_pose section keeps the upright spawn."""
    assert build_init_pose_cfg(None) is None
    assert build_init_pose_cfg({}) is None


@pytest.mark.parametrize("prob", [0.0, 1.0])
def test_build_init_pose_cfg_accepts_prob_boundaries(prob):
    """prob accepts both closed-interval boundaries."""
    assert build_init_pose_cfg({"prob": prob})["prob"] == prob


@pytest.mark.parametrize("prob", [-0.1, 1.1])
def test_build_init_pose_cfg_rejects_prob_outside_unit_interval(prob):
    """prob below zero or above one fails fast with the field named."""
    with pytest.raises(ValueError, match="init_pose.prob must be in"):
        build_init_pose_cfg({"prob": prob})


@pytest.mark.parametrize(
    ("section", "match"),
    [
        ({"roll_range": [0.0]}, "init_pose.roll_range must be"),
        ({"pitch_range": [0.0, 1.0, 2.0]}, "init_pose.pitch_range must be"),
        ({"yaw_range": [0.0]}, "init_pose.yaw_range must be"),
        ({"roll_range": [5.0, -5.0]}, "init_pose.roll_range requires lo <= hi"),
        ({"pitch_range": [1.0, 0.0]}, "init_pose.pitch_range requires lo <= hi"),
        ({"yaw_range": [10.0, -10.0]}, "init_pose.yaw_range requires lo <= hi"),
    ],
)
def test_build_init_pose_cfg_validates_euler_ranges(section, match):
    """Every euler range is exactly [lo, hi] ordered low to high."""
    with pytest.raises(ValueError, match=match):
        build_init_pose_cfg(section)


def test_build_init_pose_cfg_defaults_and_preserves_height():
    """height defaults to 0.42 while an explicit value survives normalization."""
    assert build_init_pose_cfg({"prob": 1.0})["height"] == 0.42
    assert build_init_pose_cfg({"height": 1.25})["height"] == 1.25


def test_sample_init_pose_shapes_dtype_and_device():
    """Sampling returns (N, 3) positions and (N, 4) float32 quaternions on device."""
    device = torch.device("cpu")

    pos, quat = sample_init_pose(5, [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], device=device)

    assert pos.shape == (5, 3)
    assert quat.shape == (5, 4)
    assert pos.dtype == torch.float32
    assert quat.dtype == torch.float32
    assert pos.device == device
    assert quat.device == device


def test_sample_init_pose_places_every_base_at_height():
    """Every sampled base position uses the requested spawn height on z."""
    pos, _ = sample_init_pose(4, [-180.0, 180.0], [-90.0, 90.0], [-180.0, 180.0], height=1.25)

    assert torch.all(pos[:, 2] == 1.25)


def test_sample_init_pose_quaternions_are_unit_norm():
    """Every sampled (w, x, y, z) row is a unit quaternion."""
    _, quat = sample_init_pose(8, [-180.0, 180.0], [-90.0, 90.0], [-180.0, 180.0])

    norms = torch.linalg.vector_norm(quat, dim=1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


@pytest.mark.parametrize(
    ("yaw", "expected_z"),
    [(90.0, math.sqrt(0.5)), (-90.0, -math.sqrt(0.5))],
    ids=["positive-yaw", "negative-yaw"],
)
def test_sample_init_pose_yaw_only_quaternion_order_and_sign(yaw, expected_z):
    """A yaw-only sample yields (w, 0, 0, z) with z carrying the yaw sign."""
    _, quat = sample_init_pose(2, [0.0, 0.0], [0.0, 0.0], [yaw, yaw])

    expected = torch.tensor([math.sqrt(0.5), 0.0, 0.0, expected_z])

    assert torch.allclose(quat, expected.expand(2, 4), atol=1e-6)


def test_sample_init_pose_seeded_sampling_is_reproducible():
    """The same torch seed reproduces identical positions and quaternions."""
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(1234)
        pos_a, quat_a = sample_init_pose(6, [-60.0, 60.0], [-30.0, 30.0], [-180.0, 180.0])
        torch.manual_seed(1234)
        pos_b, quat_b = sample_init_pose(6, [-60.0, 60.0], [-30.0, 30.0], [-180.0, 180.0])
    finally:
        torch.random.set_rng_state(state)

    assert torch.equal(pos_a, pos_b)
    assert torch.equal(quat_a, quat_b)
