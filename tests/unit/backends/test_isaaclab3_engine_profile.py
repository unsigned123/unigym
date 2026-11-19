"""The IsaacLab 3 engine profile is the only engine-divergent contract."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from unigym.backends.isaaclab3_newton.profile import NEWTON_PROFILE
from unigym.backends.isaaclab3_physx.profile import PHYSX_PROFILE
from unigym.backends.shared.isaaclab3_common import EngineProfile


def _profile() -> EngineProfile:
    return EngineProfile(
        name="test",
        build_physics_cfg=lambda: object(),
        renderer_cfg=None,
        depth_background=0.0,
        default_joint_ordering=None,
        get_body_com=lambda asset: None,
        set_body_com=lambda asset, com, env_ids: None,
    )


def test_engine_profile_names_the_divergence_points() -> None:
    names = {field.name for field in dataclasses.fields(EngineProfile)}
    assert names == {
        "name",
        "build_physics_cfg",
        "renderer_cfg",
        "depth_background",
        "default_joint_ordering",
        "get_body_com",
        "set_body_com",
    }


def test_engine_profile_is_frozen() -> None:
    profile = _profile()
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(profile, "name", "other")


class _CoMAsset:
    """Minimal CoM asset surface used by the profile read/write tests."""

    def __init__(self, n: int, b: int) -> None:
        self.data = SimpleNamespace(
            body_com_pos_b=torch.arange(n * b * 3, dtype=torch.float32).reshape(n, b, 3),
            body_com_pose_b=torch.cat(
                (
                    torch.arange(n * b * 3, dtype=torch.float32).reshape(n, b, 3),
                    torch.zeros(n, b, 4),
                ),
                dim=-1,
            ),
        )
        self.data.body_com_pose_b[..., 6] = 1.0
        self.calls: list[tuple[torch.Tensor, object]] = []

    def set_coms_index(self, *, coms, env_ids) -> None:
        self.calls.append((coms, env_ids))


def test_physx_profile_com_write_preserves_pose_and_uses_env_rows() -> None:
    asset = _CoMAsset(3, 2)
    env_ids = torch.tensor([0, 2])
    pos = torch.full((2, 2, 3), 7.0)

    PHYSX_PROFILE.set_body_com(asset, pos, env_ids)

    coms, written_ids = asset.calls[0]
    assert coms.shape == (2, 2, 7)
    assert torch.equal(coms[..., :3], pos)
    assert torch.equal(coms[..., 3:], asset.data.body_com_pose_b[env_ids][..., 3:])
    assert torch.equal(written_ids, env_ids)


def test_newton_profile_com_write_uses_positions_only() -> None:
    asset = _CoMAsset(3, 2)
    env_ids = torch.tensor([1])
    pos = torch.full((1, 2, 3), 5.0)

    NEWTON_PROFILE.set_body_com(asset, pos, env_ids)

    coms, written_ids = asset.calls[0]
    assert coms.shape == (1, 2, 3)
    assert torch.equal(coms, pos)
    assert torch.equal(written_ids, env_ids)
