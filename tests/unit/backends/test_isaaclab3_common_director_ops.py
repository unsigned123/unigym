"""Shared IsaacLab 3 director ops identity, bind gating and weld trait query."""

from __future__ import annotations

import pytest
import torch
from types import SimpleNamespace

from unigym.backends.registry import Trait, backend_info
from unigym.backends.shared.isaaclab3_common.director.gripper import IsaacLabGripper
from unigym.backends.shared.isaaclab3_common.director_ops import IsaacLabDirectorOps


class _EnvCarrier:
    """Minimal env carrier: ops construction touches nothing else."""

    device = "cpu"
    prop_manager = None
    robot = None


class _CanonicalOps(IsaacLabDirectorOps):
    """Backend identity subclass (what the physx/newton packages provide)."""

    backend = "isaaclab3_physx"


def test_shared_director_ops_backend_is_empty() -> None:
    ops = IsaacLabDirectorOps(_EnvCarrier())

    assert ops.backend == ""


def test_bind_none_mounts_no_gripper() -> None:
    """``bind(None)`` must not consult the backend identity or mount anything."""
    ops = IsaacLabDirectorOps(_EnvCarrier())

    assert ops.bind(None) is None
    assert ops.gripper is None


def test_bind_gripper_without_backend_identity_fails_with_a_teaching_error() -> None:
    """Mounting a gripper needs the canonical identity the shared class lacks."""
    ops = IsaacLabDirectorOps(_EnvCarrier())

    with pytest.raises(NotImplementedError, match="no backend identity"):
        ops.bind({"name": "gripper"})


def test_has_weld_is_a_static_boolean_after_the_subclass_sets_the_identity() -> None:
    """The trait query is engine-free once the backend identity is present."""
    ops = _CanonicalOps(_EnvCarrier())

    result = ops.has_weld()

    assert ops.backend == "isaaclab3_physx"
    assert isinstance(result, bool)
    assert result == backend_info("isaaclab3_physx").has(Trait.DIRECTOR_ATTACH_WELD)
    assert result is False


class _KeywordOnlyAsset:
    """IsaacLab 3 asset surface whose *_index writes are keyword-only."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...]]] = []
        self.data = SimpleNamespace(
            root_pos_w=torch.zeros(1, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

    def write_root_link_pose_to_sim_index(self, *, root_pose, env_ids) -> None:
        self.calls.append(("pose", tuple(env_ids)))

    def write_root_link_velocity_to_sim_index(self, *, root_velocity, env_ids) -> None:
        self.calls.append(("vel", tuple(env_ids)))

    def write_joint_position_to_sim_index(self, *, position, joint_ids, env_ids) -> None:
        self.calls.append(("joint_pos", tuple(env_ids)))

    def write_joint_velocity_to_sim_index(self, *, velocity, joint_ids, env_ids) -> None:
        self.calls.append(("joint_vel", tuple(env_ids)))


class _GripperEnv:
    """Minimal env carrier for the kinematic gripper runtime."""

    device = "cpu"

    def __init__(self, asset) -> None:
        self.scene = {"director_gripper": asset}


def test_shared_director_writes_use_keyword_only_index_methods() -> None:
    """il3 *_index writes are keyword-only; positional calls would TypeError."""
    ops = _CanonicalOps(_EnvCarrier())
    asset = _KeywordOnlyAsset()
    ops._assets["prop"] = asset

    ops.teleport("prop", None, None, True, {})
    ops.setvel("prop", (0.1, 0.0, 0.0), None)
    ops.halt("prop")

    assert ("pose", (0,)) in asset.calls
    assert ("vel", (0,)) in asset.calls


def test_shared_gripper_set_pose_uses_keyword_only_index_write() -> None:
    asset = _KeywordOnlyAsset()
    env = _GripperEnv(asset)
    gripper = IsaacLabGripper(env, "gripper")

    gripper.set_pose(env, (1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))

    assert ("pose", (0,)) in asset.calls
