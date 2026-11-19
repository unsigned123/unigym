"""Shared IsaacLab 3 scene ops identity, validation and offline limits."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from unigym.backends.shared.isaaclab3_common.scene import (
    IsaacLabSceneOps,
    sim_joint_limits,
)
from unigym.runtime.scene import build_scene_ops

_JOINTS = ["hip", "knee"]


@dataclass
class _ArticulationData:
    joint_limits_sim: torch.Tensor | None = None
    joint_pos_limits: torch.Tensor | None = None


def test_sim_joint_limits_prefers_joint_limits_sim():
    """The 3.x joint_limits_sim face wins when both sources are present."""
    data = _ArticulationData(
        joint_limits_sim=torch.tensor([[-1.0, 1.0], [0.0, 2.0]]),
        joint_pos_limits=torch.tensor([[-9.0, 9.0], [-9.0, 9.0]]),
    )

    assert sim_joint_limits(data, _JOINTS) == {"hip": (-1.0, 1.0), "knee": (0.0, 2.0)}


def test_sim_joint_limits_falls_back_to_joint_pos_limits():
    """The 2.3.2 joint_pos_limits face is used when the 3.x face is absent."""
    data = _ArticulationData(joint_pos_limits=torch.tensor([[-1.0, 1.0], [0.0, 2.0]]))

    assert sim_joint_limits(data, _JOINTS) == {"hip": (-1.0, 1.0), "knee": (0.0, 2.0)}


@pytest.mark.parametrize("shape", [(3, 2), (2, 3)], ids=["extra-row", "extra-column"])
def test_sim_joint_limits_shape_mismatch_returns_none(shape):
    """A tensor that does not resolve to one row per joint faces as null."""
    data = _ArticulationData(joint_limits_sim=torch.zeros(shape))

    assert sim_joint_limits(data, _JOINTS) is None


def test_sim_joint_limits_unbounded_values_become_per_joint_none():
    """PhysX unbounded sentinels null only the affected joint."""
    data = _ArticulationData(joint_limits_sim=torch.tensor([[-1e9, 1e9], [-1.0, 1.0]]))

    assert sim_joint_limits(data, _JOINTS) == {"hip": None, "knee": (-1.0, 1.0)}


@pytest.mark.parametrize("data", [None, _ArticulationData()], ids=["no-data", "no-limits"])
def test_sim_joint_limits_absent_data_returns_none(data):
    """No data object or no limit attribute both keep the honest null face."""
    assert sim_joint_limits(data, _JOINTS) is None


def test_shared_scene_ops_backend_is_empty() -> None:
    """The shared library owns no engine identity."""
    assert IsaacLabSceneOps.backend == ""


class _EnvCarrier:
    """Minimal env carrier: construction only validates ``robot_name``."""


@pytest.mark.parametrize("robot_name", [None, "", "   ", " g1 "])
def test_scene_ops_constructor_requires_explicit_robot_name(robot_name):
    """A missing or empty robot name fails fast instead of falling back to G1."""
    with pytest.raises(ValueError, match="robot_name"):
        IsaacLabSceneOps(_EnvCarrier(), robot_name=robot_name)


def test_scene_ops_constructor_accepts_explicit_robot_name():
    """A non-empty caller-owned name is stored verbatim."""
    ops = IsaacLabSceneOps(_EnvCarrier(), robot_name="robot")

    assert ops.robot_name == "robot"


@pytest.mark.parametrize("robot_name", [None, "", "   ", " g1 "])
def test_build_scene_ops_requires_explicit_robot_name(robot_name):
    """The factory validates the name before any backend lookup."""
    with pytest.raises(ValueError, match="robot_name"):
        build_scene_ops(_EnvCarrier(), "", robot_name=robot_name)


def test_build_scene_ops_rejects_the_empty_shared_backend_identity():
    """An empty backend identity is not canonical; the error teaches the choices."""
    with pytest.raises(ValueError, match="unknown backend"):
        build_scene_ops(_EnvCarrier(), "", robot_name="robot")
