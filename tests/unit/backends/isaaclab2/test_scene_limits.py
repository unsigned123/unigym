"""sim_joint_limits source chain and honest-null boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from unigym.backends.isaaclab2.scene import sim_joint_limits


@dataclass
class _ArticulationData:
    joint_limits_sim: torch.Tensor | None = None
    joint_pos_limits: torch.Tensor | None = None


_JOINTS = ["hip", "knee"]


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
