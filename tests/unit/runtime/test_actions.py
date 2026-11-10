"""NamedAction: explicit kind, tensor shape and name-column agreement."""

from __future__ import annotations

import pytest
import torch

from unigym.runtime.actions import NamedAction


@pytest.mark.parametrize("kind", ["acceleration", "", "POSITION"])
def test_named_action_rejects_illegal_kind(kind):
    """An action kind must come from the closed legal target set."""
    with pytest.raises(ValueError, match="kind must be one of"):
        NamedAction(torch.zeros(1, 1), ["joint"], kind)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ([[1.0]], "must be a torch tensor"),
        (torch.zeros(3), r"must be \(N, dim\)"),
    ],
    ids=["list", "rank-1"],
)
def test_named_action_rejects_non_tensor_or_rank_mismatched_values(values, match):
    """Values must be a torch tensor with an explicit batch and column axis."""
    with pytest.raises(ValueError, match=match):
        NamedAction(values, ["joint"], "position")


@pytest.mark.parametrize(
    ("values", "joint_names", "match"),
    [
        (torch.zeros(1, 2), ["joint"], r"2 value columns vs 1 joint_names"),
        (torch.zeros(1, 1), ["left", "right"], r"1 value columns vs 2 joint_names"),
        (torch.zeros(1, 2), ["joint", "joint"], "duplicate joint names"),
    ],
    ids=["too-few-names", "too-many-names", "duplicate-names"],
)
def test_named_action_requires_one_unique_name_per_column(values, joint_names, match):
    """Every value column carries exactly one unique joint name."""
    with pytest.raises(ValueError, match=match):
        NamedAction(values, joint_names, "position")
