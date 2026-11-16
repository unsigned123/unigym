"""NamedAction: identity, explicit kind, tensor shape and name-column agreement."""

from __future__ import annotations

import pytest
import torch

from unigym.runtime.actions import NamedAction


def _action(values=None, articulation="arm", group="body", joint_names=("joint",), kind="position"):
    """One valid NamedAction with overridable fields."""
    if values is None:
        values = torch.zeros(1, len(joint_names))
    return NamedAction(
        articulation=articulation,
        group=group,
        values=values,
        joint_names=joint_names,
        kind=kind,
    )


@pytest.mark.parametrize("kind", ["acceleration", "", "POSITION"])
def test_named_action_rejects_illegal_kind(kind):
    """An action kind must come from the closed legal target set."""
    with pytest.raises(ValueError, match="kind must be one of"):
        _action(kind=kind)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("articulation", "", "articulation must be a non-empty string"),
        ("group", "", "group must be a non-empty string"),
    ],
)
def test_named_action_requires_a_non_empty_identity(field, value, match):
    """The (articulation, group) identity is explicit and non-empty."""
    kwargs = {field: value}
    with pytest.raises(ValueError, match=match):
        _action(**kwargs)


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
        _action(values=values)


@pytest.mark.parametrize(
    ("joint_names", "match"),
    [
        ((), "joint_names must be non-empty"),
        ("joint", "joint_names must be a sequence of joint names"),
    ],
    ids=["empty", "string"],
)
def test_named_action_requires_a_joint_name_sequence(joint_names, match):
    """Joint names are a non-empty name sequence, never a bare string."""
    with pytest.raises(ValueError, match=match):
        _action(values=torch.zeros(1, 1), joint_names=joint_names)


@pytest.mark.parametrize(
    ("values", "joint_names", "match"),
    [
        (torch.zeros(1, 2), ("joint",), r"2 value columns vs 1 joint_names"),
        (torch.zeros(1, 1), ("left", "right"), r"1 value columns vs 2 joint_names"),
        (torch.zeros(1, 2), ("joint", "joint"), "duplicate joint names"),
    ],
    ids=["too-few-names", "too-many-names", "duplicate-names"],
)
def test_named_action_requires_one_unique_name_per_column(values, joint_names, match):
    """Every value column carries exactly one unique joint name."""
    with pytest.raises(ValueError, match=match):
        _action(values=values, joint_names=joint_names)


def test_named_action_zeros_carries_the_identity():
    """zeros() requires and preserves the (articulation, group) identity."""
    action = NamedAction.zeros(2, "go1", "legs", ("hip", "knee"), "position")

    assert action.articulation == "go1"
    assert action.group == "legs"
    assert action.joint_names == ("hip", "knee")
    assert tuple(action.values.shape) == (2, 2)
    assert action.num_envs == 2
    assert action.dim == 2
