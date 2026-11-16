"""NamedAction: joint-addressed identity, kind, shape and name-column agreement."""

from __future__ import annotations

import pytest
import torch

from unigym.runtime.actions import NamedAction


def _action(values=None, articulation="arm", joint_names=("joint",), kind="position"):
    """One valid NamedAction with overridable fields."""
    if values is None:
        values = torch.zeros(1, len(joint_names))
    return NamedAction(
        articulation=articulation,
        values=values,
        joint_names=joint_names,
        kind=kind,
    )


@pytest.mark.parametrize("kind", ["acceleration", "", "POSITION"])
def test_named_action_rejects_illegal_kind(kind):
    """An action kind must come from the closed legal target set."""
    with pytest.raises(ValueError, match="kind must be one of"):
        _action(kind=kind)


def test_named_action_has_no_group_field():
    """Action groups are authoring labels; the runtime value has no group."""
    assert "group" not in NamedAction.__dataclass_fields__

    with pytest.raises(TypeError):
        NamedAction(
            articulation="arm",
            group="body",
            values=torch.zeros(1, 1),
            joint_names=("joint",),
            kind="position",
        )


def test_named_action_requires_a_non_empty_articulation():
    """The articulation identity is explicit and non-empty."""
    with pytest.raises(ValueError, match="articulation must be a non-empty string"):
        _action(articulation="")


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


def test_named_action_zeros_carries_the_joint_identity():
    """zeros() requires and preserves the articulation + joint names."""
    action = NamedAction.zeros(2, "go1", ("hip", "knee"), "position")

    assert action.articulation == "go1"
    assert action.joint_names == ("hip", "knee")
    assert tuple(action.values.shape) == (2, 2)
    assert action.num_envs == 2
    assert action.dim == 2


def test_zeros_rejects_the_removed_group_keyword():
    """There is no compatibility group= keyword on zeros()."""
    with pytest.raises(TypeError):
        NamedAction.zeros(2, "go1", ("hip", "knee"), "position", group="legs")


def test_adapter_named_action_rejects_a_bare_joint_string():
    """A bare string must not become a tuple of single-character joint names."""
    from unigym.adapters.manual import ManualAdapter

    adapter = ManualAdapter()

    with pytest.raises(ValueError, match="not a bare string"):
        adapter.named_action(
            torch.zeros(1, 1),
            articulation="tiny",
            joint_names="hinge",
        )


def test_adapter_named_action_has_no_group_keyword():
    """Adapter.named_action is joint-addressed; group= is rejected."""
    from unigym.adapters.manual import ManualAdapter

    with pytest.raises(TypeError):
        ManualAdapter().named_action(
            torch.zeros(1, 1),
            articulation="tiny",
            group="tiny",
            joint_names=("hinge",),
        )


def test_no_production_code_reads_or_constructs_a_named_action_group():
    """Production routing/consumers must not use the removed runtime group field."""
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    pattern = re.compile(
        r"\b(?:entry|na|action)\.group\b|NamedAction\([^)]*\bgroup=",
        re.DOTALL,
    )
    hits = [
        str(path.relative_to(repo))
        for path in sorted((repo / "unigym").rglob("*.py"))
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert hits == []
