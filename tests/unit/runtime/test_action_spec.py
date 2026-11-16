"""OrderedActionSpec: declaration-ordered, overlap-free action templates."""

from __future__ import annotations

import pytest

from unigym.runtime.actions import ActionDim, OrderedActionGroup, OrderedActionSpec


def _group(name="body", articulation="g1", joints=("hip", "knee"), target="position"):
    return OrderedActionGroup(name=name, articulation=articulation, joints=joints, target=target)


def _spec(*groups):
    return OrderedActionSpec(groups=groups)


def test_spec_preserves_group_and_joint_order():
    """dim/dim_table flatten group order then joint order, one row per column."""
    spec = _spec(
        OrderedActionGroup("body", "go1", ("hip_joint", "knee_joint"), "position"),
        OrderedActionGroup("hand", "left_hand", ("finger_a",), "effort"),
        OrderedActionGroup("body", "go2", ("hip_joint", "knee_joint"), "velocity"),
    )

    assert spec.dim == 5
    table = spec.dim_table()
    assert all(isinstance(row, ActionDim) for row in table)
    assert [row.index for row in table] == [0, 1, 2, 3, 4]
    assert [row.articulation for row in table] == [
        "go1",
        "go1",
        "left_hand",
        "go2",
        "go2",
    ]
    assert [row.group for row in table] == [
        "body",
        "body",
        "hand",
        "body",
        "body",
    ]
    assert [row.joint for row in table] == [
        "hip_joint",
        "knee_joint",
        "finger_a",
        "hip_joint",
        "knee_joint",
    ]
    assert [row.target for row in table] == [
        "position",
        "position",
        "effort",
        "velocity",
        "velocity",
    ]


def test_same_joint_names_across_articulations_are_allowed():
    """Joint names are scoped by articulation, so go1/go2 may share names."""
    spec = _spec(
        OrderedActionGroup("legs", "go1", ("hip_joint", "knee_joint"), "position"),
        OrderedActionGroup("legs", "go2", ("hip_joint", "knee_joint"), "position"),
    )

    assert spec.dim == 4
    assert [(row.articulation, row.joint) for row in spec.dim_table()] == [
        ("go1", "hip_joint"),
        ("go1", "knee_joint"),
        ("go2", "hip_joint"),
        ("go2", "knee_joint"),
    ]


def test_group_stores_joints_as_tuple():
    """A list declaration is frozen into the tuple contract."""
    group = OrderedActionGroup("body", "g1", ["hip", "knee"], "position")

    assert group.joints == ("hip", "knee")
    assert isinstance(group.joints, tuple)


def test_duplicate_group_identity_is_rejected():
    """(articulation, name) identifies one action group."""
    with pytest.raises(ValueError, match="duplicate group identity"):
        _spec(
            _group(name="body", articulation="g1"),
            _group(name="body", articulation="g1"),
        )


def test_overlapping_joints_within_one_articulation_are_rejected():
    """Two groups of one articulation may not own the same joint."""
    with pytest.raises(ValueError, match="overlapping joints"):
        _spec(
            _group(name="body", joints=("hip", "knee")),
            _group(name="aux", joints=("knee", "ankle")),
        )


def test_spec_requires_at_least_one_group():
    with pytest.raises(ValueError, match="at least one group"):
        OrderedActionSpec(groups=())


def test_action_dim_accepts_valid_row():
    row = ActionDim(
        index=2,
        articulation="g1",
        group="hand",
        joint="finger_a",
        target="position",
    )

    assert (row.index, row.target) == (2, "position")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"index": -1}, "non-negative integer"),
        ({"index": "0"}, "non-negative integer"),
        ({"articulation": ""}, "articulation"),
        ({"group": ""}, "group"),
        ({"joint": ""}, "joint"),
        ({"target": "acceleration"}, "target must be one of"),
    ],
    ids=[
        "negative-index",
        "non-int-index",
        "empty-articulation",
        "empty-group",
        "empty-joint",
        "illegal-target",
    ],
)
def test_action_dim_rejects_invalid_fields(kwargs, match):
    fields = {
        "index": 0,
        "articulation": "g1",
        "group": "body",
        "joint": "hip",
        "target": "position",
    }
    fields.update(kwargs)

    with pytest.raises(ValueError, match=match):
        ActionDim(**fields)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"joints": ("hip", "hip")}, "duplicate joints"),
        ({"joints": ()}, "joints must be non-empty"),
        ({"joints": ("hip", "")}, "joint name"),
        ({"joints": "hip"}, "sequence"),
        ({"target": "acceleration"}, "target must be one of"),
        ({"name": ""}, "name must be a non-empty"),
        ({"articulation": ""}, "articulation must be a non-empty"),
    ],
    ids=[
        "duplicate-joint",
        "empty-joints",
        "empty-joint-name",
        "string-joints",
        "illegal-target",
        "empty-name",
        "empty-articulation",
    ],
)
def test_ordered_action_group_rejects_invalid_fields(kwargs, match):
    fields = {"name": "body", "articulation": "g1", "joints": ("hip",), "target": "position"}
    fields.update(kwargs)

    with pytest.raises(ValueError, match=match):
        OrderedActionGroup(**fields)


def test_str_renders_header_and_dimension_rows():
    """str(spec) is a readable table: header plus one row per dimension."""
    spec = _spec(
        OrderedActionGroup("body", "g1", ("hip_joint", "knee_joint"), "position"),
        OrderedActionGroup("hand", "g1", ("finger_a",), "effort"),
    )

    lines = str(spec).splitlines()
    assert lines[0].split() == ["idx", "articulation", "group", "joint", "target"]
    assert [line.split() for line in lines[1:]] == [
        ["0", "g1", "body", "hip_joint", "position"],
        ["1", "g1", "body", "knee_joint", "position"],
        ["2", "g1", "hand", "finger_a", "effort"],
    ]
    assert len(lines) == spec.dim + 1


def test_repr_is_a_concise_summary():
    spec = _spec(
        OrderedActionGroup("body", "g1", ("hip", "knee"), "position"),
        OrderedActionGroup("hand", "g1", ("finger",), "position"),
    )

    text = repr(spec)
    assert "OrderedActionSpec" in text
    assert "dim=3" in text
    assert "g1/body:2" in text
    assert "g1/hand:1" in text
