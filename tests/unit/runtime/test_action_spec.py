"""OrderedActionSpec: declaration-ordered, overlap-free action templates."""

from __future__ import annotations

import pytest
import torch

from unigym.runtime.actions import (
    ActionDim,
    NamedAction,
    OrderedActionGroup,
    OrderedActionSpec,
)


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


# --------------------------------------------------------------------------- #
# Stage A2: OrderedActionSpec <-> NamedAction conversion
# --------------------------------------------------------------------------- #
JOINTS = ("hip_joint", "knee_joint", "ankle_joint")


def _two_robot_spec():
    return _spec(
        OrderedActionGroup("body", "go1", JOINTS, "position"),
        OrderedActionGroup("body", "go2", JOINTS, "position"),
    )


def test_unflatten_slices_groups_in_declaration_order():
    spec = _two_robot_spec()
    flat = torch.arange(24, dtype=torch.float64).reshape(4, spec.dim)

    named = spec.unflatten(flat)

    assert isinstance(named, tuple)
    assert [entry.articulation for entry in named] == ["go1", "go2"]
    assert [entry.group for entry in named] == ["body", "body"]
    assert [entry.joint_names for entry in named] == [JOINTS, JOINTS]
    assert [entry.kind for entry in named] == ["position", "position"]
    assert torch.equal(named[0].values, flat[:, :3])
    assert torch.equal(named[1].values, flat[:, 3:])
    assert named[0].values.dtype == torch.float64
    assert named[1].values.dtype == torch.float64


def test_flat_to_named_to_flat_round_trip_is_bit_identical():
    spec = _two_robot_spec()
    flat = torch.randn(4, spec.dim, dtype=torch.float64)

    assert torch.equal(spec.flatten(spec.unflatten(flat)), flat)


def test_flatten_is_order_insensitive():
    spec = _two_robot_spec()
    flat = torch.randn(3, spec.dim)
    named = spec.unflatten(flat)

    assert torch.equal(spec.flatten((named[1], named[0])), flat)


def test_unflatten_preserves_dtype_and_device():
    spec = _two_robot_spec()
    flat = torch.zeros(2, spec.dim, dtype=torch.float64, device="meta")

    named = spec.unflatten(flat)

    assert all(entry.values.dtype == torch.float64 for entry in named)
    assert all(entry.values.device.type == "meta" for entry in named)


def test_flatten_rejects_missing_identity():
    spec = _two_robot_spec()
    named = spec.unflatten(torch.zeros(2, spec.dim))

    with pytest.raises(ValueError, match="unaddressed"):
        spec.flatten((named[0],))


def test_flatten_rejects_extra_identity():
    spec = _two_robot_spec()
    named = spec.unflatten(torch.zeros(2, spec.dim))
    extra = NamedAction("go3", "body", torch.zeros(2, 3), JOINTS, "position")

    with pytest.raises(ValueError, match="not declared"):
        spec.flatten((*named, extra))


def test_flatten_rejects_duplicate_identity():
    spec = _two_robot_spec()
    named = spec.unflatten(torch.zeros(2, spec.dim))

    with pytest.raises(ValueError, match="more than one NamedAction"):
        spec.flatten((named[0], named[0], named[1]))


def test_flatten_rejects_non_named_action_entries():
    spec = _two_robot_spec()
    named = spec.unflatten(torch.zeros(2, spec.dim))

    with pytest.raises(TypeError, match="sequence of NamedAction"):
        spec.flatten((named[0], object(), named[1]))


def test_flatten_accepts_joint_permutation_and_reorders_columns():
    spec = _spec(OrderedActionGroup("body", "go1", ("hip", "knee", "ankle"), "position"))
    entry = NamedAction(
        articulation="go1",
        group="body",
        values=torch.tensor([[30.0, 10.0, 20.0]]),
        joint_names=("ankle", "hip", "knee"),
        kind="position",
    )

    flat = spec.flatten((entry,))

    assert torch.equal(flat, torch.tensor([[10.0, 20.0, 30.0]]))


def test_flatten_rejects_wrong_joint_name():
    spec = _spec(OrderedActionGroup("body", "go1", ("hip", "knee", "ankle"), "position"))
    entry = NamedAction(
        articulation="go1",
        group="body",
        values=torch.zeros(1, 3),
        joint_names=("hip", "knee", "toe"),
        kind="position",
    )

    with pytest.raises(ValueError, match="permutation"):
        spec.flatten((entry,))


def test_flatten_rejects_wrong_kind():
    spec = _two_robot_spec()
    named = spec.unflatten(torch.zeros(2, spec.dim))
    bad = NamedAction("go2", "body", torch.zeros(2, 3), JOINTS, "velocity")

    with pytest.raises(ValueError, match="template expects target"):
        spec.flatten((named[0], bad))


def test_flatten_rejects_mixed_num_envs():
    spec = _two_robot_spec()
    named = spec.unflatten(torch.zeros(2, spec.dim))
    bad = NamedAction("go2", "body", torch.zeros(3, 3), JOINTS, "position")

    with pytest.raises(ValueError, match="env rows"):
        spec.flatten((named[0], bad))


def test_flatten_rejects_mixed_dtype():
    spec = _two_robot_spec()
    named = spec.unflatten(torch.zeros(2, spec.dim))
    bad = NamedAction("go2", "body", torch.zeros(2, 3, dtype=torch.float64), JOINTS, "position")

    with pytest.raises(ValueError, match="mixed dtypes"):
        spec.flatten((named[0], bad))


def test_flatten_rejects_mixed_device():
    spec = _two_robot_spec()
    named = spec.unflatten(torch.zeros(2, spec.dim))
    bad = NamedAction("go2", "body", torch.zeros(2, 3, device="meta"), JOINTS, "position")

    with pytest.raises(ValueError, match="mixed devices"):
        spec.flatten((named[0], bad))


@pytest.mark.parametrize(
    "flat",
    [torch.zeros(3), torch.zeros(2, 3, 1)],
    ids=["rank-1", "rank-3"],
)
def test_unflatten_rejects_non_matrix_shapes(flat):
    spec = _two_robot_spec()

    with pytest.raises(ValueError, match=r"expects a \(N, dim\) tensor"):
        spec.unflatten(flat)


def test_unflatten_rejects_wrong_column_count():
    spec = _two_robot_spec()

    with pytest.raises(ValueError, match="action columns"):
        spec.unflatten(torch.zeros(2, spec.dim + 1))
