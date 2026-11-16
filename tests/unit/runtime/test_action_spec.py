"""OrderedActionSpec: declaration-ordered, overlap-free action templates."""

from __future__ import annotations

import pytest
import torch

from unigym.runtime.actions import (
    ActionDim,
    NamedAction,
    OrderedActionGroup,
    OrderedActionSpec,
    PolicyBinding,
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


# --------------------------------------------------------------------------- #
# Stage A3a: PolicyBinding + declaration mapping parsers
# --------------------------------------------------------------------------- #
def _group_mapping(**overrides):
    entry = {
        "name": "body",
        "articulation": "g1",
        "joints": ["hip", "knee"],
        "target": "position",
    }
    entry.update(overrides)
    return entry


def _binding_mapping(**overrides):
    entry = {"adapter": "sonic_g1", "groups": [["g1", "body"]]}
    entry.update(overrides)
    return entry


def test_policy_binding_normalizes_groups_to_tuple_of_tuples():
    binding = PolicyBinding(
        adapter="sonic_g1",
        groups=[["g1", "body"], ("g1", "hand")],
        control_dt=1,
        agent="sonic",
    )

    assert binding.groups == (("g1", "body"), ("g1", "hand"))
    assert all(isinstance(pair, tuple) for pair in binding.groups)
    assert binding.control_dt == 1.0
    assert isinstance(binding.control_dt, float)
    assert binding.adapter == "sonic_g1"
    assert binding.agent == "sonic"


def test_policy_binding_defaults_optional_fields_to_none():
    binding = PolicyBinding("sonic_g1", (("g1", "body"),))

    assert binding.control_dt is None
    assert binding.agent is None


def test_policy_binding_allows_same_group_name_on_different_articulations():
    binding = PolicyBinding("sonic_g1", (("g1", "body"), ("g2", "body")))

    assert binding.groups == (("g1", "body"), ("g2", "body"))


def test_policy_binding_rejects_duplicate_identity():
    with pytest.raises(ValueError, match="duplicate group identity"):
        PolicyBinding("sonic_g1", (("g1", "body"), ("g1", "body")))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"adapter": ""}, "adapter must be a non-empty"),
        ({"adapter": 3}, "adapter must be a non-empty"),
        ({"groups": ()}, "at least one"),
        ({"groups": "g1"}, "must be a non-empty sequence"),
        ({"groups": (("g1",),)}, "exactly 2 items"),
        ({"groups": (("g1", "body", "hand"),)}, "exactly 2 items"),
        ({"groups": (("g1", ""),)}, "group name must be a non-empty"),
        ({"groups": (("", "body"),)}, "group articulation must be a non-empty"),
        ({"groups": ((1, "body"),)}, "group articulation must be a non-empty"),
        ({"groups": (42,)}, "entries must be 2-item"),
        ({"control_dt": 0}, "control_dt must be a positive finite"),
        ({"control_dt": -0.02}, "control_dt must be a positive finite"),
        ({"control_dt": float("inf")}, "control_dt must be a positive finite"),
        ({"control_dt": float("nan")}, "control_dt must be a positive finite"),
        ({"control_dt": True}, "control_dt must be a positive finite"),
        ({"control_dt": "0.02"}, "control_dt must be a positive finite"),
        ({"agent": ""}, "agent must be a non-empty"),
        ({"agent": 5}, "agent must be a non-empty"),
    ],
    ids=[
        "empty-adapter",
        "non-text-adapter",
        "empty-groups",
        "string-groups",
        "short-pair",
        "long-pair",
        "empty-group-name",
        "empty-articulation",
        "non-text-articulation",
        "non-sequence-pair",
        "zero-dt",
        "negative-dt",
        "infinite-dt",
        "nan-dt",
        "bool-dt",
        "string-dt",
        "empty-agent",
        "non-text-agent",
    ],
)
def test_policy_binding_rejects_invalid_fields(kwargs, match):
    fields = {"adapter": "sonic_g1", "groups": (("g1", "body"),)}
    fields.update(kwargs)

    with pytest.raises(ValueError, match=match):
        PolicyBinding(**fields)


def test_group_from_mapping_builds_and_freezes_joints():
    group = OrderedActionGroup.from_mapping(_group_mapping(joints=["hip", "knee"]))

    assert group == OrderedActionGroup("body", "g1", ("hip", "knee"), "position")
    assert isinstance(group.joints, tuple)


def test_group_from_mapping_rejects_unknown_keys():
    with pytest.raises(ValueError, match="does not accept"):
        OrderedActionGroup.from_mapping(_group_mapping(extra=1))


def test_group_from_mapping_rejects_joint_set_selector():
    entry = _group_mapping()
    entry["joint_set"] = "g1_body"

    with pytest.raises(ValueError, match="joint_set"):
        OrderedActionGroup.from_mapping(entry)


def test_group_from_mapping_requires_all_keys():
    entry = _group_mapping()
    del entry["target"]

    with pytest.raises(ValueError, match="missing required"):
        OrderedActionGroup.from_mapping(entry)


def test_group_from_mapping_requires_explicit_joints():
    with pytest.raises(ValueError, match="explicit"):
        OrderedActionGroup.from_mapping(_group_mapping(joints="hip"))


def test_group_from_mapping_requires_mapping():
    with pytest.raises(TypeError, match="expects a mapping"):
        OrderedActionGroup.from_mapping(("body", "g1", ("hip",), "position"))


def test_spec_from_mapping_accepts_mappings_and_groups_in_declared_order():
    spec = OrderedActionSpec.from_mapping(
        [
            _group_mapping(name="body", articulation="g1", joints=["hip"]),
            OrderedActionGroup("hand", "g1", ("finger",), "effort"),
            _group_mapping(name="body", articulation="g2", joints=["hip"]),
        ]
    )

    assert spec.dim == 3
    assert [(group.articulation, group.name) for group in spec.groups] == [
        ("g1", "body"),
        ("g1", "hand"),
        ("g2", "body"),
    ]
    assert spec.groups[0].joints == ("hip",)


def test_spec_from_mapping_rejects_empty_sequence():
    with pytest.raises(ValueError, match="at least one group"):
        OrderedActionSpec.from_mapping([])


@pytest.mark.parametrize(
    "entries",
    ["body", b"body", {"name": "body"}],
    ids=["str", "bytes", "mapping"],
)
def test_spec_from_mapping_rejects_strings_bytes_and_mappings(entries):
    with pytest.raises(TypeError, match="expects a sequence"):
        OrderedActionSpec.from_mapping(entries)


def test_spec_from_mapping_rejects_non_sequence():
    with pytest.raises(TypeError, match="expects a sequence"):
        OrderedActionSpec.from_mapping(42)


def test_spec_from_mapping_rejects_unknown_entry_type():
    with pytest.raises(TypeError, match="OrderedActionGroup or mapping"):
        OrderedActionSpec.from_mapping([_group_mapping(), object()])


def test_spec_from_mapping_propagates_group_validation():
    with pytest.raises(ValueError, match="duplicate group identity"):
        OrderedActionSpec.from_mapping([_group_mapping(), _group_mapping()])


def test_binding_from_mapping_normalizes_groups():
    binding = PolicyBinding.from_mapping(
        _binding_mapping(
            groups=[["g1", "body"], ["g1", "hand"]],
            control_dt=0.02,
            agent="sonic",
        )
    )

    assert binding.groups == (("g1", "body"), ("g1", "hand"))
    assert binding.control_dt == 0.02
    assert binding.agent == "sonic"


def test_binding_from_mapping_defaults_optional_fields():
    binding = PolicyBinding.from_mapping(_binding_mapping())

    assert (binding.control_dt, binding.agent) == (None, None)


def test_binding_from_mapping_rejects_unknown_keys():
    with pytest.raises(ValueError, match="does not accept"):
        PolicyBinding.from_mapping(_binding_mapping(extra=1))


@pytest.mark.parametrize("key", ["adapter", "groups"])
def test_binding_from_mapping_requires_keys(key):
    entry = _binding_mapping()
    del entry[key]

    with pytest.raises(ValueError, match="missing required"):
        PolicyBinding.from_mapping(entry)


@pytest.mark.parametrize("groups", ["g1", 3], ids=["str", "int"])
def test_binding_from_mapping_rejects_non_sequence_groups(groups):
    with pytest.raises(ValueError, match="groups must be a non-empty list"):
        PolicyBinding.from_mapping(_binding_mapping(groups=groups))


def test_binding_from_mapping_rejects_empty_groups():
    with pytest.raises(ValueError, match="at least one"):
        PolicyBinding.from_mapping(_binding_mapping(groups=[]))


@pytest.mark.parametrize(
    ("groups", "match"),
    [
        ([["g1"]], "exactly 2 items"),
        ([["g1", "body", "x"]], "exactly 2 items"),
        ([["g1", ""]], "group name must be a non-empty"),
    ],
    ids=["short-pair", "long-pair", "empty-group-name"],
)
def test_binding_from_mapping_rejects_bad_pairs(groups, match):
    with pytest.raises(ValueError, match=match):
        PolicyBinding.from_mapping(_binding_mapping(groups=groups))


def test_binding_from_mapping_requires_mapping():
    with pytest.raises(TypeError, match="expects a mapping"):
        PolicyBinding.from_mapping(("sonic_g1", [["g1", "body"]]))


def test_policy_binding_is_independent_of_run_config():
    import unigym.runtime.actions as actions

    binding = PolicyBinding.from_mapping({"adapter": "sonic_g1", "groups": [("g1", "body")]})

    assert binding.groups == (("g1", "body"),)
    assert "ControlGroup" not in vars(actions)
