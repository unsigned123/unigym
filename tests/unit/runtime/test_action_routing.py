"""Joint-addressed action routing over an opaque physical control table."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from unigym.runtime.action_routing import JointActionIndex, JointSlot, RoutedColumn
from unigym.runtime.actions import NamedAction


@dataclass(frozen=True)
class _Control:
    """Minimal ControlRuntime-like row: only the attributes the helper reads."""

    name: str
    articulation: str
    joint_names: tuple[str, ...]
    target: str = "position"
    required: bool = True
    prop: bool = False


def _action(
    articulation,
    joint_names,
    values=None,
    *,
    kind="position",
):
    if values is None:
        values = torch.zeros(1, len(joint_names))
    return NamedAction(
        articulation=articulation,
        values=values,
        joint_names=tuple(joint_names),
        kind=kind,
    )


def _index(*controls, label="test"):
    return JointActionIndex.from_controls(controls, label=label)


def test_from_controls_rejects_duplicate_articulation_joint_and_names_both_rows():
    controls = (
        _Control("arm", "g1", ("shoulder",)),
        _Control("aux", "g1", ("shoulder",)),
    )

    with pytest.raises(ValueError) as excinfo:
        _index(*controls, label="mujoco")

    message = str(excinfo.value)
    assert "duplicate joint slot" in message
    assert "'g1'" in message
    assert "'shoulder'" in message
    assert "'arm'" in message
    assert "'aux'" in message


def test_from_controls_accepts_a_mapping_control_table():
    controls = {
        ("g1", "body"): _Control("body", "g1", ("hip",)),
        ("g1", "hand"): _Control("hand", "g1", ("finger",)),
    }

    index = JointActionIndex.from_controls(controls, label="isaaclab2")
    routed = index.route((_action("g1", ("hip", "finger")),), 1)

    assert [column.slot.owner.name for column in routed] == ["body", "hand"]


def test_one_entry_spans_two_controls_of_the_same_articulation():
    index = _index(
        _Control("body", "g1", ("hip", "knee")),
        _Control("hand", "g1", ("finger",)),
    )
    entry = _action("g1", ("hip", "knee", "finger"))

    routed = index.route((entry,), 1)

    assert len(routed) == 3
    assert all(isinstance(column, RoutedColumn) for column in routed)
    assert [column.slot.owner.name for column in routed] == ["body", "body", "hand"]
    assert [column.slot.column for column in routed] == [0, 1, 0]
    assert [column.source_index for column in routed] == [0, 1, 2]
    assert routed[0].entry is entry


def test_same_control_label_and_joint_names_on_two_articulations_stay_separate():
    index = _index(
        _Control("body", "go1", ("hip", "knee")),
        _Control("body", "go2", ("hip", "knee")),
    )
    go1 = _action("go1", ("hip", "knee"), torch.tensor([[10.0, 20.0]]))
    go2 = _action("go2", ("hip", "knee"), torch.tensor([[30.0, 40.0]]))

    routed = index.route((go2, go1), 1)

    assert [column.slot.articulation for column in routed] == ["go2", "go2", "go1", "go1"]
    assert [column.entry is go2 for column in routed] == [True, True, False, False]
    assert routed[0].slot.owner is not routed[2].slot.owner


def test_unknown_joint_names_articulation_and_accepted_joints():
    index = _index(_Control("body", "g1", ("hip", "knee")))

    with pytest.raises(ValueError) as excinfo:
        index.route((_action("g1", ("toe",)),), 1)

    message = str(excinfo.value)
    assert "'g1'" in message
    assert "'toe'" in message
    assert "'hip'" in message
    assert "'knee'" in message


def test_unknown_articulation_lists_accepted_articulations():
    index = _index(_Control("body", "g1", ("hip",)))

    with pytest.raises(ValueError) as excinfo:
        index.route((_action("go9", ("hip",)),), 1)

    message = str(excinfo.value)
    assert "'go9'" in message
    assert "accepted articulations" in message
    assert "'g1'" in message


def test_wrong_kind_names_articulation_joint_and_both_kinds():
    index = _index(_Control("body", "g1", ("hip",), target="position"))

    with pytest.raises(ValueError) as excinfo:
        index.route((_action("g1", ("hip",), kind="velocity"),), 1)

    message = str(excinfo.value)
    assert "'g1'" in message
    assert "'hip'" in message
    assert "'velocity'" in message
    assert "'position'" in message


def test_duplicate_addressed_joint_fails_naming_articulation_and_joint():
    index = _index(_Control("body", "g1", ("hip",)), _Control("aux", "g1", ("knee",)))
    first = _action("g1", ("hip",))
    second = _action("g1", ("hip",))

    with pytest.raises(ValueError, match="more than once") as excinfo:
        index.route((first, second), 1)

    message = str(excinfo.value)
    assert "'g1'" in message
    assert "'hip'" in message


def test_missing_required_joints_name_articulation_and_joints():
    index = _index(
        _Control("body", "g1", ("hip", "knee")),
        _Control("hand", "g1", ("finger",)),
    )

    with pytest.raises(ValueError, match="required joints are unaddressed") as excinfo:
        index.route((_action("g1", ("hip", "knee")),), 1)

    message = str(excinfo.value)
    assert "'g1'" in message
    assert "'finger'" in message


def test_missing_required_groups_by_articulation():
    index = _index(
        _Control("body", "g1", ("hip",)),
        _Control("body", "go2", ("knee",)),
    )

    missing = index.missing_required(())

    assert missing == {"g1": ("hip",), "go2": ("knee",)}


def test_missing_required_is_empty_after_full_coverage():
    index = _index(_Control("body", "g1", ("hip", "knee")))

    routed = index.route((_action("g1", ("knee", "hip")),), 1)

    assert index.missing_required(routed) == {}


def test_optional_prop_joint_may_be_omitted():
    index = _index(
        _Control("body", "g1", ("hip",)),
        _Control("drawer", "g1", ("drawer_joint",), required=False, prop=True),
    )

    routed = index.route((_action("g1", ("hip",)),), 1)

    assert [column.slot.joint for column in routed] == ["hip"]
    assert index.missing_required(routed) == {}


def test_optional_prop_joint_routes_with_its_prop_slot():
    index = _index(
        _Control("body", "g1", ("hip",)),
        _Control("drawer", "g1", ("drawer_joint",), required=False, prop=True),
    )

    routed = index.route((_action("g1", ("drawer_joint", "hip")),), 1)

    drawer, body = routed
    assert drawer.slot.prop is True
    assert drawer.slot.owner.name == "drawer"
    assert body.slot.prop is False


def test_num_envs_mismatch_is_rejected():
    index = _index(_Control("body", "g1", ("hip",)))
    entry = _action("g1", ("hip",), torch.zeros(2, 1))

    with pytest.raises(ValueError) as excinfo:
        index.route((entry,), 1)

    message = str(excinfo.value)
    assert "2 env rows" in message
    assert "expects 1" in message


@pytest.mark.parametrize("actions", ["hip", b"hip", 3], ids=["str", "bytes", "int"])
def test_route_rejects_non_sequence_actions(actions):
    index = _index(_Control("body", "g1", ("hip",)))

    with pytest.raises(TypeError, match="sequence of NamedAction"):
        index.route(actions, 1)


def test_route_rejects_non_named_action_entries():
    index = _index(_Control("body", "g1", ("hip",)))

    with pytest.raises(TypeError, match="NamedAction"):
        index.route((_action("g1", ("hip",)), object()), 1)


def test_missing_required_rejects_non_routed_entries():
    index = _index(_Control("body", "g1", ("hip",)))

    with pytest.raises(TypeError, match="RoutedColumn"):
        index.missing_required((object(),))


def test_joint_slot_and_routed_column_are_frozen_value_objects():
    control = _Control("body", "g1", ("hip", "knee"))
    slot = JointSlot("g1", "hip", "position", control, 0, True, False)
    column = RoutedColumn(_action("g1", ("hip",)), 0, slot)

    assert (slot.articulation, slot.joint, slot.kind) == ("g1", "hip", "position")
    assert (column.source_index, column.slot) == (0, slot)
