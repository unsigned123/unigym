"""Contact group prim-path construction, independent of Kit."""

from __future__ import annotations

import pytest

from unigym.backends.isaaclab2.contacts import (
    entry_prim_path,
    filter_prim_exprs,
    general_field_names,
    general_prim_paths,
    sensor_prim_path,
    split_scope,
)
from unigym.contacts.base import ContactGroupSpec, ENV_PREFIX_SENTINEL


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("torso", (None, "torso")),
        ("left_hand/finger_1", ("left_hand", "finger_1")),
        ("left_hand/palm/finger_1", ("left_hand/palm", "finger_1")),
    ],
)
def test_split_scope_separates_nested_prim_scope(entry, expected):
    """A slash entry yields its Robot-relative scope and last-segment body."""
    assert split_scope(entry) == expected


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("torso", f"{ENV_PREFIX_SENTINEL}/Robot/(torso)"),
        ("left_hand/finger_1", f"{ENV_PREFIX_SENTINEL}/Robot/left_hand/finger_1"),
    ],
)
def test_entry_prim_path_anchors_one_sensor_body(entry, expected):
    """One sensor entry becomes a Robot-anchored prim path at its scope."""
    assert entry_prim_path(entry) == expected


def test_general_prim_paths_buckets_scopes_in_first_appearance_order():
    """A general group gets one union path per scope, in first-appearance order."""
    entries = ["left/a", "base", "right/b", "left/c"]

    assert general_prim_paths(entries) == [
        f"{ENV_PREFIX_SENTINEL}/Robot/left/(a|c)",
        f"{ENV_PREFIX_SENTINEL}/Robot/(base)",
        f"{ENV_PREFIX_SENTINEL}/Robot/right/(b)",
    ]


def test_general_field_names_suffixes_each_scope_bucket():
    """General scope buckets yield the bare name then __s1, __s2, ..."""
    assert general_field_names("feet", ["left/a", "base", "right/b"]) == [
        "feet",
        "feet__s1",
        "feet__s2",
    ]


def test_sensor_prim_path_builds_flat_union_from_spec():
    """A scope-less group unions its declared sensor bodies into one regex."""
    spec = ContactGroupSpec(name="feet", sensor_bodies=["left_foot", "right_foot"], frame="world")

    assert sensor_prim_path(spec) == f"{ENV_PREFIX_SENTINEL}/Robot/(left_foot|right_foot)"


def test_sensor_prim_path_rejects_scoped_entry():
    """A nested-scope entry cannot be represented by a flat union path."""
    spec = ContactGroupSpec(name="hands", sensor_bodies=["left_hand/finger_1"], frame="world")

    with pytest.raises(ValueError, match="scoped sensor entry"):
        sensor_prim_path(spec)


@pytest.mark.parametrize(
    ("filter_bodies", "expected"),
    [
        (["foot"], [f"{ENV_PREFIX_SENTINEL}/Robot/foot"]),
        (["/World/Table"], ["/World/Table"]),
        ([f"{ENV_PREFIX_SENTINEL}/Prop_.*"], [f"{ENV_PREFIX_SENTINEL}/Prop_.*"]),
        (
            ["left_foot", "/World/Table", f"{ENV_PREFIX_SENTINEL}/Prop_.*"],
            [
                f"{ENV_PREFIX_SENTINEL}/Robot/left_foot",
                "/World/Table",
                f"{ENV_PREFIX_SENTINEL}/Prop_.*",
            ],
        ),
    ],
    ids=["bare-robot-entry", "explicit-prim-path", "env-prefix-pass-through", "mixed-order"],
)
def test_filter_prim_exprs_prefixes_bare_entries_only(filter_bodies, expected):
    """Only bare filter bodies get the Robot prefix; explicit paths pass through."""
    spec = ContactGroupSpec(
        name="feet_props",
        sensor_bodies=["left_foot"],
        filter_bodies=filter_bodies,
        frame="world",
    )

    assert filter_prim_exprs(spec) == expected
