"""Contact group prim-path construction, independent of Kit."""

from __future__ import annotations

import pytest

from unigym.backends.isaaclab2.contacts import (
    IsaacLabContactBackend,
    entry_prim_path,
    filter_prim_exprs,
    general_field_names,
    general_prim_paths,
    resolve_all_articulation,
    resolve_articulation_leaf,
    sensor_prim_path,
    split_scope,
)
from unigym.backends.isaaclab2.naming import (
    articulation_prim_leaf,
    articulation_scene_field,
)
from unigym.contacts.base import ContactGroupSpec, ENV_PREFIX_SENTINEL
from unigym.contacts.discovery import ContactCandidateError


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


def test_scoped_paths_use_the_owned_articulation_leaf():
    """Every derived prim path is anchored at the owning articulation leaf."""
    spec = ContactGroupSpec(
        name="feet_props",
        articulation="g1",
        sensor_bodies=["left_ankle_roll_link"],
        filter_bodies=["left_foot", "/World/Table"],
        frame="world",
    )

    assert entry_prim_path("left_ankle_roll_link", robot_leaf="Robot_g1") == (
        f"{ENV_PREFIX_SENTINEL}/Robot_g1/(left_ankle_roll_link)"
    )
    assert general_prim_paths(["left/a", "base"], robot_leaf="Robot_g1") == [
        f"{ENV_PREFIX_SENTINEL}/Robot_g1/left/(a)",
        f"{ENV_PREFIX_SENTINEL}/Robot_g1/(base)",
    ]
    assert sensor_prim_path(spec, robot_leaf="Robot_g1") == (
        f"{ENV_PREFIX_SENTINEL}/Robot_g1/(left_ankle_roll_link)"
    )
    assert filter_prim_exprs(spec, robot_leaf="Robot_g1") == [
        f"{ENV_PREFIX_SENTINEL}/Robot_g1/left_foot",
        "/World/Table",
    ]


@pytest.mark.parametrize(
    ("name", "index", "field", "leaf"),
    [
        ("g1", 0, "robot", "Robot"),
        ("g1", 1, "robot_g1", "Robot_g1"),
        ("arm", 2, "robot_arm", "Robot_arm"),
    ],
)
def test_articulation_naming_rule_is_shared(name, index, field, leaf):
    """The first articulation keeps the historical robot/Robot names."""
    assert articulation_scene_field(name, index) == field
    assert articulation_prim_leaf(name, index) == leaf


def test_resolve_articulation_leaf_uses_the_declaration_order():
    """An explicit owner maps to its slot leaf; an implicit owner is Robot."""
    implicit = ContactGroupSpec(name="feet", sensor_bodies=["foot"], frame="world")
    explicit = ContactGroupSpec(
        name="feet", articulation="g1", sensor_bodies=["foot"], frame="world"
    )

    assert resolve_articulation_leaf(implicit, ("cartpole", "g1")) == "Robot"
    assert resolve_articulation_leaf(explicit, ("cartpole", "g1")) == "Robot_g1"


def test_resolve_articulation_leaf_rejects_an_unknown_owner():
    """A scoped group whose owner is not in the run order fails with the set."""
    spec = ContactGroupSpec(
        name="feet", articulation="ghost", sensor_bodies=["foot"], frame="world"
    )

    with pytest.raises(ValueError, match="not declared by the run") as err:
        resolve_articulation_leaf(spec, ("cartpole", "g1"))

    assert "ghost" in str(err.value)


def test_all_with_one_asset_keeps_the_implicit_single_articulation_owner():
    """A single all-mode asset needs no explicit owner (single-robot config)."""
    spec = ContactGroupSpec(name="all", sensor_bodies=["<all>"], frame="world")

    assert resolve_all_articulation(spec, {"cartpole": "/tmp/cartpole.usd"}) == "cartpole"


def test_all_with_multiple_assets_requires_the_explicit_owner():
    """Several all-mode assets never union into one inventory by default."""
    spec = ContactGroupSpec(name="all", sensor_bodies=["<all>"], frame="world")

    with pytest.raises(ValueError, match="ambiguous across articulations") as err:
        resolve_all_articulation(spec, {"cartpole": "/tmp/cartpole.usd", "g1": "/tmp/g1.usd"})

    assert "cartpole" in str(err.value) and "g1" in str(err.value)


def test_all_rejects_an_owner_without_a_resolved_asset():
    """An explicitly owned all group needs that articulation's USD asset."""
    spec = ContactGroupSpec(name="all", articulation="g1", sensor_bodies=["<all>"], frame="world")

    with pytest.raises(ValueError, match="no USD asset for it") as err:
        resolve_all_articulation(spec, {"cartpole": "/tmp/cartpole.usd"})

    assert "g1" in str(err.value)


ROBOT_BODIES = ["left_foot", "right_foot", "left_hand"]


def _inventory():
    return [{"prim": name, "collision": True} for name in ROBOT_BODIES]


def _backend(spec: ContactGroupSpec) -> IsaacLabContactBackend:
    return IsaacLabContactBackend([spec])


def test_bare_filter_matching_several_bodies_fails_with_the_inventory():
    """A bare filter regex that matches 2 bodies refuses to pick one."""
    spec = ContactGroupSpec(
        name="feet_props",
        sensor_bodies=["left_foot"],
        filter_bodies=["left_.*"],
        frame="world",
    )

    with pytest.raises(ContactCandidateError, match="matched 2 robot bodies") as err:
        _backend(spec)._resolve_bare_filters(spec, ROBOT_BODIES, _inventory)

    assert err.value.arg == "filter_bodies"
    assert [entry["prim"] for entry in err.value.discovered] == ROBOT_BODIES
    assert "declare filter_bodies" in err.value.hint


def test_bare_filter_matching_no_body_fails_with_the_inventory():
    """A bare filter regex matching nothing refuses with the candidates."""
    spec = ContactGroupSpec(
        name="feet_props",
        sensor_bodies=["left_foot"],
        filter_bodies=["ghost"],
        frame="world",
    )

    with pytest.raises(ContactCandidateError, match="matched 0 robot bodies") as err:
        _backend(spec)._resolve_bare_filters(spec, ROBOT_BODIES, _inventory)

    assert err.value.arg == "filter_bodies"


def test_duplicate_resolved_filter_bodies_fail_fast():
    """Two filter expressions resolved to the same body are rejected."""
    spec = ContactGroupSpec(
        name="feet_props",
        sensor_bodies=["left_foot"],
        filter_bodies=["left_foot", "left_foot"],
        frame="world",
    )

    with pytest.raises(ValueError, match=r"filter_bodies resolve to duplicate bodies"):
        _backend(spec)._resolve_bare_filters(spec, ROBOT_BODIES, _inventory)


@pytest.mark.parametrize("arg", ["sensor_bodies", "filter_bodies"])
def test_duplicate_resolved_bodies_share_one_rejection(arg):
    """The bind-time guard rejects duplicates on either resolved axis."""
    spec = ContactGroupSpec(name="feet", sensor_bodies=["left_foot"], frame="world")

    with pytest.raises(ValueError, match=rf"{arg} resolve to duplicate bodies \['left_foot'\]"):
        IsaacLabContactBackend._reject_duplicates(spec, arg, ["left_foot", "left_foot"])
