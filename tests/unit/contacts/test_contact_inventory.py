"""ContactInventory: canonical rows, explicit per-ref resolution, B/F order."""

from __future__ import annotations

import pytest

from unigym.contacts.inventory import (
    ContactBodyAddress,
    ContactInventory,
    resolve_refs,
)
from unigym.contacts.refs import ContactBodyRef


def _row(owner, scope, body, native=None, *, root=False, prim=None):
    native = body if native is None else native
    scope_part = f"{scope}/" if scope is not None else ""
    return ContactBodyAddress(
        owner=owner,
        scope=scope,
        body=body,
        native=native,
        prim_path=prim,
        collision=True,
        contact_capable=True,
        key=f"{owner}/{scope_part}{body}",
        root=root,
    )


def _inventory(body_sets=()):
    rows = (
        _row("robot", "arm", "shoulder", root=True),
        _row("robot", "arm", "elbow"),
        _row("robot", "arm", "fingertip"),
        _row("robot", "hand", "palm", root=True),
        _row("robot", "hand", "index"),
        _row("object", "crate", "crate", "Prop_crate", root=True),
        _row("object", "crate", "lid", "Prop_crate_lid"),
        _row("fixture", "microwave", "microwave_main", "Prop_microwave_microwave_main", root=True),
        _row("support", "table", "table_top", root=True),
        _row("environment", "world", "world", root=True),
    )
    return ContactInventory("mujoco", rows, tuple(body_sets))


def test_declaration_order_then_native_order_within_each_ref():
    inv = _inventory()
    refs = [
        ContactBodyRef(owner="robot", articulation="hand", body=".*"),
        ContactBodyRef(owner="robot", articulation="arm", body="elbow"),
        ContactBodyRef(owner="robot", articulation="arm", body="shoulder"),
    ]

    rows = resolve_refs(inv, refs, axis="sensor_refs", group="mixed")

    assert [row.native for row in rows] == ["palm", "index", "elbow", "shoulder"]


def test_wildcard_selects_every_body_in_scope_in_native_order():
    inv = _inventory()
    rows = inv.resolve(ContactBodyRef(owner="robot", articulation="arm", body="*"))
    assert [row.native for row in rows] == ["shoulder", "elbow", "fingertip"]


def test_exact_wildcard_is_literal():
    inv = _inventory((("arm", "literal", ("shoulder",)),))
    with pytest.raises(ValueError, match="matched no bodies"):
        inv.resolve(ContactBodyRef(owner="robot", articulation="arm", body="*", match="exact"))


def test_regex_and_exact_body_matching():
    inv = _inventory()
    assert [
        r.body for r in inv.resolve(ContactBodyRef(owner="robot", articulation="arm", body=".*tip"))
    ] == ["fingertip"]
    assert [
        r.body
        for r in inv.resolve(
            ContactBodyRef(owner="robot", articulation="arm", body="elbow", match="exact")
        )
    ] == ["elbow"]
    # regex "index" also matches "fingertip"? fullmatch says no.
    with pytest.raises(ValueError, match="matched no bodies"):
        inv.resolve(ContactBodyRef(owner="robot", articulation="arm", body="index"))


def test_body_set_expands_within_the_named_articulation():
    inv = _inventory((("arm", "grasp", ("shoulder", "fingertip")),))
    rows = inv.resolve(ContactBodyRef(owner="robot", articulation="arm", body_set="grasp"))
    assert [r.native for r in rows] == ["shoulder", "fingertip"]
    with pytest.raises(ValueError, match="unknown body_set"):
        inv.resolve(ContactBodyRef(owner="robot", articulation="arm", body_set="ghost"))


def test_scoped_owner_root_default_and_explicit_body():
    inv = _inventory()
    assert [r.native for r in inv.resolve(ContactBodyRef(owner="object", name="crate"))] == [
        "Prop_crate"
    ]
    assert [
        r.native for r in inv.resolve(ContactBodyRef(owner="object", name="crate", body="lid"))
    ] == ["Prop_crate_lid"]
    assert [
        r.native
        for r in inv.resolve(
            ContactBodyRef(owner="fixture", name="microwave", body="microwave_main")
        )
    ] == ["Prop_microwave_microwave_main"]


def test_support_uses_the_declared_scope_name():
    inv = _inventory()
    rows = inv.resolve(ContactBodyRef(owner="support", name="table"))
    assert [r.native for r in rows] == ["table_top"]


def test_environment_body_and_path_resolution():
    inv = ContactInventory(
        "isaaclab2",
        (_row("environment", "world", "world", "world", root=True, prim="/World/ground"),),
    )
    assert [r.native for r in inv.resolve(ContactBodyRef(owner="environment", body="world"))] == [
        "world"
    ]
    assert [
        r.native for r in inv.resolve(ContactBodyRef(owner="environment", path="/World/ground"))
    ] == ["world"]


def test_unknown_scope_and_articulation_list_candidate_keys():
    inv = _inventory()
    with pytest.raises(ValueError, match="unknown object scope") as err:
        inv.resolve(ContactBodyRef(owner="object", name="ghost"))
    assert "object/crate/crate" in str(err.value)

    with pytest.raises(ValueError, match="unknown articulation") as err:
        inv.resolve(ContactBodyRef(owner="robot", articulation="ghost", body=".*"))
    assert "robot/arm/shoulder" in str(err.value)


def test_robot_without_articulation_is_ambiguous_across_scopes():
    inv = _inventory()
    with pytest.raises(ValueError, match="must name articulation") as err:
        inv.resolve(ContactBodyRef(owner="robot", body=".*"))
    assert "['arm', 'hand']" in str(err.value)


def test_robot_without_articulation_is_legal_with_one_scope():
    only = ContactInventory(
        "mujoco",
        (_row("robot", "tiny", "tip"),),
    )
    rows = only.resolve(ContactBodyRef(owner="robot", body="tip"))
    assert [r.scope for r in rows] == ["tiny"]


def test_duplicate_resolved_addresses_fail_with_the_axis():
    inv = _inventory()
    refs = [
        ContactBodyRef(owner="robot", articulation="arm", body="elbow"),
        ContactBodyRef(owner="robot", articulation="arm", body="el.*"),
    ]
    with pytest.raises(ValueError, match=r"filter_refs\[1\].*duplicate body"):
        resolve_refs(inv, refs, axis="filter_refs", group="dup")


def test_empty_ref_list_is_a_general_group_axis():
    inv = _inventory()
    assert resolve_refs(inv, [], axis="filter_refs", group="general") == []


def test_inventory_body_sets_lookup_prefers_the_articulation_scope():
    inv = _inventory((("arm", "grasp", ("shoulder",)), ("hand", "grasp", ("palm",))))
    assert inv.body_set("arm", "grasp") == ("shoulder",)
    assert inv.body_set("hand", "grasp") == ("palm",)
    assert inv.body_set(None, "grasp") == ("shoulder",)
    assert inv.body_set("arm", "ghost") is None
