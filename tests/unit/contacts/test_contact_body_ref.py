"""ContactBodyRef: the only explicit contact body address (Scheme A)."""

from __future__ import annotations

import pytest

from unigym.contacts.refs import ContactBodyRef, is_wildcard


def test_robot_body_and_body_set_forms_are_accepted():
    body = ContactBodyRef(owner="robot", articulation="left_hand", body=".*")
    body_set = ContactBodyRef(owner="robot", articulation="left_arm", body_set="grasp")

    assert body.to_dict() == {
        "owner": "robot",
        "articulation": "left_hand",
        "body": ".*",
    }
    assert body_set.to_dict() == {
        "owner": "robot",
        "articulation": "left_arm",
        "body_set": "grasp",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"owner": "robot", "body": "x", "body_set": "s"},
        {"owner": "robot"},
    ],
)
def test_robot_requires_exactly_one_address_form(kwargs):
    with pytest.raises(ValueError, match="exactly one of body or body_set"):
        ContactBodyRef(**kwargs)


@pytest.mark.parametrize("key", ["name", "path"])
def test_robot_forbids_scoped_address_fields(key):
    with pytest.raises(ValueError, match="forbids"):
        ContactBodyRef(owner="robot", body="x", **{key: "bad"})


@pytest.mark.parametrize("owner", ["object", "support", "fixture"])
def test_scoped_owners_require_name_and_optional_body(owner):
    ref = ContactBodyRef(owner=owner, name="crate")
    assert ref.body is None
    assert ref.scope() == "crate"
    with pytest.raises(ValueError, match="requires name"):
        ContactBodyRef(owner=owner)


@pytest.mark.parametrize("owner", ["object", "support", "fixture"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"articulation": "g1"},
        {"body_set": "grasp"},
        {"path": "/World/x"},
    ],
)
def test_scoped_owners_forbid_foreign_address_fields(owner, kwargs):
    with pytest.raises(ValueError, match="forbids"):
        ContactBodyRef(owner=owner, name="crate", **kwargs)


def test_environment_body_or_path_is_exclusive():
    assert ContactBodyRef(owner="environment", body="world").to_dict() == {
        "owner": "environment",
        "body": "world",
    }
    assert ContactBodyRef(owner="environment", path="{ENV_REGEX_NS}/ground").to_dict() == {
        "owner": "environment",
        "path": "{ENV_REGEX_NS}/ground",
    }
    with pytest.raises(ValueError, match="exactly one of body or path"):
        ContactBodyRef(owner="environment")
    with pytest.raises(ValueError, match="exactly one of body or path"):
        ContactBodyRef(owner="environment", body="world", path="/World/ground")


@pytest.mark.parametrize("key", ["name", "articulation", "body_set"])
def test_environment_forbids_scoped_fields(key):
    with pytest.raises(ValueError, match="forbids"):
        ContactBodyRef(owner="environment", body="world", **{key: "bad"})


def test_unknown_owner_and_match_fail_fast():
    with pytest.raises(ValueError, match="owner must be one of"):
        ContactBodyRef(owner="world", body="x")
    with pytest.raises(ValueError, match="match must be one of"):
        ContactBodyRef(owner="robot", body="x", match="glob")


def test_unknown_keys_are_rejected_with_the_allowed_set():
    with pytest.raises(ValueError, match="unknown keys"):
        ContactBodyRef.from_mapping({"owner": "robot", "body": "x", "prim": "/World"})


def test_wildcard_is_literal_under_exact_match():
    assert is_wildcard("*")
    assert not is_wildcard("*", "exact")
    assert not is_wildcard("crate")

    ref = ContactBodyRef(owner="object", name="crate", body="*", match="exact")
    assert ref.to_dict() == {
        "owner": "object",
        "name": "crate",
        "body": "*",
        "match": "exact",
    }


def test_from_mapping_round_trips_and_passes_existing_refs_through():
    mapping = {"owner": "robot", "articulation": "arm", "body_set": "fingers"}
    ref = ContactBodyRef.from_mapping(mapping)
    assert ref.to_dict() == mapping
    assert ContactBodyRef.from_mapping(ref) is ref


@pytest.mark.parametrize(
    "value",
    ["crate", ["crate"], 3],
)
def test_non_mapping_refs_are_rejected(value):
    with pytest.raises(ValueError, match="must be a mapping"):
        ContactBodyRef.from_mapping(value)
