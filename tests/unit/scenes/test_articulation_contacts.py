"""Contact sensor inventory normalization and reference-validation invariants."""

from __future__ import annotations

import pytest

from unigym.robots.base import BodyBinding
from unigym.scenes.articulation import (
    ArticulationError,
    ContactSensorGroup,
    ContactSensorInventory,
    normalize_contact_sensor_inventory,
    validate_contact_sensor_inventory,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bodies": ("left_ankle_roll",), "body_set": "feet"},
        {},
        {"bodies": ()},
    ],
)
def test_contact_sensor_group_requires_exactly_one_addressing_mode(kwargs):
    """A group declares non-empty bodies or body_set, never both and never neither."""
    with pytest.raises(ArticulationError, match="exactly one"):
        ContactSensorGroup(name="feet", **kwargs)


def test_contact_all_spellings_normalize_identically():
    """contacts: all and the single named wildcard group are the same inventory."""
    short_form = normalize_contact_sensor_inventory("all")
    wildcard_form = normalize_contact_sensor_inventory([{"name": "all", "bodies": ["*"]}])

    assert short_form == ContactSensorInventory(mode="all")
    assert wildcard_form == short_form
    assert normalize_contact_sensor_inventory("none") == ContactSensorInventory(mode="none")


@pytest.mark.parametrize(
    "value",
    [
        [{"name": "all", "bodies": ["*", "base"]}],
        [{"name": "feet", "bodies": ["*"]}],
        [
            {"name": "all", "bodies": ["*"]},
            {"name": "feet", "bodies": ["base"]},
        ],
    ],
)
def test_contact_wildcard_must_be_the_only_body_in_the_only_group(value):
    """A wildcard mixed with named bodies or named groups is rejected."""
    with pytest.raises(ArticulationError, match="wildcard"):
        normalize_contact_sensor_inventory(value)


def test_normalize_contact_groups_builds_declared_inventory():
    """A declared group list normalizes bodies to canonical ids."""
    inventory = normalize_contact_sensor_inventory(
        [{"name": "feet", "bodies": ["left_ankle_roll", "right_ankle_roll"]}]
    )

    assert inventory.mode == "declared"
    assert inventory.groups == (
        ContactSensorGroup(
            name="feet",
            bodies=("left_ankle_roll", "right_ankle_roll"),
        ),
    )


def test_unknown_contact_body_set_names_declared_sets():
    """A body_set reference must resolve against the declared sets."""
    inventory = ContactSensorInventory(
        mode="declared",
        groups=(ContactSensorGroup(name="grasp", body_set="grasp.left"),),
    )

    with pytest.raises(
        ArticulationError,
        match=r"body_set 'grasp.left' is unknown.*\['grasp.right'\]",
    ):
        validate_contact_sensor_inventory(
            inventory,
            body_sets={"grasp.right": ("right_hand",)},
        )


def test_unknown_contact_body_is_rejected_against_bindings():
    """An explicit body must have a BodyBinding when bindings are supplied."""
    inventory = ContactSensorInventory(
        mode="declared",
        groups=(ContactSensorGroup(name="feet", bodies=("ghost",)),),
    )
    bindings = {"base": BodyBinding("base", {"mjcf": "base"})}

    with pytest.raises(ArticulationError, match=r"unknown bodies \['ghost'\]"):
        validate_contact_sensor_inventory(inventory, bindings=bindings)


def test_contact_body_missing_required_format_is_rejected():
    """Every referenced body must carry every explicitly listed format."""
    inventory = ContactSensorInventory(
        mode="declared",
        groups=(ContactSensorGroup(name="feet", bodies=("base",)),),
    )
    bindings = {"base": BodyBinding("base", {"mjcf": "base"})}

    with pytest.raises(
        ArticulationError,
        match=r"lack required formats.*'base'.*\['usd'\]",
    ):
        validate_contact_sensor_inventory(
            inventory,
            bindings=bindings,
            formats=("mjcf", "usd"),
        )


def test_contact_format_validation_requires_bindings():
    """Asking for format validation without bindings is a programming error."""
    inventory = ContactSensorInventory(
        mode="declared",
        groups=(ContactSensorGroup(name="feet", bodies=("base",)),),
    )

    with pytest.raises(ArticulationError, match="requires body bindings"):
        validate_contact_sensor_inventory(inventory, formats=("mjcf",))


def test_duplicate_contact_body_membership_across_groups_is_rejected():
    """One canonical body may appear in only one sensor group."""
    inventory = ContactSensorInventory(
        mode="declared",
        groups=(
            ContactSensorGroup(name="left", bodies=("base",)),
            ContactSensorGroup(name="right", bodies=("base",)),
        ),
    )

    with pytest.raises(ArticulationError, match="multiple contact sensor groups"):
        validate_contact_sensor_inventory(inventory)


def test_duplicate_contact_body_membership_inside_one_group_is_rejected():
    """A group's explicit body list has no duplicates."""
    with pytest.raises(ArticulationError, match="duplicate bodies"):
        ContactSensorGroup(name="feet", bodies=("base", "base"))


@pytest.mark.parametrize("mode", ["all", "none"])
def test_all_and_none_need_no_binding_lookup(mode):
    """all and none are mode declarations, not body references."""
    validate_contact_sensor_inventory(
        ContactSensorInventory(mode=mode),
        body_sets=None,
        bindings=None,
        formats=None,
    )


@pytest.mark.parametrize("name", ["all", "none"])
def test_contact_group_names_all_and_none_are_reserved(name):
    """all and none are mode/wildcard names, never declared group names."""
    with pytest.raises(ArticulationError, match="reserved"):
        ContactSensorGroup(name=name, bodies=("base",))


def test_contact_group_name_syntax_is_enforced():
    """Group names are limited to [A-Za-z0-9_-]+."""
    with pytest.raises(ArticulationError, match="must match"):
        ContactSensorGroup(name="bad name", bodies=("base",))


def test_contact_inventory_group_names_must_be_unique():
    """Two groups of one inventory may not share a name."""
    group = ContactSensorGroup(name="feet", bodies=("base",))

    with pytest.raises(ArticulationError, match="duplicate group names"):
        ContactSensorInventory(mode="declared", groups=(group, group))


def test_declared_contact_inventory_requires_a_group():
    """Declared mode carries at least one named group."""
    with pytest.raises(ArticulationError, match="requires at least one group"):
        ContactSensorInventory(mode="declared")


def test_all_contact_inventory_rejects_groups():
    """all and none carry no groups."""
    group = ContactSensorGroup(name="feet", bodies=("base",))

    with pytest.raises(ArticulationError, match="must not declare groups"):
        ContactSensorInventory(mode="all", groups=(group,))


def test_contact_body_set_members_are_validated_as_body_references():
    """Body-set members follow the same binding and format rules as bodies."""
    body_sets = {"grasp.left": ("left_hand.index_1",)}
    bindings = {
        "left_hand.index_1": BodyBinding(
            "left_hand.index_1",
            {"mjcf": "left_hand_index_1", "usd": "left_hand/index_1"},
        )
    }
    inventory = ContactSensorInventory(
        mode="declared",
        groups=(ContactSensorGroup(name="grasp", body_set="grasp.left"),),
    )

    validate_contact_sensor_inventory(
        inventory,
        body_sets=body_sets,
        bindings=bindings,
        formats=("mjcf", "usd"),
    )


def test_contact_body_id_errors_carry_group_field_context():
    """A malformed body id is wrapped with the contact group field context."""
    with pytest.raises(
        ArticulationError,
        match=r"contact sensor group 'feet' bodies\[0\]",
    ):
        ContactSensorGroup(name="feet", bodies=("left/hand",))


def test_contact_inventory_serializes_bodies_and_body_set_groups():
    """v4 metadata serializes each group as bodies or body_set."""
    inventory = ContactSensorInventory(
        mode="declared",
        groups=(
            ContactSensorGroup(name="feet", bodies=("left_ankle_roll",)),
            ContactSensorGroup(name="grasp", body_set="grasp.left"),
        ),
    )

    assert inventory.to_dict() == {
        "mode": "declared",
        "groups": [
            {"name": "feet", "bodies": ["left_ankle_roll"]},
            {"name": "grasp", "body_set": "grasp.left"},
        ],
    }
    assert ContactSensorInventory(mode="none").to_dict() == {"mode": "none", "groups": []}
