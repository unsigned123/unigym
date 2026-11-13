"""Run-level contact-group references resolve to native body regexes."""

from __future__ import annotations

import pytest

from unigym.contacts.base import (
    ContactGroupSpec,
    resolve_contact_entries,
    resolve_contact_group,
)


def _table():
    return {"g1": {"grasp": ("left_hand_index_0_link", "left_hand_index_1_link")}}


def test_contact_group_reference_maps_to_native_bodies():
    """The resolver maps one inventory group to its native body regexes."""
    value = {"contact_group": {"articulation": "g1", "group": "grasp"}}

    assert resolve_contact_group(_table(), value) == [
        "left_hand_index_0_link",
        "left_hand_index_1_link",
    ]


def test_contact_entries_resolve_the_reference_without_mutating_input():
    """A raw run-level entry resolves in a new dict, leaving the input intact."""
    entry = {
        "name": "grasp",
        "sensor_bodies": {"contact_group": {"articulation": "g1", "group": "grasp"}},
        "frame": "world",
    }

    resolved = resolve_contact_entries([entry], "mujoco", contact_groups=_table())

    assert resolved[0]["sensor_bodies"] == [
        "left_hand_index_0_link",
        "left_hand_index_1_link",
    ]
    assert resolved[0]["articulation"] == "g1"
    assert entry["sensor_bodies"] == {"contact_group": {"articulation": "g1", "group": "grasp"}}
    assert "articulation" not in entry


def test_contact_entries_reject_a_conflicting_explicit_owner():
    """An explicit articulation must agree with the contact_group reference."""
    entry = {
        "name": "grasp",
        "articulation": "g2",
        "sensor_bodies": {"contact_group": {"articulation": "g1", "group": "grasp"}},
        "frame": "world",
    }

    with pytest.raises(ValueError, match="conflicts with the contact_group reference owner"):
        resolve_contact_entries([entry], "mujoco", contact_groups=_table())


def test_explicit_articulation_survives_entry_resolution():
    """A direct articulation key passes through unchanged."""
    entry = {"name": "feet", "articulation": "g1", "sensor_bodies": ["foot"], "frame": "world"}

    resolved = resolve_contact_entries([entry], "mujoco", contact_groups={})

    assert resolved[0]["articulation"] == "g1"


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ({"body_parts": "grasp.left"}, "contact_group"),
        ({"contact_group": {"articulation": "g1", "group": "grasp"}, "x": 1}, "contact_group"),
        ({"contact_group": {"articulation": "g1", "group": "grasp", "x": 1}}, "exactly"),
        ({"contact_group": {"articulation": "g2", "group": "grasp"}}, "unknown.*g1.grasp"),
        ({"contact_group": {"articulation": "g1", "group": "ghost"}}, "unknown.*g1.grasp"),
    ],
)
def test_contact_group_reference_rejects_unknown_keys_and_refs(value, match):
    """Unknown reference keys or articulation.group names fail with the available set."""
    with pytest.raises(ValueError, match=match):
        resolve_contact_group(_table(), value)


def test_contact_group_spec_rejects_the_raw_reference_form():
    """from_yaml rejects unresolved references and names the resolver form."""
    with pytest.raises(ValueError, match="contact_group"):
        ContactGroupSpec.from_yaml(
            {
                "name": "grasp",
                "sensor_bodies": {"contact_group": {"articulation": "g1", "group": "grasp"}},
                "frame": "world",
            }
        )


def test_explicit_sensor_body_lists_still_resolve_unchanged():
    """An explicit native list is passed through without any table lookup."""
    entry = {"name": "feet", "sensor_bodies": ["left_foot_link"], "frame": "world"}

    resolved = resolve_contact_entries([entry], "mujoco", contact_groups={})

    assert resolved[0]["sensor_bodies"] == ["left_foot_link"]
