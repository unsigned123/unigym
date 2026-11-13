"""The contacts-all sentinel and its resolution helpers."""

from __future__ import annotations

import pytest

from unigym.contacts.base import (
    ALL_BODIES,
    ContactGroupSpec,
    expand_all_bodies,
    is_all_bodies,
    resolve_contact_group,
)


def test_all_bodies_sentinel_is_a_typed_value():
    """The all sentinel is one fixed string, never a body-name regex."""
    assert ALL_BODIES == "<all>"
    assert is_all_bodies([ALL_BODIES])
    assert is_all_bodies((ALL_BODIES,))
    assert not is_all_bodies(["pelvis"])
    assert not is_all_bodies([ALL_BODIES, "pelvis"])


def test_expand_all_bodies_calls_the_backend_resolver_once():
    """The sentinel expands through the resolver; plain lists pass through."""
    calls = []

    def resolver():
        calls.append(True)
        return ["pelvis", "left_ankle_roll_link"]

    assert expand_all_bodies([ALL_BODIES], resolver) == [
        "pelvis",
        "left_ankle_roll_link",
    ]
    assert expand_all_bodies(["left_ankle_roll_link"], resolver) == ["left_ankle_roll_link"]
    assert calls == [True]


def test_contact_group_resolves_the_all_inventory_group():
    """A run-level reference to the synthetic all group returns the sentinel."""
    value = {"contact_group": {"articulation": "g1", "group": "all"}}

    assert resolve_contact_group({"g1": {"all": (ALL_BODIES,)}}, value) == [ALL_BODIES]


def test_contact_spec_accepts_only_the_single_all_sentinel():
    """The sentinel cannot be mixed with body names or used as a filter."""
    spec = ContactGroupSpec(name="all", sensor_bodies=[ALL_BODIES], frame="world")
    assert spec.sensor_bodies == [ALL_BODIES]

    with pytest.raises(ValueError, match="must be the only sensor body"):
        ContactGroupSpec(name="all", sensor_bodies=[ALL_BODIES, "pelvis"], frame="world")

    with pytest.raises(ValueError, match="not a valid filter body"):
        ContactGroupSpec(
            name="all",
            sensor_bodies=["pelvis"],
            filter_bodies=[ALL_BODIES],
            frame="world",
        )
