"""ArticulationSpec body overrides, contact inventory and spawn validation."""

from __future__ import annotations

import pytest

from unigym.scenes.articulation import (
    ArticulationError,
    ArticulationSpec,
    ContactSensorGroup,
    ContactSensorInventory,
    normalize_articulations,
)


def test_body_fields_default_to_inherit():
    """Omitted body overrides stay None (inherit the robot profile)."""
    spec = ArticulationSpec(name="g1")

    assert spec.base_body is None
    assert spec.foot_bodies is None
    assert spec.body_sets is None
    assert spec.contacts is None
    assert spec.spawn == {}


def test_explicit_empty_body_fields_stay_empty():
    """An explicit empty tuple/mapping means empty, not inherit."""
    spec = ArticulationSpec(
        name="g1",
        base_body=None,
        foot_bodies=(),
        body_sets={},
    )

    assert spec.foot_bodies == ()
    assert spec.body_sets == {}


def test_body_fields_normalize_canonical_ids():
    """Base/feet/body-set members normalize to canonical ids."""
    spec = ArticulationSpec(
        name="g1",
        base_body="base",
        foot_bodies=("left_ankle_roll", "right_ankle_roll"),
        body_sets={"grasp.left": ("left_hand.index_0",)},
    )

    assert spec.base_body == "base"
    assert spec.foot_bodies == ("left_ankle_roll", "right_ankle_roll")
    assert spec.body_sets == {"grasp.left": ("left_hand.index_0",)}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"base_body": "left/base"}, "base_body"),
        ({"foot_bodies": ("left/foot",)}, "foot_bodies"),
        ({"body_sets": {"grasp": ()}}, "at least one body"),
        ({"body_sets": {"grasp": ("base", "base")}}, "duplicate body"),
        ({"body_sets": {"": ("base",)}}, "names must be non-empty"),
    ],
)
def test_malformed_body_fields_fail_fast(kwargs, match):
    """Malformed canonical ids/sets fail with field context."""
    with pytest.raises(ArticulationError, match=match):
        ArticulationSpec(name="g1", **kwargs)


def test_contacts_normalize_a_group_list_into_the_inventory_ir():
    """A raw contacts group list normalizes to the contact sensor inventory."""
    spec = ArticulationSpec(
        name="g1",
        contacts=[{"name": "feet", "bodies": ["left_ankle_roll"]}],
    )

    assert spec.contacts == ContactSensorInventory(
        mode="declared",
        groups=(ContactSensorGroup(name="feet", bodies=("left_ankle_roll",)),),
    )


@pytest.mark.parametrize("value", ["all", "none"])
def test_contacts_accept_the_mode_spellings(value):
    """all/none normalize to the closed inventory modes."""
    spec = ArticulationSpec(name="g1", contacts=value)

    assert spec.contacts is not None
    assert spec.contacts.mode == value


def test_spawn_requires_pos_when_present_and_defaults_yaw_to_zero():
    """A present spawn requires a finite 3-vector pos; yaw defaults to 0."""
    spec = ArticulationSpec(name="g1", spawn={"pos": [1.0, 2.0, 0.8]})

    assert spec.spawn == {"pos": [1.0, 2.0, 0.8], "yaw": 0.0}

    with_yaw = ArticulationSpec(name="g1", spawn={"pos": [0.0, 0.0, 1.0], "yaw": 0.5})
    assert with_yaw.spawn == {"pos": [0.0, 0.0, 1.0], "yaw": 0.5}


@pytest.mark.parametrize(
    ("spawn", "match"),
    [
        ({"yaw": 0.2}, "requires `pos`"),
        ({"pos": [0.0, 0.0]}, "3-vector"),
        ({"pos": [0.0, 0.0, float("inf")]}, "finite"),
        ({"pos": [0.0, 0.0, 1.0], "roll": 0.1}, r"unknown keys \['roll'\]"),
        ({"pos": [0.0, 0.0, 1.0], "yaw": float("nan")}, "finite"),
    ],
)
def test_malformed_spawn_fails_fast(spawn, match):
    """Spawn rejects missing pos, non-finite values and unknown keys."""
    with pytest.raises(ArticulationError, match=match):
        ArticulationSpec(name="g1", spawn=spawn)


def test_normalize_articulations_accepts_the_neutral_body_contact_keys():
    """The config surface carries the new body/contact/spawn fields."""
    specs = normalize_articulations(
        [
            {
                "name": "g1",
                "base_body": "base",
                "foot_bodies": ["left_ankle_roll"],
                "body_sets": {"grasp.left": ["left_hand.index_0"]},
                "contacts": [{"name": "feet", "bodies": ["left_ankle_roll"]}],
                "spawn": {"pos": [0.0, 0.0, 0.8]},
            }
        ]
    )

    spec = specs[0]
    assert spec.base_body == "base"
    assert spec.foot_bodies == ("left_ankle_roll",)
    assert spec.body_sets == {"grasp.left": ("left_hand.index_0",)}
    assert spec.contacts == ContactSensorInventory(
        mode="declared",
        groups=(ContactSensorGroup(name="feet", bodies=("left_ankle_roll",)),),
    )
    assert spec.spawn == {"pos": [0.0, 0.0, 0.8], "yaw": 0.0}
