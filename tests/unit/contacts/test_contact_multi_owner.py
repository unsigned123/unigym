"""Mixed-owner groups, derived metadata and legacy-key rejection."""

from __future__ import annotations

import json

import pytest

from unigym.contacts.base import ContactGroupSpec
from unigym.contacts.manager import build_contact_manager
from unigym.contacts.refs import ContactBodyRef


def _mixed_group():
    return ContactGroupSpec(
        name="crate_hand",
        sensor_refs=[
            ContactBodyRef(owner="object", name="crate"),
            ContactBodyRef(owner="robot", articulation="hand", body=".*"),
        ],
        filter_refs=[ContactBodyRef(owner="support", name="table")],
        frame="world",
        format="contact",
        threshold=0.5,
        aggregate="any",
        aggregation="sum_magnitudes",
    )


def test_owner_is_derived_metadata_and_mixed():
    spec = _mixed_group()
    assert spec.owners() == ["object", "robot"]
    assert spec.owner == "mixed"
    assert spec.mode == "pairwise"
    assert spec.articulation_names() == ["hand"]
    assert spec.prop_names() == ["crate", "table"]


def test_metadata_carries_refs_and_owners_and_is_json_safe():
    metadata = _mixed_group().metadata()
    assert metadata["owner"] == "mixed"
    assert metadata["owners"] == ["object", "robot"]
    assert metadata["mode"] == "pairwise"
    assert metadata["sensor_refs"][0] == {"owner": "object", "name": "crate"}
    assert metadata["filter_refs"] == [{"owner": "support", "name": "table"}]
    assert json.loads(json.dumps(metadata)) == metadata


def test_mode_is_derived_from_filter_refs():
    general = ContactGroupSpec(
        name="g", sensor_refs=[{"owner": "object", "name": "crate"}], frame="world"
    )
    pairwise = ContactGroupSpec(
        name="p",
        sensor_refs=[{"owner": "object", "name": "crate"}],
        filter_refs=[{"owner": "robot", "body": "tip"}],
        frame="world",
    )
    assert general.mode == "general"
    assert pairwise.mode == "pairwise"


@pytest.mark.parametrize(
    "key",
    ["sensor_bodies", "filter_bodies", "articulation", "mode", "owner"],
)
def test_from_yaml_rejects_every_removed_key_with_a_teaching_error(key):
    entry = {
        "name": "x",
        "sensor_refs": [{"owner": "object", "name": "crate"}],
        "frame": "world",
        key: ["legacy"],
    }
    with pytest.raises(ValueError, match="removed keys"):
        ContactGroupSpec.from_yaml(entry)


def test_from_yaml_rejects_the_legacy_reference_mappings():
    for field in ("sensor_refs", "filter_refs"):
        entry = {
            "name": "x",
            "sensor_refs": [{"owner": "object", "name": "crate"}],
            "frame": "world",
        }
        entry[field] = {"prop": "crate"}
        with pytest.raises(ValueError, match="legacy reference forms"):
            ContactGroupSpec.from_yaml(entry)


def test_from_yaml_rejects_string_body_lists():
    with pytest.raises(ValueError, match="must be a mapping"):
        ContactGroupSpec.from_yaml({"name": "x", "sensor_refs": ["Prop_crate"], "frame": "world"})


def test_from_yaml_requires_sensor_refs_and_frame():
    with pytest.raises(ValueError, match="missing required keys"):
        ContactGroupSpec.from_yaml({"name": "x", "frame": "world"})
    with pytest.raises(ValueError, match="missing required keys"):
        ContactGroupSpec.from_yaml(
            {"name": "x", "sensor_refs": [{"owner": "object", "name": "crate"}]}
        )


def test_multi_articulation_robot_refs_must_name_their_owner():
    with pytest.raises(ValueError, match="every robot ref must name its owner") as err:
        build_contact_manager(
            [
                {
                    "name": "feet",
                    "sensor_refs": [{"owner": "robot", "body": "foot"}],
                    "frame": "world",
                }
            ],
            backend="isaaclab2",
            articulation_names=("cartpole", "g1"),
        )
    assert "cartpole" in str(err.value) and "g1" in str(err.value)


def test_unknown_articulation_in_a_ref_is_a_teaching_error():
    with pytest.raises(ValueError, match="not declared by the run"):
        build_contact_manager(
            [
                {
                    "name": "feet",
                    "sensor_refs": [{"owner": "robot", "articulation": "ghost", "body": "foot"}],
                    "frame": "world",
                }
            ],
            backend="isaaclab2",
            articulation_names=("g1",),
        )


def test_single_articulation_robot_ref_without_owner_is_legal():
    manager = build_contact_manager(
        [
            {
                "name": "feet",
                "sensor_refs": [{"owner": "robot", "body": "foot"}],
                "frame": "world",
            }
        ],
        backend="isaaclab2",
        articulation_names=("cartpole",),
    )
    assert manager is not None
    assert manager.specs[0].sensor_refs[0].articulation is None
