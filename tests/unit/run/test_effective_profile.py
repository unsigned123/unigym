"""Effective-profile precedence and contact-group native resolution."""

from __future__ import annotations

import pytest

from unigym.backends.registry import BackendInfo
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.runtime.controllers import (
    ArticulationRuntime,
    ControlRuntime,
    articulation_facts,
)
from unigym.run.build import (
    DATA_SCHEMA_VERSION,
    _contact_group_table,
    _effective_profile,
    _required_body_ids,
)
from unigym.scenes.articulation import (
    ArticulationSpec,
    ContactSensorGroup,
    ContactSensorInventory,
)


def _binding(name: str, *, mjcf: str | None = None, usd: str | None = None) -> BodyBinding:
    names = {}
    if mjcf is not None:
        names["mjcf"] = mjcf
    if usd is not None:
        names["usd"] = usd
    return BodyBinding(name, names)


def _profile() -> RobotProfile:
    bindings = {
        "base": _binding("base", mjcf="pelvis", usd="pelvis"),
        "left_foot": _binding("left_foot", mjcf="left_foot_link", usd="left_foot_link"),
        "right_foot": _binding("right_foot", mjcf="right_foot_link", usd="right_foot_link"),
        "finger": _binding("finger", mjcf="finger_link", usd="hand/finger_link"),
        "palm": _binding("palm", mjcf="palm_link", usd="hand/palm_link"),
    }
    return RobotProfile(
        name="arm",
        joint_sets={"body": ("j",)},
        default_joint_pos={"j": 0.0},
        assets=None,
        action_faces={},
        base_body="base",
        foot_bodies=("left_foot",),
        body_sets={"grasp": ("finger",)},
        body_bindings=bindings,
    )


def _info(asset_format: str) -> BackendInfo:
    return BackendInfo("mujoco", (0, 0), frozenset(), asset_format=asset_format)


def test_data_schema_version_is_v4():
    """The run data schema is the explicit v4 cutover version."""
    assert DATA_SCHEMA_VERSION == 4


def test_spec_overrides_win_over_profile_body_facts():
    """Explicit spec fields replace the profile; None inherits."""
    profile = _profile()
    spec = ArticulationSpec(
        name="arm",
        base_body="palm",
        foot_bodies=("right_foot",),
        body_sets={"grasp": ("finger", "palm")},
    )

    effective = _effective_profile(spec, profile, _info("mjcf"))

    assert effective.base_body == "palm"
    assert effective.foot_bodies == ("right_foot",)
    assert effective.body_sets == {"grasp": ("finger", "palm")}
    # the source profile is not mutated
    assert profile.base_body == "base"
    assert profile.foot_bodies == ("left_foot",)


def test_none_spec_fields_inherit_the_profile():
    """Omitted (None) spec fields inherit every profile body fact."""
    profile = _profile()
    spec = ArticulationSpec(name="arm")

    effective = _effective_profile(spec, profile, _info("usd"))

    assert effective.base_body == "base"
    assert effective.foot_bodies == ("left_foot",)
    assert effective.body_sets == {"grasp": ("finger",)}


def test_selected_backend_format_must_exist_on_every_required_binding():
    """The selected backend format is required separately from formats=None."""
    profile = _profile()
    bad = RobotProfile(
        name="arm",
        joint_sets={"body": ("j",)},
        default_joint_pos={"j": 0.0},
        assets=None,
        action_faces={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": _binding("base", mjcf="pelvis")},
    )

    with pytest.raises(ValueError, match="lack the selected asset format 'usd'"):
        _effective_profile(ArticulationSpec(name="arm"), bad, _info("usd"))
    # multiformat profile still passes when the required format is present
    assert (
        _effective_profile(ArticulationSpec(name="arm"), profile, _info("usd")).base_body == "base"
    )


def test_contact_references_extend_the_required_body_ids():
    """Declared contact bodies and body_set members are required bindings."""
    profile = _profile()
    inventory = ContactSensorInventory(
        mode="declared",
        groups=(
            ContactSensorGroup(name="feet", bodies=("left_foot",)),
            ContactSensorGroup(name="grasp", body_set="grasp"),
        ),
    )

    required = _required_body_ids(profile, inventory)

    assert required == ["base", "left_foot", "finger"]


def test_unknown_contact_body_set_fails_fast():
    """A contact group body_set must resolve against the effective body sets."""
    profile = _profile()
    inventory = ContactSensorInventory(
        mode="declared",
        groups=(ContactSensorGroup(name="ghost", body_set="missing"),),
    )

    with pytest.raises(ValueError, match="body_set 'missing' is unknown"):
        _required_body_ids(profile, inventory)


def test_contact_group_table_maps_canonical_bodies_through_the_selected_format():
    """The run layer (and only it) maps inventory bodies to native names."""
    profile = _profile()
    inventory = ContactSensorInventory(
        mode="declared",
        groups=(
            ContactSensorGroup(name="feet", bodies=("left_foot",)),
            ContactSensorGroup(name="grasp", body_set="grasp"),
        ),
    )
    spec = ArticulationSpec(name="arm", contacts=inventory)
    effective = _effective_profile(spec, profile, _info("usd"))
    builds = {"arm": {"profile": effective, "params": {}, "spec": spec}}

    table = _contact_group_table(builds, _info("usd"))

    assert table == {
        "arm": {
            "feet": ("left_foot_link",),
            "grasp": ("hand/finger_link",),
        }
    }


@pytest.mark.parametrize("mode", ["all", "none"])
def test_all_and_none_contribute_no_resolvable_groups(mode):
    """D2 leaves omitted-contacts all/none materialization to D6."""
    profile = _profile()
    spec = ArticulationSpec(name="arm", contacts=ContactSensorInventory(mode=mode))
    effective = _effective_profile(spec, profile, _info("mjcf"))
    builds = {"arm": {"profile": effective, "params": {}, "spec": spec}}

    assert _contact_group_table(builds, _info("mjcf")) == {"arm": {}}


def test_v4_articulation_facts_serialize_body_and_contact_vocabulary():
    """The meta/summary row carries canonical body facts and inventory groups."""
    inventory = ContactSensorInventory(
        mode="declared",
        groups=(ContactSensorGroup(name="feet", bodies=("left_foot",)),),
    )
    art = ArticulationRuntime(
        name="arm",
        joint_names=("j",),
        q0=(0.0,),
        base_body="pelvis",
        base_body_id="base",
        foot_bodies=("left_foot",),
        body_sets={"grasp": ("finger",)},
        contacts=inventory,
    )
    control = ControlRuntime(
        name="arm",
        action_face="arm",
        adapter="manual",
        articulation="arm",
        joint_names=("j",),
        q0=(0.0,),
    )

    facts = articulation_facts(art, control)

    assert facts == {
        "joint_names": ["j"],
        "asset_joint_names": ["j"],
        "action_face": "arm",
        "base_body": "base",
        "foot_bodies": ["left_foot"],
        "body_sets": {"grasp": ["finger"]},
        "contacts": {
            "mode": "declared",
            "groups": [{"name": "feet", "bodies": ["left_foot"]}],
        },
    }
