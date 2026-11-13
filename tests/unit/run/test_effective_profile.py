"""Effective-profile precedence and contact-group native resolution."""

from __future__ import annotations

import pytest

from unigym.assets.asset_ref import AssetRef
from unigym.backends.registry import BackendInfo
from unigym.contacts.base import ALL_BODIES
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.runtime.controllers import (
    ArticulationRuntime,
    ControlRuntime,
    articulation_facts,
)
from unigym.run.build import (
    DATA_SCHEMA_VERSION,
    _camera_base_mount,
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
    return BackendInfo("mujoco", (0, 0), frozenset(), asset_formats=(asset_format,))


def _build(profile: RobotProfile, spec: ArticulationSpec, asset_format: str) -> dict:
    return {
        "profile": profile,
        "params": {},
        "spec": spec,
        "asset_format": asset_format,
    }


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

    effective, asset_format = _effective_profile(spec, profile, _info("mjcf"))

    assert asset_format == "mjcf"
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

    effective, asset_format = _effective_profile(spec, profile, _info("usd"))

    assert asset_format == "usd"
    assert effective.base_body == "base"
    assert effective.foot_bodies == ("left_foot",)
    assert effective.body_sets == {"grasp": ("finger",)}


def test_asset_format_selection_is_per_articulation():
    """Each articulation selects its own first compatible declared format."""
    info = BackendInfo("isaaclab2", (2, 3), frozenset(), asset_formats=("usd", "urdf", "mjcf"))

    usd_profile, usd_format = _effective_profile(
        ArticulationSpec(name="usd_arm", model=AssetRef(usd="/assets/arm.usd")),
        _profile(),
        info,
    )
    mjcf_profile, mjcf_format = _effective_profile(
        ArticulationSpec(name="mjcf_arm", model=AssetRef(mjcf="/assets/arm.xml")),
        _profile(),
        info,
    )

    assert usd_format == "usd"
    assert mjcf_format == "mjcf"
    assert usd_profile.assets is not None
    assert usd_profile.assets.usd == "/assets/arm.usd"
    assert mjcf_profile.assets is not None
    assert mjcf_profile.assets.mjcf == "/assets/arm.xml"


def test_no_effective_asset_uses_the_first_declared_binding_format():
    """A provider/scene-owned asset has no AssetRef; the binding key is the head."""
    info = BackendInfo("fixture", (0, 0), frozenset(), asset_formats=("usd", "urdf", "mjcf"))

    effective, asset_format = _effective_profile(ArticulationSpec(name="arm"), _profile(), info)

    assert effective.assets is None
    assert asset_format == "usd"


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
    effective, _ = _effective_profile(ArticulationSpec(name="arm"), profile, _info("usd"))
    assert effective.base_body == "base"


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
    effective, asset_format = _effective_profile(spec, profile, _info("usd"))
    builds = {"arm": _build(effective, spec, asset_format)}

    table, assets = _contact_group_table(builds)

    assert table == {
        "arm": {
            "feet": ("left_foot_link",),
            "grasp": ("hand/finger_link",),
        }
    }
    assert assets == {}


def test_all_contributes_one_sentinel_group_and_the_selected_asset():
    """contacts: all maps to the typed sentinel plus the resolved asset path."""
    profile = RobotProfile(
        name="arm",
        joint_sets={"body": ("j",)},
        default_joint_pos={"j": 0.0},
        assets=AssetRef(usd="/tmp/arm.usd"),
        action_faces={},
        base_body=None,
        foot_bodies=(),
        body_sets={},
        body_bindings={},
    )
    spec = ArticulationSpec(name="arm", contacts=ContactSensorInventory(mode="all"))
    builds = {"arm": _build(profile, spec, "usd")}

    table, assets = _contact_group_table(builds)

    assert table == {"arm": {"all": (ALL_BODIES,)}}
    assert assets == {"arm": "/tmp/arm.usd"}


def test_all_uses_the_per_articulation_selected_asset_format():
    """contacts: all passes the selected format's path, not a backend-wide one."""
    profile = RobotProfile(
        name="arm",
        joint_sets={"body": ("j",)},
        default_joint_pos={"j": 0.0},
        assets=AssetRef(usd="/tmp/arm.usd", urdf="/tmp/arm.urdf"),
        action_faces={},
        base_body=None,
        foot_bodies=(),
        body_sets={},
        body_bindings={},
    )
    spec = ArticulationSpec(name="arm", contacts=ContactSensorInventory(mode="all"))
    builds = {"arm": _build(profile, spec, "urdf")}

    _, assets = _contact_group_table(builds)

    assert assets == {"arm": "/tmp/arm.urdf"}


def test_omitted_contacts_with_a_contacts_section_default_to_all(capsys):
    """An omitted inventory under a run-level contacts section becomes all."""
    profile = _profile()
    spec = ArticulationSpec(name="arm")
    builds = {"arm": _build(profile, spec, "mjcf")}

    table, assets = _contact_group_table(builds, run_contacts=True)

    assert table == {"arm": {"all": (ALL_BODIES,)}}
    assert assets == {}
    warning = capsys.readouterr().out
    assert "omits `contacts`" in warning
    assert "ContactSensorCfg per rigid body" in warning


def test_none_suppresses_the_implicit_all_default():
    """contacts: none carries no group even under a run-level section."""
    profile = _profile()
    spec = ArticulationSpec(name="arm", contacts=ContactSensorInventory(mode="none"))
    builds = {"arm": _build(profile, spec, "mjcf")}

    table, assets = _contact_group_table(builds, run_contacts=True)

    assert table == {}
    assert assets == {}


def test_camera_base_mount_uses_the_per_articulation_selected_format():
    """A body rig attach resolves through that articulation's selected format."""
    profile = RobotProfile(
        name="arm",
        joint_sets={"body": ("j",)},
        default_joint_pos={"j": 0.0},
        assets=None,
        action_faces={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={
            "base": BodyBinding("base", {"mjcf": "pelvis_mjcf", "urdf": "pelvis_urdf"}),
        },
    )
    cameras = [{"name": "chase", "rig": {"type": "body"}}]

    out = _camera_base_mount(cameras, {"arm": profile}, {"arm": "urdf"})

    assert out[0]["rig"]["attach"] == "pelvis_urdf"


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
