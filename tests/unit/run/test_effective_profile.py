"""Effective-profile precedence and contact-group native resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from unigym.assets.asset_ref import AssetRef, AssetRefError
from unigym.capabilities.specs import BackendSpec
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.runtime.controllers import (
    ArticulationRuntime,
    ControlRuntime,
    articulation_facts,
)
from unigym.run.build import (
    DATA_SCHEMA_VERSION,
    _camera_base_mount,
    _contact_declarations,
    _effective_profile,
    _fixture_prop_names,
    _prepare_articulation_models,
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
        extra_action_sets={},
        base_body="base",
        foot_bodies=("left_foot",),
        body_sets={"grasp": ("finger",)},
        body_bindings=bindings,
    )


def _info(asset_format: str) -> BackendSpec:
    return BackendSpec("mujoco", frozenset(), asset_formats=(asset_format,))


def _build(profile: RobotProfile, spec: ArticulationSpec, asset_format: str) -> dict:
    return {
        "profile": profile,
        "params": {},
        "spec": spec,
        "asset_format": asset_format,
    }


def test_data_schema_version_is_v6():
    """The run data schema is the explicit v6 cutover version."""
    assert DATA_SCHEMA_VERSION == 6


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
    info = BackendSpec("isaaclab2", frozenset(), asset_formats=("usd", "urdf", "mjcf"))

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
    info = BackendSpec("fixture", frozenset(), asset_formats=("usd", "urdf", "mjcf"))

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
        extra_action_sets={},
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


def test_contact_declarations_map_canonical_bodies_through_the_selected_format():
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

    table, assets = _contact_declarations(builds)

    assert table == {
        "arm": {
            "feet": ("left_foot_link",),
            "grasp": ("hand/finger_link",),
        }
    }
    assert assets == {}


def test_all_keeps_the_selected_asset_path_for_backend_inventory():
    """contacts: all carries no synthetic group; the asset path stays available."""
    profile = RobotProfile(
        name="arm",
        joint_sets={"body": ("j",)},
        default_joint_pos={"j": 0.0},
        assets=AssetRef(usd="/tmp/arm.usd"),
        extra_action_sets={},
        base_body=None,
        foot_bodies=(),
        body_sets={},
        body_bindings={},
    )
    spec = ArticulationSpec(name="arm", contacts=ContactSensorInventory(mode="all"))
    builds = {"arm": _build(profile, spec, "usd")}

    body_sets, assets = _contact_declarations(builds)

    assert body_sets == {}
    assert assets == {"arm": "/tmp/arm.usd"}


def test_all_uses_the_per_articulation_selected_asset_format():
    """contacts: all passes the selected format's path, not a backend-wide one."""
    profile = RobotProfile(
        name="arm",
        joint_sets={"body": ("j",)},
        default_joint_pos={"j": 0.0},
        assets=AssetRef(usd="/tmp/arm.usd", urdf="/tmp/arm.urdf"),
        extra_action_sets={},
        base_body=None,
        foot_bodies=(),
        body_sets={},
        body_bindings={},
    )
    spec = ArticulationSpec(name="arm", contacts=ContactSensorInventory(mode="all"))
    builds = {"arm": _build(profile, spec, "urdf")}

    _, assets = _contact_declarations(builds)

    assert assets == {"arm": "/tmp/arm.urdf"}


def test_declared_contacts_groups_and_profile_body_sets_map_to_native_names():
    """Robot refs resolve body_set names from profile/contact declarations."""
    profile = RobotProfile(
        name="arm",
        joint_sets={"body": ("j",)},
        default_joint_pos={"j": 0.0},
        assets=None,
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={"grasp": ("finger",)},
        body_bindings={
            "base": BodyBinding("base", {"mjcf": "base_link"}),
            "finger": BodyBinding("finger", {"mjcf": "finger_link"}),
            "palm": BodyBinding("palm", {"mjcf": "palm_link"}),
        },
    )
    spec = ArticulationSpec(
        name="arm",
        contacts=ContactSensorInventory(
            mode="declared",
            groups=(ContactSensorGroup(name="hand", bodies=("palm",)),),
        ),
    )
    builds = {"arm": _build(profile, spec, "mjcf")}

    body_sets, _ = _contact_declarations(builds)

    assert body_sets == {"arm": {"grasp": ("finger_link",), "hand": ("palm_link",)}}


def test_none_and_omitted_contacts_declare_no_body_sets():
    """contacts: none / omitted carry no groups; an asset may still be listed."""
    profile = _profile()
    none_spec = ArticulationSpec(name="arm", contacts=ContactSensorInventory(mode="none"))
    omitted = ArticulationSpec(name="arm")

    body_sets, assets = _contact_declarations({"arm": _build(profile, none_spec, "mjcf")})
    assert body_sets == {"arm": {"grasp": ("finger_link",)}}
    assert assets == {}

    omitted_sets, _ = _contact_declarations({"arm": _build(profile, omitted, "mjcf")})
    assert omitted_sets == {"arm": {"grasp": ("finger_link",)}}


def test_camera_base_mount_uses_the_per_articulation_selected_format():
    """A body rig attach resolves through that articulation's selected format."""
    profile = RobotProfile(
        name="arm",
        joint_sets={"body": ("j",)},
        default_joint_pos={"j": 0.0},
        assets=None,
        extra_action_sets={},
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


def test_effective_profile_respects_asset_backends() -> None:
    """A format restricted to other backends is skipped, not selected."""
    profile = replace(
        _profile(),
        assets=AssetRef(mjcf="/assets/arm.xml", usd="/assets/arm.usd"),
        asset_backends={"mjcf": ("mjwarp",), "usd": ("mujoco",)},
    )
    info = BackendSpec("mujoco", frozenset(), asset_formats=("mjcf", "usd"))

    effective, asset_format = _effective_profile(ArticulationSpec(name="arm"), profile, info)

    assert asset_format == "usd"
    assert effective.assets is not None
    assert effective.assets.usd == "/assets/arm.usd"


def test_effective_profile_rejects_formats_restricted_to_other_backends() -> None:
    """No allowed declared format raises the teaching selection error."""
    profile = replace(
        _profile(),
        assets=AssetRef(mjcf="/assets/arm.xml"),
        asset_backends={"mjcf": ("isaaclab2",)},
    )
    info = BackendSpec("mujoco", frozenset(), asset_formats=("mjcf",))

    with pytest.raises(AssetRefError, match="restrictions"):
        _effective_profile(ArticulationSpec(name="arm"), profile, info)


def test_effective_profile_drops_metadata_for_undefined_override_formats() -> None:
    """An inline model override keeps only metadata for formats it declares."""
    profile = replace(
        _profile(),
        assets=AssetRef(mjcf="/assets/arm.xml", usd="/assets/arm.usd"),
        asset_backends={"mjcf": ("mujoco",), "usd": ("isaaclab2",)},
        asset_options={"usd": {"scale": 1.0}},
    )
    spec = ArticulationSpec(name="arm", model=AssetRef(mjcf="/override/arm.xml"))
    info = BackendSpec("mujoco", frozenset(), asset_formats=("mjcf",))

    effective, asset_format = _effective_profile(spec, profile, info)

    assert asset_format == "mjcf"
    assert dict(effective.asset_backends) == {"mjcf": ("mujoco",)}
    assert dict(effective.asset_options) == {}


@dataclass(frozen=True)
class _Plan:
    """Minimal asset boundary: records which reference was resolved."""

    def resolve(self, reference: str) -> str:
        return f"/resolved/{reference}"


@dataclass(frozen=True)
class _Spec:
    """RunSpec stand-in carrying only what model preparation reads."""

    articulations: tuple[ArticulationSpec, ...]
    robot_profiles: dict[str, RobotProfile] | None = None


def test_prepare_articulation_models_respects_asset_backends() -> None:
    """Profile-backed models resolve the format the backend is allowed to use."""
    profile = replace(
        _profile(),
        assets=AssetRef(mjcf="arm.xml", usd="arm.usd"),
        asset_backends={"mjcf": ("mjwarp",), "usd": ("mujoco",)},
    )
    spec = _Spec(
        articulations=(ArticulationSpec(name="arm"),),
        robot_profiles={"arm": profile},
    )
    info = BackendSpec("mujoco", frozenset(), asset_formats=("mjcf", "usd"))

    prepared = _prepare_articulation_models(spec, info, _Plan())

    model = prepared.articulations[0].model
    assert model is not None
    assert model.usd == "/resolved/arm.usd"
    assert model.mjcf == "arm.xml"


def test_prepare_articulation_models_leaves_forbidden_formats_uncomposed() -> None:
    """A profile whose only format is restricted elsewhere is not injected."""
    profile = replace(
        _profile(),
        assets=AssetRef(mjcf="arm.xml"),
        asset_backends={"mjcf": ("isaaclab2",)},
    )
    spec = _Spec(
        articulations=(ArticulationSpec(name="arm"),),
        robot_profiles={"arm": profile},
    )
    info = BackendSpec("mujoco", frozenset(), asset_formats=("mjcf",))

    prepared = _prepare_articulation_models(spec, info, _Plan())

    assert prepared is spec
    assert prepared.articulations[0].model is None


def test_v5_articulation_facts_serialize_body_and_contact_vocabulary():
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
        adapter="manual",
        articulation="arm",
        joint_names=("j",),
        q0=(0.0,),
    )

    facts = articulation_facts(art, control)

    assert facts == {
        "joint_names": ["j"],
        "asset_joint_names": ["j"],
        "action_group": "arm",
        "base_body": "base",
        "foot_bodies": ["left_foot"],
        "body_sets": {"grasp": ["finger"]},
        "contacts": {
            "mode": "declared",
            "groups": [{"name": "feet", "bodies": ["left_foot"]}],
        },
    }


def test_fixture_contact_ownership_is_anchored_articulated_props_only():
    from types import SimpleNamespace

    props = SimpleNamespace(
        specs=[
            SimpleNamespace(name="microwave", type="articulation"),
            SimpleNamespace(name="door", type="usd_articulation"),
            SimpleNamespace(name="laptop", type="free_articulation"),
            SimpleNamespace(name="crate", type="rigid_object"),
        ]
    )
    assert _fixture_prop_names(props) == {"microwave", "door"}
    assert _fixture_prop_names(None) == set()
