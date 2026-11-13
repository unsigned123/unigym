"""Generic robot builder: asset dispatch, spawn/init state and actuator seam."""

from __future__ import annotations

import math

import pytest

from unigym.assets.asset_ref import AssetRef
from unigym.backends.isaaclab2.robots.builder import build_articulation_cfg
from unigym.env.naming import ENV_REGEX_NS
from unigym.robots.base import RobotProfile
from unigym.scenes.articulation import ArticulationSpec

_DEFAULT_ASSETS = AssetRef(usd="/assets/arm.usd", urdf="/assets/arm.urdf")
_ABSENT = object()


class _Cfg:
    """Recording cfg constructor: the builder only forwards kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class _InitialStateCfg(_Cfg):
    pass


class _ArticulationCfg(_Cfg):
    InitialStateCfg = _InitialStateCfg


class _UsdFileCfg(_Cfg):
    pass


class _UrdfFileCfg(_Cfg):
    pass


class _RigidBodyPropertiesCfg(_Cfg):
    pass


class _ArticulationRootPropertiesCfg(_Cfg):
    pass


_CLASSES = {
    "ArticulationCfg": _ArticulationCfg,
    "UsdFileCfg": _UsdFileCfg,
    "UrdfFileCfg": _UrdfFileCfg,
    "RigidBodyPropertiesCfg": _RigidBodyPropertiesCfg,
    "ArticulationRootPropertiesCfg": _ArticulationRootPropertiesCfg,
}


class _AssetPlan:
    """Resolve every ref to a fixed absolute path and record the calls."""

    def __init__(self, resolved):
        self.resolved = resolved
        self.calls = []

    def resolve(self, value):
        self.calls.append(value)
        return self.resolved


def _profile(*, name="arm", assets=_DEFAULT_ASSETS):
    return RobotProfile(
        name=name,
        joint_sets={"body": ("joint",)},
        default_joint_pos={"joint": 0.25},
        assets=assets,
        action_faces={},
        base_body=None,
        foot_bodies=(),
        body_sets={},
        body_bindings={},
    )


def _spec(*, name="arm", root="free", spawn=_ABSENT, model=_ABSENT):
    kwargs = {"name": name, "root": root}
    if spawn is not _ABSENT:
        kwargs["spawn"] = spawn
    if model is not _ABSENT:
        kwargs["model"] = model
    return ArticulationSpec(**kwargs)


def _actuator_plan(profile, spec):
    return {"body": ("actuator", profile.name, spec.name)}


def _build(
    *,
    profile=None,
    spec=None,
    params=None,
    asset_format="usd",
    actuator_plan=_actuator_plan,
    asset_plan=None,
    classes=None,
    default_pos=None,
    spawn_options=None,
    soft_joint_pos_limit_factor=1.0,
):
    return build_articulation_cfg(
        _profile() if profile is None else profile,
        spec=_spec(spawn={"pos": [0.0, 0.0, 1.0]}) if spec is None else spec,
        params={} if params is None else params,
        asset_format=asset_format,
        actuator_plan=actuator_plan,
        asset_plan=asset_plan,
        classes=_CLASSES if classes is None else classes,
        default_pos=default_pos,
        spawn_options=spawn_options,
        soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    )


def test_usd_asset_dispatch_builds_usd_spawn():
    """The usd format selects UsdFileCfg with the contact-sensor opt-in."""
    profile = _profile()

    cfg = _build(profile=profile)

    assert isinstance(cfg, _ArticulationCfg)
    assert isinstance(cfg.kwargs["spawn"], _UsdFileCfg)
    assert cfg.kwargs["spawn"].kwargs == {
        "usd_path": "/assets/arm.usd",
        "activate_contact_sensors": True,
    }
    state = cfg.kwargs["init_state"]
    assert state.kwargs["joint_pos"] == dict(profile.default_joint_pos)
    assert state.kwargs["joint_vel"] == {".*": 0.0}
    assert cfg.kwargs["collision_group"] == 0
    assert cfg.kwargs["debug_vis"] is False
    assert cfg.kwargs["soft_joint_pos_limit_factor"] == 1.0


@pytest.mark.parametrize(
    ("root", "expected"),
    [("fixed", True), ("welded", True), ("free", False)],
    ids=["fixed", "welded", "free"],
)
def test_urdf_asset_dispatch_sets_fix_base(root, expected):
    """The urdf format selects UrdfFileCfg and fixes only declared base roots."""
    cfg = _build(
        spec=_spec(root=root),
        asset_format="urdf",
        default_pos=(0.0, 0.0, 1.0),
    )

    assert isinstance(cfg.kwargs["spawn"], _UrdfFileCfg)
    assert cfg.kwargs["spawn"].kwargs == {
        "asset_path": "/assets/arm.urdf",
        "fix_base": expected,
        "merge_fixed_joints": False,
        "collision_from_visuals": False,
        "make_instanceable": True,
    }


def test_missing_asset_fails():
    """A profile without an asset and a spec without a model fail fast."""
    with pytest.raises(ValueError, match="declares no asset"):
        _build(profile=_profile(assets=None))


def test_missing_selected_format_fails():
    """The selected format must carry a reference."""
    profile = _profile(assets=AssetRef(usd="/assets/arm.usd"))

    with pytest.raises(ValueError, match="asset format 'urdf' has no reference"):
        _build(profile=profile, asset_format="urdf")


def test_empty_selected_ref_fails():
    """An empty selected reference is a missing asset, not a path."""
    profile = _profile(assets=AssetRef(usd=""))

    with pytest.raises(ValueError, match="assets.usd=''"):
        _build(profile=profile)


def test_unsupported_asset_format_fails():
    """A format the backend does not declare fails with the supported set."""
    with pytest.raises(ValueError, match="unsupported asset_format 'mjcf'"):
        _build(asset_format="mjcf")


@pytest.mark.parametrize(
    "ref",
    ["pack:arm/arm.usd", "assets/arm.usd", "${ARM_USD}"],
    ids=["pack", "relative", "placeholder"],
)
def test_unresolved_ref_without_asset_plan_fails(ref):
    """Pack, relative and placeholder refs need the composition asset plan."""
    with pytest.raises(ValueError, match="needs an asset plan"):
        _build(profile=_profile(assets=AssetRef(usd=ref)))


def test_absolute_ref_without_asset_plan_passes_verbatim():
    """An absolute local reference is already resolved."""
    cfg = _build(profile=_profile(assets=AssetRef(usd="/opt/assets/arm.usd")))

    assert cfg.kwargs["spawn"].kwargs["usd_path"] == "/opt/assets/arm.usd"


def test_asset_plan_resolves_the_selected_ref():
    """A provided asset plan owns reference resolution."""
    plan = _AssetPlan("/resolved/arm.usd")

    cfg = _build(profile=_profile(assets=AssetRef(usd="pack:arm/arm.usd")), asset_plan=plan)

    assert plan.calls == ["pack:arm/arm.usd"]
    assert cfg.kwargs["spawn"].kwargs["usd_path"] == "/resolved/arm.usd"


def test_default_prim_path_uses_the_env_regex_namespace():
    """The default prim path is the framework's per-env Robot convention."""
    cfg = _build()

    assert cfg.kwargs["prim_path"] == f"{ENV_REGEX_NS}/Robot"


def test_explicit_prim_path_wins():
    """A robot-section prim_path param replaces the default."""
    cfg = _build(params={"prim_path": "/World/envs/env_0/Arm"})

    assert cfg.kwargs["prim_path"] == "/World/envs/env_0/Arm"


def test_non_string_prim_path_fails():
    """A non-string prim_path is a configuration error."""
    with pytest.raises(TypeError, match="'prim_path' must be a string"):
        _build(params={"prim_path": 7})


def test_spawn_present_sets_init_pos_and_yaw_quaternion():
    """spec.spawn supplies the init position and the yaw quaternion."""
    spec = _spec(spawn={"pos": [1.0, 2.0, 3.0], "yaw": math.pi / 2})

    state = _build(spec=spec).kwargs["init_state"]

    assert state.kwargs["pos"] == (1.0, 2.0, 3.0)
    assert state.kwargs["rot"] == pytest.approx(
        (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    )


def test_spawn_present_without_yaw_defaults_zero():
    """A spawn position without yaw keeps the identity orientation."""
    state = _build(spec=_spec(spawn={"pos": [1.0, 2.0, 3.0]})).kwargs["init_state"]

    assert state.kwargs["rot"] == (1.0, 0.0, 0.0, 0.0)


def test_spawn_absent_uses_explicit_default_pos():
    """An absent spawn uses the caller's explicit default position."""
    state = _build(spec=_spec(), default_pos=(0.1, 0.2, 0.9)).kwargs["init_state"]

    assert state.kwargs["pos"] == (0.1, 0.2, 0.9)
    assert state.kwargs["rot"] == (1.0, 0.0, 0.0, 0.0)


def test_spawn_absent_without_default_pos_fails():
    """The builder never invents a spawn height."""
    with pytest.raises(ValueError, match="no default_pos"):
        _build(spec=_spec())


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
def test_non_finite_default_pos_fails(value):
    """A non-finite default position is rejected."""
    with pytest.raises(ValueError, match="must be finite"):
        _build(spec=_spec(), default_pos=(0.0, 0.0, value))


def test_actuator_plan_receives_profile_and_spec():
    """The actuator seam receives the neutral pair and its mapping is passed."""
    profile = _profile()
    spec = _spec()
    seen = []
    planned = {"body": object()}

    def plan(given_profile, given_spec):
        seen.append((given_profile, given_spec))
        return planned

    cfg = _build(
        profile=profile,
        spec=spec,
        actuator_plan=plan,
        default_pos=(0.0, 0.0, 1.0),
    )

    assert seen == [(profile, spec)]
    assert cfg.kwargs["actuators"] == planned


def test_missing_actuator_plan_fails():
    """There is no empty-actuator fallback."""
    with pytest.raises(TypeError, match="actuator_plan must be a callable"):
        _build(actuator_plan=None)


def test_spawn_options_merge_into_the_spawn_constructor():
    """The escape hatch merges flat props over the builder's spawn kwargs."""
    cfg = _build(spawn_options={"visible": False, "mass": 3.0})

    assert cfg.kwargs["spawn"].kwargs == {
        "usd_path": "/assets/arm.usd",
        "activate_contact_sensors": True,
        "visible": False,
        "mass": 3.0,
    }


def test_nested_spawn_props_become_schema_cfgs_and_flat_props_stay_verbatim():
    """rigid_props/articulation_props mappings convert; flat props pass through."""
    cfg = _build(
        spawn_options={
            "rigid_props": {"linear_damping": 0.1, "max_depenetration_velocity": 1.0},
            "articulation_props": {"enabled_self_collisions": False},
            "visible": False,
        }
    )

    rigid = cfg.kwargs["spawn"].kwargs["rigid_props"]
    root = cfg.kwargs["spawn"].kwargs["articulation_props"]
    assert isinstance(rigid, _RigidBodyPropertiesCfg)
    assert rigid.kwargs == {"linear_damping": 0.1, "max_depenetration_velocity": 1.0}
    assert isinstance(root, _ArticulationRootPropertiesCfg)
    assert root.kwargs == {"enabled_self_collisions": False}
    assert cfg.kwargs["spawn"].kwargs["visible"] is False


def test_explicit_soft_joint_pos_limit_factor_reaches_the_cfg():
    """The builder forwards the soft joint-limit scale to ArticulationCfg."""
    cfg = _build(soft_joint_pos_limit_factor=0.9)

    assert cfg.kwargs["soft_joint_pos_limit_factor"] == 0.9


def test_nested_prop_cfg_objects_pass_through_unchanged():
    """An already-built schema cfg is not rebuilt from a mapping."""
    rigid = _RigidBodyPropertiesCfg(linear_damping=0.2)

    cfg = _build(spawn_options={"rigid_props": rigid})

    assert cfg.kwargs["spawn"].kwargs["rigid_props"] is rigid


@pytest.mark.parametrize(
    ("asset_format", "key"),
    [("usd", "usd_path"), ("usd", "activate_contact_sensors"), ("urdf", "asset_path")],
    ids=["usd-path", "usd-contact-sensors", "urdf-path"],
)
def test_duplicate_spawn_option_fails(asset_format, key):
    """A spawn option cannot override a builder-owned spawn key."""
    with pytest.raises(ValueError, match="duplicate spawn keys"):
        _build(asset_format=asset_format, spawn_options={key: "override"})


@pytest.mark.parametrize(
    "missing_key",
    [
        "ArticulationCfg",
        "UsdFileCfg",
        "UrdfFileCfg",
        "RigidBodyPropertiesCfg",
        "ArticulationRootPropertiesCfg",
    ],
)
def test_missing_class_key_fails(missing_key):
    """Every declared class key is required."""
    classes = {key: value for key, value in _CLASSES.items() if key != missing_key}

    with pytest.raises(ValueError, match=f"missing \\['{missing_key}'\\]"):
        _build(classes=classes)


def test_spec_model_overrides_profile_assets():
    """A run-level model override wins over the profile asset in the builder."""
    profile = _profile(assets=AssetRef(usd="/assets/profile.usd", urdf="/assets/profile.urdf"))
    spec = _spec(model=AssetRef(usd="/assets/run.usd", urdf="/assets/run.urdf"))

    cfg = _build(profile=profile, spec=spec, default_pos=(0.0, 0.0, 1.0))

    assert cfg.kwargs["spawn"].kwargs["usd_path"] == "/assets/run.usd"
