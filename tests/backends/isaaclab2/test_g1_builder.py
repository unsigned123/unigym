"""G1 realizes through the generic IsaacLab builder over the 43-DoF USD."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import replace

import pytest

pytestmark = [pytest.mark.sim, pytest.mark.usefixtures("simulation_app")]

ARTICULATION = "g1"


def _profile():
    from unigym.robots import build_robot_profile

    return build_robot_profile({"name": "g1"})


def _actuation():
    """Four explicit standard_pd groups covering the 29 policy joints."""
    from unigym.actuators import ActuationGroup
    from unigym.robots import g1_constants as g1c

    order = list(g1c.G1_JOINT_ORDER_IL)
    legs = [name for name in order if "hip" in name or "knee" in name]
    feet = [name for name in order if "ankle" in name]
    waist = [name for name in order if name.startswith("waist_")]
    arms = [name for name in order if "shoulder" in name or "elbow" in name or "wrist" in name]
    assert len(legs) + len(feet) + len(waist) + len(arms) == len(order)

    def group(name, joints, kp, kd, effort):
        return ActuationGroup(
            name=name,
            law="standard_pd",
            joints=tuple(joints),
            gains={"kp": {joint: kp for joint in joints}, "kd": {joint: kd for joint in joints}},
            effort_limit={joint: effort for joint in joints},
        )

    return (
        group("legs", legs, 100.0, 2.0, 139.0),
        group("feet", feet, 40.0, 1.0, 50.0),
        group("waist", waist, 200.0, 5.0, 88.0),
        group("arms", arms, 40.0, 1.0, 25.0),
    )


def _spec(g1_usd, *, actuation=None, spawn=None, passive_joints=()):
    from unigym.assets.asset_ref import AssetRef
    from unigym.scenes.articulation import ArticulationSpec

    kwargs = {
        "name": ARTICULATION,
        "model": AssetRef(usd=str(g1_usd)),
        "actuation": _actuation() if actuation is None else actuation,
        "passive_joints": tuple(passive_joints),
    }
    if spawn is not None:
        kwargs["spawn"] = spawn
    return ArticulationSpec(**kwargs)


def _realize(spec, *, params=None, profile=None):
    from unigym.backends.isaaclab2.robots import build_robot_realization

    return build_robot_realization(
        _profile() if profile is None else profile,
        spec=spec,
        params={} if params is None else params,
        asset_plan=None,
    )


def test_g1_realization_builds_the_generic_articulation_cfg(g1_usd):
    """The real G1 cfg carries prim, spawn, q0, actuators, groups and limits."""
    from isaaclab.assets import ArticulationCfg

    profile = _profile()
    spawn_options = {
        "rigid_props": {
            "linear_damping": 0.15,
            "max_depenetration_velocity": 1.0,
            "max_linear_velocity": 100.0,
            "max_angular_velocity": 100.0,
        },
        "articulation_props": {
            "enabled_self_collisions": False,
            "solver_position_iteration_count": 8,
            "solver_velocity_iteration_count": 4,
        },
        "visible": False,
    }
    cfg = _realize(
        _spec(g1_usd, spawn={"pos": [1.0, 2.0, 0.91], "yaw": math.pi / 2}),
        params={
            "prim_path": "/World/envs/env_0/G1",
            "spawn_options": spawn_options,
            "soft_joint_pos_limit_factor": 0.9,
        },
    )

    assert isinstance(cfg, ArticulationCfg)
    assert cfg.prim_path == "/World/envs/env_0/G1"
    assert cfg.spawn.usd_path == str(g1_usd)
    assert cfg.spawn.activate_contact_sensors is True
    assert cfg.spawn.visible is False
    assert cfg.init_state.pos == (1.0, 2.0, 0.91)
    assert cfg.init_state.rot == pytest.approx(
        (math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0))
    )
    assert cfg.init_state.joint_pos == dict(profile.default_joint_pos)
    assert cfg.collision_group == 0
    assert cfg.soft_joint_pos_limit_factor == 0.9
    assert set(cfg.actuators) == {"legs", "feet", "waist", "arms"}
    legs = cfg.actuators["legs"]
    assert legs.stiffness["left_hip_pitch_joint"] == 100.0
    assert legs.damping["left_hip_pitch_joint"] == 2.0
    assert legs.effort_limit_sim["left_hip_pitch_joint"] == 139.0


def test_g1_realization_converts_nested_spawn_props_to_schema_cfgs(g1_usd):
    """The run-level rigid/articulation props become the real schema cfg types."""
    from isaaclab.sim.schemas import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg

    cfg = _realize(
        _spec(g1_usd),
        params={
            "spawn_options": {
                "rigid_props": {"max_depenetration_velocity": 1.0},
                "articulation_props": {"enabled_self_collisions": False},
            }
        },
    )

    assert isinstance(cfg.spawn.rigid_props, RigidBodyPropertiesCfg)
    assert cfg.spawn.rigid_props.max_depenetration_velocity == 1.0
    assert isinstance(cfg.spawn.articulation_props, ArticulationRootPropertiesCfg)
    assert cfg.spawn.articulation_props.enabled_self_collisions is False


def test_g1_realization_reads_the_explicit_soft_factor_and_collision_group(g1_usd):
    """Realizer-only params override the generic defaults."""
    cfg = _realize(
        _spec(g1_usd),
        params={"collision_group": 3, "soft_joint_pos_limit_factor": 0.8, "debug_vis": True},
    )

    assert cfg.collision_group == 3
    assert cfg.soft_joint_pos_limit_factor == 0.8
    assert cfg.debug_vis is True


def test_g1_realization_absent_spawn_uses_the_profile_base_height(g1_usd):
    """An absent run-level spawn uses the profile base_height and zero yaw."""
    cfg = _realize(_spec(g1_usd))

    assert cfg.init_state.pos[0] == 0.0
    assert cfg.init_state.pos[1] == 0.0
    assert cfg.init_state.pos[2] == pytest.approx(0.78)
    assert cfg.init_state.rot == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_generic_realizer_defaults_the_soft_limit_factor_to_one(g1_usd):
    """Without an explicit param the generic realizer uses the 1.0 default."""
    cfg = _realize(_spec(g1_usd))

    assert cfg.soft_joint_pos_limit_factor == 1.0


def test_realizer_builds_a_renamed_profile_through_the_generic_path(g1_usd):
    """A profile whose name is not 'g1' builds the same generic cfg."""
    from unigym.env.naming import ENV_REGEX_NS

    profile = replace(_profile(), name="custom_humanoid")
    cfg = _realize(_spec(g1_usd), profile=profile)

    assert cfg.prim_path == f"{ENV_REGEX_NS}/Robot"
    assert cfg.spawn.usd_path == str(g1_usd)
    assert cfg.init_state.joint_pos == dict(profile.default_joint_pos)
    assert set(cfg.actuators) == {"legs", "feet", "waist", "arms"}


def test_realizer_absent_spawn_without_base_height_requires_an_explicit_spawn(g1_usd):
    """base_height=None with no spawn fails in the builder, not the realizer."""
    profile = replace(_profile(), base_height=None)

    with pytest.raises(ValueError, match="no spawn and the builder received no default_pos"):
        _realize(_spec(g1_usd), profile=profile)


def test_realizer_absent_base_height_with_an_explicit_spawn_builds(g1_usd):
    """base_height=None is legal when the run-level spawn supplies the pose."""
    profile = replace(_profile(), base_height=None)
    cfg = _realize(
        _spec(g1_usd, spawn={"pos": [0.5, -0.25, 0.4]}),
        profile=profile,
    )

    assert cfg.init_state.pos == (0.5, -0.25, 0.4)


def test_g1_realization_rejects_a_non_standard_law(g1_usd):
    """The IsaacLab path has no mapping for non-standard_pd laws."""
    from unigym.actuators import ActuationGroup

    non_standard = (
        ActuationGroup(
            name="legs",
            law="computed_torque_pd",
            joints=("left_hip_pitch_joint",),
            gains={"kp": 100.0, "kd": 2.0},
        ),
    )

    with pytest.raises(ValueError, match="no native Isaac Lab actuator mapping"):
        _realize(_spec(g1_usd, actuation=non_standard))


def test_g1_realization_requires_run_level_actuation(g1_usd):
    """The G1 path has no hardcoded actuator table to fall back to."""
    from unigym.scenes.articulation import ArticulationSpec
    from unigym.assets.asset_ref import AssetRef

    spec = ArticulationSpec(name=ARTICULATION, model=AssetRef(usd=str(g1_usd)))

    with pytest.raises(ValueError, match="requires explicit.*actuation"):
        _realize(spec)


def test_g1_realization_has_no_old_module_or_template():
    """The G1-specific realizer module and G1_29DOF_CFG template are gone."""
    import unigym.backends.isaaclab2.robots as robots

    assert importlib.util.find_spec("unigym.backends.isaaclab2.robots.g1") is None
    assert not hasattr(robots, "G1_29DOF_CFG")


def test_g1_contact_all_materializes_one_sensor_per_body(g1_usd):
    """contacts: all uses the pre-clone USD inventory and reads per-body rows."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    from unigym.adapters.manual import ManualAdapter
    from unigym.backends.isaaclab2.env import IsaaclabEnv
    from unigym.backends.isaaclab2.usd import robot_rigid_bodies
    from unigym.contacts.base import ALL_BODIES
    from unigym.contacts.manager import build_contact_manager
    from unigym.robots import g1_constants as g1c
    from unigym.runtime.controllers import build_control_runtimes
    from unigym.runtime.controls import ControlGroup

    inventory = robot_rigid_bodies(str(g1_usd))
    paths = [body.path for body in inventory]
    assert "pelvis" in paths
    assert "left_ankle_roll_link" in paths
    assert "right_ankle_roll_link" in paths
    assert "left_hand/left_hand_index_0_link" in paths
    assert "right_hand/right_hand_thumb_2_link" in paths
    assert "left_hand/left_hand_palm_link" not in paths
    assert "right_hand/right_hand_palm_link" not in paths

    manager = build_contact_manager(
        [{"name": "all", "sensor_bodies": [ALL_BODIES], "frame": "world"}],
        backend="isaaclab2",
        contact_assets={"g1": str(g1_usd)},
    )
    assert manager is not None

    profile = _profile()
    spec = _spec(g1_usd, passive_joints=g1c.G1_HAND_JOINTS)
    realization = _realize(spec)
    group = ControlGroup.from_mapping(
        {
            "name": ARTICULATION,
            "adapter": "manual",
            "articulation": ARTICULATION,
            "joints": list(g1c.G1_JOINT_ORDER_IL),
            "drive": "position_pd",
            "target": "position",
        }
    )
    controls = build_control_runtimes((group,), (spec,), {ARTICULATION: profile})
    terrain_cfg = {
        "ground": AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    }
    sim_utils.create_new_stage()
    env = IsaaclabEnv(
        robot_profile=profile,
        adapter=ManualAdapter(),
        terrain_cfg=terrain_cfg,
        num_envs=1,
        episode_length_s=100.0,
        decimation=4,
        sim_dt=0.005,
        device="cuda:0",
        articulations=(spec,),
        controls=controls,
        profiles={ARTICULATION: profile},
        robot_realizations={ARTICULATION: realization},
        contact_manager=manager,
    )
    try:
        assert manager.backend.group_axes(manager.specs[0]) == (len(paths), 0)

        value = manager.read()["contact/all"]
        assert tuple(value.shape) == (1, len(paths), 3)

        assert env.reset_all().tolist() == [0]
        after_reset = manager.read()["contact/all"]
        assert tuple(after_reset.shape) == (1, len(paths), 3)
        assert after_reset[0, paths.index("pelvis")].tolist() == [0.0, 0.0, 0.0]
    finally:
        env.close()


def test_g1_builds_and_resets_a_real_isaaclab_env(g1_usd):
    """A real G1 env loads the 43-DoF asset and resets over 29 driven joints."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    from unigym.adapters.manual import ManualAdapter
    from unigym.backends.isaaclab2.env import IsaaclabEnv
    from unigym.robots import g1_constants as g1c
    from unigym.runtime.controllers import build_control_runtimes
    from unigym.runtime.controls import ControlGroup

    profile = _profile()
    spec = _spec(g1_usd, passive_joints=g1c.G1_HAND_JOINTS)
    realization = _realize(spec)
    group = ControlGroup.from_mapping(
        {
            "name": ARTICULATION,
            "adapter": "manual",
            "articulation": ARTICULATION,
            "joints": list(g1c.G1_JOINT_ORDER_IL),
            "drive": "position_pd",
            "target": "position",
        }
    )
    controls = build_control_runtimes((group,), (spec,), {ARTICULATION: profile})
    terrain_cfg = {
        "ground": AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    }
    sim_utils.create_new_stage()
    env = IsaaclabEnv(
        robot_profile=profile,
        adapter=ManualAdapter(),
        terrain_cfg=terrain_cfg,
        num_envs=1,
        episode_length_s=100.0,
        decimation=4,
        sim_dt=0.005,
        device="cuda:0",
        articulations=(spec,),
        controls=controls,
        profiles={ARTICULATION: profile},
        robot_realizations={ARTICULATION: realization},
    )
    try:
        art = env.articulations[ARTICULATION]
        assert len(art.joint_names) == 43
        assert set(art.joint_names) == set(g1c.G1_JOINT_ORDER_FULL)
        assert len(env.controls[ARTICULATION].joint_names) == 29

        bodies = [body.name for body in env.model_set.articulations[ARTICULATION].bodies]
        assert "pelvis" in bodies
        assert "left_ankle_roll_link" in bodies
        assert "right_ankle_roll_link" in bodies

        assert env.reset_all().tolist() == [0]
    finally:
        env.close()
