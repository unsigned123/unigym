"""Two real articulations own independent contact groups on the dog lane."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.sim,
    pytest.mark.usefixtures("simulation_app", "g1_pack"),
    pytest.mark.requires_test_packs("g1", "isaaclab_cartpole"),
]

CARTPOLE = "cartpole"
G1 = "g1"


def _cartpole_profile():
    from unigym.robots.base import RobotProfile

    joints = ("slider_to_cart", "cart_to_pole")
    return RobotProfile(
        name=CARTPOLE,
        joint_sets={"body": joints},
        default_joint_pos={joint: 0.0 for joint in joints},
        assets=None,
        action_faces={},
        base_body=None,
        foot_bodies=(),
        body_sets={},
        body_bindings={},
    )


def _cartpole_spec():
    from unigym.actuators import ActuationGroup
    from unigym.scenes.articulation import ArticulationSpec

    return ArticulationSpec(
        name=CARTPOLE,
        actuation=(
            ActuationGroup(
                name=CARTPOLE,
                law="standard_pd",
                joints=("slider_to_cart", "cart_to_pole"),
                gains={"kp": 1.0, "kd": 0.1},
            ),
        ),
    )


def _g1_profile():
    from unigym.robots import build_robot_profile

    return build_robot_profile({"name": G1})


def _g1_actuation(profile):
    from unigym.actuators import ActuationGroup

    order = list(profile.joint_sets["body"])
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
            gains={
                "kp": {joint: kp for joint in joints},
                "kd": {joint: kd for joint in joints},
            },
            effort_limit={joint: effort for joint in joints},
        )

    return (
        group("legs", legs, 100.0, 2.0, 139.0),
        group("feet", feet, 40.0, 1.0, 50.0),
        group("waist", waist, 200.0, 5.0, 88.0),
        group("arms", arms, 40.0, 1.0, 25.0),
    )


def _g1_spec(g1_usd, profile):
    from unigym.assets.asset_ref import AssetRef
    from unigym.scenes.articulation import ArticulationSpec

    return ArticulationSpec(
        name=G1,
        model=AssetRef(usd=str(g1_usd)),
        actuation=_g1_actuation(profile),
        passive_joints=tuple(profile.joint_sets["hand"]),
    )


def _g1_realization(spec, profile):
    from unigym.backends.isaaclab2.robots import build_robot_realization

    return build_robot_realization(profile, spec=spec, params={}, asset_plan=None)


def _control(name, profile):
    from unigym.runtime.controls import ControlGroup

    return ControlGroup.from_mapping(
        {
            "name": name,
            "adapter": "manual",
            "articulation": name,
            "joints": list(profile.joint_sets["body"]),
            "drive": "position_pd",
            "target": "position",
        }
    )


def test_two_real_articulations_read_independent_per_articulation_contact_all(
    g1_usd, cartpole_usd, cartpole_articulation_cfg
):
    """Each articulation expands its own pre-clone inventory; never one union."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    from unigym.adapters.manual import ManualAdapter
    from unigym.backends.isaaclab2.env import IsaaclabEnv
    from unigym.backends.isaaclab2.usd import robot_rigid_bodies
    from unigym.contacts.base import ALL_BODIES
    from unigym.contacts.manager import build_contact_manager
    from unigym.runtime.controllers import build_control_runtimes

    cart_profile = _cartpole_profile()
    g1_profile = _g1_profile()
    cart_spec = _cartpole_spec()
    g1_spec = _g1_spec(g1_usd, g1_profile)
    g1_realization = _g1_realization(g1_spec, g1_profile)

    manager = build_contact_manager(
        [
            {
                "name": "cart_all",
                "sensor_bodies": {"contact_group": {"articulation": CARTPOLE, "group": "all"}},
                "frame": "world",
            },
            {
                "name": "humanoid_all",
                "sensor_bodies": {"contact_group": {"articulation": G1, "group": "all"}},
                "frame": "world",
            },
            {
                "name": "humanoid_feet",
                "articulation": G1,
                "sensor_bodies": ["left_ankle_roll_link", "right_ankle_roll_link"],
                "frame": "world",
            },
        ],
        backend="isaaclab2",
        contact_groups={
            CARTPOLE: {"all": (ALL_BODIES,)},
            G1: {"all": (ALL_BODIES,)},
        },
        contact_assets={CARTPOLE: str(cartpole_usd), G1: str(g1_usd)},
        articulation_names=(CARTPOLE, G1),
    )
    assert manager is not None

    controls = build_control_runtimes(
        (_control(CARTPOLE, cart_profile), _control(G1, g1_profile)),
        (cart_spec, g1_spec),
        {CARTPOLE: cart_profile, G1: g1_profile},
    )
    terrain_cfg = {
        "ground": AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    }
    sim_utils.create_new_stage()
    env = IsaaclabEnv(
        robot_profile=cart_profile,
        adapter=ManualAdapter(),
        terrain_cfg=terrain_cfg,
        num_envs=1,
        episode_length_s=100.0,
        decimation=4,
        sim_dt=0.005,
        device="cuda:0",
        articulations=(cart_spec, g1_spec),
        controls=controls,
        profiles={CARTPOLE: cart_profile, G1: g1_profile},
        robot_realizations={CARTPOLE: cartpole_articulation_cfg, G1: g1_realization},
        contact_manager=manager,
    )
    try:
        cart_inventory = robot_rigid_bodies(str(cartpole_usd))
        g1_inventory = robot_rigid_bodies(str(g1_usd))
        assert len(g1_inventory) > len(cart_inventory)
        g1_paths = [body.path for body in g1_inventory]
        assert "left_hand/left_hand_index_0_link" in g1_paths
        assert "left_hand/left_hand_palm_link" not in g1_paths

        assert manager.backend.group_axes(manager.specs[0]) == (len(cart_inventory), 0)
        assert manager.backend.group_axes(manager.specs[1]) == (len(g1_inventory), 0)
        assert manager.backend.group_axes(manager.specs[2]) == (2, 0)

        cart_names = {body.path.rsplit("/", 1)[-1] for body in cart_inventory}
        g1_names = {body.path.rsplit("/", 1)[-1] for body in g1_inventory}
        assert cart_names.isdisjoint(g1_names)

        for i, body in enumerate(cart_inventory):
            sensor = env.scene[f"contact_cart_all_b{i}"]
            assert set(sensor.body_names) == {body.path.rsplit("/", 1)[-1]}
            assert "/Robot/" in sensor.cfg.prim_path
        for i, body in enumerate(g1_inventory):
            sensor = env.scene[f"contact_humanoid_all_b{i}"]
            assert set(sensor.body_names) == {body.path.rsplit("/", 1)[-1]}
            assert "/Robot_g1/" in sensor.cfg.prim_path

        values = manager.read()
        assert tuple(values["contact/cart_all"].shape) == (1, len(cart_inventory), 3)
        assert tuple(values["contact/humanoid_all"].shape) == (1, len(g1_inventory), 3)
        assert tuple(values["contact/humanoid_feet"].shape) == (1, 2, 3)
        assert tuple(values["contact/cart_all"].shape) != (
            1,
            len(cart_inventory) + len(g1_inventory),
            3,
        )

        assert env.reset_all().tolist() == [0]
        after_reset = manager.read()
        assert tuple(after_reset["contact/cart_all"].shape) == (1, len(cart_inventory), 3)
        assert tuple(after_reset["contact/humanoid_all"].shape) == (1, len(g1_inventory), 3)
        assert tuple(after_reset["contact/humanoid_feet"].shape) == (1, 2, 3)
    finally:
        env.close()
