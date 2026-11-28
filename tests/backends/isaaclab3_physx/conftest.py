"""Real IsaacLab 3 PhysX Cartpole fixtures from installed test packs."""

from __future__ import annotations

from unigym.backends.compose import backend_runtime

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tests._support.envs import require_optional_module

# The rc1 launcher requires the EULA to be accepted non-interactively.  This is
# lane-local because the runner does not inject it.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

ARTICULATION = "cartpole"
JOINT_ORDER = ("slider_to_cart", "cart_to_pole")


@pytest.fixture(scope="session", autouse=True)
def _stack() -> None:
    for module in ("isaaclab", "isaaclab_physx"):
        require_optional_module(module, reason="the IsaacLab 3 PhysX lane")


@pytest.fixture(scope="session")
def simulation_app() -> Iterator[object]:
    """Launch Kit headless before importing isaaclab-dependent unigym modules."""
    require_optional_module("isaaclab", reason="the IsaacLab 3 PhysX lane")
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, enable_cameras=True)
    yield launcher.app
    launcher.app.close()


def _cartpole_articulation_cfg(usd: Path):
    """The built-in Cartpole articulation, pointed at the provisioned USD."""
    from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

    from unigym.env.naming import ENV_REGEX_NS

    return CARTPOLE_CFG.replace(
        prim_path=f"{ENV_REGEX_NS}/Robot",
        spawn=CARTPOLE_CFG.spawn.replace(
            usd_path=str(usd),
            activate_contact_sensors=True,
        ),
    )


def _build_env(
    usd: Path,
    *,
    num_envs: int = 1,
    contact_manager=None,
    terrain_cfg=None,
    events_cfg=None,
    prop_manager=None,
    director_cfg=None,
    camera_manager=None,
):
    """Construct the IsaacLab 3 PhysX backend over the Cartpole asset."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    from unigym.actuators import ActuationGroup
    from unigym.adapters.manual import ManualAdapter
    from unigym.backends.isaaclab3_physx.env import Isaaclab3PhysxEnv
    from unigym.env.obs import ObservationConfig
    from unigym.robots.base import RobotProfile
    from unigym.runtime.actions import OrderedActionGroup, OrderedActionSpec, PolicyBinding
    from unigym.runtime.controllers import build_control_runtimes
    from unigym.scenes.articulation import ArticulationSpec

    articulation_cfg = _cartpole_articulation_cfg(usd)
    profile = RobotProfile(
        name=ARTICULATION,
        joint_sets={"body": JOINT_ORDER},
        default_joint_pos={name: 0.0 for name in JOINT_ORDER},
        assets=None,
        extra_action_sets={},
        base_body=None,
        foot_bodies=(),
        body_sets={},
        body_bindings={},
    )
    articulation = ArticulationSpec(
        name=ARTICULATION,
        actuation=(
            ActuationGroup(
                name=ARTICULATION,
                law="standard_pd",
                joints=JOINT_ORDER,
                gains={"kp": 1.0, "kd": 0.1},
            ),
        ),
    )
    group = OrderedActionGroup(
        name=ARTICULATION,
        articulation=ARTICULATION,
        joints=JOINT_ORDER,
        target="position",
        role="body",
    )
    from unigym.actuators.ir import resolve_actuation_groups

    plans = {
        ARTICULATION: resolve_actuation_groups(
            articulation.actuation, articulation=ARTICULATION, joints=JOINT_ORDER
        )
    }
    controls = build_control_runtimes(
        OrderedActionSpec((group,)),
        (PolicyBinding(adapter="manual", groups=((ARTICULATION, ARTICULATION),)),),
        (articulation,),
        {ARTICULATION: profile},
        plans,
    )
    if terrain_cfg is None:
        terrain_cfg = {
            "ground": AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.CuboidCfg(
                    size=(10.0, 10.0, 0.1),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
            )
        }
    sim_utils.create_new_stage()
    return Isaaclab3PhysxEnv(
        robot_profile=profile,
        adapter=ManualAdapter(),
        terrain_cfg=terrain_cfg,
        num_envs=num_envs,
        decimation=4,
        sim_dt=0.005,
        events_cfg=events_cfg,
        device="cuda:0",
        articulations=(articulation,),
        observations=ObservationConfig(robot=(ARTICULATION,)),
        controls=controls,
        profiles={ARTICULATION: profile},
        robot_realizations={ARTICULATION: articulation_cfg},
        contact_manager=contact_manager,
        prop_manager=prop_manager,
        director_cfg=director_cfg,
        camera_manager=camera_manager,
    )


@pytest.fixture
def cartpole_usd(pack_runtime) -> Path:
    """Resolve the Cartpole USD through the installed test pack."""
    require_optional_module("isaaclab", reason="the IsaacLab 3 PhysX lane")
    runtime = pack_runtime("isaaclab_cartpole")
    return runtime.resolve("assets", "isaaclab_cartpole.usd")


@pytest.fixture
def env_factory(cartpole_usd: Path, simulation_app: object) -> Iterator[Callable[..., object]]:
    """Build real PhysX envs and close every one of them at teardown."""
    created = []

    def make(num_envs: int = 1, **kwargs):
        env = _build_env(cartpole_usd, num_envs=num_envs, **kwargs)
        created.append(env)
        return env

    yield make

    for env in reversed(created):
        env.close()


@pytest.fixture
def cartpole_env(env_factory: Callable[..., object]):
    """One constructed IsaacLab 3 PhysX Cartpole env, closed at teardown."""
    return env_factory(1)


@pytest.fixture
def crate_prop_manager() -> Callable[..., object]:
    """Build a fresh ``box`` prop manager for the PhysX prop lane."""
    from unigym.props import build_prop_manager

    def make(num_envs: int = 2):
        return build_prop_manager(
            {
                "types": [
                    {
                        "name": "crate",
                        "type": "box",
                        "size": [0.1, 0.1, 0.1],
                        "aabb": {
                            "center": [0.0, 0.0, 0.05],
                            "half_extents": [0.05, 0.05, 0.05],
                        },
                        "placement": {
                            "mode": "fixed",
                            "x_range": [0.5, 0.5],
                            "y_range": [0.0, 0.0],
                            "z_range": [0.3, 0.3],
                        },
                    }
                ]
            },
            num_envs=num_envs,
            env_spacing=4.0,
            device="cuda:0",
            backend="isaaclab3_physx",
            prop_backend=backend_runtime("isaaclab3_physx").bindings.prop_backend,
            mujoco_prop_backend=backend_runtime("mujoco").bindings.prop_backend,
        )

    return make
