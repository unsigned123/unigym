"""Real IsaacLab 2 Cartpole and G1 fixtures from installed test packs."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from tests._support.envs import require_optional_module

ARTICULATION = "cartpole"
JOINT_ORDER = ("slider_to_cart", "cart_to_pole")

#: Process-lifetime strong reference to the Kit app.  ``SimulationApp.close()``
#: ends the process natively (``exit_group(0)``) during fixture teardown, which
#: drops pytest's final summary, so the lane deliberately never closes it.  The
#: module keeps the app alive instead of relying on the fixture frame.
_APP = None


@pytest.fixture(scope="session")
def simulation_app() -> object:
    """Start Kit once for the process lifetime; it is intentionally unclosed."""
    global _APP
    if _APP is not None:
        return _APP
    require_optional_module("isaaclab", reason="the IsaacLab 2 backend lane")
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    args.headless = True
    launcher = AppLauncher(args)
    _APP = launcher.app
    return _APP


@pytest.fixture
def fresh_stage(simulation_app: object) -> None:
    """Give one test a fresh USD stage before it spawns scene prims.

    Tests that build through ``env_factory`` already call
    ``sim_utils.create_new_stage()`` per build; this named fixture covers the
    direct-builder tests (``test_sim_facade.py``) whose builds reuse the shared
    Kit stage.  Without it, a ``/World/light`` prim left by a previous test
    collides with the next spawn.

    Stage provisioning only: ``new_stage()`` also stops the Kit timeline as a
    side effect, so the next ``SimulationContext`` never starts while the
    timeline is still playing (the multi-build hang, see
    ``plans/artifacts/kit-multi-build-hang.md``).
    """
    import isaaclab.sim as sim_utils

    sim_utils.create_new_stage()


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
    scene_provider: str | None = None,
    scene_provider_params: Mapping[str, object] | None = None,
    scene_provider_preset: str | None = None,
    asset_plan=None,
    artifact_dir: str | Path | None = None,
    static_shapes=None,
):
    """Construct the single-articulation backend over the Cartpole asset."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    from unigym.actuators import ActuationGroup
    from unigym.adapters.manual import ManualAdapter
    from unigym.backends.isaaclab2.env import IsaaclabEnv
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
    return IsaaclabEnv(
        robot_profile=profile,
        adapter=ManualAdapter(),
        terrain_cfg=terrain_cfg,
        num_envs=num_envs,
        decimation=4,
        sim_dt=0.005,
        device="cuda:0",
        articulations=(articulation,),
        observations=ObservationConfig(robot=(ARTICULATION,)),
        controls=controls,
        profiles={ARTICULATION: profile},
        robot_realizations={ARTICULATION: articulation_cfg},
        scene_provider=scene_provider,
        scene_provider_params=scene_provider_params,
        scene_provider_preset=scene_provider_preset,
        asset_plan=asset_plan,
        artifact_dir=artifact_dir,
        static_shapes=static_shapes,
    )


@pytest.fixture
def cartpole_usd(pack_runtime) -> Path:
    """Resolve the Cartpole USD through the installed test pack."""
    require_optional_module("isaaclab", reason="the IsaacLab 2 backend lane")
    runtime = pack_runtime("isaaclab_cartpole")
    return runtime.resolve("assets", "isaaclab_cartpole.usd")


@pytest.fixture
def cartpole_pack(pack_runtime):
    """The installed Cartpole pack runtime for file-backed asset fixtures."""
    require_optional_module("isaaclab", reason="the IsaacLab 2 backend lane")
    return pack_runtime("isaaclab_cartpole")


@pytest.fixture
def cartpole_articulation_cfg(cartpole_usd: Path):
    """The real Cartpole ArticulationCfg with contact sensors enabled."""
    return _cartpole_articulation_cfg(cartpole_usd)


@pytest.fixture
def g1_pack(pack_runtime):
    """Activate the G1 pack's data-only robot provider once per process."""
    return pack_runtime("g1").activate("g1")


@pytest.fixture
def g1_usd(pack_runtime) -> Path:
    """Resolve the 43-DoF G1 USD through the installed test pack."""
    require_optional_module("isaaclab", reason="the IsaacLab 2 backend lane")
    runtime = pack_runtime("g1")
    return runtime.resolve("assets", "g1.usd")


@pytest.fixture
def env_factory(cartpole_usd: Path, simulation_app: object) -> Callable[..., object]:
    """Build real backends and close every one of them at teardown."""
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
    """One constructed IsaacLab 2 Cartpole env, closed at teardown."""
    return env_factory(1)


@pytest.fixture
def two_env(env_factory: Callable[..., object]):
    """Two constructed IsaacLab 2 Cartpole envs, closed at teardown."""
    return env_factory(2)
