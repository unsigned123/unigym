"""Real IsaacLab 2 Cartpole fixture from a provisioned asset cache."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tests._support.envs import require_optional_module

CACHE_ENV = "UNIGYM_ISAACLAB2_ASSET_CACHE"
CACHE_FILE = "cartpole.usd"
ARTICULATION = "cartpole"
JOINT_ORDER = ("slider_to_cart", "cart_to_pole")


def _cartpole_usd() -> Path:
    """The provisioned Cartpole USD; a missing cache fails the lane."""
    cache = os.environ.get(CACHE_ENV)
    if not cache:
        pytest.fail(
            f"{CACHE_ENV} is not set: point it at a provisioned directory "
            f"containing {CACHE_FILE}; the backend lane never downloads assets"
        )
    usd = Path(cache) / CACHE_FILE
    if not usd.is_file():
        pytest.fail(
            f"{CACHE_ENV}={cache!r} has no {CACHE_FILE}; provision the "
            "Cartpole asset cache before running the IsaacLab 2 lane"
        )
    return usd


@pytest.fixture(scope="session")
def simulation_app() -> Iterator[object]:
    """Start Kit before importing any isaaclab-dependent unigym module."""
    require_optional_module("isaaclab", reason="the IsaacLab 2 backend lane")
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    args.headless = True
    launcher = AppLauncher(args)
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


def _build_env(usd: Path, *, num_envs: int = 1):
    """Construct the single-articulation backend over the Cartpole asset."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    from unigym.adapters.manual import ManualAdapter
    from unigym.backends.isaaclab2.env import IsaaclabEnv
    from unigym.robots.base import RobotProfile
    from unigym.runtime.controllers import build_control_runtimes
    from unigym.runtime.controls import ControlGroup
    from unigym.scenes.articulation import ArticulationSpec

    articulation_cfg = _cartpole_articulation_cfg(usd)
    profile = RobotProfile(
        name=ARTICULATION,
        joint_sets={"body": JOINT_ORDER},
        default_joint_pos={name: 0.0 for name in JOINT_ORDER},
        assets=None,
        action_faces={},
        base_body=None,
        foot_bodies=(),
        body_sets={},
        body_bindings={},
    )
    articulation = ArticulationSpec(name=ARTICULATION)
    group = ControlGroup.from_mapping(
        {
            "name": ARTICULATION,
            "adapter": "manual",
            "articulation": ARTICULATION,
            "joints": list(JOINT_ORDER),
            "drive": "position_pd",
            "target": "position",
        }
    )
    controls = build_control_runtimes((group,), (articulation,), {ARTICULATION: profile})
    terrain_cfg = {
        "ground": AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    }
    sim_utils.create_new_stage()
    return IsaaclabEnv(
        robot_profile=profile,
        adapter=ManualAdapter(),
        terrain_cfg=terrain_cfg,
        num_envs=num_envs,
        episode_length_s=100.0,
        decimation=4,
        sim_dt=0.005,
        device="cuda:0",
        articulations=(articulation,),
        controls=controls,
        profiles={ARTICULATION: profile},
        robot_realizations={ARTICULATION: articulation_cfg},
    )


@pytest.fixture(scope="session")
def cartpole_usd() -> Path:
    """Resolve the optional lane and cached USD before Kit is launched."""
    require_optional_module("isaaclab", reason="the IsaacLab 2 backend lane")
    return _cartpole_usd()


@pytest.fixture
def env_factory(cartpole_usd: Path, simulation_app: object) -> Callable[..., object]:
    """Build real backends and close every one of them at teardown."""
    created = []

    def make(num_envs: int = 1):
        env = _build_env(cartpole_usd, num_envs=num_envs)
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
