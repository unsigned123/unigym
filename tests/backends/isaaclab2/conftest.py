"""Real IsaacLab 2 Cartpole fixture from a provisioned asset cache."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests._support.envs import require_optional_module

CACHE_ENV = "UNIGYM_ISAACLAB2_ASSET_CACHE"
CACHE_FILE = "cartpole.usd"
ARTICULATION = "cartpole"
JOINT_ORDER = ("slider_to_cart", "cart_to_pole")
FOOT_BODY = "cart"


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


def _build_env(usd: Path):
    """Construct the single-articulation backend over the Cartpole asset."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    from unigym.adapters.manual import ManualAdapter
    from unigym.backends.isaaclab2.env import IsaaclabEnv
    from unigym.robots.base import RobotRecipe
    from unigym.runtime.controllers import build_control_runtimes
    from unigym.runtime.controls import ControlGroup
    from unigym.scenes.articulation import ArticulationSpec

    body_parts = {"feet": [FOOT_BODY], "base": FOOT_BODY}
    profile = RobotRecipe(
        articulation_cfg=_cartpole_articulation_cfg(usd),
        joint_sets={"body": JOINT_ORDER},
        default_joint_pos={name: 0.0 for name in JOINT_ORDER},
        foot_bodies=[FOOT_BODY],
        body_parts=body_parts,
    )
    articulation = ArticulationSpec(name=ARTICULATION, body_parts=body_parts)
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
    return IsaaclabEnv(
        robot_profile=profile,
        adapter=ManualAdapter(),
        terrain_cfg=terrain_cfg,
        num_envs=1,
        episode_length_s=100.0,
        decimation=4,
        sim_dt=0.005,
        device="cuda:0",
        articulations=(articulation,),
        controls=controls,
        recipes={ARTICULATION: profile},
    )


@pytest.fixture(scope="session")
def cartpole_usd() -> Path:
    """Resolve the cached Cartpole USD before Kit is launched."""
    return _cartpole_usd()


@pytest.fixture
def cartpole_env(cartpole_usd, simulation_app) -> Iterator[object]:
    """One constructed IsaacLab 2 Cartpole env, closed at teardown."""
    env = _build_env(cartpole_usd)
    yield env
    env.close()
