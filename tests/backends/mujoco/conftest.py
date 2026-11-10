"""Real MuJoCo backend fixtures over the shared tiny MJCF asset."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

ARTICULATION = "tiny"
CONTROL = "tiny"
JOINT_ORDER = ("elbow", "shoulder")
KP = 20.0
KD = 1.0


def build_tiny_env(
    mjcf: Path,
    *,
    num_envs: int = 1,
    drive: str = "position_pd",
    target: str = "position",
):
    """Construct the real backend from the fixture MJCF (CPU, no managers)."""
    from unigym.adapters.manual import ManualAdapter
    from unigym.backends.mujoco.env import MujocoEnv
    from unigym.robots.base import RobotRecipe
    from unigym.runtime.controllers import build_control_runtimes
    from unigym.runtime.controls import ControlGroup
    from unigym.scenes.articulation import ArticulationSpec

    profile = RobotRecipe(
        articulation_cfg=None,
        joint_sets={"body": JOINT_ORDER},
        default_joint_pos={name: 0.0 for name in JOINT_ORDER},
        foot_bodies=[],
        body_parts={"base": "base"},
    )
    articulation = ArticulationSpec(
        name=ARTICULATION,
        spawn={"pos": [0.0, 0.0, 0.8]},
        body_parts={"base": "base"},
    )
    group = ControlGroup.from_mapping(
        {
            "name": CONTROL,
            "adapter": "manual",
            "articulation": ARTICULATION,
            "joints": list(JOINT_ORDER),
            "drive": drive,
            "target": target,
            "gains": {"kp": KP, "kd": KD} if drive == "position_pd" else {},
        }
    )
    controls = build_control_runtimes((group,), (articulation,), {ARTICULATION: profile})
    return MujocoEnv(
        robot_profile=profile,
        adapter=ManualAdapter(),
        num_envs=num_envs,
        episode_length_s=100.0,
        decimation=4,
        sim_dt=0.005,
        world_base={"mjcf": str(mjcf)},
        backend_params={},
        device="cpu",
        articulations=(articulation,),
        controls=controls,
        recipes={ARTICULATION: profile},
    )


@pytest.fixture
def tiny_env_factory(tiny_mjcf: Path) -> Callable[..., object]:
    """Build real backends and close every one of them at teardown."""
    created = []

    def make(
        num_envs: int = 1,
        *,
        drive: str = "position_pd",
        target: str = "position",
    ):
        env = build_tiny_env(tiny_mjcf, num_envs=num_envs, drive=drive, target=target)
        created.append(env)
        return env

    yield make

    for env in created:
        env.close()


@pytest.fixture
def tiny_env(tiny_env_factory: Callable[..., object]):
    """One single-env real MuJoCo backend over the shared tiny MJCF."""
    return tiny_env_factory()
