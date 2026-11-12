"""Real MuJoCo Warp backend fixtures over the shared tiny MJCF asset."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests._support.envs import require_optional_module

ARTICULATION = "tiny"
CONTROL = "tiny"
JOINT_ORDER = ("elbow", "shoulder")
KP = 20.0
KD = 1.0
_WARP_REASON = (
    "MuJoCo Warp is an optional extra (unigym[warp]); install it or run this "
    "lane in the warp environment"
)


def build_tiny_env(mjcf: Path, *, num_envs: int = 1):
    """Construct the real warp backend from the fixture MJCF (CPU, no managers)."""
    for module_name in ("mujoco_warp", "warp"):
        require_optional_module(module_name, reason=_WARP_REASON)

    from unigym.adapters.manual import ManualAdapter
    from unigym.backends.mjwarp.env import MjwarpEnv
    from unigym.robots.base import BodyBinding, RobotProfile
    from unigym.runtime.controllers import build_control_runtimes
    from unigym.runtime.controls import ControlGroup
    from unigym.scenes.articulation import ArticulationSpec

    profile = RobotProfile(
        name=ARTICULATION,
        joint_sets={"body": JOINT_ORDER},
        default_joint_pos={name: 0.0 for name in JOINT_ORDER},
        assets=None,
        action_faces={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
    )
    articulation = ArticulationSpec(
        name=ARTICULATION,
        spawn={"pos": [0.0, 0.0, 0.8]},
    )
    group = ControlGroup.from_mapping(
        {
            "name": CONTROL,
            "adapter": "manual",
            "articulation": ARTICULATION,
            "joints": list(JOINT_ORDER),
            "drive": "position_pd",
            "target": "position",
            "gains": {"kp": KP, "kd": KD},
        }
    )
    controls = build_control_runtimes((group,), (articulation,), {ARTICULATION: profile})
    return MjwarpEnv(
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
        profiles={ARTICULATION: profile},
    )


@pytest.fixture
def tiny_env_factory(tiny_mjcf: Path) -> Callable[..., object]:
    """Build real warp backends and close every one of them at teardown."""
    created = []

    def make(num_envs: int = 1):
        env = build_tiny_env(tiny_mjcf, num_envs=num_envs)
        created.append(env)
        return env

    yield make

    for env in created:
        env.close()


@pytest.fixture
def tiny_env(tiny_env_factory: Callable[..., object]):
    """One single-env real warp backend over the shared tiny MJCF."""
    return tiny_env_factory()
