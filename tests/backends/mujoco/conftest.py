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
    law: str = "standard_pd",
    gains: dict | None = None,
    parts: tuple[str, ...] | None = None,
    plans: bool = True,
    params: dict | None = None,
    contact_manager=None,
):
    """Construct the real backend from the fixture MJCF (CPU, no managers)."""
    from unigym.actuators import ActuationGroup
    from unigym.actuators.ir import resolve_actuation_groups
    from unigym.adapters.manual import ManualAdapter
    from unigym.backends.mujoco.env import MujocoEnv
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
    actuation = ActuationGroup(
        name="body",
        law=law,
        joints=JOINT_ORDER,
        gains={"kp": KP, "kd": KD} if gains is None else gains,
        parts=parts,
    )
    articulation = ArticulationSpec(
        name=ARTICULATION,
        spawn={"pos": [0.0, 0.0, 0.8]},
        actuation=(actuation,),
    )
    group = ControlGroup.from_mapping(
        {
            "name": CONTROL,
            "adapter": "manual",
            "articulation": ARTICULATION,
            "joints": list(JOINT_ORDER),
            "drive": drive,
            "target": target,
        }
    )
    controls = build_control_runtimes((group,), (articulation,), {ARTICULATION: profile})
    actuation_plans = (
        {
            ARTICULATION: resolve_actuation_groups(
                articulation.actuation,
                articulation=ARTICULATION,
                joints=JOINT_ORDER,
            )
        }
        if plans
        else None
    )
    return MujocoEnv(
        robot_profile=profile,
        adapter=ManualAdapter(),
        num_envs=num_envs,
        episode_length_s=100.0,
        decimation=4,
        sim_dt=0.005,
        world_base={"mjcf": str(mjcf)},
        backend_params=params or {},
        device="cpu",
        articulations=(articulation,),
        controls=controls,
        profiles={ARTICULATION: profile},
        actuation_plans=actuation_plans,
        contact_manager=contact_manager,
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
        law: str = "standard_pd",
        gains: dict | None = None,
        parts: tuple[str, ...] | None = None,
        plans: bool = True,
        params: dict | None = None,
        mjcf: Path | None = None,
        contact_manager=None,
    ):
        env = build_tiny_env(
            tiny_mjcf if mjcf is None else mjcf,
            num_envs=num_envs,
            drive=drive,
            target=target,
            law=law,
            gains=gains,
            parts=parts,
            plans=plans,
            params=params,
            contact_manager=contact_manager,
        )
        created.append(env)
        return env

    yield make

    for env in created:
        env.close()


@pytest.fixture
def tiny_env(tiny_env_factory: Callable[..., object]):
    """One single-env real MuJoCo backend over the shared tiny MJCF."""
    return tiny_env_factory()
