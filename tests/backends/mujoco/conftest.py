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
    spawn: dict | None = None,
    episode_length_s: float | None = 100.0,
    drive: str = "position_pd",
    target: str = "position",
    law: str = "standard_pd",
    gains: dict | None = None,
    parts: tuple[str, ...] | None = None,
    control_joints: tuple[str, ...] = JOINT_ORDER,
    plans: bool = True,
    params: dict | None = None,
    contact_manager=None,
    prop_manager=None,
    asset_plan=None,
    artifact_dir=None,
    adapter=None,
    observations=None,
):
    """Construct the real backend from the fixture MJCF (CPU, no managers)."""
    from unigym.actuators import ActuationGroup
    from unigym.actuators.ir import resolve_actuation_groups
    from unigym.adapters.manual import ManualAdapter
    from unigym.env.obs import ObservationConfig
    from unigym.backends.mujoco.env import MujocoEnv
    from unigym.robots.base import BodyBinding, RobotProfile
    from unigym.runtime.actions import OrderedActionGroup, OrderedActionSpec, PolicyBinding
    from unigym.runtime.controllers import build_control_runtimes
    from unigym.scenes.articulation import ArticulationSpec

    profile = RobotProfile(
        name=ARTICULATION,
        joint_sets={"body": JOINT_ORDER},
        default_joint_pos={name: 0.0 for name in JOINT_ORDER},
        assets=None,
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
    )
    actuation = ActuationGroup(
        name="body",
        law=law,
        joints=control_joints,
        gains={"kp": KP, "kd": KD} if gains is None else gains,
        parts=parts,
    )
    articulation = ArticulationSpec(
        name=ARTICULATION,
        spawn={"pos": [0.0, 0.0, 0.8]} if spawn is None else spawn,
        actuation=(actuation,),
        passive_joints=tuple(j for j in JOINT_ORDER if j not in control_joints),
    )
    group = OrderedActionGroup(
        name=CONTROL,
        articulation=ARTICULATION,
        joints=control_joints,
        target=target,
        role="body",
    )
    resolved_plan = resolve_actuation_groups(
        articulation.actuation,
        articulation=ARTICULATION,
        joints=JOINT_ORDER,
        passive_joints=tuple(j for j in JOINT_ORDER if j not in control_joints),
    )
    controls = build_control_runtimes(
        OrderedActionSpec((group,)),
        (PolicyBinding(adapter="manual", groups=((ARTICULATION, CONTROL),)),),
        (articulation,),
        {ARTICULATION: profile},
        {ARTICULATION: resolved_plan},
    )
    actuation_plans = {ARTICULATION: resolved_plan} if plans else None
    return MujocoEnv(
        robot_profile=profile,
        adapter=ManualAdapter() if adapter is None else adapter,
        num_envs=num_envs,
        episode_length_s=episode_length_s,
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
        prop_manager=prop_manager,
        asset_plan=asset_plan,
        artifact_dir=artifact_dir,
        observations=(
            observations
            if observations is not None
            else ObservationConfig(robot=(ARTICULATION,), command=True, prev_action=True)
        ),
    )


@pytest.fixture
def tiny_env_factory(tiny_mjcf: Path) -> Callable[..., object]:
    """Build real backends and close every one of them at teardown."""
    created = []

    def make(
        num_envs: int = 1,
        *,
        spawn: dict | None = None,
        episode_length_s: float | None = 100.0,
        drive: str = "position_pd",
        target: str = "position",
        law: str = "standard_pd",
        gains: dict | None = None,
        parts: tuple[str, ...] | None = None,
        control_joints: tuple[str, ...] = JOINT_ORDER,
        plans: bool = True,
        params: dict | None = None,
        mjcf: Path | None = None,
        contact_manager=None,
        prop_manager=None,
        asset_plan=None,
        artifact_dir=None,
        adapter=None,
        observations=None,
    ):
        env = build_tiny_env(
            tiny_mjcf if mjcf is None else mjcf,
            num_envs=num_envs,
            spawn=spawn,
            episode_length_s=episode_length_s,
            drive=drive,
            target=target,
            law=law,
            gains=gains,
            parts=parts,
            control_joints=control_joints,
            plans=plans,
            params=params,
            contact_manager=contact_manager,
            prop_manager=prop_manager,
            asset_plan=asset_plan,
            artifact_dir=artifact_dir,
            adapter=adapter,
            observations=observations,
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
