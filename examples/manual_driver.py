"""Manual external-driver example: step a tiny MJCF with named actions.

Run from the repository root::

    PYTHONPATH=. python examples/manual_driver.py

No robot asset, policy adapter, or task pack is involved.  The manual
adapter deliberately has no ``act`` forward; the caller owns the clock and
supplies ``dict[str, NamedAction]`` through ``ExternalRunner.step``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from unigym.adapters.manual import ManualAdapter
from unigym.backends.mujoco.env import MujocoEnv
from unigym.robots.base import RobotProfile
from unigym.runner import ExternalRunner
from unigym.runtime.actions import NamedAction
from unigym.runtime.controllers import build_control_runtimes
from unigym.runtime.controls import ControlGroup
from unigym.scenes.articulation import ArticulationSpec
from unigym.terminations import TerminationGroup

MJCF = Path(__file__).resolve().parent / "tiny_arm.xml"


class PrintSink:
    """Minimal runner sink: print the action target of every tick."""

    def on_step(self, t, bundle):
        """Print the target carried by this tick's named action."""
        values = next(iter(bundle["targets"].values())).values.tolist()
        print(f"t={t} target={values[0]}")

    def on_episode_end(self, env_ids, bundle):
        """No episode bookkeeping in this example."""

    def on_step_end(self, t, bundle):
        """No per-step bookkeeping in this example."""

    def on_run_end(self):
        """Report the fixed rollout length of this example."""
        return {"steps": 5}


def make_env():
    """Build the tiny free-base arm and its manual control face."""
    profile = RobotProfile(
        name="tiny",
        joint_sets={"body": ("hinge",)},
        default_joint_pos={"hinge": 0.0},
        foot_bodies=[],
        body_parts={"base": "base"},
    )
    articulation = ArticulationSpec(
        name="tiny",
        spawn={"pos": [0.0, 0.0, 0.8]},
        body_parts={"base": "base"},
    )
    group = ControlGroup.from_mapping(
        {
            "name": "tiny",
            "adapter": "manual",
            "articulation": "tiny",
            "joints": ["hinge"],
            "drive": "position_pd",
            "target": "position",
            "gains": {"kp": 40.0, "kd": 2.0},
        }
    )
    controls = build_control_runtimes((group,), (articulation,), {"tiny": profile})
    env = MujocoEnv(
        robot_profile=profile,
        adapter=ManualAdapter(),
        num_envs=1,
        episode_length_s=100.0,
        decimation=4,
        sim_dt=0.005,
        world_base={"mjcf": str(MJCF)},
        backend_params={},
        device="cpu",
        articulations=(articulation,),
        controls=controls,
        profiles={"tiny": profile},
    )
    env.adapter_groups = {"manual": ("tiny",)}
    return env


def main() -> None:
    """Drive five ticks with caller-supplied position targets."""
    env = make_env()
    adapter = ManualAdapter()
    adapter._unigym_name = "manual"
    spec = SimpleNamespace(cfg={"collect": {"steps": 5}}, runner={}, adapter_name="manual")
    instance = SimpleNamespace(
        spec=spec,
        profile=None,
        adapter=adapter,
        sampler=None,
        term_group=TerminationGroup(),
        env=env,
        sink=PrintSink(),
        replay=None,
        reset_on_termination=False,
    )
    runner = ExternalRunner.from_instance(instance)
    try:
        for step in range(5):
            target = 0.1 * (step + 1)
            action = NamedAction(
                torch.tensor([[target]], dtype=torch.float32), ["hinge"], kind="position"
            )
            runner.step({"tiny": action})
        print("final hinge qpos:", float(env.mj_data.qpos[-1]))
    finally:
        env.close()


if __name__ == "__main__":
    main()
