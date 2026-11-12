"""Manual adapter + a tiny self-authored MJCF: external action smoke."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unigym.actuators import ActuationGroup
from unigym.actuators.ir import resolve_actuation_groups
from unigym.adapters.manual import ManualAdapter
from unigym.backends.mujoco.env import MujocoEnv
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.runner import ExternalRunner
from unigym.runtime.actions import NamedAction
from unigym.runtime.controllers import build_control_runtimes
from unigym.runtime.controls import ControlGroup
from unigym.scenes.articulation import ArticulationSpec
from unigym.terminations import TerminationGroup

TINY_MJCF = """<?xml version="1.0"?>
<!-- Self-authored MIT-clean one-joint model; no external assets. -->
<mujoco model="tiny_arm">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.8">
      <freejoint name="root"/>
      <geom name="base_geom" type="box" size="0.06 0.06 0.06" mass="1.0"/>
      <body name="link" pos="0 0 -0.12">
        <joint name="hinge" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <geom name="link_geom" type="capsule" fromto="0 0 0 0 0 -0.2"
              size="0.02" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hinge_motor" joint="hinge" ctrlrange="-20 20"/>
  </actuator>
</mujoco>
"""


class _Sink:
    """Collects the per-tick action bundles."""

    def __init__(self):
        self.bundles = []

    def on_step(self, t, bundle):
        self.bundles.append((t, bundle))

    def on_episode_end(self, env_ids, bundle):
        """No episode bookkeeping in this smoke."""

    def on_step_end(self, t, bundle):
        """No per-step bookkeeping in this smoke."""

    def on_run_end(self):
        return {"steps": len(self.bundles)}


def _spy_adapter():
    class SpyManualAdapter(ManualAdapter):
        """ManualAdapter that counts (illegal) policy forwards."""

        def __init__(self):
            super().__init__()
            self.act_calls = 0

        def act(self, obs):
            self.act_calls += 1
            return super().act(obs)

    return SpyManualAdapter()


def _make_env(tmp_path):
    mjcf = tmp_path / "tiny_arm.xml"
    mjcf.write_text(TINY_MJCF, encoding="utf-8")
    profile = RobotProfile(
        name="tiny",
        joint_sets={"body": ("hinge",)},
        default_joint_pos={"hinge": 0.0},
        assets=None,
        action_faces={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
    )
    actuation = ActuationGroup(
        name="body",
        law="standard_pd",
        joints=("hinge",),
        gains={"kp": 40.0, "kd": 2.0},
    )
    articulation = ArticulationSpec(
        name="tiny",
        spawn={"pos": [0.0, 0.0, 0.8]},
        actuation=(actuation,),
    )
    actuation_plans = {
        "tiny": resolve_actuation_groups(
            articulation.actuation,
            articulation="tiny",
            joints=("hinge",),
        )
    }
    group = ControlGroup.from_mapping(
        {
            "name": "tiny",
            "adapter": "manual",
            "articulation": "tiny",
            "joints": ["hinge"],
            "drive": "position_pd",
            "target": "position",
        }
    )
    controls = build_control_runtimes((group,), (articulation,), {"tiny": profile})
    adapter = _spy_adapter()
    env = MujocoEnv(
        robot_profile=profile,
        adapter=adapter,
        num_envs=1,
        episode_length_s=100.0,
        decimation=4,
        sim_dt=0.005,
        world_base={"mjcf": str(mjcf)},
        backend_params={},
        device="cpu",
        articulations=(articulation,),
        controls=controls,
        profiles={"tiny": profile},
        actuation_plans=actuation_plans,
    )
    env.adapter_groups = {"manual": ("tiny",)}
    return env, adapter


def _runner(env, adapter, sink):
    adapter._unigym_name = "manual"
    spec = SimpleNamespace(
        cfg={"collect": {"steps": 4}},
        runner={},
        adapter_name="manual",
    )
    instance = SimpleNamespace(
        spec=spec,
        profile=None,
        adapter=adapter,
        sampler=None,
        term_group=TerminationGroup(),
        env=env,
        sink=sink,
        replay=None,
        reset_on_termination=False,
    )
    return ExternalRunner.from_instance(instance)


def _action(value):
    return {
        "tiny": NamedAction(
            torch.tensor([[value]], dtype=torch.float32), ["hinge"], kind="position"
        )
    }


def test_manual_adapter_act_is_a_loud_error():
    """The manual adapter never fakes a policy forward."""
    with pytest.raises(RuntimeError, match="ExternalRunner"):
        ManualAdapter().act(None)


def test_manual_driver_external_actions_advance_the_model(tmp_path):
    """Caller-supplied named actions step the tiny model without act()."""
    env, adapter = _make_env(tmp_path)
    sink = _Sink()
    runner = _runner(env, adapter, sink)
    try:
        for step in range(4):
            result = runner.step(_action(0.1 * (step + 1)))
            assert result.t == step
        assert adapter.act_calls == 0
        assert len(sink.bundles) == 4
        assert sink.bundles[0][1]["raw"] is None
        assert float(env.mj_data.qpos[-1]) != 0.0
    finally:
        env.close()


def test_external_run_without_action_fn_refuses(tmp_path):
    """run() has no action source unless action_fn is supplied."""
    env, adapter = _make_env(tmp_path)
    try:
        runner = _runner(env, adapter, _Sink())
        with pytest.raises(RuntimeError, match="step"):
            runner.run()
    finally:
        env.close()
