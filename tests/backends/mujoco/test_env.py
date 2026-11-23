"""MuJoCo backend-local construction, IR, limits, permutation and teardown."""

from __future__ import annotations

import pytest
import torch

from tests._support.fakes.closing import FailingClose, RecordingClose
from unigym.adapters.base import Adapter
from unigym.env.backends.base import assert_backend_contract
from unigym.runtime.actions import NamedAction


def _zero_action(env):
    control = env.controls[("tiny", "tiny")]
    return (
        NamedAction.zeros(
            env.num_envs,
            control.articulation,
            list(control.joint_names),
            control.target,
        ),
    )


def test_env_constructs_from_tiny_mjcf_and_satisfies_backend_contract(tiny_env):
    """The real MuJoCo env loads the tiny MJCF and passes the live contract."""
    assert_backend_contract(tiny_env, exercise_step=True)
    assert tiny_env.num_envs == 1
    assert str(tiny_env.device) == "cpu"


def test_model_ir_publishes_expected_body_joint_and_actuator_names(tiny_env):
    """The compiled IR exposes the tiny model's declared topology and names."""
    ir = tiny_env.model_set.articulations["tiny"]

    assert [body.name for body in ir.bodies] == ["world", "base", "link", "tip"]
    assert [joint.name for joint in ir.joints] == ["root", "shoulder", "elbow"]
    assert [actuator.name for actuator in ir.actuators] == [
        "shoulder_motor",
        "elbow_motor",
    ]
    assert [actuator.target for actuator in ir.actuators] == ["shoulder", "elbow"]

    bodies = {body.name: body for body in ir.bodies}
    assert bodies["link"].parent == "base"
    assert bodies["tip"].parent == "link"
    assert bodies["base"].parent == "world"


def test_env_loads_only_the_tmp_mjcf_without_external_assets(tiny_mjcf, tiny_env_factory):
    """Construction consumes the tmp asset only: no mesh, texture or hfield."""
    assert [path.name for path in tiny_mjcf.parent.iterdir()] == ["tiny_arm.xml"]

    env = tiny_env_factory()
    model = env.mj_model
    assert model.nmesh == 0
    assert model.ntex == 0
    assert model.nhfield == 0

    ir = env.model_set.articulations["tiny"]
    assert list(ir.materials) == []
    if ir.mj is not None:
        assert list(ir.mj.textures) == []


def test_policy_joint_pos_limits_follow_the_control_order(tiny_env):
    """Limits are (N, J) float32 bounds in the control's joint order."""
    lower, upper = tiny_env.policy_joint_pos_limits
    assert tiny_env.controls[("tiny", "tiny")].joint_names == ("elbow", "shoulder")

    for bound in (lower, upper):
        assert tuple(bound.shape) == (tiny_env.num_envs, 2)
        assert bound.dtype == torch.float32
        assert str(bound.device) == str(tiny_env.device)

    torch.testing.assert_close(lower, torch.tensor([[-0.5, -1.57]]))
    torch.testing.assert_close(upper, torch.tensor([[1.0, 1.57]]))
    assert bool((lower <= upper).all())


def test_articulation_joint_pos_limits_follow_the_record_order(tiny_env):
    """The runtime limits use the record ``joint_pos`` order, not asset order."""
    art = tiny_env.articulations["tiny"]
    assert art.joint_names == ("shoulder", "elbow")
    assert tiny_env.controls[("tiny", "tiny")].joint_names == ("elbow", "shoulder")
    assert art.joint_pos_limits is not None

    lower, upper = art.joint_pos_limits
    torch.testing.assert_close(lower, torch.tensor([[-0.5, -1.57]]))
    torch.testing.assert_close(upper, torch.tensor([[1.0, 1.57]]))
    assert tiny_env.joint_pos_limits_by_articulation["tiny"] is art.joint_pos_limits
    assert tiny_env.policy_joint_pos_limits is art.joint_pos_limits


def test_joint_permutation_maps_il_and_policy_order_both_ways(tiny_env):
    """The IL<->policy permutation is a real reorder and round-trips exactly."""
    assert tiny_env.perm_il2pol.tolist() == [1, 0]
    assert tiny_env.perm_pol2il.tolist() == [1, 0]

    policy_values = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    il_values = tiny_env.to_il_order(policy_values)
    assert il_values.tolist() == [[20.0, 10.0], [40.0, 30.0]]
    assert torch.equal(tiny_env.to_policy_order(il_values), policy_values)

    assert tiny_env.to_policy_order(torch.tensor([[1.0, 2.0]])).tolist() == [[2.0, 1.0]]


def test_close_is_idempotent_and_post_close_step_fails(tiny_env):
    """Closing twice is a no-op; stepping a closed env raises the closed error."""
    tiny_env.close()
    tiny_env.close()
    with pytest.raises(RuntimeError, match="environment is closed"):
        tiny_env.step(_zero_action(tiny_env))


def test_reset_envs_after_close_raises_the_closed_error(tiny_env):
    """A subset reset on a closed env reports the closed backend, not a handle crash."""
    tiny_env.close()
    with pytest.raises(RuntimeError) as excinfo:
        tiny_env.reset_envs(torch.tensor([0], dtype=torch.int64))
    assert str(excinfo.value) == "mujoco backend: environment is closed"


def test_reset_all_after_close_raises_the_closed_error(tiny_env):
    """A full reset on a closed env reports the closed backend, not a handle crash."""
    tiny_env.close()
    with pytest.raises(RuntimeError) as excinfo:
        tiny_env.reset_all()
    assert str(excinfo.value) == "mujoco backend: environment is closed"


def test_close_failure_leaves_backend_guarded_and_retryable(tiny_env, monkeypatch):
    """A manager teardown failure blocks use but keeps the backend retryable."""
    camera = RecordingClose()
    failing_prop = FailingClose()
    monkeypatch.setattr(tiny_env, "camera_manager", camera)
    monkeypatch.setattr(tiny_env, "prop_backend", failing_prop)

    with pytest.raises(RuntimeError, match="prop backend teardown failed"):
        tiny_env.close()

    assert camera.close_calls == 1
    assert tiny_env.mj_datas is not None
    assert tiny_env.mj_data is not None
    assert tiny_env.mj_model is not None

    action = _zero_action(tiny_env)

    def _step():
        return tiny_env.step(action)

    def _reset_subset():
        return tiny_env.reset_envs(torch.tensor([0], dtype=torch.int64))

    for call in (_step, _reset_subset, tiny_env.reset_all):
        with pytest.raises(RuntimeError) as excinfo:
            call()
        assert str(excinfo.value) == "mujoco backend: environment is closed"

    monkeypatch.setattr(failing_prop, "close", lambda: None)
    tiny_env.close()

    assert camera.close_calls == 2
    assert tiny_env.mj_datas is None
    assert tiny_env.mj_data is None
    assert tiny_env.mj_model is None


_BASE_XML = """<?xml version="1.0"?>
<mujoco model="base_scene">
  <compiler angle="radian"/>
  <worldbody>
    <geom name="ground" type="plane" size="4 4 0.1"/>
  </worldbody>
</mujoco>
"""


def _two_articulation_env(
    tmp_path, tiny_mjcf, *, group_name=None, names=("left", "right"), contact_manager=None
):
    from unigym.actuators import ActuationGroup
    from unigym.actuators.ir import resolve_actuation_groups
    from unigym.adapters.manual import ManualAdapter
    from unigym.env.obs import ObservationConfig
    from unigym.backends.mujoco.env import MujocoEnv
    from unigym.robots.base import BodyBinding, RobotProfile
    from unigym.runtime.actions import OrderedActionGroup, OrderedActionSpec, PolicyBinding
    from unigym.runtime.controllers import build_control_runtimes
    from unigym.scenes.articulation import ArticulationSpec

    base = tmp_path / "base_scene.xml"
    base.write_text(_BASE_XML, encoding="utf-8")
    joints = ("elbow", "shoulder")
    source = "\n".join(
        line
        for line in tiny_mjcf.read_text(encoding="utf-8").splitlines()
        if 'geom name="floor"' not in line
    )
    specs = []
    profiles = {}
    groups = []
    controls = []
    for index, name in enumerate(names):
        sign = -1.0 if index == 0 else 1.0
        robot = tmp_path / f"tiny_arm_{name}.xml"
        robot.write_text(
            source.replace('name="base_geom"', f'name="{name}_base_geom"')
            .replace('name="link_geom"', f'name="{name}_link_geom"')
            .replace('name="tip_geom"', f'name="{name}_tip_geom"'),
            encoding="utf-8",
        )
        profiles[name] = RobotProfile(
            name=name,
            joint_sets={"body": joints},
            default_joint_pos={joint: 0.0 for joint in joints},
            assets=None,
            extra_action_sets={},
            base_body="base",
            foot_bodies=(),
            body_sets={},
            body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
        )
        actuation = (
            ActuationGroup(
                name="body",
                law="computed_torque_pd",
                joints=joints,
                gains={
                    "kp": {"elbow": 20.0, "shoulder": 30.0},
                    "kd": {"elbow": 1.0, "shoulder": 2.0},
                },
                parts=("elbow", "shoulder"),
            )
            if name == "left"
            else ActuationGroup(
                name="body",
                law="standard_pd",
                joints=joints,
                gains={"kp": 20.0, "kd": 1.0},
            )
        )
        specs.append(
            ArticulationSpec(
                name=name,
                model={"mjcf": str(robot)},
                spawn={"pos": [sign, 0.0, 0.8]},
                actuation=(actuation,),
            )
        )
        groups.append(
            OrderedActionGroup(
                name=name if group_name is None else group_name,
                articulation=name,
                joints=tuple(joints),
                target="position",
                role="body",
            )
        )
    plans = {
        spec.name: resolve_actuation_groups(
            spec.actuation,
            articulation=spec.name,
            joints=joints,
        )
        for spec in specs
    }
    controls = build_control_runtimes(
        OrderedActionSpec(tuple(groups)),
        (PolicyBinding(adapter="manual", groups=tuple((g.articulation, g.name) for g in groups)),),
        tuple(specs),
        profiles,
        plans,
    )
    env = MujocoEnv(
        robot_profile=profiles[names[0]],
        adapter=ManualAdapter(),
        num_envs=1,
        episode_length_s=100.0,
        decimation=4,
        sim_dt=0.005,
        world_base={"mjcf": str(base)},
        backend_params={},
        backend_options={"compose": {"auto_prefix": True}},
        device="cpu",
        articulations=tuple(specs),
        controls=controls,
        profiles=profiles,
        actuation_plans=plans,
        contact_manager=contact_manager,
        observations=ObservationConfig(robot=tuple(names)),
    )
    return env


def test_auto_prefix_multi_articulation_plans_translate_names_gains_and_parts(tmp_path, tiny_mjcf):
    """The seam maps neutral joints/gain keys/parts to compiled prefixed names."""
    env = _two_articulation_env(tmp_path, tiny_mjcf)
    try:
        actions = tuple(
            NamedAction(
                articulation=name,
                values=torch.tensor([[sign * 0.4, sign * -0.2]], dtype=torch.float32),
                joint_names=("elbow", "shoulder"),
                kind="position",
            )
            for name, sign in (("left", 1.0), ("right", -1.0))
        )

        step = env.step(actions)

        assert step.record["articulation/left/joint_pos"][0, 0] > 0.0
        assert step.record["articulation/right/joint_pos"][0, 0] < 0.0
        assert env.controls[("left", "left")].joint_names == ("elbow", "shoulder")
        assert env.controls[("right", "right")].joint_names == ("elbow", "shoulder")
    finally:
        env.close()


def _legs_action(articulation: str, target: float) -> NamedAction:
    """One body-group action over the tiny arm's joints (``legs``)."""
    values = torch.tensor([[target, target]], dtype=torch.float32)
    return NamedAction(articulation, values, ("elbow", "shoulder"), "position")


def test_two_articulations_may_share_the_group_name_legs(tmp_path, tiny_mjcf):
    """Duplicate group names are disambiguated by articulation across two steps."""
    env = _two_articulation_env(tmp_path, tiny_mjcf, group_name="legs", names=("go1", "go2"))
    try:
        assert set(env.controls) == {("go1", "legs"), ("go2", "legs")}

        first = env.step((_legs_action("go1", 0.4), _legs_action("go2", -0.4)))
        assert (
            first.record["articulation/go1/joint_pos"][0, 0]
            > first.record["articulation/go2/joint_pos"][0, 0]
        )
        assert first.obs.robot("go1").joint_pos[0, 0] > first.obs.robot("go2").joint_pos[0, 0]

        second = env.step((_legs_action("go1", -0.4), _legs_action("go2", 0.4)))
        assert (
            second.record["articulation/go2/joint_pos"][0, 0]
            > second.record["articulation/go1/joint_pos"][0, 0]
        )
    finally:
        env.close()


class _TwoDogAdapter(Adapter):
    """A real composite adapter emitting both ``legs`` groups every tick."""

    generates_command = False
    _unigym_name = "two_dog"

    def __init__(self) -> None:
        self.calls = 0

    @property
    def obs_dim(self) -> int:
        return 1

    @property
    def control_dt(self) -> float:
        return 0.02

    def compute_obs(self, obs, env):
        """Return a finite one-column frame for the runner's alive check."""
        return torch.zeros(env.num_envs, 1)

    def act(self, obs):
        """Return the tick index (the composite postprocess reads it)."""
        return torch.tensor([float(self.calls)])

    def postprocess_action(self, raw):
        """Return both articulation-scoped NamedActions (swapping the sign)."""
        step = int(float(raw.reshape(-1)[0]))
        self.calls += 1
        target = 0.4 if step == 0 else -0.4
        return (
            _legs_action("go1", target),
            _legs_action("go2", -target),
        )

    def on_reset(self, env_ids) -> None:
        """No history to clear."""


class _CaptureSink:
    """Collect the per-tick extras the composite runner steps."""

    def __init__(self) -> None:
        self.extras = []

    def on_step(self, t, bundle) -> None:
        self.extras.append(bundle["record"])

    def on_episode_end(self, env_ids, bundle) -> None:
        """No episode bookkeeping."""

    def on_step_end(self, t, bundle) -> None:
        """No metrics bookkeeping."""

    def on_run_end(self) -> dict:
        return {"steps": len(self.extras)}


def test_composite_adapter_sync_runner_routes_both_legs_groups(tmp_path, tiny_mjcf):
    """One adapter owns ``(go1, legs)`` + ``(go2, legs)``; the runner steps both."""
    from unigym.runner.sync import SyncRunner
    from unigym.terminations import TerminationGroup

    env = _two_articulation_env(tmp_path, tiny_mjcf, group_name="legs", names=("go1", "go2"))
    adapter = _TwoDogAdapter()
    env.adapter_groups = {"two_dog": (("go1", "legs"), ("go2", "legs"))}
    env.adapter_joint_targets = {
        "two_dog": {
            ("go1", "elbow"): "position",
            ("go1", "shoulder"): "position",
            ("go2", "elbow"): "position",
            ("go2", "shoulder"): "position",
        }
    }
    sink = _CaptureSink()
    runner = SyncRunner(
        cfg={"collect": {"steps": 2}},
        profile=None,
        adapter=adapter,
        sampler=None,
        term_group=TerminationGroup(),
        env=env,
        sink=sink,
        adapter_joint_targets=env.adapter_joint_targets["two_dog"],
    )
    try:
        assert runner.run() == {"steps": 2}
        first, second = sink.extras
        assert first["articulation/go1/joint_pos"][0, 0] > first["articulation/go2/joint_pos"][0, 0]
        assert (
            second["articulation/go2/joint_pos"][0, 0] > second["articulation/go1/joint_pos"][0, 0]
        )
        assert env.controls[("go1", "legs")].name == "legs"
        assert env.controls[("go2", "legs")].name == "legs"
    finally:
        env.close()
