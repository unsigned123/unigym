"""MuJoCo Warp backend-local named-action stepping, failure paths and resets."""

from __future__ import annotations

import pytest
import torch

from unigym.env.backends.base import check_extra_info, check_obs_cache
from unigym.runtime.actions import NamedAction

pytestmark = pytest.mark.sim


def _action(env, values):
    control = env.controls["tiny"]
    names = list(control.joint_names)
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(env.num_envs, len(names))
    return {"tiny": NamedAction(tensor, names, control.target)}


def test_named_action_step_advances_joints_and_returns_contract_shaped_output(tiny_env):
    """One named-position tick moves the joints and honors the obs/extra contract."""
    control = tiny_env.controls["tiny"]

    obs, extra = tiny_env.step(_action(tiny_env, [[0.3, -0.2]]))

    check_obs_cache(obs, tiny_env.num_envs, device=tiny_env.device)
    check_obs_cache(tiny_env.obs_cache, tiny_env.num_envs, device=tiny_env.device)
    check_extra_info(
        extra,
        tiny_env.num_envs,
        len(control.joint_names),
        device=tiny_env.device,
    )

    joint_pos = extra["articulation/tiny/joint_pos"]
    assert 0.0 < joint_pos[0, 0] <= 0.3
    assert -0.2 <= joint_pos[0, 1] < 0.0
    assert extra["should_reset"] is False
    assert tuple(extra["reset_env_ids"].shape) == (0,)
    assert extra["episode_id"].tolist() == [1]


def test_named_action_step_keeps_batch_rows_independent(tiny_env_factory):
    """Different per-row targets drive different per-row joints in one batch."""
    env = tiny_env_factory(num_envs=2)

    _, extra = env.step(_action(env, [[0.4, -0.3], [0.2, 0.1]]))

    joint_pos = extra["articulation/tiny/joint_pos"]
    assert joint_pos[0, 0] > joint_pos[1, 0] > 0.0
    assert joint_pos[0, 1] < 0.0 < joint_pos[1, 1]
    assert extra["episode_id"].tolist() == [1, 1]


def test_reset_envs_resets_only_the_selected_row_and_returns_its_id(tiny_env_factory):
    """A subset reset restores that row only and returns the requested ids."""
    env = tiny_env_factory(num_envs=2)
    env.step(_action(env, [[0.4, -0.3], [0.2, 0.1]]))

    returned = env.reset_envs(torch.tensor([1], dtype=torch.int64))

    assert returned.tolist() == [1]
    assert returned.dtype == torch.int64

    _, extra = env.step(_action(env, [[0.0, 0.0], [0.0, 0.0]]))
    joint_pos = extra["articulation/tiny/joint_pos"]
    assert float(joint_pos[1].abs().max()) < 0.01
    assert float(joint_pos[0].abs().max()) > 0.01
    assert extra["episode_id"].tolist() == [1, 2]
    assert extra["time_in_episode"].tolist() == pytest.approx([0.04, 0.02])


def test_reset_all_resets_every_row_and_returns_the_full_id_range(tiny_env_factory):
    """reset_all restores every row and returns arange(num_envs)."""
    env = tiny_env_factory(num_envs=2)
    env.step(_action(env, [[0.4, -0.3], [0.2, 0.1]]))

    returned = env.reset_all()

    assert returned.tolist() == [0, 1]
    assert returned.dtype == torch.int64

    _, extra = env.step(_action(env, [[0.0, 0.0], [0.0, 0.0]]))
    assert float(extra["articulation/tiny/joint_pos"].abs().max()) < 0.01
    assert extra["episode_id"].tolist() == [2, 2]
    assert extra["time_in_episode"].tolist() == pytest.approx([0.02, 0.02])


def test_reset_envs_empty_is_a_noop_and_preserves_requested_id_order(tiny_env_factory):
    """An empty reset changes nothing; a subset reset returns ids in input order."""
    env = tiny_env_factory(num_envs=2)
    env.step(_action(env, [[0.4, -0.3], [0.2, 0.1]]))

    empty = env.reset_envs(torch.empty(0, dtype=torch.int64))
    assert empty.numel() == 0
    assert empty.dtype == torch.int64

    _, extra = env.step(_action(env, [[0.0, 0.0], [0.0, 0.0]]))
    assert extra["episode_id"].tolist() == [1, 1]
    assert extra["time_in_episode"].tolist() == pytest.approx([0.04, 0.04])

    returned = env.reset_envs(torch.tensor([1, 0], dtype=torch.int64))
    assert returned.tolist() == [1, 0]
    assert returned.dtype == torch.int64


@pytest.mark.parametrize(
    "joint_names",
    [
        pytest.param(["elbow", "wrist"], id="nonmatching-joints"),
        pytest.param(["elbow", "shoulder"], id="matching-joints"),
    ],
)
def test_step_rejects_an_unknown_action_face(tiny_env, joint_names):
    """An unknown face is rejected even when its joints match a control."""
    entry = NamedAction(torch.zeros(tiny_env.num_envs, 2), joint_names, "position")

    with pytest.raises(ValueError) as excinfo:
        tiny_env.step({"wrist": entry})

    message = str(excinfo.value)
    assert "action entry 'wrist' is not a declared control face" in message
    assert "accepted faces: ['tiny']" in message


@pytest.mark.parametrize("kind", ["torque", "velocity", "effort", "current"])
def test_step_rejects_an_action_kind_that_differs_from_the_control_target(tiny_env, kind):
    """A mismatched NamedAction.kind is rejected against the control's target."""
    control = tiny_env.controls["tiny"]
    entry = NamedAction.zeros(tiny_env.num_envs, list(control.joint_names), kind)

    with pytest.raises(ValueError) as excinfo:
        tiny_env.step({"tiny": entry})

    message = str(excinfo.value)
    assert "action entry 'tiny'" in message
    assert f"kind '{kind}'" in message
    assert "control 'tiny' expects target 'position'" in message


def test_direct_torque_writes_the_target_torque(tiny_env_factory):
    """A torque control runs the compiled motors with the requested torque."""
    env = tiny_env_factory(drive="torque", target="torque", law="direct_torque", gains={})

    _, extra = env.step(_action(env, [[2.0, -1.0]]))

    applied = extra["articulation/tiny/applied_torque"]
    assert applied[0].tolist() == pytest.approx([2.0, -1.0])
    joint_vel = extra["articulation/tiny/joint_vel"]
    assert joint_vel[0, 0] > 0.0
    assert joint_vel[0, 1] < 0.0


def test_direct_velocity_writes_the_target_velocity(tiny_env_factory, tiny_velocity_mjcf):
    """A velocity control runs the compiled velocity servos at the target."""
    env = tiny_env_factory(
        mjcf=tiny_velocity_mjcf,
        drive="velocity",
        target="velocity",
        law="direct_velocity",
        gains={},
    )

    for _ in range(10):
        _, extra = env.step(_action(env, [[1.0, -1.0]]))

    joint_vel = extra["articulation/tiny/joint_vel"]
    assert joint_vel[0, 0] > 0.5
    assert joint_vel[0, 1] < -0.5


def test_direct_torque_requires_a_compiled_motor(tiny_env_factory, tiny_velocity_mjcf):
    """direct_torque over velocity servos is a plant mismatch, not a fallback."""
    with pytest.raises(ValueError, match=r"direct_torque.*compiled law is 'velocity_servo'"):
        tiny_env_factory(
            mjcf=tiny_velocity_mjcf,
            drive="torque",
            target="torque",
            law="direct_torque",
            gains={},
        )


def test_direct_velocity_requires_a_compiled_velocity_servo(tiny_env_factory):
    """direct_velocity over torque motors is a plant mismatch, not a fallback."""
    with pytest.raises(ValueError, match=r"direct_velocity.*compiled law is 'motor'"):
        tiny_env_factory(drive="velocity", target="velocity", law="direct_velocity", gains={})


def test_direct_torque_rejects_a_non_unit_gain_product(tiny_env_factory, tiny_mjcf, tmp_path):
    """A gain-2 bias-free general actuator doubles ctrl: not joint torque."""
    variant = tmp_path / "tiny_arm_gain2.xml"
    variant.write_text(
        tiny_mjcf.read_text(encoding="utf-8").replace(
            '<motor name="shoulder_motor" joint="shoulder" ctrlrange="-20 20"/>',
            '<general name="shoulder_motor" joint="shoulder" biastype="none" '
            'gainprm="2 0 0" ctrlrange="-20 20"/>',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        tiny_env_factory(
            mjcf=variant,
            drive="torque",
            target="torque",
            law="direct_torque",
            gains={},
        )

    message = str(excinfo.value)
    assert "articulation 'tiny'" in message
    assert "actuation group 'body'" in message
    assert "joint 'shoulder'" in message
    assert "law 'direct_torque'" in message
    assert "gain*gear is 2.0" in message
    assert "gainprm[0]=2.0" in message
    assert "gear=1.0" in message


def test_direct_torque_accepts_a_unit_gain_general_actuator(tiny_env_factory, tiny_mjcf, tmp_path):
    """A general gain-1 bias-free actuator is still a plain torque motor."""
    variant = tmp_path / "tiny_arm_gain1.xml"
    variant.write_text(
        tiny_mjcf.read_text(encoding="utf-8").replace(
            '<motor name="shoulder_motor" joint="shoulder" ctrlrange="-20 20"/>',
            '<general name="shoulder_motor" joint="shoulder" biastype="none" '
            'gainprm="1 0 0" ctrlrange="-20 20"/>',
        ),
        encoding="utf-8",
    )

    env = tiny_env_factory(
        mjcf=variant,
        drive="torque",
        target="torque",
        law="direct_torque",
        gains={},
    )

    _, extra = env.step(_action(env, [[2.0, -1.0]]))
    applied = extra["articulation/tiny/applied_torque"]
    assert applied[0].tolist() == pytest.approx([2.0, -1.0])


def test_settle_feeds_a_neutral_target_to_direct_torque(tiny_env_factory):
    """Settle must not command torque=q0: a direct torque law settles on zero."""
    env = tiny_env_factory(drive="torque", target="torque", law="direct_torque", gains={})
    control = env.controls["tiny"]
    control.q0 = control.q0 + 5.0

    env._settle_substeps(1)

    slots = env._actuators[0].bound_slots
    assert env._active_data.ctrl[slots].tolist() == pytest.approx([0.0, 0.0])


def test_settle_feeds_a_neutral_target_to_direct_velocity(tiny_env_factory, tiny_velocity_mjcf):
    """Settle must not command velocity=q0: a direct velocity law settles on zero."""
    env = tiny_env_factory(
        mjcf=tiny_velocity_mjcf,
        drive="velocity",
        target="velocity",
        law="direct_velocity",
        gains={},
    )
    control = env.controls["tiny"]
    control.q0 = control.q0 + 5.0

    env._settle_substeps(1)

    slots = env._actuators[0].bound_slots
    assert env._active_data.ctrl[slots].tolist() == pytest.approx([0.0, 0.0])


def test_settle_keeps_the_q0_goal_for_position_drives(tiny_env_factory, monkeypatch):
    """Settle still feeds q0 to position laws (the unchanged reset semantics)."""
    env = tiny_env_factory()
    control = env.controls["tiny"]
    control.q0 = control.q0 + 0.5
    seen = []
    actuator = env._actuators[0]
    original = actuator.set_targets

    def record(targets):
        seen.append(list(targets))
        original(targets)

    monkeypatch.setattr(actuator, "set_targets", record)

    env._settle_substeps(1)

    assert seen[0] == pytest.approx([0.5, 0.5])


def test_direct_torque_keeps_batch_rows_independent(tiny_env_factory):
    """A batched direct-torque step writes each row's own target."""
    env = tiny_env_factory(
        num_envs=2, drive="torque", target="torque", law="direct_torque", gains={}
    )

    _, extra = env.step(_action(env, [[1.0, -1.0], [2.0, 0.5]]))

    applied = extra["articulation/tiny/applied_torque"]
    assert applied[0].tolist() == pytest.approx([1.0, -1.0])
    assert applied[1].tolist() == pytest.approx([2.0, 0.5])


@pytest.mark.parametrize(
    ("drive", "target"),
    [("effort", "effort"), ("current", "current")],
)
def test_mjwarp_rejects_unserved_direct_drives_at_build(tiny_env_factory, drive, target):
    """effort/current have no native direct-drive law and fail before routing."""
    with pytest.raises(ValueError, match="no native direct-drive actuation law"):
        tiny_env_factory(drive=drive, target=target)


def test_computed_torque_pd_is_a_supported_non_standard_law(tiny_env_factory):
    """A computed_torque_pd group builds the seam and still tracks a target."""
    env = tiny_env_factory(law="computed_torque_pd")
    _, extra = env.step(_action(env, [[0.3, -0.2]]))

    joint_pos = extra["articulation/tiny/joint_pos"]
    assert 0.0 < joint_pos[0, 0] <= 0.3

    env.reset_all()


def test_native_servo_requires_a_compiled_position_servo(tiny_env_factory):
    """A native_servo law over torque motors is a plant mismatch, not silent."""
    with pytest.raises(ValueError, match=r"native_servo.*compiled law is 'motor'"):
        tiny_env_factory(law="native_servo", gains={}, drive="position_servo")


def test_direct_env_with_controls_but_no_plan_fails(tiny_env_factory):
    """A direct env must not silently leave driven joints unactuated."""
    with pytest.raises(ValueError, match="no actuation plan"):
        tiny_env_factory(plans=False)


@pytest.mark.parametrize("params", [{"actuators": {}}, {"kp": {"shoulder": 1.0}}])
def test_direct_env_rejects_removed_gain_env_params(tiny_env_factory, params):
    """The removed env actuator/gain channels fail at construction."""
    with pytest.raises(ValueError, match="removed actuator/gain keys"):
        tiny_env_factory(params=params)


@pytest.mark.parametrize(
    ("joint_names", "fragment"),
    [
        pytest.param(["elbow"], "leaves joints unaddressed: ['shoulder']", id="missing-joint"),
        pytest.param(["elbow", "wrist"], "names joints ['wrist']", id="unowned-joint"),
    ],
)
def test_step_rejects_an_action_that_does_not_match_the_control_joint_set(
    tiny_env, joint_names, fragment
):
    """An action's joint columns must address exactly the control's joints."""
    entry = NamedAction(torch.zeros(tiny_env.num_envs, len(joint_names)), joint_names, "position")

    with pytest.raises(ValueError, match=fragment.replace("[", r"\[").replace("]", r"\]")):
        tiny_env.step({"tiny": entry})


def test_step_rejects_a_non_mapping_action_payload(tiny_env):
    """The step entry point requires dict[str, NamedAction], not a sequence."""
    entry = _action(tiny_env, [[0.0, 0.0]])["tiny"]

    with pytest.raises(TypeError, match=r"dict\[str, NamedAction\]"):
        tiny_env.step([("tiny", entry)])
