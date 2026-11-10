"""IsaacLab 2 Cartpole named-action stepping, extra contract and routing errors."""

from __future__ import annotations

import pytest
import torch

from unigym.runtime.actions import NamedAction

pytestmark = pytest.mark.sim

ARTICULATION = "cartpole"
CART_JOINTS = ("slider_to_cart", "cart_to_pole")

ARTICULATION_EXTRA_KEYS = (
    "joint_pos",
    "joint_vel",
    "applied_torque",
    "gravity_b",
    "base_pos_w",
    "base_quat_w",
    "base_lin_vel_w",
    "base_ang_vel_w",
    "base_lin_vel_b",
    "base_ang_vel_b",
    "foot_contact",
    "foot_contact_force_z",
    "foot_pos_z",
)
GLOBAL_EXTRA_KEYS = ("time_in_episode", "episode_id", "reset_env_ids", "should_reset")
EXPECTED_EXTRA_KEYS = {
    *(f"articulation/{ARTICULATION}/{name}" for name in ARTICULATION_EXTRA_KEYS),
    *GLOBAL_EXTRA_KEYS,
}

ONE_ENV_TENSOR_SPECS = {
    f"articulation/{ARTICULATION}/joint_pos": ((1, 2), torch.float32),
    f"articulation/{ARTICULATION}/joint_vel": ((1, 2), torch.float32),
    f"articulation/{ARTICULATION}/applied_torque": ((1, 2), torch.float32),
    f"articulation/{ARTICULATION}/gravity_b": ((1, 3), torch.float32),
    f"articulation/{ARTICULATION}/base_pos_w": ((1, 3), torch.float32),
    f"articulation/{ARTICULATION}/base_quat_w": ((1, 4), torch.float32),
    f"articulation/{ARTICULATION}/base_lin_vel_w": ((1, 3), torch.float32),
    f"articulation/{ARTICULATION}/base_ang_vel_w": ((1, 3), torch.float32),
    f"articulation/{ARTICULATION}/base_lin_vel_b": ((1, 3), torch.float32),
    f"articulation/{ARTICULATION}/base_ang_vel_b": ((1, 3), torch.float32),
    f"articulation/{ARTICULATION}/foot_contact": ((1, 1), torch.bool),
    f"articulation/{ARTICULATION}/foot_contact_force_z": ((1, 1), torch.float32),
    f"articulation/{ARTICULATION}/foot_pos_z": ((1, 1), torch.float32),
    "time_in_episode": ((1,), torch.float32),
    "episode_id": ((1,), torch.int64),
}

TWO_ENV_TENSOR_SHAPES = {
    f"articulation/{ARTICULATION}/joint_pos": (2, 2),
    f"articulation/{ARTICULATION}/base_pos_w": (2, 3),
    f"articulation/{ARTICULATION}/base_quat_w": (2, 4),
    f"articulation/{ARTICULATION}/foot_contact": (2, 1),
    "time_in_episode": (2,),
    "episode_id": (2,),
}


def _zero_action(env):
    return {ARTICULATION: NamedAction.zeros(env.num_envs, list(CART_JOINTS), "position")}


def test_zero_action_advances_exactly_one_policy_step(cartpole_env):
    """A zero named action advances one step and returns the (obs, extra) pair."""
    obs, extra = cartpole_env.step(_zero_action(cartpole_env))

    assert isinstance(obs, dict)
    assert isinstance(extra, dict)
    assert cartpole_env.step_dt == pytest.approx(4 * 0.005)
    assert extra["time_in_episode"].tolist() == pytest.approx([cartpole_env.step_dt])


def test_extra_key_set_is_the_cartpole_contract(cartpole_env):
    """The step extra carries exactly the Cartpole articulation and bookkeeping keys."""
    _, extra = cartpole_env.step(_zero_action(cartpole_env))

    assert set(extra) == EXPECTED_EXTRA_KEYS


def test_extra_tensors_carry_their_contracted_shapes_and_dtypes(cartpole_env):
    """Every recorded extra tensor has the contracted batch shape and dtype."""
    _, extra = cartpole_env.step(_zero_action(cartpole_env))

    actual = {key: (tuple(extra[key].shape), extra[key].dtype) for key in ONE_ENV_TENSOR_SPECS}
    assert actual == ONE_ENV_TENSOR_SPECS


def test_first_step_requests_no_reset(cartpole_env):
    """A long episode's first step reports empty reset ids and no reset flag."""
    _, extra = cartpole_env.step(_zero_action(cartpole_env))

    assert extra["reset_env_ids"].dtype == torch.int64
    assert extra["reset_env_ids"].tolist() == []
    assert extra["should_reset"] is False


def test_two_env_step_batches_every_extra_over_both_rows(two_env):
    """A two-env step batches every extra over both env rows."""
    _, extra = two_env.step(_zero_action(two_env))

    shapes = {key: tuple(extra[key].shape) for key in TWO_ENV_TENSOR_SHAPES}
    assert shapes == TWO_ENV_TENSOR_SHAPES


@pytest.mark.parametrize("face", ["g1", "wrist"])
def test_unknown_action_face_fails_fast(cartpole_env, face):
    """An action entry naming no control face is rejected with the accepted faces."""
    entry = NamedAction.zeros(cartpole_env.num_envs, list(CART_JOINTS), "position")

    with pytest.raises(ValueError) as excinfo:
        cartpole_env.step({face: entry})

    message = str(excinfo.value)
    assert face in message
    assert f"accepted action faces: ['{ARTICULATION}']" in message


def test_missing_action_entry_fails_fast(cartpole_env):
    """Leaving a required control unaddressed is rejected."""
    with pytest.raises(ValueError) as excinfo:
        cartpole_env.step({})

    message = str(excinfo.value)
    assert "leave controls unaddressed" in message
    assert f"['{ARTICULATION}']" in message


def test_wrong_env_row_count_fails_fast(cartpole_env):
    """An entry with the wrong batch height is rejected against num_envs."""
    entry = NamedAction.zeros(cartpole_env.num_envs + 1, list(CART_JOINTS), "position")

    with pytest.raises(ValueError, match=r"has 2 env rows, expected 1"):
        cartpole_env.step({ARTICULATION: entry})


def test_wrong_joint_count_fails_fast(cartpole_env):
    """An entry addressing fewer joints than the action term is rejected."""
    entry = NamedAction.zeros(cartpole_env.num_envs, list(CART_JOINTS[:1]), "position")

    with pytest.raises(ValueError, match=r"addresses 1 joints but term 'cartpole' covers 2"):
        cartpole_env.step({ARTICULATION: entry})


def test_unknown_joint_name_fails_fast(cartpole_env):
    """An entry column naming a joint the action term does not own is rejected."""
    entry = NamedAction.zeros(cartpole_env.num_envs, [CART_JOINTS[0], "wrist"], "position")

    with pytest.raises(ValueError) as excinfo:
        cartpole_env.step({ARTICULATION: entry})

    message = str(excinfo.value)
    assert "unknown joint name(s) ['wrist']" in message
    assert f"accepted names: {list(CART_JOINTS)}" in message
