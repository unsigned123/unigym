"""The fake backend satisfies the shared EnvBackend contract and its semantics."""

from __future__ import annotations

import pytest
import torch

from tests._support.fakes.backend import (
    ARTICULATION_NAME,
    CONTROL_NAME,
    FakeBackend,
)
from unigym.env.backends.base import assert_backend_contract
from unigym.runtime.actions import NamedAction

_BOOKKEEPING_KEYS = frozenset({"time_in_episode", "episode_id"})


@pytest.fixture(params=(1, 2), ids=("N1", "N2"))
def env(request):
    backend = FakeBackend(num_envs=request.param)
    try:
        yield backend
    finally:
        backend.close()


def _named_action(env, values):
    control = env.controls[(ARTICULATION_NAME, CONTROL_NAME)]
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(
        env.num_envs, len(control.joint_names)
    )
    return NamedAction(
        control.articulation,
        tensor,
        control.joint_names,
        control.target,
    )


def _step(env, values, **kwargs):
    return env.step((_named_action(env, values),), **kwargs)


@pytest.mark.parametrize("num_envs", (1, 2))
def test_fake_backend_satisfies_env_backend_contract(num_envs):
    """The fake passes the structural and live-step contract validator."""
    backend = FakeBackend(num_envs=num_envs)
    try:
        assert_backend_contract(backend, exercise_step=True)
    finally:
        backend.close()


def test_step_publishes_schema_shaped_obs_and_record(env):
    """Obs/record keys, shapes and dtypes follow the observation schema."""
    control = env.controls[(ARTICULATION_NAME, CONTROL_NAME)]
    step = _step(env, [[0.0]] * env.num_envs)

    assert step.obs.schema is env.obs.schema
    for key, value in step.obs.values.items():
        assert isinstance(value, torch.Tensor)
        assert value.shape[0] == env.num_envs
    assert tuple(step.obs.robot(ARTICULATION_NAME).joint_pos.shape) == (
        env.num_envs,
        len(control.joint_names),
    )
    assert step.should_reset is False
    assert step.reset_env_ids.numel() == 0
    assert set(step.record) >= _BOOKKEEPING_KEYS
    assert f"articulation/{ARTICULATION_NAME}/joint_pos" in step.record


def test_step_applies_named_action_and_passes_raw_action_through(env):
    """A named action drives the joint state; raw_action lands in prev_action."""
    control = env.controls[(ARTICULATION_NAME, CONTROL_NAME)]
    values = torch.arange(1, env.num_envs + 1, dtype=torch.float32).reshape(env.num_envs, 1) / 10
    raw_action = values + 10.0

    action = NamedAction(
        control.articulation,
        values.clone(),
        control.joint_names,
        control.target,
    )
    step = env.step((action,), raw_action=raw_action)

    assert torch.equal(step.obs.robot(ARTICULATION_NAME).joint_pos, values)
    assert torch.equal(step.obs.prev_action.value, raw_action)
    assert torch.equal(step.record[f"articulation/{ARTICULATION_NAME}/joint_pos"], values)


def test_step_publishes_command_only_when_supplied(env):
    """A supplied command lands in the registry; no command -> absent slot."""
    command = torch.arange(env.num_envs, dtype=torch.float32).reshape(env.num_envs, 1)

    step = _step(env, [[0.0]] * env.num_envs)
    assert step.obs.command.value is None

    step = _step(env, [[0.0]] * env.num_envs, cmd=command)
    assert torch.equal(step.obs.command.value, command)
    assert torch.equal(env.obs.command.value, command)


def test_step_is_deterministic_for_same_state_and_actions(env):
    """Two fresh fakes produce identical observations and extra info."""
    values = torch.arange(1, env.num_envs + 1, dtype=torch.float32).reshape(env.num_envs, 1) / 4
    control = env.controls[(ARTICULATION_NAME, CONTROL_NAME)]
    action = NamedAction(
        control.articulation,
        values.clone(),
        control.joint_names,
        control.target,
    )
    actions = (action,)

    other = FakeBackend(num_envs=env.num_envs)
    try:
        step_a = env.step(actions)
        step_b = other.step(actions)
    finally:
        other.close()

    assert set(step_a.obs.values) == set(step_b.obs.values)
    for key in step_a.obs.values:
        assert torch.equal(step_a.obs.values[key], step_b.obs.values[key]), key
    assert set(step_a.record) == set(step_b.record)
    for key in step_a.record:
        if isinstance(step_a.record[key], torch.Tensor):
            assert torch.equal(step_a.record[key], step_b.record[key]), key
        else:
            assert step_a.record[key] == step_b.record[key], key


def test_identity_permutation_round_trips(env):
    """The fake publishes inverse identity permutations and a trivial round-trip."""
    control = env.controls[(ARTICULATION_NAME, CONTROL_NAME)]
    joint_count = len(control.joint_names)
    expected = list(range(joint_count))
    assert env.perm_il2pol.tolist() == expected
    assert env.perm_pol2il.tolist() == expected

    values = torch.arange(env.num_envs * joint_count, dtype=torch.float32).reshape(
        env.num_envs, joint_count
    )
    assert torch.equal(env.to_policy_order(values), values)
    assert torch.equal(env.to_il_order(values), values)
    assert torch.equal(env.to_policy_order(env.to_il_order(values)), values)


def test_policy_joint_pos_limits_have_policy_shape(env):
    """Limits are a (N, J) float32 lower/upper pair in policy order."""
    control = env.controls[(ARTICULATION_NAME, CONTROL_NAME)]
    joint_count = len(control.joint_names)
    limits = env.policy_joint_pos_limits

    assert limits is not None
    lower, upper = limits
    for bound in (lower, upper):
        assert bound.dtype == torch.float32
        assert tuple(bound.shape) == (env.num_envs, joint_count)
        assert str(bound.device) == env.device
    assert bool((lower <= upper).all())


def test_close_is_idempotent():
    """Closing the fake twice is a no-op, not an error."""
    backend = FakeBackend(num_envs=2)
    backend.close()
    backend.close()
