"""The fake backend satisfies the shared EnvBackend contract and its semantics."""

from __future__ import annotations

import pytest
import torch

from tests._support.fakes.backend import (
    ARTICULATION_NAME,
    CONTROL_NAME,
    FakeBackend,
)
from unigym.env.backends.base import (
    RECORD_KEYS,
    assert_backend_contract,
    check_extra_info,
    check_obs_cache,
)
from unigym.runtime.actions import NamedAction

_BOOKKEEPING_KEYS = frozenset({"time_in_episode", "episode_id", "reset_env_ids", "should_reset"})


@pytest.fixture(params=(1, 2), ids=("N1", "N2"))
def env(request):
    backend = FakeBackend(num_envs=request.param)
    try:
        yield backend
    finally:
        backend.close()


def _named_action(env, values):
    control = env.controls[CONTROL_NAME]
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(
        env.num_envs, len(control.joint_names)
    )
    return NamedAction(tensor, list(control.joint_names), control.target)


def _step(env, values, **kwargs):
    control = env.controls[CONTROL_NAME]
    return env.step({control.action_face: _named_action(env, values)}, **kwargs)


@pytest.mark.parametrize("num_envs", (1, 2))
def test_fake_backend_satisfies_env_backend_contract(num_envs):
    """The fake passes the structural and live-step contract validator."""
    backend = FakeBackend(num_envs=num_envs)
    try:
        assert_backend_contract(backend, exercise_step=True)
    finally:
        backend.close()


def test_step_publishes_contract_shaped_obs_and_extra(env):
    """Obs/extra keys, shapes, dtypes and the record namespace are valid."""
    control = env.controls[CONTROL_NAME]
    obs, extra = _step(env, [[0.0]] * env.num_envs)

    check_obs_cache(obs, env.num_envs, device=env.device)
    check_obs_cache(env.obs_cache, env.num_envs, device=env.device)
    check_extra_info(extra, env.num_envs, len(control.joint_names), env.device)

    assert set(obs) == {"joint_pos", "prev_action"}
    assert tuple(obs["joint_pos"].shape) == (env.num_envs, len(control.joint_names))
    assert tuple(obs["prev_action"].shape) == (env.num_envs, len(control.joint_names))

    namespaced = {key for key in RECORD_KEYS if key not in _BOOKKEEPING_KEYS}
    expected = {f"articulation/{ARTICULATION_NAME}/{key}" for key in namespaced}
    expected |= _BOOKKEEPING_KEYS
    assert set(extra) == expected
    assert extra["should_reset"] is False
    assert tuple(extra["reset_env_ids"].shape) == (0,)
    assert extra["reset_env_ids"].dtype == torch.int64


def test_step_applies_named_action_and_passes_raw_action_through(env):
    """A named action drives the joint state; raw_action lands in prev_action."""
    control = env.controls[CONTROL_NAME]
    values = torch.arange(1, env.num_envs + 1, dtype=torch.float32).reshape(env.num_envs, 1) / 10
    raw_action = values + 10.0

    action = NamedAction(values.clone(), list(control.joint_names), control.target)
    obs, extra = env.step({control.action_face: action}, raw_action=raw_action)

    assert torch.equal(obs["joint_pos"], values)
    assert torch.equal(obs["prev_action"], raw_action)
    assert torch.equal(extra[f"articulation/{ARTICULATION_NAME}/joint_pos"], values)


def test_step_publishes_command_only_when_supplied(env):
    """A supplied command lands in obs_cache; absent command keeps the cache key absent."""
    command = torch.arange(env.num_envs, dtype=torch.float32).reshape(env.num_envs, 1)

    obs, _ = _step(env, [[0.0]] * env.num_envs)
    assert "command" not in obs

    obs, _ = _step(env, [[0.0]] * env.num_envs, cmd=command)
    assert torch.equal(obs["command"], command)
    assert torch.equal(env.obs_cache["command"], command)


def test_step_is_deterministic_for_same_state_and_actions(env):
    """Two fresh fakes produce identical observations and extra info."""
    values = torch.arange(1, env.num_envs + 1, dtype=torch.float32).reshape(env.num_envs, 1) / 4
    control = env.controls[CONTROL_NAME]
    action = NamedAction(values.clone(), list(control.joint_names), control.target)
    actions = {control.action_face: action}

    other = FakeBackend(num_envs=env.num_envs)
    try:
        obs_a, extra_a = env.step(actions)
        obs_b, extra_b = other.step(actions)
    finally:
        other.close()

    assert set(obs_a) == set(obs_b)
    for key in obs_a:
        assert torch.equal(obs_a[key], obs_b[key]), key
    assert set(extra_a) == set(extra_b)
    for key in extra_a:
        if isinstance(extra_a[key], torch.Tensor):
            assert torch.equal(extra_a[key], extra_b[key]), key
        else:
            assert extra_a[key] == extra_b[key], key


def test_identity_permutation_round_trips(env):
    """The fake publishes inverse identity permutations and a trivial round-trip."""
    control = env.controls[CONTROL_NAME]
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
    control = env.controls[CONTROL_NAME]
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
