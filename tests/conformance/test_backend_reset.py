"""The fake backend resets exactly the selected env rows."""

from __future__ import annotations

import pytest
import torch

from tests._support.fakes.backend import CONTROL_NAME, FakeBackend
from unigym.runtime.actions import NamedAction


def _named_action(env, values):
    control = env.controls[CONTROL_NAME]
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(
        env.num_envs, len(control.joint_names)
    )
    return NamedAction(tensor, list(control.joint_names), control.target)


def _step(env, values, **kwargs):
    control = env.controls[CONTROL_NAME]
    return env.step({control.action_face: _named_action(env, values)}, **kwargs)


def test_reset_envs_resets_only_selected_rows():
    """A subset reset restores its rows and leaves the other rows running."""
    backend = FakeBackend(num_envs=2)
    try:
        control = backend.controls[CONTROL_NAME]
        values = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
        raw_action = values + 10.0
        backend.step(
            {
                control.action_face: NamedAction(
                    values.clone(), list(control.joint_names), control.target
                )
            },
            raw_action=raw_action,
        )

        returned = backend.reset_envs(torch.tensor([1], dtype=torch.int64))

        assert torch.equal(returned, torch.tensor([1], dtype=torch.int64))
        assert returned.dtype == torch.int64
        expected_joint = values.clone()
        expected_joint[1] = 0.0
        expected_prev = raw_action.clone()
        expected_prev[1] = 0.0
        assert torch.equal(backend.obs_cache["joint_pos"], expected_joint)
        assert torch.equal(backend.obs_cache["prev_action"], expected_prev)

        _, extra = _step(backend, [[0.0], [0.0]])
        assert extra["episode_id"].tolist() == [0, 1]
        assert extra["time_in_episode"].tolist() == [2.0, 1.0]
    finally:
        backend.close()


@pytest.mark.parametrize("num_envs", (1, 2))
def test_reset_all_resets_every_row(num_envs):
    """reset_all restores every env and reports the full id range."""
    backend = FakeBackend(num_envs=num_envs)
    try:
        values = torch.full((num_envs, 1), 0.3, dtype=torch.float32)
        _step(backend, values, raw_action=values + 1.0)

        returned = backend.reset_all()

        expected = torch.arange(num_envs, dtype=torch.int64)
        assert torch.equal(returned, expected)
        assert torch.equal(backend.obs_cache["joint_pos"], torch.zeros(num_envs, 1))
        assert torch.equal(backend.obs_cache["prev_action"], torch.zeros(num_envs, 1))

        _, extra = _step(backend, [[0.0]] * num_envs)
        assert extra["episode_id"].tolist() == [1] * num_envs
        assert extra["time_in_episode"].tolist() == [1.0] * num_envs
    finally:
        backend.close()


def test_reset_envs_preserves_id_order_and_empty_is_noop():
    """Reset returns ids in the given order; an empty selection changes nothing."""
    backend = FakeBackend(num_envs=2)
    try:
        _step(backend, [[0.5], [0.6]])
        before = backend.obs_cache["joint_pos"].clone()

        empty = backend.reset_envs(torch.empty(0, dtype=torch.int64))
        assert tuple(empty.shape) == (0,)
        assert torch.equal(backend.obs_cache["joint_pos"], before)

        returned = backend.reset_envs(torch.tensor([1, 0], dtype=torch.int64))
        assert returned.tolist() == [1, 0]
        assert torch.equal(backend.obs_cache["joint_pos"], torch.zeros(2, 1))
    finally:
        backend.close()


def test_reset_envs_rejects_out_of_range_ids():
    """Ids outside [0, num_envs) fail fast with the offending id."""
    backend = FakeBackend(num_envs=2)
    try:
        with pytest.raises(ValueError, match="cannot reset env ids"):
            backend.reset_envs(torch.tensor([2], dtype=torch.int64))
        with pytest.raises(ValueError, match="cannot reset env ids"):
            backend.reset_envs(torch.tensor([-1], dtype=torch.int64))
    finally:
        backend.close()
