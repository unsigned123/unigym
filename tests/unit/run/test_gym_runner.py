"""Gym facades keep the documented reset behavior for open and done envs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests._support.fakes.backend import ARTICULATION_NAME, FakeBackend
from tests._support.fakes.external_sink import LifecycleProbeSink
from unigym.adapters.manual import ManualAdapter
from unigym.runner.gym import GymEnvRunner, GymVectorRunner
from unigym.runtime.actions import NamedAction
from unigym.terminations import TerminationGroup


def _actions(num_envs: int = 1) -> tuple[NamedAction, ...]:
    return (
        NamedAction(
            ARTICULATION_NAME,
            torch.zeros(num_envs, 1),
            ("shoulder",),
            "position",
        ),
    )


def _instance(env: FakeBackend):
    spec = SimpleNamespace(
        cfg={"collect": {"steps": 4}},
        runner={},
        adapter_name="manual",
    )
    return SimpleNamespace(
        spec=spec,
        profile=None,
        adapter=ManualAdapter(),
        sampler=None,
        term_group=TerminationGroup(),
        env=env,
        sink=LifecycleProbeSink(),
        replay=None,
        reset_on_termination=False,
    )


def test_gym_env_reset_after_done_starts_the_next_episode():
    """A truncated single-env episode resets normally and can step again."""
    env = FakeBackend(due_rows=(0,))
    runner = GymEnvRunner(_instance(env))
    try:
        _, _, terminated, truncated, _ = runner.step(_actions())
        assert terminated is False
        assert truncated is True
        with pytest.raises(RuntimeError, match="call reset"):
            runner.step(_actions())

        obs, _info = runner.reset()

        assert isinstance(obs, dict)
        assert env.record()["episode_id"].tolist() == [1]
        assert env.record()["time_in_episode"].tolist() == [0.0]
        runner.step(_actions())
    finally:
        env.close()


def test_gym_env_reset_of_an_open_env_keeps_the_documented_behavior():
    """A mid-run reset restarts the episode without finalizing it."""
    env = FakeBackend()
    runner = GymEnvRunner(_instance(env))
    try:
        runner.step(_actions())
        assert env.record()["time_in_episode"].tolist() == [1.0]

        _obs, info = runner.reset()

        assert info["episode_id"].tolist() == [1]
        assert info["time_in_episode"].tolist() == [0.0]
        assert runner.ticks_done == 1
    finally:
        env.close()


def test_gym_vector_reset_of_an_open_batch_restarts_every_row():
    """The batched facade also resets open rows mid-run."""
    env = FakeBackend(num_envs=2)
    runner = GymVectorRunner(_instance(env))
    try:
        runner.step(_actions(num_envs=2))
        assert env.record()["time_in_episode"].tolist() == [1.0, 1.0]

        obs, info = runner.reset()

        assert isinstance(obs, dict)
        assert env.record()["episode_id"].tolist() == [1, 1]
        assert env.record()["time_in_episode"].tolist() == [0.0, 0.0]
        assert info["episode_id"].tolist() == [1, 1]
    finally:
        env.close()
