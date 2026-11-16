"""Gym facades reject the removed mapping/dict action form loudly."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._support.fakes.backend import FakeBackend
from unigym.runner.gym import GymEnvRunner, GymVectorRunner
from unigym.terminations import TerminationGroup


def _instance(num_envs=1):
    spec = SimpleNamespace(
        cfg={"collect": {"steps": 1}},
        runner={},
        adapter_name="manual",
    )
    return SimpleNamespace(
        spec=spec,
        profile=None,
        adapter=SimpleNamespace(_unigym_name="manual"),
        sampler=None,
        term_group=TerminationGroup(),
        env=FakeBackend(num_envs=num_envs),
        sink=SimpleNamespace(),
        replay=None,
        reset_on_termination=False,
    )


def test_single_env_facade_rejects_a_mapping_even_with_action_fn():
    """A dict must not fall through to the per-env action_fn hook."""
    runner = GymEnvRunner(_instance(), action_fn=lambda action: ())

    with pytest.raises(TypeError, match="mapping action"):
        runner.step({"arm": object()})


def test_vector_facade_rejects_a_mapping_even_with_action_fn():
    """A dict must not be treated as a per-env action by the vector facade."""
    runner = GymVectorRunner(_instance(num_envs=2), action_fn=lambda action: ())

    with pytest.raises(TypeError, match="mapping action"):
        runner.step({"arm": object()})
