"""Gym facades reject the removed mapping/dict action form loudly."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

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


def _entry(value, joints=("shoulder",)):
    from unigym.runtime.actions import NamedAction

    values = torch.zeros(1, len(joints), dtype=torch.float32)
    values[0, 0] = value
    return NamedAction("arm", values, tuple(joints), "position")


def test_vector_facade_merges_per_env_entries_by_joint_identity():
    """Per-env joint-addressed entries merge on the env axis by (articulation, kind, joints)."""
    runner = GymVectorRunner(_instance(num_envs=2), action_fn=lambda action: action)
    merged = runner._vector_actions([(_entry(0.1),), (_entry(0.2),)])

    assert len(merged) == 1
    assert merged[0].articulation == "arm"
    assert merged[0].kind == "position"
    assert merged[0].joint_names == ("shoulder",)
    assert torch.allclose(merged[0].values, torch.tensor([[0.1], [0.2]]))


def test_vector_facade_rejects_inconsistent_ordered_joint_names():
    """Same joint set with different column order cannot be concatenated."""
    runner = GymVectorRunner(_instance(num_envs=2), action_fn=lambda action: action)
    two = ("shoulder", "elbow")

    with pytest.raises(ValueError, match="inconsistent ordered joint_names"):
        runner._vector_actions([(_entry(0.1, two),), (_entry(0.2, tuple(reversed(two))),)])
