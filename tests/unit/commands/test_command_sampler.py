"""CommandSampler lifecycle hooks and the read-only EpisodeResultView."""

from __future__ import annotations

import dataclasses
import operator

import pytest
import torch

from unigym.commands.base import CommandSampler
from unigym.core.episode_result import EpisodeResultView


class _ConstantSampler(CommandSampler):
    """Minimal sampler proving the base-class hook defaults."""

    def __init__(self, dim: int = 2, num_envs: int = 1) -> None:
        super().__init__(dim, "cpu", num_envs)

    def reset(self, env_ids=None) -> torch.Tensor:
        self.cmd[self._resolve_ids(env_ids)] = 1.0
        return self.cmd

    def update(self, step_dt: float, cmd: torch.Tensor) -> torch.Tensor:
        return self.cmd


def test_command_sampler_defaults_declare_no_rewind_capability():
    """A plain sampler is neither rewind-inert nor rewind-capable."""
    sampler = _ConstantSampler()

    assert sampler.rewind_inert is False
    assert sampler.rewind_capable is False


def test_command_sampler_lifecycle_hooks_default_to_no_ops():
    """bind_env and on_episode_end are optional no-ops for built-in samplers."""
    sampler = _ConstantSampler()

    assert sampler.bind_env(object()) is None
    assert sampler.on_episode_end(torch.tensor([0]), {}) is None


@pytest.mark.parametrize("method_name", ["rewind_state", "apply_rewind_state"])
def test_command_sampler_rewind_defaults_name_the_capability_rule(method_name):
    """Unimplemented rewind methods fail with the rewind_capable teaching rule."""
    sampler = _ConstantSampler()
    method = getattr(sampler, method_name)
    arguments = () if method_name == "rewind_state" else ({},)

    with pytest.raises(NotImplementedError, match="rewind_capable = True"):
        method(*arguments)


def test_episode_result_view_copies_metrics_and_fixes_its_fields():
    """The view owns a private read-only metrics copy and cannot be reassigned."""
    metrics = {"score": 1.5}
    view = EpisodeResultView(task="acme/reach", success=True, metrics=metrics)

    metrics["score"] = 9.0

    assert view.task == "acme/reach"
    assert view.success is True
    assert dict(view.metrics) == {"score": 1.5}
    with pytest.raises(TypeError):
        operator.setitem(view.metrics, "score", 2.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(view, "success", False)
