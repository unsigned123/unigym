"""Custom command-sampler factory doubles for build and lifecycle tests.

Boundary: :class:`unigym.commands.base.CommandSampler` built through the
documented ``command.module`` factory seam.  The sampler only records which
lifecycle calls happened; it declares no rewind capability, so it must not be
used as a rewind model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from unigym.commands.base import CommandSampler

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from unigym.signals import SignalView


class RecordingSampler(CommandSampler):
    """Sampler double recording factory parameters and lifecycle calls."""

    def __init__(
        self,
        dim: int,
        device: str,
        num_envs: int,
        *,
        seats: int = 1,
    ) -> None:
        super().__init__(dim, device, num_envs)
        self.seats = seats
        self.bound_envs: list = []
        self.signal_views: list[SignalView] = []
        self.episode_calls: list = []

    def bind_env(self, env) -> None:
        self.bound_envs.append(env)

    def bind_signals(self, signals: SignalView) -> None:
        self.signal_views.append(signals)

    def on_episode_end(self, env_ids, results) -> None:
        self.episode_calls.append((env_ids.tolist(), results))

    def reset(self, env_ids=None) -> torch.Tensor:
        self.cmd[self._resolve_ids(env_ids)] = float(self.seats)
        return self.cmd

    def update(self, step_dt: float, cmd: torch.Tensor) -> torch.Tensor:
        del cmd
        return self.cmd


def make_sampler(*, dim: int, device: str, num_envs: int, **params) -> CommandSampler:
    """Factory matching the documented external sampler signature."""
    return RecordingSampler(dim, device, num_envs, **params)


def make_not_a_sampler(*, dim: int, device: str, num_envs: int, **params) -> object:
    """A factory that violates the CommandSampler return contract."""
    del dim, device, num_envs, params
    return object()
