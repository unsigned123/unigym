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
    from unigym.values import ValueView


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
        self.value_views: list[ValueView] = []
        self.episode_calls: list = []

    def bind_env(self, env) -> None:
        self.bound_envs.append(env)

    def bind_values(self, values: ValueView) -> None:
        self.value_views.append(values)

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


class ClosedLoopValueSampler(CommandSampler):
    """Sampler double that turns one published value into the next command.

    It records, per ``update`` call, the value observed through the bound
    :class:`~unigym.values.ValueView` and the command the runner passed in
    (the value the tick's ``env.step`` used).  ``None`` means the value was
    absent or the runner had no command yet.  Values are kept as isolated
    clones; the double declares no rewind capability.
    """

    def __init__(
        self,
        dim: int,
        device: str,
        num_envs: int,
        *,
        value: str = "acme/probe",
    ) -> None:
        super().__init__(dim, device, num_envs)
        self.value = value
        self.value_views: list[ValueView] = []
        self.observations: list[torch.Tensor | None] = []
        self.used_commands: list[torch.Tensor | None] = []

    def bind_values(self, values: ValueView) -> None:
        self.value_views.append(values)

    def reset(self, env_ids=None) -> torch.Tensor:
        self.cmd[self._resolve_ids(env_ids)] = 0.0
        return self.cmd

    def update(self, step_dt: float, cmd: torch.Tensor) -> torch.Tensor:
        del step_dt
        if not self.value_views:
            raise RuntimeError("ClosedLoopValueSampler.update ran before bind_values")
        observed = self.value_views[0].get(self.value)
        self.observations.append(None if observed is None else observed.detach().clone())
        self.used_commands.append(None if cmd is None else cmd.detach().clone())
        if observed is None:
            self.cmd = torch.zeros(self.num_envs, self.dim, dtype=torch.float32, device=self.device)
        else:
            self.cmd = observed.to(device=self.device, dtype=torch.float32).reshape(
                self.num_envs, self.dim
            )
            self.cmd = self.cmd.clone()
        return self.cmd


def make_not_a_sampler(*, dim: int, device: str, num_envs: int, **params) -> object:
    """A factory that violates the CommandSampler return contract."""
    del dim, device, num_envs, params
    return object()
