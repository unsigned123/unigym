"""Fixed (constant) command sampler."""
from __future__ import annotations

import torch

from .base import CommandSampler


class FixedSampler(CommandSampler):
    """Constant command vector, identical for every env and every step.

    ``rewind_inert`` (v056 checkpoint ring): the explicit class
    attribute the rewind gate consults -- this sampler carries NO
    state outside the snapshot (constants only, zero RNG), so a run
    with an ignored fixed command (agent_g1 session configs) keeps its
    checkpoint/undo face."""

    rewind_inert = True

    def __init__(self, dim: int, device, num_envs: int = 1, values: list | None = None):
        super().__init__(dim, device, num_envs)
        if values is None:
            values = [0.0] * dim
        if len(values) != dim:
            raise ValueError(f"FixedSampler: expected {dim} values, got {len(values)}")
        self.values = torch.tensor(values, dtype=torch.float32, device=device)

    def reset(self, env_ids=None) -> torch.Tensor:
        ids = self._resolve_ids(env_ids)
        self.cmd[ids] = self.values
        return self.cmd

    def update(self, step_dt: float, cmd: torch.Tensor) -> torch.Tensor:
        return self.cmd  # constant by construction
