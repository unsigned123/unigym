"""CommandSampler interface (torch-only, unit-testable)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class CommandSampler(ABC):
    """Per-env command vector sampler with optional timed resampling.

    Holds a buffer ``self.cmd`` of shape (num_envs, dim). ``reset`` resamples
    the given envs; ``update`` is called once per policy step with the step
    dt and may resample envs whose timer expired.
    """

    def __init__(self, dim: int, device, num_envs: int = 1):
        self.dim = dim
        self.device = device
        self.num_envs = num_envs
        self.cmd = torch.zeros(num_envs, dim, dtype=torch.float32, device=device)

    @abstractmethod
    def reset(self, env_ids=None) -> torch.Tensor:
        """(Re)sample commands for env_ids (None = all) and return the buffer."""

    @abstractmethod
    def update(self, step_dt: float, cmd: torch.Tensor) -> torch.Tensor:
        """Advance timers by step_dt, resample expired envs, return the buffer."""

    def _resolve_ids(self, env_ids) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device)
        return env_ids
