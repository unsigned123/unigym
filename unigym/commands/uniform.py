"""Uniform command sampler with per-env resample timers."""

from __future__ import annotations

import torch

from .base import CommandSampler


class UniformSampler(CommandSampler):
    """Uniform random commands.

    Args:
        ranges: list of [lo, hi] per dim (length == dim).
        resample_interval_s: (lo, hi) seconds drawn uniformly per env per
            resample, or None to sample once and keep forever.
    """

    def __init__(
        self,
        dim: int,
        device,
        num_envs: int = 1,
        ranges: list | None = None,
        resample_interval_s: tuple | None = None,
    ):
        super().__init__(dim, device, num_envs)
        if ranges is None or len(ranges) != dim:
            raise ValueError(f"UniformSampler: need {dim} [lo,hi] ranges, got {ranges!r}")
        ranges_t = torch.tensor(ranges, dtype=torch.float32, device=device)
        self._lo = ranges_t[:, 0]
        self._hi = ranges_t[:, 1]
        self.resample_interval_s = resample_interval_s
        if resample_interval_s is not None:
            self._int_lo = float(resample_interval_s[0])
            self._int_hi = float(resample_interval_s[1])
        else:
            self._int_lo, self._int_hi = 0.0, 0.0
        self._timers = torch.zeros(num_envs, device=device)

    def _sample(self, ids: torch.Tensor) -> None:
        n = ids.shape[0]
        self.cmd[ids] = self._lo + (self._hi - self._lo) * torch.rand(
            n, self.dim, device=self.device
        )
        if self.resample_interval_s is None:
            self._timers[ids] = float("inf")
        else:
            self._timers[ids] = self._int_lo + (self._int_hi - self._int_lo) * torch.rand(
                n, device=self.device
            )

    def reset(self, env_ids=None) -> torch.Tensor:
        ids = self._resolve_ids(env_ids)
        self._sample(ids)
        return self.cmd

    def update(self, step_dt: float, cmd: torch.Tensor) -> torch.Tensor:
        if self.resample_interval_s is not None:
            self._timers -= step_dt
            due = torch.nonzero(self._timers <= 0.0).squeeze(-1)
            if due.numel() > 0:
                self._sample(due)
        return self.cmd
