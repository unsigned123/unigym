"""Streaming Latin Hypercube Sampling (LHS) command sampler.

Classic LHS draws an N x dim matrix at once (each column is a permutation of
strata). For open-ended rollout collection we need a *streaming* variant:
per (env, dim) we keep a "deck" of strata -- a shuffled permutation of
``0..num_strata-1`` -- and every draw pops the next stratum plus a uniform
jitter inside it. When the deck is exhausted it is reshuffled. This preserves
the LHS guarantee over any window of ``num_strata`` consecutive draws (each
stratum hit exactly once) while never blocking on a batch boundary.

Long-run marginal properties vs plain uniform sampling:
  * every dimension is stratified into ``num_strata`` layers -> no random
    clustering gaps larger than ``1/num_strata`` of the range;
  * each env keeps its own deck per dimension, so envs are independent.
"""

from __future__ import annotations

import torch

from .base import CommandSampler


class LHSSampler(CommandSampler):
    """Streaming Latin-hypercube sampler with per-env resample timers.

    Args:
        dim: command dimension.
        ranges: list of [lo, hi] per dim (length == dim).
        resample_interval_s: (lo, hi) seconds drawn uniformly per env per
            resample, or None to sample once and keep forever.
        num_strata: strata per dimension (deck size). Default 20.
    """

    def __init__(
        self,
        dim: int,
        device,
        num_envs: int = 1,
        ranges: list | None = None,
        resample_interval_s: tuple | None = None,
        num_strata: int = 20,
    ):
        super().__init__(dim, device, num_envs)
        if ranges is None or len(ranges) != dim:
            raise ValueError(f"LHSSampler: need {dim} [lo,hi] ranges, got {ranges!r}")
        if num_strata < 1:
            raise ValueError(f"LHSSampler: num_strata must be >= 1, got {num_strata}")
        ranges_t = torch.tensor(ranges, dtype=torch.float32, device=device)
        self._lo = ranges_t[:, 0]
        self._hi = ranges_t[:, 1]
        self.num_strata = int(num_strata)
        self.resample_interval_s = resample_interval_s
        if resample_interval_s is not None:
            self._int_lo = float(resample_interval_s[0])
            self._int_hi = float(resample_interval_s[1])
        else:
            self._int_lo, self._int_hi = 0.0, 0.0
        self._timers = torch.zeros(num_envs, device=device)

        # strata decks: (num_envs, dim, num_strata), each row a permutation
        self._decks = torch.argsort(
            torch.rand(num_envs, dim, self.num_strata, device=device), dim=-1
        )
        # next card index per (env, dim)
        self._ptr = torch.zeros(num_envs, dim, dtype=torch.int64, device=device)

    def _sample(self, ids: torch.Tensor) -> None:
        n = ids.shape[0]
        # pop next card from each (env, dim) deck
        cards = self._decks[ids].gather(2, self._ptr[ids].unsqueeze(-1)).squeeze(-1)  # (n, dim)
        u = torch.rand(n, self.dim, device=self.device)
        x01 = (cards.to(torch.float32) + u) / self.num_strata
        self.cmd[ids] = self._lo + (self._hi - self._lo) * x01
        # advance pointers; reshuffle exhausted decks
        new_ptr = self._ptr[ids] + 1
        exhausted = new_ptr >= self.num_strata
        self._ptr[ids] = torch.where(exhausted, torch.zeros_like(new_ptr), new_ptr)
        fresh = torch.argsort(torch.rand(n, self.dim, self.num_strata, device=self.device), dim=-1)
        self._decks[ids] = torch.where(exhausted.unsqueeze(-1), fresh, self._decks[ids])
        # resample timers (same policy as UniformSampler)
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
