"""Episode duration spec parsing / sampling (torch-only, unit-testable).

``episode_length_s`` yaml field accepts:
  * scalar (number)  -> fixed duration for every env every episode (the P1 shorthand)
  * [lo, hi]         -> per-env per-episode uniform sample in [lo, hi]
  * null             -> no core episode clock; only valid with
    ``termination_mode: external``.  Backends handle null before calling
    :func:`parse_episode_length`, so this parser stays numeric-only.
"""

from __future__ import annotations

import numbers

import torch


def parse_episode_length(spec) -> tuple[float, float]:
    """Normalize the yaml field to a (lo, hi) tuple with validation.

    ``None`` is the external-mode "no core clock" spelling; it is rejected
    here because backends must branch on it before any duration is sampled.
    """
    if spec is None:
        raise ValueError(
            "episode_length_s: null disables the core episode clock and is "
            "only valid with termination_mode: external; the backend handles "
            "it without sampling a duration"
        )
    if isinstance(spec, numbers.Real):
        lo = hi = float(spec)
    elif isinstance(spec, (list, tuple)) and len(spec) == 2:
        lo, hi = float(spec[0]), float(spec[1])
    else:
        raise ValueError(f"episode_length_s must be a scalar or [lo, hi], got {spec!r}")
    if not (0.0 < lo <= hi):
        raise ValueError(f"episode_length_s requires 0 < lo <= hi, got ({lo}, {hi})")
    return lo, hi


def sample_durations(lo: float, hi: float, n: int, device) -> torch.Tensor:
    """Draw n per-env episode durations, uniform in [lo, hi]."""
    if lo == hi:
        return torch.full((n,), lo, dtype=torch.float32, device=device)
    return lo + (hi - lo) * torch.rand(n, device=device)
