"""Per-world MuJoCo Warp buffer sizing (v094 P3).

``mujoco_warp`` sizes its contact (``nconmax``/``naconmax``) and constraint
(``njmax``) buffers at data-construction time; overflow is reported from
inside the kernel (it cannot be caught).  The P0 probe measured the budget a
world actually uses on a CPU ``mj_step`` rollout, so this module sizes the
device buffers from that probe with a safety factor and exposes a retry
ladder for worlds that saturate later.

Measured baselines (P0, 100 CPU steps): the 45-dof tabletop worlds need
288/932, ``kitchen_layout009`` 768/2508.  The floor is 64 contacts / 128
constraints so a contact-free world still gets a usable buffer.

The retry ladder is deliberately engine-free (``build``/``saturated`` are
callables): the backend supplies the warp data pair, while tests can drive
the ladder with fakes.

Capacity is always **per world**: ``nconmax``/``njmax`` are passed to
``mjw.put_data(..., nworld=N)`` unchanged.  Warp's contact arena is the
batched ``naconmax``, which defaults to ``nconmax * nworld`` (and
``naccdmax = nccdmax * nworld``); the backend asserts the allocated
``naconmax`` covers ``N`` so a batch can never silently overflow.  Memory is
therefore N x the per-world budget, not N².
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

import mujoco

MIN_NCONMAX = 64
"""Floor for the per-world contact buffer."""

MIN_NJMAX = 128
"""Floor for the per-world constraint buffer."""

SAFETY_FACTOR = 4
"""CPU-probe budget multiplier (P0: no saturation at 100 steps)."""

RETRY_FACTOR = 2
"""Per-attempt buffer growth when a world saturates."""

MAX_ATTEMPTS = 4
"""Retry-ladder depth (64 -> 128 -> 256 -> 512 ... of the initial budget)."""

T = TypeVar("T")


@dataclass(frozen=True)
class Capacity:
    """One per-world ``(nconmax, njmax)`` pair, with the CPU probe facts."""

    nconmax: int
    njmax: int
    probe_ncon: int = 0
    probe_nefc: int = 0

    def doubled(self) -> Capacity:
        """The next rung of the retry ladder (``RETRY_FACTOR`` growth)."""
        return Capacity(self.nconmax * RETRY_FACTOR,
                        self.njmax * RETRY_FACTOR,
                        self.probe_ncon, self.probe_nefc)


def probe_peak(mjm, steps: int = 100) -> tuple[int, int]:
    """Run a CPU rollout and return ``max(ncon), max(nefc)`` over it.

    Args:
        mjm: The compiled ``mujoco.MjModel`` (the same model warp consumes).
        steps: Number of ``mj_step`` samples (at least one).

    Returns:
        The peak per-step contact and constraint counts.
    """
    data = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, data)
    ncon = nefc = 0
    for _ in range(max(1, int(steps))):
        mujoco.mj_step(mjm, data)
        ncon = max(ncon, int(data.ncon))
        nefc = max(nefc, int(data.nefc))
    return ncon, nefc


def capacity_for(mjm, steps: int = 100) -> Capacity:
    """Size one world from the CPU probe (``SAFETY_FACTOR``, floor applied)."""
    ncon, nefc = probe_peak(mjm, steps)
    return Capacity(max(MIN_NCONMAX, ncon * SAFETY_FACTOR),
                    max(MIN_NJMAX, nefc * SAFETY_FACTOR),
                    probe_ncon=ncon, probe_nefc=nefc)


def build_with_retry(
        build: Callable[[Capacity], T],
        capacity: Capacity,
        saturated: Callable[[T, Capacity], bool],
        max_attempts: int = MAX_ATTEMPTS,
) -> tuple[T, Capacity]:
    """Build a resource, doubling the capacity until it is not saturated.

    Args:
        build: ``capacity -> resource`` (the warp data pair in the backend).
        capacity: The first rung (normally :func:`capacity_for`).
        saturated: ``(resource, capacity) -> bool``; True means the buffers
            filled and the resource must be rebuilt with more room.
        max_attempts: Ladder depth before giving up.

    Returns:
        ``(resource, capacity)`` for the first non-saturated build.

    Raises:
        RuntimeError: Every attempt saturated; the message carries the last
            capacity so the caller can report the world's requirement.
    """
    cap = capacity
    for _ in range(max(1, int(max_attempts))):
        resource = build(cap)
        if not saturated(resource, cap):
            return resource, cap
        cap = cap.doubled()
    raise RuntimeError(
        "mjwarp capacity saturation persisted after "
        f"{max_attempts} attempts (last nconmax={cap.nconmax}, "
        f"njmax={cap.njmax}); the world needs a larger per-world buffer "
        "(increase SAFETY_FACTOR or pass an explicit Capacity)")


__all__ = [
    "Capacity",
    "MAX_ATTEMPTS",
    "MIN_NCONMAX",
    "MIN_NJMAX",
    "RETRY_FACTOR",
    "SAFETY_FACTOR",
    "build_with_retry",
    "capacity_for",
    "probe_peak",
]
