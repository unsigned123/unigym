"""Episode timeline / deadline / run identity helpers (benchmark contract).

Canonical integer tick axis shared by recorders, benchmark sinks and parity
probes::

    reset sample:          tick=0, time=0
    first post-step frame: tick=1, time=policy_dt
    recorded frame i:      tick=i+1, time=(i+1)*policy_dt

The deadline has exactly ONE authority: an integer ``timeout_steps``.
``timeout_s`` is only a declaration convenience resolved with floor
semantics (``floor(timeout_s / policy_dt + eps)``); declaring both requires
agreement.

The run identity types are core-neutral facts owned by
:mod:`unigym.core.identity`; this module re-exports them for the historical
import path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from unigym._typing import JsonValue
from unigym.core.identity import (
    RUN_IDENTITY_SCHEMA_VERSION,
    RobotIdentity,
    RunIdentity,
)

__all__ = [
    "EpisodeLimits",
    "RobotIdentity",
    "RunIdentity",
    "RUN_IDENTITY_SCHEMA_VERSION",
    "EPS_STEP",
    "step_index_for_frame",
    "time_for_frame",
    "timeline_step_index",
    "timeline_time_s",
]

#: Tolerance used by the floor division (guards 19.999999999 floor noise).
EPS_STEP = 1e-9


@dataclass(frozen=True)
class EpisodeLimits:
    """Declared episode deadline (one integer authority).

    Exactly one of ``timeout_s`` / ``timeout_steps`` may be authoritative.
    When both are supplied they must agree under floor semantics.  A value of
    ``None`` for both means "no task deadline" (the core clock, when enabled,
    remains the only horizon).
    """

    timeout_s: float | None = None
    timeout_steps: int | None = None

    def __post_init__(self) -> None:
        if self.timeout_s is not None:
            value = float(self.timeout_s)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"timeout_s must be finite and > 0, got {value!r}")
            object.__setattr__(self, "timeout_s", value)
        if self.timeout_steps is not None:
            if not isinstance(self.timeout_steps, int) or self.timeout_steps < 1:
                raise ValueError(f"timeout_steps must be an int >= 1, got {self.timeout_steps!r}")

    def validate_against(self, policy_dt: float) -> int | None:
        """Resolve/validate the deadline for *policy_dt*; returns steps.

        ``None`` means no deadline.  When both fields are declared they must
        resolve to the same integer; otherwise fail loud (two deadlines are
        an ambiguity, not a preference).
        """
        dt = float(policy_dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"policy_dt must be finite and > 0, got {policy_dt!r}")
        from_s = (
            None
            if self.timeout_s is None
            else int(math.floor(float(self.timeout_s) / dt + EPS_STEP))
        )
        if self.timeout_s is not None and from_s is not None and from_s < 1:
            raise ValueError(
                f"timeout_s={self.timeout_s!r} resolves to {from_s} steps at "
                f"policy_dt={dt!r}; the deadline must be at least one step"
            )
        if self.timeout_steps is not None and from_s is not None:
            if from_s != self.timeout_steps:
                raise ValueError(
                    f"conflicting episode deadline: timeout_s={self.timeout_s!r} "
                    f"resolves to {from_s} steps but timeout_steps="
                    f"{self.timeout_steps}; declare exactly one authority or "
                    "make them agree"
                )
        return self.timeout_steps if self.timeout_steps is not None else from_s

    def to_json(self) -> dict[str, JsonValue]:
        """JSON-safe declaration document."""
        return {"timeout_s": self.timeout_s, "timeout_steps": self.timeout_steps}


# --------------------------------------------------------------------------- #
# canonical timeline mapping
# --------------------------------------------------------------------------- #
def step_index_for_frame(frame: int) -> int:
    """0-based recorded frame -> canonical tick (frame 0 is tick 1)."""
    if not isinstance(frame, int) or frame < 0:
        raise ValueError(f"frame must be a non-negative int, got {frame!r}")
    return int(frame) + 1


def time_for_frame(frame: int, policy_dt: float) -> float:
    """0-based recorded frame -> canonical time ``(frame + 1) * policy_dt``."""
    dt = float(policy_dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"policy_dt must be finite and > 0, got {policy_dt!r}")
    return float(step_index_for_frame(frame)) * dt


def timeline_step_index(length: int) -> np.ndarray:
    """``int64`` tick array ``1..length`` for one recorded episode."""
    count = int(length)
    if count < 0:
        raise ValueError(f"length must be >= 0, got {length!r}")
    return np.arange(1, count + 1, dtype=np.int64)


def timeline_time_s(length: int, policy_dt: float) -> np.ndarray:
    """Authoritative per-frame times ``step_index * policy_dt``."""
    indices = timeline_step_index(length)
    dt = float(policy_dt)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"policy_dt must be finite and > 0, got {policy_dt!r}")
    return indices.astype(np.float64) * dt
