"""Step-clock capability double for the runtime metrics writer.

``Bridge`` receives the engine's step clock at construction; ``MetricsWriter``
reads only ``step_dt`` (and, tolerantly, ``decimation``).  This double carries
exactly that capability so metrics tests can choose the period without starting
a simulator.  It holds no physics, observation or control state.
"""

from __future__ import annotations


class StepClockEnv:
    """Minimal step-clock capability (``step_dt`` + optional ``decimation``)."""

    def __init__(self, *, step_dt: float = 0.02, decimation: int | None = 4) -> None:
        if isinstance(step_dt, bool) or not isinstance(step_dt, (int, float)) or step_dt <= 0:
            raise ValueError(f"step_dt must be a positive number, got {step_dt!r}")
        self.step_dt = float(step_dt)
        if decimation is not None:
            if isinstance(decimation, bool) or not isinstance(decimation, int) or decimation <= 0:
                raise ValueError(f"decimation must be a positive int, got {decimation!r}")
            self.decimation = decimation


__all__ = ["StepClockEnv"]
