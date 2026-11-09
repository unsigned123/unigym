"""Fall termination: low base height OR excessive body tilt (robot-agnostic).

Pure-height
thresholds are geometry-dependent (a lying robot's base_z depends on body
geometry and size); body tilt from projected
gravity is geometry-independent: upright is ``gravity_b ≈ [0, 0, -1]`` for
ANY robot, and a fallen robot (side/back/face down) has a body z-axis
tilted > ~60 deg from vertical.  The combined rule

    fallen = (base_z < height_threshold) OR (-gravity_b_z < cos(tilt_threshold))

with defaults 0.15 m / 0.8 rad (~46 deg) separates normal locomotion
(< ~30 deg tilt) from fallen states (> ~60 deg) with margin.

A term judges exactly one articulation; ``articulation`` is a
required param and the state is read from ``state.articulations[name]``.
"""

from __future__ import annotations

import math

from .articulation import ArticulationTermination
from .base import TerminationState, body_tilt_exceeded

DEFAULT_HEIGHT_THRESHOLD = 0.15  # m below which the height term fires
DEFAULT_TILT_THRESHOLD = 0.8  # rad (~46 deg) body tilt from vertical
COS_DEFAULT_TILT = math.cos(DEFAULT_TILT_THRESHOLD)


class FallTermination(ArticulationTermination):
    """Fire when the base is too low OR the body is tilted past a threshold.

    params:
        articulation: REQUIRED articulation name; the term reads
            ``articulation/<name>/...`` from the state (no flat
            fallback).
        height_threshold: base-z below which the term fires (default 0.15).
        tilt_threshold: body-tilt threshold in rad (default 0.8);
            ``null`` disables the tilt term (height-only rule).
            Skipped automatically when the state carries no gravity data.
    """

    def __init__(self, params: dict | None = None, name: str | None = None):
        super().__init__(params, name)
        self.height_threshold = float(self.params.get("height_threshold", DEFAULT_HEIGHT_THRESHOLD))
        tilt = self.params.get("tilt_threshold", DEFAULT_TILT_THRESHOLD)
        self.tilt_threshold = None if tilt is None else float(tilt)
        self._cos_tilt = None if self.tilt_threshold is None else math.cos(self.tilt_threshold)

    def check(self, state: TerminationState):
        source = self.source(state)
        base_pos_w = source.get("base_pos_w")
        gravity_b = source.get("gravity_b")
        if base_pos_w is None:
            raise ValueError(
                f"FallTermination requires {self.articulation!r} base_pos_w " f"in state"
            )
        fallen = base_pos_w[..., 2] < self.height_threshold
        if self._cos_tilt is not None and gravity_b is not None:
            fallen = fallen | body_tilt_exceeded(gravity_b, self._cos_tilt)
        return fallen
