"""daemon.media_telemetry -- frame-path telemetry counters .

publish/consumed rates + dropped-frame (gap) accounting for the media
gateway's shared :class:`~unigym.daemon.media.RunFrames` reader.  Extracted from media.py --
which is deliberately self-contained but already at scale -- the same
spirit as webrtc.py: pure bookkeeping over (fid, t) observations, no
I/O and no SHM access of its own, fully unit-testable with fake fid
sequences.

Semantics (design §3):
* ``publish_rate`` -- fid velocity over a sliding window (fid delta /
  wall delta; design §3 offers sliding-window or EMA -- the window is unbiased
  under the consumers' uneven sampling cadence).  With too few points
  in the window the last computed velocity is held (a sparse status
  poller must not see the number flicker to 0).
* ``consumed_rate`` -- distinct new-fid observations per second over
  the same window (the reader-side demand; bounded by the consumers'
  sampling cadence).
* ``gap_total`` / ``last_gap`` / ``last_gap_ts`` -- Σ(fid jump − 1):
  frames the reader never saw (published then overwritten between two
  observations).  The baseline resets whenever the reader is
  (re)created -- a daemon restart or a fresh attach must never
  fabricate a gap (design §3 resilience gate).
"""

from __future__ import annotations

import time
from collections import deque

__all__ = ["FrameTelemetry", "WINDOW_S"]

WINDOW_S = 2.0  # velocity window (design §3: sliding window)
MIN_POINTS = 2  # window needs >= 2 points for a velocity


class FrameTelemetry:
    """fid-observation bookkeeping for one RunFrames reader."""

    def __init__(self, window: float = WINDOW_S):
        self.window = float(window)
        self.publish_rate = 0.0
        self.consumed_rate = 0.0
        self.gap_total = 0
        self.last_gap = 0
        self.last_gap_ts = 0.0
        self._prev_fid = -1  # -1 = baseline unset (no fake gap)
        self._prev_t = 0.0
        self._points: deque = deque()  # (t, fid) new-fid observations
        self._held = (0.0, 0.0)  # last computed (publish, consumed)

    def observe(self, fid: int, now: float | None = None) -> int:
        """Record one new-fid observation; returns the observed gap.

        Callers must serialise observations (RunFrames does, under its
        fid lock) so each new fid is counted exactly once.
        """
        t = time.time() if now is None else now
        gap = 0
        if self._prev_fid >= 0:
            gap = max(0, fid - self._prev_fid - 1)  # Σ(fid jump − 1)
            self.gap_total += gap
            if gap:
                self.last_gap = gap
                self.last_gap_ts = t
        self._prev_fid = fid
        self._prev_t = t
        self._points.append((t, fid))
        while len(self._points) > MIN_POINTS and t - self._points[0][0] > self.window:
            self._points.popleft()
        return gap

    def _velocity(self) -> tuple[float, float] | None:
        """(publish_rate, consumed_rate) over the window, or None."""
        if len(self._points) < MIN_POINTS:
            return None
        (t0, f0), (t1, f1) = self._points[0], self._points[-1]
        if t1 <= t0:
            return None
        n = len(self._points)
        return (f1 - f0) / (t1 - t0), (n - 1) / (t1 - t0)

    def snapshot(self) -> dict:
        """The API-face counters (publish/consumed rates + gap fields).

        last_fid/stale are reader state and are attached by
        RunFrames.media, design §3.
        """
        v = self._velocity()
        if v is not None:
            self._held = v
        return {
            "publish_rate": round(self._held[0], 2),
            "consumed_rate": round(self._held[1], 2),
            "gap_total": self.gap_total,
            "last_gap": self.last_gap,
            "last_gap_ts": self.last_gap_ts,
        }
