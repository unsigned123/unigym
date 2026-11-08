"""Sliding-window + full-run statistics aggregator (numpy-only, unit-testable).

Fed per-step scalar metrics; produces window means (for live progress prints),
overall means, and p5/p50/p95 distributions for key quantities.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class StatsCollector:
    """Sliding-window + full-run statistics aggregator.

    Parameters
    ----------
    window_size
        Number of recent steps for the sliding-window mean.
    dist_keys
        Keys for which full per-step values are retained so that
        percentile distributions can be computed at the end.
    """

    def __init__(self, window_size: int = 500,
                 dist_keys: tuple[str, ...] = ("vx", "base_z", "alive")):
        self._window_size = window_size
        self._dist_keys = set(dist_keys)
        self._window: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size))
        self._running_sum: dict[str, float] = defaultdict(float)
        self._running_count: dict[str, int] = defaultdict(int)
        self._dist_data: dict[str, list[float]] = defaultdict(list)

    def update(self, metrics: dict) -> None:
        """Record one step's metrics (``{key: scalar_value}``)."""
        for k, v in metrics.items():
            v = float(v)
            self._window[k].append(v)
            self._running_sum[k] += v
            self._running_count[k] += 1
            if k in self._dist_keys:
                self._dist_data[k].append(v)

    @property
    def window_means(self) -> dict[str, float]:
        """Mean of the most recent *window_size* steps per key."""
        return {k: sum(v) / len(v) for k, v in self._window.items()
                if len(v) > 0}

    @property
    def overall_means(self) -> dict[str, float]:
        """Mean over all recorded steps per key."""
        return {k: self._running_sum[k] / self._running_count[k]
                for k in self._running_count if self._running_count[k] > 0}

    @property
    def distributions(self) -> dict[str, dict]:
        """p5/p50/p95/mean/min/max for keys in *dist_keys*."""
        result: dict[str, dict] = {}
        for k in self._dist_keys:
            data = self._dist_data.get(k)
            if not data:
                continue
            arr = np.asarray(data, dtype=np.float64)
            result[k] = {
                "p5": float(np.percentile(arr, 5)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "mean": float(arr.mean()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "count": int(arr.size),
            }
        return result

    def raw_values(self, key: str) -> np.ndarray:
        """All stored values for *key* (empty array if not a dist_key or no data)."""
        return np.asarray(self._dist_data.get(key, []), dtype=np.float64)

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._window.clear()
        self._running_sum.clear()
        self._running_count.clear()
        self._dist_data.clear()
