"""Generic series operations over a recorded shard key.

The L2 call-site escape hatch (``--series KEY --op TEMPLATE``): dtype-preserving
reductions over any shard key, independent of any task or evaluation vocabulary.
Pure numpy/stdlib; zero sim/daemon imports.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SERIES_OPS", "SeriesError", "apply_op"]

# dtype-preserving like the campaign's inline numpy: the series stays
# as-loaded, only the final scalar is float()
SERIES_OPS = (
    "max",
    "min",
    "abs_max",
    "abs_min",
    "mean",
    "std",
    "argmax",
    "argmin",
    "norm_max",
    "norm_mean",
    "rel_max",
    "rel_min",
    "rel_argmax",
    "rel_argmin",
    "first_gt",
    "first_lt",
    "first_ge",
    "first_le",
)


class SeriesError(ValueError):
    """A series selector or operation is invalid."""

    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint or "(none)"


def apply_op(x: np.ndarray, op: str, value: float | None = None) -> tuple[float | None, int | None]:
    """One whitelist op over a (T, [...]) series; returns (value, aux).

    dtype-preserving (the campaign's inline-numpy convention): float32
    shards stay float32 through the op -- only the final scalar converts.
    Semantics: ``rel_*`` subtracts the first frame (x - x[0]) before the
    reduction; ``norm_*`` is the per-frame L2 over the remaining axes;
    multi-column inputs reduce per frame (max for arg/first ops, full
    flatten for value ops); ``first_*`` returns the first tick crossing
    ``value`` (aux; value=None when unreachable).
    """
    if op not in SERIES_OPS:
        raise SeriesError(f"unknown --op {op!r} (whitelist: {', '.join(SERIES_OPS)})")
    if x.ndim == 0 or x.shape[0] == 0:
        raise SeriesError("empty series (T=0)")
    if op.startswith("rel_"):
        x = x - x[0]  # broadcast over trailing axes
    if op.startswith("norm_"):
        s = np.linalg.norm(x.reshape(x.shape[0], -1).astype(np.float64), axis=1)
        return (float(s.max()) if op == "norm_max" else float(s.mean()), None)
    if op in ("argmax", "rel_argmax"):
        s = x if x.ndim == 1 else x.reshape(x.shape[0], -1).max(axis=1)
        return None, int(np.argmax(s))
    if op in ("argmin", "rel_argmin"):
        s = x if x.ndim == 1 else x.reshape(x.shape[0], -1).min(axis=1)
        return None, int(np.argmin(s))
    if op.startswith("first_"):
        if value is None:
            raise SeriesError(f"--op {op} requires --value THRESHOLD")
        s = x if x.ndim == 1 else x.reshape(x.shape[0], -1).max(axis=1)
        cmp = {
            "first_gt": np.greater,
            "first_lt": np.less,
            "first_ge": np.greater_equal,
            "first_le": np.less_equal,
        }[op]
        hits = np.nonzero(cmp(s, value))[0]
        return None, (int(hits[0]) if hits.size else None)
    series = x if x.ndim == 1 else x.reshape(x.shape[0], -1)
    table = {
        "max": np.max,
        "min": np.min,
        "mean": np.mean,
        "std": np.std,
        "rel_max": np.max,
        "rel_min": np.min,
    }
    if op in ("abs_max", "abs_min"):
        return float((np.abs(series).max() if op == "abs_max" else np.abs(series).min())), None
    return float(table[op](series)), None
