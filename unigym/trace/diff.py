"""Trace diff engine (docs/contracts/trace_contract.md §4).

Pure numpy over two :class:`~unigym.trace.io.TraceReader` roots:
per-segment max|d|, per-tick BITWISE/eps/DIVERGED verdicts, first
divergence short-circuit, missing-segment N/A intersection semantics,
shape-mismatch guards.  No task-specific segment knowledge anywhere
(the segment registry is each trace's manifest).
"""

from __future__ import annotations

import numpy as np

from .io import TraceReader

BITWISE, EPS, DIVERGED, NA, SHAPE = "BITWISE", "eps", "DIVERGED", "N/A", "SHAPE"
TOP_K = 5


def _verdict(worst: float, tol: float) -> str:
    if worst == 0.0:
        return BITWISE
    return EPS if worst <= tol else DIVERGED


def diff_traces(
    a: TraceReader, b: TraceReader, *, tol: float = 1e-6, ticks: int | None = None
) -> dict:
    """Full diff document (machine shape; the CLI renders tables from it).

    ``segments`` rows are keyed by segment name; ``per_tick`` is a list of
    {tick, segments: {name: max|d| or None}, verdict}.  ``first`` is the
    first DIVERGED tick document or None.  Only ticks present in BOTH
    traces are compared (record-order index, contract §4).
    """
    common = sorted(set(a.indices) & set(b.indices))
    if ticks is not None:
        common = common[: max(0, int(ticks))]
    seg_a, seg_b = a.segments(), b.segments()
    only_a = sorted(set(seg_a) - set(seg_b) - {"tick"})
    only_b = sorted(set(seg_b) - set(seg_a) - {"tick"})

    rows: dict[str, dict] = {}
    per_tick: list[dict] = []
    first: dict | None = None
    for idx in common:
        ta, tb = a.load_tick(idx), b.load_tick(idx)
        names = sorted((set(ta) | set(tb)) - {"tick"})
        worst_name, worst_val = None, 0.0
        tick_row: dict = {
            "tick_index": idx,
            "policy_tick": _tick_scalar(tb),
            "segments": {},
            "verdict": BITWISE,
        }
        for name in names:
            va, vb = ta.get(name), tb.get(name)
            if va is None or vb is None:
                tick_row["segments"][name] = None  # N/A: missing side
                rows.setdefault(name, {"present": "one-side", "max": None})
                continue
            if tuple(va.shape) != tuple(vb.shape):
                tick_row["segments"][name] = SHAPE
                rows.setdefault(name, {"present": "both", "max": None})["shape_mismatch"] = True
                if tick_row["verdict"] == BITWISE:
                    tick_row["verdict"] = SHAPE
                continue
            d = (
                float(np.max(np.abs(va.astype(np.float64) - vb.astype(np.float64))))
                if va.size
                else 0.0
            )
            tick_row["segments"][name] = d
            row = rows.setdefault(name, {"present": "both", "max": 0.0})
            if not row.get("shape_mismatch"):
                row["max"] = max(row["max"] or 0.0, d)
            if d > worst_val:
                worst_name, worst_val = name, d
        v = _verdict(worst_val, tol)
        if tick_row["verdict"] != SHAPE:
            tick_row["verdict"] = v
        else:
            tick_row["verdict"] = SHAPE
        tick_row["worst"] = {"segment": worst_name, "max": worst_val}
        per_tick.append(tick_row)
        if first is None and v == DIVERGED:
            first = _first_doc(ta, tb, idx, _tick_scalar(tb), worst_name, worst_val, tol)
    return {
        "a": str(a.root),
        "b": str(b.root),
        "transport_a": a.transport,
        "transport_b": b.transport,
        "tol": tol,
        "ticks_compared": len(common),
        "only_in_a": only_a,
        "only_in_b": only_b,
        "segments": {k: v for k, v in sorted(rows.items())},
        "per_tick": per_tick,
        "first": first,
    }


def _tick_scalar(tick_arrays: dict) -> int | None:
    v = tick_arrays.get("tick")
    return int(v) if v is not None else None


def _first_doc(
    ta: dict,
    tb: dict,
    idx: int,
    policy_tick: int | None,
    worst_name: str | None,
    worst_val: float,
    tol: float,
) -> dict:
    """First-divergence document: segment table + top-k dims (§4)."""
    segs = {}
    for name in sorted((set(ta) & set(tb)) - {"tick"}):
        va, vb = ta.get(name), tb.get(name)
        if va is None or vb is None or tuple(va.shape) != tuple(vb.shape):
            continue
        d = np.abs(va.astype(np.float64) - vb.astype(np.float64))
        segs[name] = float(d.max()) if d.size else 0.0
    top = []
    if worst_name is not None:
        d = np.abs(ta[worst_name].astype(np.float64) - tb[worst_name].astype(np.float64)).reshape(
            -1
        )
        order = np.argsort(d)[::-1][:TOP_K]
        top = [{"dim": int(i), "d": float(d[i])} for i in order if d[i] > 0][:TOP_K]
    return {
        "tick_index": idx,
        "policy_tick": policy_tick,
        "segments": segs,
        "worst": {"segment": worst_name, "max": worst_val},
        "tol": tol,
        "top_dims": top,
    }


def by_segment(doc: dict) -> list[dict]:
    """``--by-segment`` rows: segment x all-ticks aggregation (§4)."""
    out = []
    for name, row in doc["segments"].items():
        if row.get("max") is None:
            verdict = NA
        elif row.get("shape_mismatch"):
            verdict = SHAPE
        else:
            verdict = _verdict(row["max"], doc["tol"])
        out.append({"segment": name, "max": row.get("max"), "verdict": verdict})
    return out
