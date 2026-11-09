"""Gate metrics + series face.

"What gets analyzed" is decided by declarations, not hardcoded:

* **L0 platform invariants** stay in behavior.py (tick/episode segmentation, finite
  checks, fall axis);
* **L1 run declarations** (THIS module's gate face): rows = the benchmark section's
  declared metrics + predicate-tree leaves, resolved through the SAME pack
  index the engine used (inline formula specs pass through; ``<pack>/<task>``
  references resolve via benchmarks.packs).  Values come from the engine's
  own ``results.json`` -- zero re-computation, zero task vocabulary here
  (a pack task run renders its own predicate set with zero analyze
  changes);
* **L2 call-site expressions** (``--series KEY --op TEMPLATE``): generic series ops
  over any shard key -- the ~20 inline-numpy campaign moves, first-classed.
  Task-specific quantities live in declarations and call sites, never in
  code.

Pure numpy/stdlib; zero sim/daemon imports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS_JSON = "results.json"
METRICS_NPZ = "metrics.npz"

# leaf-label vocabulary (formula comparators, benchmark_contract §Formula)
_LABEL_OP = {"gte": ">=", "lte": "<=", "gt": ">", "lt": "<", "eq": "=="}

# L2 series ops (whitelist; dtype-preserving like the campaign's inline
# numpy: the series stays as-loaded, only the final scalar is float())
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


class GateError(ValueError):
    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint or "(none)"


# --------------------------------------------------------------------------- #
# L1: results.json + declaration-driven gate doc
# --------------------------------------------------------------------------- #
def load_results(run_dir: str | Path) -> dict | None:
    """results.json face: dict when present, None when absent; corrupt
    files raise GateError (the renderer degrades to `na`, it never guesses)."""
    p = Path(run_dir) / RESULTS_JSON
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text())
    except ValueError as e:
        raise GateError(
            f"{p}: unreadable ({e})",
            hint="results.json is engine-written; a partial " "file means the run died mid-write",
        ) from e
    if not isinstance(doc, dict) or doc.get("schema") != "results.v1":
        raise GateError(
            f"{p}: schema {doc.get('schema')!r} != results.v1"
            if isinstance(doc, dict)
            else f"{p}: not a mapping"
        )
    return doc


def resolve_task_specs(config: dict | None) -> list[tuple[str, dict | None]]:
    """(task_name, formula_spec|None) per declared benchmark task.

    Inline formula sections pass through verbatim; ``<pack>/<task>`` names
    resolve through the SAME pack index the engine used (build_tasks).
    Python-registry tasks resolve to None (no declarative rows).
    """
    bench = (config or {}).get("benchmark") or {}
    out: list[tuple[str, dict | None]] = []
    sections = bench.get("tasks") or []
    if sections and all(isinstance(s, str) for s in sections):
        sections = [{"name": s} for s in sections]
    for section in sections:
        if not isinstance(section, dict):
            continue
        name = str(section.get("name") or "formula")
        if "metrics" in section:  # inline formula spec
            out.append((name, section))
            continue
        try:
            from unigym.benchmarks.packs import discover_packs

            index = discover_packs(bench.get("packs"))
            def_, _pack = index.resolve_task(name)
            spec = def_.spec if getattr(def_, "kind", "") == "formula" else None
        except Exception:  # noqa: BLE001
            spec = None
        out.append((name, spec))
    return out


def _leaf_rows(spec: dict | None) -> list[dict]:
    """Declared predicate leaves -> rows (id/label from the declaration)."""
    if not spec or not isinstance(spec.get("success"), dict):
        return []
    from unigym.benchmarks.formula import leaf_keys

    rows = []
    for key, (_path, node) in leaf_keys(spec["success"]).items():
        mid = str(node.get("metric"))
        for op, sym in _LABEL_OP.items():
            if op in node:
                rows.append(
                    {
                        "kind": "predicate",
                        "id": key,
                        "metric": mid,
                        "label": f"{mid}{sym}{node[op]}",
                        "op": op,
                        "value": node[op],
                    }
                )
                break
        else:
            lo, hi = node.get("between", (None, None))
            rows.append(
                {
                    "kind": "predicate",
                    "id": key,
                    "metric": mid,
                    "label": f"{mid}in[{lo},{hi}]",
                    "op": "between",
                    "value": [lo, hi],
                }
            )
    return rows


def gate_doc(run_dir: str | Path, config: dict | None = None) -> dict:
    """The gate section: rows = config declarations, values = results.json.

    Degradation is honest: no results.json -> ``available=False`` with
    reason ``na`` (missing results); corrupt -> ``na`` (corrupt results).
    """
    meta = {}
    try:
        meta = json.loads((Path(run_dir) / "meta.json").read_text())
    except (OSError, ValueError):
        pass
    cfg = config if config is not None else meta.get("config")
    doc: dict = {
        "available": False,
        "status": "ok",
        "columns": [],
        "rows": [],
        "episodes": [],
        "summary": None,
        "gate": None,
    }
    try:
        results = load_results(run_dir)
    except GateError as e:
        doc["status"] = f"na(results 残缺: {e})"
        return doc
    if results is None:
        doc["status"] = (
            "na(results 缺失: no results.json or scored 0 "
            "episodes -- gate rows need a benchmark run)"
        )
        return doc
    eps = results.get("episodes") or []
    doc["available"] = True
    doc["episodes"] = eps
    doc["columns"] = [f"e{ep.get('env', i)}/g={ep.get('gate')}" for i, ep in enumerate(eps)]
    gates = [str(ep.get("gate")) for ep in eps]
    doc["gate"] = gates[0] if gates else None
    n_ok = sum(1 for ep in eps if ep.get("verdict") == "success")
    doc["summary"] = (
        f"n={len(eps)} success={n_ok}/{len(eps)}" f" gate={doc['gate']}" if eps else "n=0"
    )
    specs = dict(resolve_task_specs(cfg))
    rows: list[dict] = []
    for task_name, spec in specs.items():
        for m in (spec or {}).get("metrics") or []:
            if isinstance(m, dict) and m.get("id"):
                rows.append(
                    {
                        "kind": "metric",
                        "id": str(m["id"]),
                        "label": f"{m['id']}({m.get('term')})",
                        "task": task_name,
                    }
                )
        for leaf in _leaf_rows(spec):
            leaf["task"] = task_name
            rows.append(leaf)
    for row in rows:
        vals = []
        for ep in eps:
            if str(ep.get("gate")) != row["task"]:
                vals.append(None)
                continue
            src = ep.get("metrics") if row["kind"] == "metric" else ep.get("predicates")
            v = (src or {}).get(row["id"])
            vals.append(v if isinstance(v, (int, float, bool)) else None)
        row["values"] = vals
    doc["rows"] = rows
    return doc


def render_gate_md(doc: dict) -> list[str]:
    """The md face of the gate section (rows x episodes table)."""
    if not doc.get("available"):
        return ["", "## gate", f"status: {doc['status']}"]
    L = ["", "## gate", f"{doc['summary']}  ({doc['status']})"]
    header = "row | " + " | ".join(doc["columns"]) or "row"
    L += [header, "---|" + "---|" * max(0, len(doc["columns"]) - 1)]
    for row in doc["rows"]:
        cells = []
        for v in row["values"]:
            if v is None:
                cells.append("-")
            elif isinstance(v, bool):
                cells.append("T" if v else "F")
            else:
                cells.append(f"{v:.4g}")
        L.append(f"{row['label']} | " + " | ".join(cells))
    return L


# --------------------------------------------------------------------------- #
# L2: --series escape hatch + threshold back-query (--metric)
# --------------------------------------------------------------------------- #
def apply_op(x: np.ndarray, op: str, value: float | None = None):
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
        raise GateError(f"unknown --op {op!r} (whitelist: {', '.join(SERIES_OPS)})")
    if x.ndim == 0 or x.shape[0] == 0:
        raise GateError("empty series (T=0)")
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
            raise GateError(f"--op {op} requires --value THRESHOLD")
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


def thresholds_for(spec: dict | None, metric_id: str) -> list[dict]:
    """Predicate-tree back-query: every comparator constant referencing
    ``metric_id`` -> --metric chart threshold lines (e.g. dz>=0.09)."""
    out: list[dict] = []
    tree = (spec or {}).get("success")
    if not isinstance(tree, dict):
        return out
    from unigym.benchmarks.formula import leaf_keys

    for _key, (_path, node) in leaf_keys(tree).items():
        if str(node.get("metric")) != metric_id:
            continue
        for op, sym in _LABEL_OP.items():
            if op in node:
                out.append(
                    {"value": float(node[op]), "label": f"{metric_id} {sym} {node[op]}", "op": op}
                )
                break
        else:
            if "between" in node:
                lo, hi = node["between"]
                out.append({"value": float(lo), "label": f"{metric_id} >= {lo}"})
                out.append({"value": float(hi), "label": f"{metric_id} <= {hi}"})
    return out


def metric_frames(run_dir: str | Path, metric_id: str) -> dict[str, np.ndarray]:
    """Retained per-frame series for one metric: ``{e<i>: (T,)}`` from
    metrics.npz (empty when the run did not retain)."""
    p = Path(run_dir) / METRICS_NPZ
    if not p.is_file():
        return {}
    out: dict[str, np.ndarray] = {}
    with np.load(p) as z:
        for key in z.files:
            parts = key.split("/")
            if len(parts) == 3 and parts[0] == "metric" and parts[2] == metric_id:
                out[parts[1]] = np.asarray(z[key], dtype=np.float64)
    return out
