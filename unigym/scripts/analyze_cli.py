"""`unigym analyze` CLI (toolkit C): behavior report over existing shards.

Thin argparse-command layer over :mod:`unigym.analysis.behavior`
(contract: docs/contracts/analyze_contract.md).  Accepts a run OUTPUT
DIR directly or a run name resolved under the output root
(``data_out/<name>*``: exact hit else latest timestamped match).

``--series/--op`` (the L2 escape hatch -- generic ops over any
shard key), ``--metric`` (retained per-frame metric + predicate
threshold lines), ``--no-gate``.  Report rows are declaration-driven
(gate.py); the CLI carries zero task vocabulary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "data_out"


def resolve_run_dir(name_or_dir: str) -> Path:
    """Dir as-is; else `<output_root>/<name>` else latest `<name>_*`."""
    p = Path(name_or_dir)
    if p.is_dir():
        return p
    if p.is_absolute():
        raise FileNotFoundError(f"no run dir at {p}")
    exact = OUTPUT_ROOT / name_or_dir
    if exact.is_dir():
        return exact
    hits = sorted(OUTPUT_ROOT.glob(f"{p.name}*"))
    if hits:
        return hits[-1]              # latest timestamped variant
    raise FileNotFoundError(
        f"no run dir for {name_or_dir!r} under {OUTPUT_ROOT} "
        f"(pass the output dir explicitly)")


def _series_face(a: argparse.Namespace, run_dir: Path) -> int:
    """`--series KEY [--op T] [--axis N] [--ep N]`: the L2 escape hatch."""
    from unigym.analysis import behavior, gate
    try:
        loaded = behavior.load_run(run_dir, env=a.env, max_ticks=a.max_ticks,
                                   extra_keys={a.series})
    except behavior.AnalyzeError as e:
        print(f"[unigym analyze] FAILED: {e}\n  hint: {e.hint}",
              file=sys.stderr)
        return 1
    data = loaded["data"]
    if a.series not in data:
        avail = ", ".join(sorted(data))
        print(f"[unigym analyze] FAILED: key {a.series!r} not in shards\n"
              f"  hint: available keys: {avail}", file=sys.stderr)
        return 1
    x = data[a.series]
    tick_lo, tick_hi = 0, loaded["ticks"]
    ep_note = "all"
    if a.ep is not None:
        eps = behavior._episodes(data["episode_id"])
        if not 0 <= a.ep < len(eps):
            print(f"[unigym analyze] FAILED: --ep {a.ep} out of range "
                  f"(0..{len(eps) - 1})", file=sys.stderr)
            return 1
        tick_lo, tick_hi = eps[a.ep]["start"], eps[a.ep]["end"]
        x = x[tick_lo:tick_hi]
        ep_note = f"ep{a.ep} (ticks {tick_lo}..{tick_hi - 1})"
    if a.axis is not None:
        if a.axis >= x.shape[-1]:
            print(f"[unigym analyze] FAILED: --axis {a.axis} out of range "
                  f"(last axis size {x.shape[-1]})", file=sys.stderr)
            return 1
        x = x[..., a.axis]
    try:
        value, aux = gate.apply_op(x, a.op, a.value)
    except gate.GateError as e:
        print(f"[unigym analyze] FAILED: {e}\n  hint: {e.hint}",
              file=sys.stderr)
        return 1
    doc = {"ok": True, "run": str(run_dir), "series": a.series,
           "axis": a.axis, "op": a.op, "ep": a.ep, "value": value,
           "aux": aux}
    if a.json:
        print(json.dumps(doc))
        return 0
    shown = f"{value:.6g}" if value is not None else "n/a"
    if a.op.startswith("first_"):
        extra = f" (first crossing t={aux})"
    elif value is None and aux is not None:
        extra = f" (tick {aux})"
    else:
        extra = ""
    print(f"[unigym analyze] {run_dir.name}: {a.series}"
          f"[{f'axis={a.axis}' if a.axis is not None else 'all'}] "
          f"op={a.op} {ep_note} -> {shown}{extra}")
    return 0


def _metric_chart(a: argparse.Namespace, run_dir: Path, out: Path) -> int:
    """`--metric ID`: retained per-frame series + predicate threshold lines
    (back-queried from the declared predicate trees; analyze_contract §6)."""
    from unigym.analysis import gate
    from unigym.analysis.charts import line_chart
    frames = gate.metric_frames(run_dir, a.metric)
    if not frames:
        print(f"[unigym analyze] no retained frames for metric "
              f"{a.metric!r} (metrics.npz absent or metric not declared) "
              "-- benchmark.retain_metric_frames: true records them",
              file=sys.stderr)
        return 1
    try:
        meta = json.loads((run_dir / "meta.json").read_text())
    except (OSError, ValueError):
        meta = {}
    thresholds: list[dict] = []
    for _name, spec in gate.resolve_task_specs(meta.get("config")):
        thresholds += gate.thresholds_for(spec, a.metric)
    series = [(k, v) for k, v in sorted(frames.items())]
    hlines = [(t["value"], t["label"]) for t in thresholds]
    f = line_chart(out / f"metric__{a.metric.replace('/', '_')}.png",
                   f"gate metric: {a.metric} (thresholds from predicates)",
                   series, hlines=hlines)
    print(f"[unigym analyze] metric chart: {f} "
          f"(threshold lines: {len(hlines)})")
    return 0


def cmd_analyze(a: argparse.Namespace, extra: list[str]) -> int:
    from unigym.analysis.behavior import AnalyzeError, analyze_run
    try:
        run_dir = resolve_run_dir(a.run)
    except FileNotFoundError as e:
        if a.json:
            print(json.dumps({"ok": False, "error": f"{e}",
                              "hint": "pass a collect run dir "
                                      "(meta.json + shard_*.npz)"}))
        else:
            print(f"[unigym analyze] FAILED: {e}", file=sys.stderr)
        return 1
    if a.series:
        return _series_face(a, run_dir)
    try:
        doc = analyze_run(run_dir, out_dir=a.out, env=a.env,
                          max_ticks=a.max_ticks, charts=not a.no_charts,
                          gate=not a.no_gate)
    except (FileNotFoundError, AnalyzeError) as e:
        hint = getattr(e, "hint", "pass a collect run dir "
                                  "(meta.json + shard_*.npz)")
        if a.json:
            print(json.dumps({"ok": False, "error": f"{e}", "hint": hint}))
        else:
            print(f"[unigym analyze] FAILED: {e}\n  hint: {hint}",
                  file=sys.stderr)
        return 1
    if a.metric:
        out_dir = Path(a.out) if a.out else run_dir
        _metric_chart(a, run_dir, out_dir)
    if a.json:
        print(json.dumps({"ok": True, "report": doc}))
        return 0
    r = doc["run"]
    g = doc.get("gate") or {}
    head = (f"; gate: {g['summary']}" if g.get("available")
            else (f"; gate: {g.get('status')}" if g.get("status") else ""))
    print(f"[unigym analyze] {r['name'] or r['dir']}: "
          f"{r['ticks']} ticks, {len(doc['episodes'])} episodes{head}")
    for row in doc["episodes"][:8]:
        md = "; ".join(f"{k.split(' <-> ')[-1]}={v['min']:.3f}m"
                       for k, v in row["min_dist"].items())
        fc = "; ".join(f"{k.replace('contact/', '')}@t{t}"
                       for k, t in row["first_contact"].items()
                       if t is not None) or "none"
        print(f"  ep{row['ep']}: {row['verdict']}  "
              f"min_dist[{md or '-'}]  first_contact[{fc}]")
    print(f"  report: {Path(r['dir']) / 'behavior_report.md'} "
          f"(+ .json, {len(doc['charts'])} png)")
    return 0


def add_analyze_subparser(sub) -> None:
    """Wire `unigym analyze` into the launcher (launch.py)."""
    an = sub.add_parser("analyze", help="behavior report over existing "
                                        "run shards (zero collection; "
                                        "docs/contracts/analyze_contract.md)")
    an.add_argument("run", help="run output dir (or run name under "
                                "data_out; latest timestamped match wins)")
    an.add_argument("--out", default=None,
                    help="report directory (default: the run dir; flat "
                         "files so the portal artifact face serves them)")
    an.add_argument("--env", type=int, default=0,
                    help="env index for multi-env shards (default 0)")
    an.add_argument("--max-ticks", type=int, default=None,
                    help="cap ticks read (prefix analysis)")
    an.add_argument("--no-charts", action="store_true",
                    help="skip PNG curves")
    an.add_argument("--no-gate", action="store_true", dest="no_gate",
                    help="skip the declaration-driven gate section "
                         "default renders it when the run "
                         "carries results.json)")
    an.add_argument("--series", default=None,
                    help="L2 escape hatch: any shard key "
                         "(object/<n>/pos, contact/<g>, base_pos, "
                         "action_target*, extra/*); prints one "
                         "op-series line instead of the full report")
    an.add_argument("--op", default=None, dest="op",
                    help="series op (whitelist: max min abs_max abs_min "
                         "mean std argmax argmin norm_max norm_mean "
                         "rel_max rel_min rel_argmax rel_argmin "
                         "first_gt first_lt first_ge first_le; "
                         "default max)")
    an.add_argument("--axis", type=int, default=None,
                    help="series column selector (last axis, e.g. "
                         "--axis 2 = z of object/<n>/pos)")
    an.add_argument("--value", type=float, default=None,
                    help="threshold for first_gt/first_lt/first_ge/"
                         "first_le")
    an.add_argument("--ep", type=int, default=None,
                    help="restrict the series to episode N")
    an.add_argument("--metric", default=None, dest="metric",
                    help="draw a retained gate metric's per-frame series "
                         "(metrics.npz) + predicate threshold lines")
    an.add_argument("--json", action="store_true",
                    help="single JSON document on stdout")
    an.set_defaults(fn=cmd_analyze)
