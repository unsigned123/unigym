"""`unigym trace` CLI: record / diff / import-shard / diverge (debug toolkit A).

Thin argparse-command layer over :mod:`unigym.trace` (contract:
docs/contracts/trace_contract.md).  stdout shapes are the CLI contract;
``--json`` swaps them for a single JSON document (repo discipline).
"""
from __future__ import annotations

import argparse
import json
import sys

from unigym.trace import TraceReader
from unigym.trace import diff as tdiff
from unigym.trace import import_shard as timport_shard
from unigym.trace import record as trecord


def _fmt(x) -> str:
    if x is None:
        return "N/A"
    return f"{x:.3e}"


# --------------------------------------------------------------------------- #
# record
# --------------------------------------------------------------------------- #
def cmd_record(a: argparse.Namespace, extra: list[str]) -> int:
    import time as _time
    ops = trecord.DaemonOps(timeout=a.timeout)
    try:
        doc = trecord.record_trace(
            ops, run=a.run, out=a.out,
            name=a.name or _time.strftime("%Y%m%dT%H%M%SZ",
                                          _time.gmtime()),
            ticks=a.ticks, every=a.every, keep_paused=a.keep_paused)
    except Exception as e:                         # noqa: BLE001 (CLI face)
        if a.json:
            print(json.dumps({"ok": False, "error": f"{e}"}))
        else:
            print(f"[unigym trace record] FAILED: {e}", file=sys.stderr)
        return 1
    if a.json:
        print(json.dumps({"ok": True, **doc}))
        return 0
    print(f"[unigym trace record] {doc['recorded']} tick shard(s) "
          f"(every={doc['every']}) -> {doc['trace_dir']}")
    print(f"  policy ticks {doc['first_tick']}..{doc['last_tick']} "
          f"segments={doc['segments']} "
          f"step_overshoot={doc['step_overshoot']}")
    print(f"  manifest: {doc['manifest']}")
    if not doc["restored"]:
        print("  gate: left PAUSED (was paused before / --keep-paused)")
    return 0


# --------------------------------------------------------------------------- #
# diff
# --------------------------------------------------------------------------- #
def cmd_diff(a: argparse.Namespace, extra: list[str]) -> int:
    try:
        ra = TraceReader(a.trace_a)
        rb = TraceReader(a.trace_b)
    except (FileNotFoundError, ValueError) as e:
        print(f"[unigym trace diff] {e}", file=sys.stderr)
        return 1
    doc = tdiff.diff_traces(ra, rb, tol=a.tol,
                            ticks=None if a.ticks is None else a.ticks)
    if a.json:
        print(json.dumps(doc))
        return 0
    print(f"=== trace diff: {doc['a']}  vs  {doc['b']} ===")
    print(f"  transport: {doc['transport_a']} vs {doc['transport_b']}  "
          f"tol={doc['tol']:g}  ticks_compared={doc['ticks_compared']}")
    if doc["only_in_a"] or doc["only_in_b"]:
        print(f"  only in A: {doc['only_in_a'] or '-'}")
        print(f"  only in B: {doc['only_in_b'] or '-'}")
    first = doc["first"]
    if first is not None:
        print(f"=== TICK {first['tick_index']} "
              f"(policy tick {first['policy_tick']}; first divergence) ===")
        for name, d in first["segments"].items():
            flag = "  <<<" if d > doc["tol"] else ""
            print(f"  {name:<28} max|d|={_fmt(d)}{flag}")
        worst = first["worst"]
        print(f"  top dims [{worst['segment']}]: " + "  ".join(
            f"{t['dim']}({t['d']:.1e})" for t in first["top_dims"]))
    print("=== PER-TICK (segment worst) ===")
    print(f"  {'tick':>5}  {'policy_tick':>10}  {'worst segment':<24} "
          f"{'max|d|':>9}  verdict")
    for row in doc["per_tick"]:
        w = row.get("worst") or {}
        print(f"  {row['tick_index']:>5}  {str(row['policy_tick']):>10}  "
              f"{str(w.get('segment') or '-'):<24} "
              f"{_fmt(w.get('max')):>9}  {row['verdict']}")
    if first is not None:
        w = first["worst"]
        print(f"first divergence: tick {first['tick_index']} segment "
              f"{w['segment']} ({w['max']:.3e} > tol {doc['tol']:g})")
    else:
        print("no divergence (all compared ticks within tol)")
    if a.by_segment:
        print("=== BY SEGMENT ===")
        print(f"  {'segment':<32} {'max|d|':>9}  verdict")
        for row in tdiff.by_segment(doc):
            print(f"  {row['segment']:<32} {_fmt(row['max']):>9}  "
                  f"{row['verdict']}")
    return 0


# --------------------------------------------------------------------------- #
# diverge (the p14 divergence profile, one command; maps-declared)
# --------------------------------------------------------------------------- #
def cmd_diverge(a: argparse.Namespace, extra: list[str]) -> int:
    from unigym.trace.diverge import diverge
    try:
        doc = diverge(a.trace_ref, a.trace_replayed, map_name=a.map,
                      parts=(a.parts.split(",") if a.parts else None),
                      group=a.group, tol=a.tol)
    except (FileNotFoundError, ValueError) as e:
        hint = getattr(e, "hint", "both inputs are trace dirs "
                                  "(manifest + tick_*.npz); produce the "
                                  "replayed side with `unigym trace "
                                  "import-shard RUN --out DIR`")
        if a.json:
            print(json.dumps({"ok": False, "error": str(e), "hint": hint}))
        else:
            print(f"[unigym trace diverge] {e}\n  hint: {hint}",
                  file=sys.stderr)
        return 1
    if a.json:
        print(json.dumps(doc))
        return 0
    print(f"=== trace diverge: {doc['a']}  vs  {doc['b']} ===")
    print(f"  map={doc['map']}  tol={doc['tol']:g}  ticks={doc['ticks']}  "
          f"parts={','.join(doc['parts'])}")
    ach = doc["achieved"]
    live = [p for p in doc["parts"] if ach.get(p)]
    if live:
        head = f"  {'ticks':>12} " + " ".join(
            f"{p + '_rms':>10}{p + '_max':>10}" for p in live)
        print(head)
        rows = [ach[p] for p in live]
        for i, b in enumerate(rows[0]):
            cells = " ".join(f"{r[i]['rms']:10.3f}{r[i]['max']:10.3f}"
                             for r in rows)
            print(f"  {b['bucket']:>12} {cells}")
    for p in doc["parts"]:
        if p in ach and ach[p] is None:
            print(f"  {p}: N/A (not in map layout)")
    tvp = doc["target_vs_plan"]
    if tvp:
        print("  target-vs-plan (executed arm targets vs official plan):")
        for p, v in tvp.items():
            if v is None:
                continue
            print(f"    {p}: rms={v['rms']:.3f} max={v['max']:.3f}")
    first = doc["first"]
    if first is not None:
        tops = "  ".join(f"{t['dim']}({t['d']:.1e})"
                         for t in first["top_dims"]) or "-"
        print(f"  first divergence: tick {first['tick']} [{tops}]")
    print(f"  verdict: {doc['verdict']} (worst |d| = {doc['worst']:.3e}, "
          f"tol {doc['tol']:g})")
    return 0


# --------------------------------------------------------------------------- #
# import-shard (toolkit phase 2, brainstorm ⑤: existing runs -> trace tools)
# --------------------------------------------------------------------------- #
def cmd_import_shard(a: argparse.Namespace, extra: list[str]) -> int:
    try:
        doc = timport_shard.import_shard(a.run_dir, a.out,
                                         max_ticks=a.ticks)
    except (FileNotFoundError, ValueError) as e:
        if a.json:
            print(json.dumps({"ok": False, "error": f"{e}",
                              "hint": "pass a collect run dir "
                                      "(meta.json + shard_*.npz)"}))
        else:
            print(f"[unigym trace import-shard] FAILED: {e}\n"
                  f"  hint: pass a collect run dir "
                  f"(meta.json + shard_*.npz)", file=sys.stderr)
        return 1
    if a.json:
        print(json.dumps({"ok": True, **doc}))
        return 0
    print(f"[unigym trace import-shard] {doc['ticks']} tick(s) from "
          f"{doc['shards']} shard(s) -> {doc['trace_dir']}")
    if doc["capped_at"] is not None:
        print(f"  truncated at --ticks {doc['capped_at']} "
              "(manifest ticks.capped_at)")
    print(f"  segments={doc['segments']}  manifest: {doc['manifest']}")
    return 0


def add_trace_subparsers(sub) -> None:
    """Wire the `unigym trace` family into the launcher (launch.py)."""
    tr = sub.add_parser(
        "trace", help="debug trace toolkit: record / diff / import-shard "
                      "/ diverge (docs/contracts/trace_contract.md)")
    trs = tr.add_subparsers(dest="trace_cmd", required=True)

    rec = trs.add_parser("record", help="record per-tick bundles from a "
                                        "live run (pause -> step -> "
                                        "console eval-dump)")
    rec.add_argument("run", help="run id (or 'latest')")
    rec.add_argument("--out", default=None,
                     help="trace directory (default: "
                          "<run output_dir>/trace/<name>)")
    rec.add_argument("--name", default=None,
                     help="trace name when --out is omitted "
                          "(default: <UTC timestamp>)")
    rec.add_argument("--ticks", type=int, default=100,
                     help="policy ticks to cover (default 100)")
    rec.add_argument("--every", type=int, default=1,
                     help="sample one shard every N policy ticks "
                          "(default 1; >1 may overshoot <= N-1 ticks)")
    rec.add_argument("--keep-paused", action="store_true",
                     dest="keep_paused",
                     help="leave the run PAUSED after recording "
                          "(default: restore the prior gate)")
    rec.add_argument("--timeout", type=float, default=15.0)
    rec.add_argument("--json", action="store_true")
    rec.set_defaults(fn=cmd_record)

    dif = trs.add_parser("diff", help="segment-wise diff of two traces "
                                      "(first-divergence report)")
    dif.add_argument("trace_a")
    dif.add_argument("trace_b")
    dif.add_argument("--by-segment", action="store_true",
                     dest="by_segment",
                     help="append the segment x all-ticks table")
    dif.add_argument("--tol", type=float, default=1e-6,
                     help="eps band; 0 = bitwise-only (default 1e-6)")
    dif.add_argument("--ticks", type=int, default=None,
                     help="compare only the first N common ticks")
    dif.add_argument("--json", action="store_true")
    dif.set_defaults(fn=cmd_diff)

    ish = trs.add_parser("import-shard",
                         help="migrate a collect run dir (meta.json + "
                              "shard_*.npz) into a trace (contract §8); "
                              "pair with `trace diverge` for the replay "
                              "divergence profile")
    ish.add_argument("run_dir", help="collect run output dir")
    ish.add_argument("--out", required=True, help="trace directory to write")
    ish.add_argument("--ticks", type=int, default=None,
                     help="import only the first N ticks "
                          "(default: all; truncation recorded as "
                          "ticks.capped_at)")
    ish.add_argument("--json", action="store_true")
    ish.set_defaults(fn=cmd_import_shard)

    # ---- diverge ------------
    dv = trs.add_parser("diverge", help="replay divergence profile "
                                        "(per-part bucketed RMS + "
                                        "target-vs-plan + first "
                                        "divergence; contract §9; "
                                        "inputs: `trace import-shard` "
                                        "the replayed run first)")
    dv.add_argument("trace_ref", help="reference trace dir (e.g. the "
                                      "official imported trace)")
    dv.add_argument("trace_replayed", help="replayed trace dir (usually "
                                           "from `trace import-shard`)")
    dv.add_argument("--map", required=True,
                    help="trace map name (maps/<name>.yaml)")
    dv.add_argument("--parts", default=None,
                    help="comma list of body parts (default: all "
                         "declared non-empty; e.g. "
                         "legs,waist,armL,armR,finger)")
    dv.add_argument("--group", type=int, default=50,
                    help="bucket size in ticks (default 50)")
    dv.add_argument("--tol", type=float, default=1e-6,
                    help="eps band for the BITWISE/eps/DIVERGED verdict "
                         "(default 1e-6)")
    dv.add_argument("--json", action="store_true")
    dv.set_defaults(fn=cmd_diverge)
