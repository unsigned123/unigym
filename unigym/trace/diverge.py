"""`unigym trace diverge` -- replay divergence profile, one command
(absorbs p14_divergence_profile.py --
probe retired in git history, v063 cleanup).

Inputs are two trace bundles (REPLAYED typically produced by
`trace import-shard`).  Per-tick the declared ref/rep segments are
aligned (hold-last on the ref side where the official stack only
refreshes on chunk boundaries), then:

* per-part bucketed RMS table over the body-domain groups declared in
  maps/<name>.yaml (IL29: legs 12 / waist 3 / armL 7 / armR 7; a part
  absent from the layout renders N/A, never guessed);
* the target-vs-plan comparison (executed arm targets vs official chunk
  plan rows through the declared permutation);
* first-divergence tick + top-k dims (aligned with `trace diff`
  semantics) and the three-state verdict BITWISE / eps / DIVERGED.

Zero layout/task knowledge in code -- every segment name, dim group and
stride comes from the map yaml.  Pure numpy/stdlib.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .io import TraceReader

MAPS_DIR = Path(__file__).resolve().parent / "maps"
TOP_K = 5


class DivergeError(ValueError):
    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint or "(none)"


def load_map(name: str) -> dict:
    """maps/<name>.yaml -> dict (fail-fast with the available set)."""
    path = MAPS_DIR / f"{name}.yaml"
    if not path.is_file():
        avail = ", ".join(sorted(p.stem for p in MAPS_DIR.glob("*.yaml")))
        raise DivergeError(f"no trace map {name!r} at {path}",
                           hint=f"available maps: {avail}")
    with open(path) as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict) or "comparisons" not in doc:
        raise DivergeError(f"{path}: not a trace map (comparisons missing)")
    return doc



def _resolve_segment(reader, segment: str) -> str:
    """Resolve a historical bare ``action/target`` to its v3 face.

    v103.5 records every NamedAction face as ``action/target/<face>``.  A
    frozen anchor spec (or a single-policy trace) still names the historical
    ``action/target``; when exactly ONE face exists, that segment IS the same
    series (a cardinality rule, never a "first binding").  Multiple faces
    stay unresolved and surface the existing not-found error with the
    available list.
    """
    if segment in reader.segments():
        return segment
    if segment == "action/target":
        matches = [key for key in reader.segments()
                   if key.startswith("action/target/")]
        if len(matches) == 1:
            return matches[0]
    return segment


def _series(reader: TraceReader, segment: str, n_ticks: int,
            hold_last: bool = False) -> np.ndarray:
    """(n_ticks, *shape) float64 matrix of one segment; hold_last repeats
    the last seen row for ticks where the segment is absent (the official
    chunk-refresh convention).  Raises DivergeError when the segment
    never appears or shapes drift."""
    segment = _resolve_segment(reader, segment)
    rows: dict[int, np.ndarray] = {}
    shape = None
    for idx in reader.indices:
        if idx >= n_ticks:
            break
        tick = reader.load_tick(idx)
        if segment in tick:
            v = np.asarray(tick[segment], dtype=np.float64).reshape(-1)
            if shape is None:
                shape = v.shape
            elif v.shape != shape:
                raise DivergeError(
                    f"{reader.root}:{segment} shape drift at tick {idx}: "
                    f"{v.shape} != {shape}")
            rows[idx] = v
    if shape is None:
        avail = ", ".join(sorted(
            k for k in reader.segments() if k != "tick")) or "(none)"
        raise DivergeError(
            f"{reader.root}: segment {segment!r} not found",
            hint=f"available segments: {avail}")
    out = np.zeros((n_ticks, shape[0]), dtype=np.float64)
    last = np.zeros(shape[0], dtype=np.float64)
    for t in range(n_ticks):
        if t in rows:
            last = rows[t]
        elif not hold_last:
            raise DivergeError(
                f"{reader.root}:{segment} missing at tick {t} "
                "(hold_last=false)")
        out[t] = last
    return out


def _first_and_top(diffs: dict[str, np.ndarray], tol: float) -> dict | None:
    """First tick with any |d| > tol + top-k dims across the named dims."""
    first_t = None
    for t in range(next(iter(diffs.values())).shape[0]):
        if any(float(np.abs(d[t]).max()) > tol for d in diffs.values()):
            first_t = t
            break
    if first_t is None:
        return None
    flat = np.concatenate([np.abs(d[first_t]).reshape(-1)
                           for d in diffs.values()])
    names = []
    for name, d in diffs.items():
        n = d[first_t].size
        names += [f"{name}[{i}]" for i in range(n)]
    order = np.argsort(flat)[::-1][:TOP_K]
    top = [{"dim": names[i], "d": float(flat[i])}
           for i in order if flat[i] > 0][:TOP_K]
    return {"tick": first_t, "top_dims": top}


def diverge(ref_dir: str | Path, rep_dir: str | Path, map_name: str,
            parts: list[str] | None = None, group: int = 50,
            tol: float = 1e-6) -> dict:
    """Full divergence document (machine shape; the CLI renders tables)."""
    m = load_map(map_name)
    layout_parts: dict[str, list[int]] = {k: list(v or []) for k, v
                                          in (m.get("parts") or {}).items()}
    want = [p.strip() for p in parts] if parts else \
        [p for p, dims in layout_parts.items() if dims]
    unknown = [p for p in want if p not in layout_parts]
    if unknown:
        raise DivergeError(
            f"--parts {unknown}: not in map {map_name!r} "
            f"(declared: {sorted(layout_parts)})")

    a, b = TraceReader(ref_dir), TraceReader(rep_dir)
    n = min(len(a.indices), len(b.indices))
    if n == 0:
        raise DivergeError("zero common ticks")

    diffs: dict[str, np.ndarray] = {}
    achieved_doc: dict[str, list | None] = {}
    target_doc: dict[str, dict | None] = {}
    specs = m.get("comparisons") or {}

    ach = specs.get("achieved")
    if ach:
        ref = _series(a, ach["ref"]["segment"], n,
                      hold_last=bool(ach["ref"].get("hold_last")))
        rep = _series(b, ach["rep"]["segment"], n)
        if ref.shape != rep.shape:
            raise DivergeError(
                f"achieved: shape mismatch ref {ref.shape} vs rep "
                f"{rep.shape} (layout drift? map {map_name!r})")
        gsize = max(1, int(group))
        for part in want:
            dims = layout_parts.get(part) or []
            if not dims:
                achieved_doc[part] = None          # N/A (finger in IL29)
                continue
            d = ref[:, dims] - rep[:, dims]
            diffs[f"achieved:{part}"] = d
            buckets = []
            for b0 in range(0, n, gsize):
                b1 = min(b0 + gsize, n)
                buckets.append({
                    "bucket": f"{b0}-{b1 - 1}",
                    "rms": float(np.sqrt((d[b0:b1] ** 2).mean())),
                    "max": float(np.abs(d[b0:b1]).max())})
            achieved_doc[part] = buckets

    tvp = specs.get("target_vs_plan")
    if tvp:
        ref_spec = tvp["ref"]
        refresh = int(ref_spec.get("refresh", 1))   # chunk refresh cadence
        pitch = int(ref_spec.get("pitch", refresh))  # stored row width
        width = int(ref_spec.get("width", pitch))    # consumed row width
        rep_seg = _series(b, tvp["rep"]["segment"], n)
        # hold-last keeps the CURRENT chunk on non-boundary ticks
        ref_seg = _series(a, ref_spec["segment"], n, hold_last=True)
        for part in want:
            g = (tvp.get("groups") or {}).get(part)
            if g is None:
                target_doc[part] = None            # no declared pairing
                continue
            rd = [int(x) for x in g["rep_dims"]]
            fd = [int(x) for x in g["ref_dims"]]
            if max(fd) >= width:
                raise DivergeError(f"target_vs_plan:{part}: ref dim "
                                   f"{max(fd)} outside width {width}")
            plan = np.zeros((n, len(fd)), dtype=np.float64)
            for t in range(n):
                chunk = ref_seg[t].reshape(refresh, pitch)
                plan[t] = chunk[t % refresh][:width][fd]
            d = rep_seg[:, rd] - plan
            diffs[f"target:{part}"] = d
            target_doc[part] = {"rms": float(np.sqrt((d ** 2).mean())),
                                "max": float(np.abs(d).max())}

    worst = max((float(np.abs(d).max()) for d in diffs.values()), default=0.0)
    verdict = "BITWISE" if worst == 0.0 else ("eps" if worst <= tol
                                              else "DIVERGED")
    return {"a": str(a.root), "b": str(b.root), "map": map_name,
            "tol": tol, "group": int(group), "ticks": n,
            "parts": want, "achieved": achieved_doc,
            "target_vs_plan": target_doc, "verdict": verdict,
            "worst": worst, "first": _first_and_top(diffs, tol)}
