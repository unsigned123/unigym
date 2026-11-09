"""shard -> trace migrator: existing runs enter the trace tool chain (§9-⑤).

`unigym trace import-shard <run_dir> --out <trace_dir>` reads a collect
run's ``meta.json + shard_*.npz`` (zero collection) and writes a contract
trace.  Key map (declared in the manifest ``source.map_table``; everything
else passes through verbatim -- no cosmetic renames):

    obs                  -> obs/policy_frame   (the dataset policy frame)
    action_raw           -> action/raw
    action_target/<face> -> action/target/<face> (v3, EVERY NamedAction face)
    action_target        -> action/target          (legacy v2 shard migration)

Robot state keys are namespaced (``articulation/<name>/...`` in v3) and
pass through verbatim, so importing a legacy v2 run still yields the same
trace segments as before.  Ticks are the GLOBAL step index across shards
(episode resets included:
shards are flat).  Truncation via ``max_ticks`` is recorded in
``ticks.capped_at`` (status stays ``complete`` for the imported range).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .io import tick_path, write_manifest

MAP = "collect_shard.v1"
_KEY_RENAME = {
    "obs": "obs/policy_frame",
    "action_raw": "action/raw",
    # historical anchors: v2 shards carry the single primary column
    "action_target": "action/target",
}
_ACTION_PREFIX = "action_target/"


def _map_key(k: str) -> str:
    if k in _KEY_RENAME:
        return _KEY_RENAME[k]
    if k.startswith(_ACTION_PREFIX):
        return "action/target/" + k[len(_ACTION_PREFIX) :]
    return k


def import_shard(run_dir: str | Path, out: str | Path, max_ticks: int | None = None) -> dict:
    """One collect run dir -> one trace bundle (fail-fast, contract §3)."""
    run = Path(run_dir)
    meta_path = run / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"{run}: no meta.json (not a unigym collect run dir?)")
    meta = json.loads(meta_path.read_text())
    shards = sorted(run.glob("shard_*.npz"))
    if not shards:
        raise ValueError(f"{run}: zero shard_*.npz files")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    segments: dict[str, dict] = {}
    total = 0
    skipped: dict[str, int] = {}
    capped_at = None
    for shard in shards:
        with np.load(shard) as z:
            rows = int(z[z.files[0]].shape[0])
            for r in range(rows):
                if max_ticks is not None and total >= int(max_ticks):
                    capped_at = total
                    break
                arrs: dict[str, Any] = {"tick": np.int64(total)}
                for k in z.files:
                    if k in ("tick",):
                        continue
                    v = z[k][r]
                    if not isinstance(v, np.ndarray) or v.dtype.kind not in "bfiu":
                        skipped[k] = skipped.get(k, 0) + 1
                        continue
                    name = _map_key(k)
                    arrs[name] = np.ascontiguousarray(v)
                    segments[name] = {"shape": [int(d) for d in v.shape], "dtype": str(v.dtype)}
                np.savez(tick_path(out, total), **arrs)
                total += 1
            if capped_at is not None:
                break
    if total == 0:
        raise ValueError(f"{run}: shards hold zero rows")

    cfg = meta.get("config") or {}
    manifest = {
        "schema": 1,
        "run_id": run.name,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "transport": MAP,
        "git": {"rev": None, "dirty": True},
        "config": cfg or None,
        "seed": meta.get("seed"),
        "profile": next(
            (
                art.get("name")
                for art in (cfg.get("articulations") or [])
                if isinstance(art, dict) and (art.get("control") or {}).get("mode") == "policy"
            ),
            None,
        ),
        "backend": (cfg.get("env") or {}).get("backend"),
        "cameras": {},
        "segments": segments,
        "ticks": {
            "recorded": total,
            "every": 1,
            "first": 0,
            "last": total - 1,
            "sim_ts": [],
            "step_overshoot": 0,
            **({"capped_at": capped_at} if capped_at is not None else {}),
        },
        "source": {
            "kind": "unigym_collect",
            "map": MAP,
            "run_dir": str(run),
            "map_table": {
                "obs": "obs/policy_frame",
                "action_raw": "action/raw",
                "action_target/<face>": "action/target/<face>",
                "(others)": "verbatim",
            },
            **({"skipped_non_array": skipped} if skipped else {}),
        },
        "status": "complete",
    }
    write_manifest(out, manifest)
    return {
        "trace_dir": str(out),
        "ticks": total,
        "shards": len(shards),
        "segments": len(segments),
        "capped_at": capped_at,
        "manifest": str(out / "manifest.json"),
    }
