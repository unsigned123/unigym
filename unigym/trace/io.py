"""Trace bundle IO: manifest + tick npz shards (docs/contracts/trace_contract.md §1/§3).

Pure numpy/stdlib.  A trace directory is::

    <dir>/manifest.json
    <dir>/tick_000000.npz ...

The manifest is the segment registry (segment == array key); the diff
engine reads segment shapes from it WITHOUT loading any tick data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

MANIFEST_NAME = "manifest.json"
TICK_RE = re.compile(r"^tick_(\d{6,})\.npz$")
TICK_FMT = "tick_{:06d}.npz"
SCHEMA = 1


def tick_path(trace_dir: str | Path, index: int) -> Path:
    """Canonical npz path of recorded tick *index* (record order)."""
    return Path(trace_dir) / TICK_FMT.format(int(index))


def write_manifest(trace_dir: str | Path, manifest: dict) -> Path:
    """Write manifest.json (schema-pinned; fail-fast on missing keys)."""
    for key in ("schema", "run_id", "segments", "source"):
        if key not in manifest:
            raise ValueError(f"manifest missing required key {key!r}")
    if int(manifest.get("schema", 0)) != SCHEMA:
        raise ValueError(f"manifest schema must be {SCHEMA}, "
                         f"got {manifest.get('schema')!r}")
    path = Path(trace_dir) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, default=str))
    return path


def load_manifest(trace_dir: str | Path) -> dict:
    """Read + validate one manifest.json (fail-fast, contract §3)."""
    path = Path(trace_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no manifest at {path} (not a trace dir?)")
    doc = json.loads(path.read_text())
    if int(doc.get("schema", 0)) != SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r} != {SCHEMA}")
    return doc


class TraceReader:
    """Read-side face over one trace directory (lazy per-tick loads)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = load_manifest(self.root)
        self.indices: list[int] = sorted(
            int(TICK_RE.match(p.name).group(1))
            for p in self.root.glob("tick_*.npz") if TICK_RE.match(p.name))
        if not self.indices:
            raise ValueError(f"{self.root}: manifest but zero tick shards")

    @property
    def transport(self) -> str:
        return str(self.manifest.get("transport")
                   or self.manifest.get("source", {}).get("map", "unknown"))

    def segments(self) -> dict:
        """The manifest segment registry (name -> {shape, dtype})."""
        return dict(self.manifest.get("segments") or {})

    def load_tick(self, index: int) -> dict[str, np.ndarray]:
        """One tick shard as ``{key: array}`` (file must exist)."""
        path = tick_path(self.root, index)
        if not path.is_file():
            raise FileNotFoundError(f"no tick shard {path}")
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    def segment_arrays(self, index: int) -> dict[str, np.ndarray]:
        """Alias of :meth:`load_tick` (a shard IS the segment map)."""
        return self.load_tick(index)
