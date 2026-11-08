"""Photo metadata sidecar: every image artifact gets ``<name>.jpg.json``.

Toolkit D item ① (design doc §6; contract: camera_contract §sidecar).
One writer (:func:`write_sidecar`) -- the CLI capture paths (snapshot /
capture / camera get / orbit) call it right after writing the JPEG; the
existing stdout behavior is untouched (the sidecar is an ADDITIVE file).

Schema (v1)::

    {"schema": 1, "camera", "rig", "mount",
     "world": {"pos", "quat_wxyz", "fwd", "up"} | null,
     "world_error": "...| null,
     "convention", "intrinsics": {"w", "h", "hfov_deg", "model"} | null,
     "tick", "sim_ts", "wall_ts", "run_id", "phase", "source"}

Posed snapshots (``--pos/--lookat``) MUST pass ``world=posed_world(...)``
-- the forensic pose is restored inside the run before any read-back, so
the computed pose is the only record (bit-identical math: it reuses
``cameras.rigs.lookat_quat``, the same function the run-side write path
uses).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .rigs import lookat_quat
from .inspect import quat_axes

SCHEMA = 1
SOURCES = ("snapshot", "capture", "camera_get", "orbit")


def posed_world(pos, lookat) -> dict:
    """World doc for a transient posed capture (contract §sidecar bitwise anchor):
    quat from the repo's own lookat compose; fwd/up derived from it."""
    q = lookat_quat(np.asarray(pos, dtype=np.float64),
                    np.asarray(lookat, dtype=np.float64))
    fwd, up = quat_axes(q)
    return {"pos": [float(v) for v in pos],
            "quat_wxyz": [float(v) for v in q],
            "fwd": fwd, "up": up,
            "lookat": [float(v) for v in lookat]}


def write_sidecar(jpg_path: str | Path, *, source: str, camera: str,
                  run_id: str | None = None, tick: int | None = None,
                  sim_ts: float | None = None, phase: str | None = None,
                  world: dict | None = None, intrinsics: dict | None = None,
                  convention: str | None = None, rig: str | None = None,
                  mount: dict | None = None,
                  world_error: str | None = None,
                  extra: dict | None = None) -> Path:
    """Write ``<jpg_path>.json`` (fail-fast on unknown source); returns it."""
    if source not in SOURCES:
        raise ValueError(f"sidecar source must be one of {SOURCES}, "
                         f"got {source!r}")
    doc = {
        "schema": SCHEMA,
        "camera": camera,
        "rig": rig,
        "mount": mount,
        "world": world,
        "world_error": world_error,
        "convention": convention,
        "intrinsics": intrinsics,
        "tick": tick,
        "sim_ts": sim_ts,
        "wall_ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "phase": phase,
        "source": source,
    }
    if extra:
        doc.update(extra)
    path = Path(str(jpg_path) + ".json")
    path.write_text(json.dumps(doc, indent=1, default=str))
    return path


def read_sidecar(jpg_path: str | Path) -> dict | None:
    """Read one sidecar (None when absent; ValueError on schema drift)."""
    path = Path(str(jpg_path) + ".json")
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    if int(doc.get("schema", 0)) != SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r} != {SCHEMA}")
    return doc
