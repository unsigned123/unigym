"""`unigym orbit` -- circular multi-pose capture around a target (toolkit D ②).

Pure orchestration over EXISTING faces (design doc §7): pause -> resolve
target (one console eval) -> per azimuth: free-cam absolute pose set
(``pos/lookat`` face) + snapshot + sidecar -> manifest ->
restore gate.  Zero engine/daemon changes; PAUSED semantics keep physics
frozen while the camera flies (scene-review use case).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from unigym.cameras.inspect import build_info_expr, doc_to_sidecar_facts, \
    unwrap
from unigym.cameras.sidecar import posed_world, write_sidecar


def _op(cli, run: str, op: str, body: dict, timeout: float) -> dict:
    return cli._op(run, op, body, timeout=timeout)


def resolve_target_expr(target: str) -> tuple[str, str, str | None]:
    """``robot | prop/<name> | x,y,z`` -> (kind, name, eval_expr|None)."""
    if "," in target:
        parts = [p for p in target.split(",") if p != ""]
        if len(parts) != 3:
            raise ValueError(f"--target literal expects x,y,z, got "
                             f"{target!r}")
        return "point", "literal", None
    if target == "robot":
        expr = ("_src = dict((extra or {}) if isinstance(extra, dict) "
                "else {}); _src.update(obs_cache if isinstance(obs_cache, "
                "dict) else {}); _v = _src.get('base_pos_w'); "
                "[float(x) for x in (_v[0] if hasattr(_v, 'ndim') and "
                "_v.ndim == 2 else _v)]")
        return "robot", target, expr
    if target.startswith("prop/"):
        name = target[len("prop/"):]
        key = f"object/{name}/pos"
        expr = ("_src = dict((extra or {}) if isinstance(extra, dict) "
                "else {}); _v = _src.get(" + repr(key) + "); "
                "[float(x) for x in (_v[0] if hasattr(_v, 'ndim') and "
                "_v.ndim == 2 else _v)]")
        return "prop", name, expr
    raise ValueError(f"--target must be 'robot', 'prop/<name>' or 'x,y,z', "
                     f"got {target!r}")


def ring_positions(target, radius: float, height: float, count: int,
                   from_deg: float, to_deg: float) -> list[dict]:
    """Azimuth ring geometry (unit-locked): constant radius/height, lookat
    pinned to the target; angles sweep from_deg..to_deg inclusive."""
    count = max(1, int(count))
    azs = [from_deg + (to_deg - from_deg) * i / count for i in range(count)]
    frames = []
    for az in azs:
        r = math.radians(az)
        pos = [target[0] + radius * math.cos(r),
               target[1] + radius * math.sin(r),
               float(height)]
        frames.append({"az_deg": az, "pos": pos, "lookat": list(target)})
    return frames


def _unwrap_ack(doc: dict):
    """daemon op ack -> strip ok -> strip the Inspector format envelope."""
    return unwrap({k: v for k, v in doc.items() if k != "ok"})


def _eval(cli, run: str, expr: str, timeout: float):
    return _unwrap_ack(_op(cli, run, "inspect",
                           {"target": "eval", "expr": expr}, timeout))


def cmd_orbit(a: argparse.Namespace, extra: list[str]) -> int:
    from unigym.scripts import client as cli

    rid = _resolve_run_id(cli, a.run)
    kind, tname, texpr = resolve_target_expr(a.target)
    ops_timeout = a.timeout
    st = _op(cli, rid, "status", {}, ops_timeout)
    was_paused = bool(st.get("paused"))
    if not was_paused and not a.no_pause:
        _op(cli, rid, "pause", {}, ops_timeout)

    doc: dict = {"ok": True, "run_id": rid,
                 "created": datetime.now(timezone.utc).isoformat(
                     timespec="seconds"),
                 "target": {"kind": kind, "name": tname, "pos": None},
                 "radius": a.radius, "height": a.height, "count": a.count,
                 "from_deg": a.from_deg, "to_deg": a.to_deg,
                 "camera": a.camera, "frames": []}
    out_dir = None
    try:
        out_dir = Path(a.out) if a.out else _default_out(cli, rid)
        out_dir.mkdir(parents=True, exist_ok=True)
        if texpr is not None:
            res = _eval(cli, rid, texpr, ops_timeout)
            pos = res if isinstance(res, list) else res.get("inline", res)
            doc["target"]["pos"] = [float(v) for v in pos]
        else:
            doc["target"]["pos"] = [float(v) for v in
                                    a.target.split(",")]
        target = doc["target"]["pos"]

        # one camera-info fetch for the whole orbit (intrinsics/convention)
        info = _eval(cli, rid, build_info_expr([a.camera]), ops_timeout)
        facts = doc_to_sidecar_facts(info, a.camera)

        frames = ring_positions(target, a.radius, a.height, a.count,
                                a.from_deg, a.to_deg)
        tick = st.get("step")
        sim_ts = float(st.get("ts") or 0.0)
        for f in frames:
            i = len(doc["frames"])
            jpg = out_dir / f"orbit_{i:03d}.jpg"
            _op(cli, rid, "camera",
                {"camera_op": "set", "name": a.camera, "params": {},
                 "pos": f["pos"], "lookat": f["lookat"],
                 "timeout": ops_timeout}, timeout=ops_timeout)
            _op(cli, rid, "snapshot",
                {"out": str(jpg), "camera": a.camera,
                 "env": a.env, "timeout": ops_timeout},
                timeout=ops_timeout)
            world = posed_world(f["pos"], f["lookat"])
            meta = write_sidecar(
                jpg, source="orbit", camera=a.camera, run_id=rid,
                tick=tick, sim_ts=sim_ts, phase=st.get("phase"),
                world=world, intrinsics=facts["intrinsics"],
                convention=facts["convention"], rig=facts["rig"],
                mount=facts["mount"], world_error=facts["world_error"],
                extra={"orbit": {"az_deg": f["az_deg"],
                                 "lookat": f["lookat"]}})
            doc["frames"].append(
                {"i": i, "az_deg": f["az_deg"],
                 "cam": {"pos": f["pos"], "lookat": f["lookat"]},
                 "jpg": str(jpg), "meta": str(meta), "tick": tick})
    except Exception as e:                         # noqa: BLE001 (fail-fast)
        doc["ok"] = False
        doc["error"] = f"{type(e).__name__}: {e}"
        _restore(cli, rid, was_paused, a)
        if out_dir is not None:
            (out_dir / "manifest.json").write_text(
                json.dumps(doc, indent=1, default=str))
            doc["manifest"] = str(out_dir / "manifest.json")
        return _finish(a, out_dir or Path("."), doc)
    _restore(cli, rid, was_paused, a)
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps(doc, indent=1, default=str))
    doc["manifest"] = str(manifest)
    return _finish(a, out_dir, doc)


def _restore(cli, run: str, was_paused: bool, a) -> None:
    if not was_paused and not a.no_pause and not a.keep_paused:
        try:
            _op(cli, run, "resume", {}, a.timeout)
        except Exception as e:                 # noqa: BLE001 (audit #23:
            # best effort -- but the run may stay PAUSED, the user
            # must know (`unigym resume <run>` finishes the restore)
            print(f"[unigym orbit] WARNING: resume failed, run {run} may "
                  f"stay paused: {type(e).__name__}: {e}", file=sys.stderr)


def _finish(a, out_dir: Path, doc: dict) -> int:
    n = len(doc.get("frames") or [])
    if a.json:
        print(json.dumps(doc))
        return 0 if doc.get("ok") else 1
    if not doc.get("ok"):
        print(f"[unigym orbit] FAILED after {n} frame(s): "
              f"{doc.get('error')}", file=sys.stderr)
    print(f"[unigym orbit] {n}/{doc.get('count')} frame(s) around "
          f"{doc['target']['kind']}:{doc['target']['name']} "
          f"(r={doc.get('radius')}, h={doc.get('height')}) "
          f"-> {out_dir}")
    if doc.get("manifest"):
        print(f"  manifest: {doc['manifest']}")
    if not doc.get("ok"):
        return 1
    return 0


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_out(cli, run: str) -> Path:
    rid = _resolve_run_id(cli, run)
    view = cli._api("GET", f"/api/runs/{quote(rid, safe='')}",
                    timeout=15).get("run") or {}
    out = view.get("output_dir")
    if not out:
        raise ValueError(f"run {run!r} has no output_dir in its registry "
                         f"-- pass --out")
    p = Path(out)
    if not p.is_absolute():                 # registry convention: repo-root
        p = _REPO_ROOT / p                  # relative (session.py semantics)
    return p / "orbit" / time.strftime("%Y%m%d_%H%M%S")


def _resolve_run_id(cli, run: str) -> str:
    """`latest` / unique-suffix -> concrete id (the view route does not
    resolve aliases); same semantics as DaemonOps.resolve_run."""
    rows = cli._api("GET", "/api/runs", timeout=15).get("runs") or []
    ids = [r.get("run_id") for r in rows if r.get("run_id")]
    if run in ids:
        return run
    if run == "latest":
        if not ids:
            raise ValueError("no live bridge run")
        return ids[-1]
    hits = [i for i in ids if i.endswith(run)]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"run {run!r}: {len(hits)} live match(es)")


def add_orbit_subparser(sub) -> None:
    """Register the ``orbit`` command (lazy CLI dispatch)."""
    ob = sub.add_parser("orbit", help="circle a target with the free camera "
                                      "and capture N sidecar-tagged frames "
                                      "+ manifest")
    ob.add_argument("run", help="run id (or 'latest')")
    ob.add_argument("--target", required=True,
                    help="'robot' | 'prop/<name>' | 'x,y,z' (world frame)")
    ob.add_argument("--radius", type=float, default=2.5)
    ob.add_argument("--height", type=float, default=1.2,
                    help="camera z (absolute; the ring keeps this height)")
    ob.add_argument("--count", type=int, default=12)
    ob.add_argument("--from-deg", type=float, default=0.0, dest="from_deg")
    ob.add_argument("--to-deg", type=float, default=360.0, dest="to_deg")
    ob.add_argument("--camera", default="free",
                    help="fixed/free rig camera to drive (default 'free')")
    ob.add_argument("--out", default=None,
                    help="output dir (default: <run output_dir>/orbit/<ts>)")
    ob.add_argument("--no-pause", action="store_true", dest="no_pause",
                    help="do not pause the run first (default: pause -> "
                         "capture -> resume; live-perturbing otherwise)")
    ob.add_argument("--keep-paused", action="store_true",
                    dest="keep_paused")
    ob.add_argument("--env", type=int, default=None)
    ob.add_argument("--timeout", type=float, default=15.0)
    ob.add_argument("--json", action="store_true")
    ob.set_defaults(fn=cmd_orbit)
