"""Trace recorder: live-run attach over the console faces (contract §0/§5).

Transport ``console_eval_dump.v1`` -- the recorder parks the engine
(``pause``), then per sampled tick: ``step(every) -> status -> console
eval-dump``.  The eval expression (one multi-statement block, trailing
expression returned) assembles the tick bundle from the run's injected
namespace (``np torch obs_cache bundle``) and writes the tick npz DIRECTLY
in contract §2 key layout -- one req-ack per tick, zero run-side changes.

The loop talks to an :class:`RecorderOps` transport so unit tests can
drive it against a fake; the daemon implementation lives in
:class:`DaemonOps` (unigym.scripts.client HTTP faces only).
"""
from __future__ import annotations

import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .io import tick_path, write_manifest

TRANSPORT = "console_eval_dump.v1"
SOURCE_KIND = "unigym"
REPO_ROOT = Path(__file__).resolve().parents[2]

# One console eval per sampled tick.  Contract keys (§2): obs/<key> from
# obs_cache (images excluded: ndim >= 3), extra/<key> from bundle['extra'],
# action/<name> from bundle['targets'] (+ action/raw), obs/policy_frame
# from bundle['obs_frame'], tick scalar.  Trailing expression = the ack.
_DUMP_EXPR = """\
_arrs = {{}}
def _as_np(v):
    return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
_b = bundle if isinstance(bundle, dict) else {{}}
_oc = obs_cache if isinstance(obs_cache, dict) else {{}}
for _k, _v in _oc.items():
    try:
        _a = _as_np(_v)
    except Exception:
        continue
    if _a.ndim <= 2 and _a.dtype.kind in 'bfiu':
        _arrs['obs/' + str(_k)] = _a
for _k, _v in (_b.get('extra') or {{}}).items():
    if _v is None or isinstance(_v, (str, bytes, bool, int, float)):
        continue
    try:
        _a = _as_np(_v)
    except Exception:
        continue
    if _a.ndim <= 2 and _a.dtype.kind in 'bfiu':
        _arrs['extra/' + str(_k)] = _a
for _k, _v in (_b.get('targets') or {{}}).items():
    try:
        _arrs['action/' + str(_k)] = _as_np(getattr(_v, 'values', _v))
    except Exception:
        continue
if _b.get('raw') is not None:
    _arrs['action/raw'] = _as_np(_b['raw'])
if _b.get('obs_frame') is not None:
    _arrs['obs/policy_frame'] = _as_np(_b['obs_frame'])
_arrs['tick'] = np.int64(_b.get('t', -1))
np.savez(r'{path}', **_arrs)
{{'keys': len(_arrs), 'tick': int(_b.get('t', -1))}}
"""


def build_dump_expr(path: str) -> str:
    """The per-tick console-eval expression (unit-locked against fakes)."""
    return _DUMP_EXPR.format(path=str(path))


# --------------------------------------------------------------------------- #
# transports
# --------------------------------------------------------------------------- #
class RecorderOps:
    """The seven control faces the recorder needs (see DaemonOps)."""

    def status(self, run: str) -> dict: ...            # noqa: D102
    def step(self, run: str, n: int) -> dict: ...      # noqa: D102
    def pause(self, run: str) -> dict: ...             # noqa: D102
    def resume(self, run: str) -> dict: ...            # noqa: D102
    def eval(self, run: str, expr: str) -> dict: ...   # noqa: D102
    def config(self, run: str) -> dict | None: ...      # noqa: D102
    def run_view(self, run: str) -> dict: ...          # noqa: D102


class DaemonOps(RecorderOps):
    """Daemon control-API transport (thin; unigym.scripts.client faces)."""

    def __init__(self, timeout: float = 15.0):
        from unigym.scripts import client as cli
        self._cli = cli
        self.timeout = float(timeout)

    def resolve_run(self, run: str) -> str:
        """`latest` / unique-suffix -> the concrete run id (GET /api/runs;
        the view route itself does not resolve aliases)."""
        rows = self._cli._api("GET", "/api/runs", timeout=self.timeout) \
            .get("runs") or []
        ids = [r.get("run_id") for r in rows if r.get("run_id")]
        if run in ids:
            return run
        if run == "latest":
            if not ids:
                raise ValueError("no live bridge run (start with: "
                                 "unigym run CONFIG ...)")
            return ids[-1]
        hits = [i for i in ids if i.endswith(run)]
        if len(hits) == 1:
            return hits[0]
        raise ValueError(f"run {run!r}: {len(hits)} live match(es) "
                         f"({hits or 'none'}) -- use the full id")

    def status(self, run: str) -> dict:
        return self._cli._op(self.resolve_run(run), "status", {},
                             timeout=self.timeout)

    def step(self, run: str, n: int) -> dict:
        return self._cli._op(self.resolve_run(run), "step", {"n": int(n)},
                             timeout=self.timeout)

    def pause(self, run: str) -> dict:
        return self._cli._op(self.resolve_run(run), "pause", {},
                             timeout=self.timeout)

    def resume(self, run: str) -> dict:
        return self._cli._op(self.resolve_run(run), "resume", {},
                             timeout=self.timeout)

    def eval(self, run: str, expr: str) -> dict:
        return self._cli._op(self.resolve_run(run), "inspect",
                             {"target": "eval", "expr": expr},
                             timeout=self.timeout)

    def config(self, run: str) -> dict | None:
        return self._cli._op(self.resolve_run(run), "config", {},
                             timeout=self.timeout).get("config")

    def run_view(self, run: str) -> dict:
        from urllib.parse import quote
        rid = self.resolve_run(run)
        return self._cli._api("GET", f"/api/runs/{quote(rid, safe='')}",
                              timeout=self.timeout)

    def director_status(self, run: str) -> dict | None:
        """Director face ; None when the run rejects the op."""
        try:
            return self._cli._op(self.resolve_run(run), "direct/status",
                                 {}, timeout=self.timeout)
        except Exception:                     # noqa: BLE001 (best effort)
            return None


# --------------------------------------------------------------------------- #
# manifest metadata (record side)
# --------------------------------------------------------------------------- #
def git_snapshot(repo_root: Path = REPO_ROOT) -> dict:
    """``{"rev", "dirty"}`` of the recording checkout (best-effort rev,
    fail-soft dirty flag: metadata must never abort a recording)."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root,
            capture_output=True, text=True, timeout=10).stdout.strip() != ""
    except (OSError, subprocess.SubprocessError):
        return {"rev": None, "dirty": True}
    return {"rev": rev or None, "dirty": dirty}


def _default_out_dir(ops: RecorderOps, run: str, name: str) -> Path:
    view = ops.run_view(run).get("run") or {}
    out = view.get("output_dir")
    if not out:
        raise ValueError(
            f"run {run!r} has no output_dir in its registry -- pass --out")
    p = Path(out)
    if not p.is_absolute():                 # registry convention: repo-root
        p = REPO_ROOT / p                   # relative (session.py semantics)
    return p / "trace" / name


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def _policy_profile_name(cfg: dict) -> str | None:
    """The policy articulation's name from a run config."""
    for art in cfg.get("articulations") or []:
        if isinstance(art, dict) and \
                (art.get("control") or {}).get("mode") == "policy":
            return art.get("name")
    return None


def record_trace(ops: RecorderOps, *, run: str, out: str | None,
                 name: str, ticks: int, every: int = 1,
                 keep_paused: bool = False) -> dict:
    """Record *ticks*// *every* tick shards from a live run (contract §5).

    Restores the pre-recording gate (RUNNING -> resume) unless
    ``keep_paused``.  Returns the stdout summary dict; raises on any op
    failure AFTER writing an ``aborted`` manifest (fail-fast, shards kept).
    """
    every = int(every)
    if every < 1:
        raise ValueError(f"--every must be >= 1, got {every}")
    n_chunks = int(math.ceil(int(ticks) / every))

    st0 = ops.status(run)
    was_paused = bool(st0.get("paused"))
    if not was_paused:
        ops.pause(run)

    # resolved before the manifest exists so the abort path below can
    # always restore the gate (out-dir failure must never wedge a run)
    try:
        trace_dir = Path(out) if out else _default_out_dir(ops, run, name)
        trace_dir.mkdir(parents=True, exist_ok=True)
    except Exception:  # cleanup: unpause before propagating (never swallowed)
        if not was_paused and not keep_paused:
            ops.resume(run)
        raise

    manifest: dict = {
        "schema": 1,
        "run_id": run,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "transport": TRANSPORT,
        "git": git_snapshot(),
        "config": None,
        "seed": None, "profile": None, "backend": None,
        "cameras": {},
        "segments": {},
        "ticks": {"recorded": 0, "every": every, "first": None,
                  "last": None, "sim_ts": [], "step_overshoot": 0},
        "source": {"kind": SOURCE_KIND, "map": TRANSPORT},
        "status": "aborted",
        # director face (contract §7): filled below; a directed
        # trace carries the command-log reference (the reproducible
        # definition of the directed session)
        "director": None,
        "directed": False,
    }
    try:
        view = (ops.run_view(run).get("run") or {})
        cfg = ops.config(run)
        manifest["config"] = cfg
        if isinstance(cfg, dict):
            manifest["seed"] = cfg.get("seed")
            manifest["profile"] = _policy_profile_name(cfg)
            manifest["backend"] = (cfg.get("env") or {}).get("backend")
            cams = cfg.get("cameras") or []
            manifest["cameras"] = {
                "declared": [c.get("name") for c in cams
                             if isinstance(c, dict) and c.get("name")]}
        manifest["run_id"] = view.get("run_id", run)
        dst = getattr(ops, "director_status", None)
        if dst is not None:
            doc = dst(run)
            if doc is not None:
                manifest["director"] = {
                    "enabled": bool(doc.get("enabled")),
                    "commands": doc.get("commands"),
                    "log": doc.get("log")}
                manifest["directed"] = bool(doc.get("commands"))

        segments: dict[str, dict] = {}
        first_tick = last_tick = None
        expected_tick = None
        step_overshoot = 0
        import numpy as np
        for i in range(n_chunks):
            ops.step(run, every)
            st = ops.status(run)
            if str(st.get("phase")).upper() == "DONE":
                raise RuntimeError(
                    f"run finished at tick shard {i}/{n_chunks} -- "
                    f"re-record with fewer ticks")
            path = tick_path(trace_dir, i)
            expr = build_dump_expr(str(path))
            ops.eval(run, expr)

            def _file_tick() -> int:
                if not path.is_file():          # same-host promise
                    raise RuntimeError(
                        f"eval-dump acked but {path} missing (daemon and "
                        f"CLI must share one host -- contract §0)")
                with np.load(path) as z:
                    return int(z["tick"]) if "tick" in z.files else -1

            # The step op acks on DISPATCH, not tick completion: an eval
            # served by the park loop at the PREVIOUS tick's serve point
            # lands a duplicate shard.  The dumped FILE carries the tick
            # it captured -- re-dump until current (bounded, fail-fast).
            bt = _file_tick()
            retries = 0
            while expected_tick is not None and 0 <= bt != expected_tick:
                if retries >= 50:
                    raise RuntimeError(
                        f"eval-dump stuck at tick {bt} (expected "
                        f"{expected_tick}) after {retries} re-dumps")
                ops.eval(run, expr)
                bt = _file_tick()
                retries += 1
            with np.load(path) as z:            # segment registry
                for k in z.files:
                    arr = z[k]
                    segments[k] = {"shape": [int(d) for d in arr.shape],
                                   "dtype": str(arr.dtype)}
            if first_tick is None:
                first_tick = bt
            expected_tick = bt + every if bt >= 0 else None
            last_tick = bt
            manifest["ticks"]["sim_ts"].append(
                float(st.get("ts") or 0.0))
            step_overshoot = max(0, (i + 1) * every - int(ticks))
        manifest["segments"] = segments
        manifest["ticks"].update({"recorded": n_chunks, "first": first_tick,
                                  "last": last_tick,
                                  "step_overshoot": step_overshoot})
        manifest["status"] = "complete"
        write_manifest(trace_dir, manifest)
    except Exception as e:                          # noqa: BLE001 (contract §5)
        manifest["error"] = f"{type(e).__name__}: {e}"
        write_manifest(trace_dir, manifest)
        if not was_paused and not keep_paused:
            try:
                ops.resume(run)
            except Exception:                       # noqa: BLE001 (best effort)
                pass
        raise
    if not was_paused and not keep_paused:
        ops.resume(run)
    return {"trace_dir": str(trace_dir), "recorded": n_chunks,
            "every": every, "first_tick": first_tick,
            "last_tick": last_tick, "segments": len(segments),
            "step_overshoot": step_overshoot,
            "manifest": str(Path(trace_dir) / "manifest.json"),
            "restored": (not was_paused and not keep_paused),
            "wall_hint": time.strftime("%Y-%m-%d %H:%M:%S")}
