"""Trace replay source: drives ``unigym run --replay <trace>`` (contract §7).

Reads a trace bundle (TraceReader) and answers the engine's two injection
points per policy tick (docs/contracts/trace_contract.md §7):

* action mode (default): :meth:`raw_for` returns the recorded raw so the
  engine can swap it in AFTER ``adapter.act`` and BEFORE
  ``postprocess_action`` -- physics runs live, actions follow the trace.
* obs mode: :meth:`obs_for` returns the recorded policy frame BEFORE
  ``act`` -- the adapter consumes recorded observations, its LIVE raw is
  compared against the trace (the downstream-regression verdict; physics
  is bypassed in the information loop).

Alignment is TICK-LABEL based (contract §7.2, design §3.2): live tick t
consumes the shard whose ``tick`` label equals t.  Traces recorded
through the step-gate carry labels offset by the born-park (first=1):
live ticks BEFORE the first label are pre-roll (live actions, WARN
counted -- for a same-config deterministic stack the pre-roll action is
itself the recorded one, so the closed loop still closes bitwise).
In-stride ticks hold the previous shard's value.  Past the last label
the run STOPS (§7.3: stop, never hold-last past the horizon).

Pure numpy + lazy torch conversion at injection time (dtype/device follow
the live tensors); zero engine/daemon coupling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import TraceReader

RAW_KEY_DEFAULT = "action/raw"
OBS_KEY = "obs/policy_frame"


class ReplayError(ValueError):
    """Fail-fast replay face; ``.hint`` carries the recovery pointer."""

    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint or "(none)"


class TraceReplay:
    """Per-tick injection values over one trace bundle (contract §7)."""

    def __init__(self, trace_dir: str | Path, mode: str = "action",
                 strict: bool = False, key: str | None = None):
        if mode not in ("action", "obs"):
            raise ReplayError(
                f"replay mode {mode!r} unknown",
                hint="one of: action (inject recorded raw, physics live) "
                     "| obs (inject recorded policy frame, live raw is "
                     "the regression test)")
        self.reader = TraceReader(trace_dir)
        self.mode = mode
        self.strict = bool(strict)
        # injection bookkeeping (init before any validation warning path)
        self.held_ticks = 0
        self.injected = 0
        self.warnings: list[str] = []
        self._shard_cache: tuple[int, dict] | None = None
        # obs-mode live-raw comparison (the regression verdict)
        self.cmp_ticks = 0
        self.cmp_max = 0.0
        self.cmp_bitwise = True
        self.cmp_first: dict | None = None
        manifest = self.reader.manifest
        if str(manifest.get("status")) != "complete":
            raise ReplayError(
                f"{trace_dir}: manifest status "
                f"{manifest.get('status')!r} != 'complete'",
                hint="replay only accepts complete traces -- re-record")
        ticks = manifest.get("ticks") or {}
        self.every = max(1, int(ticks.get("every") or 1))
        self.recorded = int(ticks.get("recorded") or len(self.reader.indices))
        first = ticks.get("first")
        self.first = int(first) if first is not None else 0
        last = ticks.get("last")
        if last is None:
            last = self.first + self.every * self.recorded - 1
        self.last = int(last)
        self.coverage = self.last - self.first + 1   # injectable tick span
        self.preroll = self.first                    # live-action ticks 0..first-1
        segs = self.reader.segments()
        self.raw_key = key or (RAW_KEY_DEFAULT
                               if RAW_KEY_DEFAULT in segs else None)
        if mode == "action" and self.raw_key is None:
            action_segs = sorted(k for k in segs
                                 if k.startswith("action/"))
            raise ReplayError(
                f"{trace_dir}: no {RAW_KEY_DEFAULT} segment",
                hint="pass --replay-key <segment> (available action/* "
                     f"segments: {action_segs} "
                     "| non-action segments are not injectable raws)")
        if mode == "obs" and OBS_KEY not in segs:
            raise ReplayError(
                f"{trace_dir}: no {OBS_KEY} segment (obs mode needs the "
                "recorded policy frame)",
                hint="re-record without image exclusion or use action mode")
        if mode == "obs" and self.raw_key is None:
            self._warn(f"no {RAW_KEY_DEFAULT} in trace: the obs-mode "
                       "comparison verdict is N/A (nothing to compare)")

    # ------------------------------------------------------------------ #
    # face used by the engine
    # ------------------------------------------------------------------ #
    def has_tick(self, t: int) -> bool:
        """False past the last recorded label (engine stops, §7.3).

        Pre-roll ticks (t < first label) are served as live-action ticks,
        not stops."""
        return int(t) <= self.last

    def obs_for(self, t: int, live_obs) -> object | None:
        """Obs mode: the recorded policy frame for tick *t* (§7.2).

        None in action mode or pre-roll (live obs feeds act)."""
        if self.mode != "obs":
            return None
        shard = self._shard_for(t)
        if shard is None:
            return None
        frame = shard.get(OBS_KEY)
        if frame is None:
            raise ReplayError(f"tick {t}: {OBS_KEY} missing in shard",
                              hint="trace is inconsistent with its manifest")
        return self._as_live(frame, live_obs, OBS_KEY, t)

    def raw_for(self, t: int, raw_live) -> object | None:
        """Action mode: the recorded raw for tick *t*.

        None in obs mode (live raw is the test face) or pre-roll (live
        action runs, WARN-counted)."""
        if self.mode != "action":
            self._compare(t, raw_live)
            return None
        raw_key = self.raw_key
        if raw_key is None:
            raise ReplayError(
                f"tick {t}: no raw key configured",
                hint="action mode needs a raw segment")
        shard = self._shard_for(t)
        if shard is None:
            return None
        raw = shard.get(raw_key)
        if raw is None:
            raise ReplayError(
                f"tick {t}: {raw_key} missing in shard",
                hint="trace is inconsistent with its manifest")
        self.injected += 1
        return self._as_live(raw, raw_live, raw_key, t)

    # ------------------------------------------------------------------ #
    # rng freeze (opt-in, §7.4: adapter RNG is OUTSIDE the guarantee face)
    # ------------------------------------------------------------------ #
    def freeze_rng(self) -> dict:
        """Re-seed the GLOBAL rng faces from the trace manifest seed."""
        import torch
        seed = self.reader.manifest.get("seed")
        if seed is None:
            raise ReplayError(
                f"{self.reader.root}: manifest has no seed",
                hint="freeze-rng needs the recording seed -- re-record "
                     "with a config that pins one")
        seed = int(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        return {"seed": seed, "faces": ["torch", "numpy.random"]}

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _shard_for(self, t: int) -> dict | None:
        """Shard holding tick *t*'s values by label; None on pre-roll.

        In-stride ticks (every > 1) hold the previous shard (WARN)."""
        t = int(t)
        off = t - self.first
        if off < 0:
            self.preroll_ticks = getattr(self, "preroll_ticks", 0) + 1
            if self.strict:
                raise ReplayError(
                    f"tick {t}: pre-roll (trace first label "
                    f"{self.first})",
                    hint="--replay-strict is on: the live action that "
                         "fills the pre-roll fails fast")
            if self.preroll_ticks <= 3:
                self.warnings.append(
                    f"tick {t}: pre-roll (before first label "
                    f"{self.first}) runs the LIVE action")
            return None
        k = off // self.every
        if off % self.every:
            self._note_hold(t)
        k = min(k, len(self.reader.indices) - 1)
        if self._shard_cache is None or self._shard_cache[0] != k:
            self._shard_cache = (k, self.reader.load_tick(
                self.reader.indices[k]))
        return self._shard_cache[1]

    def _note_hold(self, t: int) -> None:
        """In-stride tick: the previous shard's value is held (§7.2)."""
        self.held_ticks += 1
        if self.strict:
            raise ReplayError(
                f"tick {t}: hold over a record stride (every={self.every})",
                hint="--replay-strict is on: record with --every 1 or "
                     "replay without --replay-strict")
        if len(self.warnings) < 6:
            self.warnings.append(
                f"tick {t}: held previous shard value "
                f"(every={self.every})")

    def _as_live(self, arr: np.ndarray, live, key: str, t: int):
        import torch
        if isinstance(live, torch.Tensor):
            if tuple(live.shape) != tuple(np.shape(arr)):
                raise ReplayError(
                    f"tick {t}: {key} shape {tuple(np.shape(arr))} != live "
                    f"{tuple(live.shape)}",
                    hint="trace was recorded against a different "
                         "adapter/env face -- replay needs the recording "
                         "config")
            return torch.as_tensor(arr, dtype=live.dtype, device=live.device)
        # no live tensor to mirror (e.g. obs mode's first tick): the
        # recorded dtype is the contract face
        return torch.as_tensor(np.asarray(arr, dtype=np.float32)) \
            if np.asarray(arr).dtype.kind == "f" else torch.as_tensor(arr)

    def _compare(self, t: int, raw_live) -> None:
        if self.raw_key is None or raw_live is None:
            return
        shard = self._shard_for(t)
        if shard is None:
            return
        rec = shard.get(self.raw_key)
        if rec is None:
            return
        import torch
        live = raw_live.detach().cpu().numpy() \
            if isinstance(raw_live, torch.Tensor) else np.asarray(raw_live)
        if tuple(live.shape) != tuple(np.shape(rec)):
            self._warn(f"tick {t}: comparison skipped -- live raw "
                       f"{live.shape} vs recorded {np.shape(rec)}")
            return
        d = float(np.abs(live - np.asarray(rec, dtype=live.dtype)).max()) \
            if live.size else 0.0
        self.cmp_ticks += 1
        self.cmp_max = max(self.cmp_max, d)
        if d > 0.0:
            self.cmp_bitwise = False
            if self.cmp_first is None:
                self.cmp_first = {"tick": int(t), "segment": self.raw_key,
                                  "max": d}

    def _warn(self, msg: str) -> None:
        self.warnings.append(msg)

    # ------------------------------------------------------------------ #
    # bind + report
    # ------------------------------------------------------------------ #
    def bind(self, num_envs: int) -> dict:
        """Fail-fast shape check against the live run (call before run())."""
        segs = self.reader.segments()
        key = (self.raw_key if self.mode == "action" else OBS_KEY)
        shape = (segs.get(key) or {}).get("shape") or []
        if shape and int(shape[0]) != int(num_envs):
            raise ReplayError(
                f"{key} was recorded with num_envs={shape[0]}, this run "
                f"has num_envs={num_envs}",
                hint="replay requires the recording env count (the trace "
                     "is a flat sequence over that env layout)")
        return {"trace": str(self.reader.root), "mode": self.mode,
                "every": self.every, "coverage": self.coverage,
                "raw_key": self.raw_key}

    def report(self) -> dict:
        doc = {"mode": self.mode, "trace": str(self.reader.root),
               "first_label": self.first, "last_label": self.last,
               "preroll": getattr(self, "preroll_ticks", 0),
               "coverage": self.coverage, "injected": self.injected,
               "held": self.held_ticks, "warnings": list(
                   getattr(self, "warnings", []))}
        if self.mode == "obs":
            doc["comparison"] = {
                "ticks": self.cmp_ticks, "bitwise": self.cmp_bitwise,
                "max_abs": self.cmp_max, "first": self.cmp_first}
        return doc

    def report_lines(self) -> list[str]:
        r = self.report()
        lines = [f"[replay] mode={r['mode']} trace={r['trace']}",
                 f"[replay] labels {r['first_label']}..{r['last_label']} "
                 f"coverage={r['coverage']} injected={r['injected']} "
                 f"preroll={r['preroll']} held={r['held']} "
                 f"warnings={len(r['warnings'])}"]
        for w in r["warnings"][:3]:
            lines.append(f"[replay] WARN: {w}")
        if "comparison" in r:
            c = r["comparison"]
            first = (f"first at tick {c['first']['tick']} "
                     f"(max|d|={c['first']['max']:.3e})"
                     if c["first"] else "none")
            lines.append(f"[replay] live-raw vs trace: ticks={c['ticks']} "
                         f"bitwise={c['bitwise']} max|d|={c['max_abs']:.3e} "
                         f"{first}")
        return lines
