"""Runtime metrics block: bounded structured telemetry .

The run publishes a FIXED-SIZE metrics block in its SHM segment (layout
constants live in ``bridge.py`` next to the other blocks; the wire contract
is docs/contracts/runtime_contract.md §2) so the daemon can answer typed
control queries (command ids 10-12) without any run-side state of its own:

* tick level   -- timing scalars only (fps EMA / sim_time), one seqlock
                 cycle per step, microseconds (the B9 budget holds);
* episode end  -- the typed document: task identity, last episode
                 {reward, success, steps} and the bounded reward/success
                 rings (fixed capacity, head pointer -- never grows with
                 run time, red line §1.3);
* queries      -- ``metrics`` composes the document + live status counters;
                 ``config`` returns the effective-config snapshot captured
                 ONCE at startup (oversize = fail-fast there, not at query
                 time); ``report`` mirrors the benchmark draft report via
                 the duck-typed metrics source (None without a benchmark).

Everything here is stdlib-only and offline-testable; the seqlock primitives
are the bridge module's own (single-sourcing discipline, runtime_contract §3).
"""
from __future__ import annotations

import json
import time

from unigym.runtime.bridge import (M_CK, M_FPS_P, M_FPS_S, M_GEN, M_JSON,
                                   M_SEQ, M_SIMT, M_TS, METRICS_JSON,
                                   METRICS_OFF, BridgeError, _f64,
                                   _payload_ck, _put_bytes, _seqlock_read,
                                   _u64)

__all__ = ["MetricsRing", "MetricsWriter", "read_metrics_block",
           "RING_CAP"]

RING_CAP = 50                    # bounded episode rings (red line §1.3)

#: EMA weight of the newest fps sample (a light low-pass over step periods)
_FPS_EMA_A = 0.9


class MetricsRing:
    """Fixed-capacity FIFO (head pointer + preallocated list -- no growth)."""

    __slots__ = ("_cap", "_buf", "_head", "_n")

    def __init__(self, cap: int = RING_CAP):
        self._cap = int(cap)
        self._buf: list = [None] * self._cap
        self._head = 0
        self._n = 0

    def push(self, value) -> None:
        """Append one item (oldest silently drops once full)."""
        self._buf[self._head] = value
        self._head = (self._head + 1) % self._cap
        self._n = min(self._n + 1, self._cap)

    def to_list(self) -> list:
        """Chronological snapshot (oldest first, at most ``cap`` items)."""
        if self._n < self._cap:
            return list(self._buf[:self._n])
        return self._buf[self._head:] + self._buf[:self._head]

    def __len__(self) -> int:
        return self._n


class MetricsWriter:
    """Run-side single writer of the SHM metrics block.

    One seqlock guards the whole block: tick updates touch only the scalar
    prefix, episode updates rewrite the JSON document -- both close their
    odd/even window before any reader looks.  Python-side copies mirror the
    published state so the in-process query handler never re-reads its own
    SHM, and the direct block read (:func:`read_metrics_block`) stays
    available to every other process.
    """

    def __init__(self, buf):
        self._buf = buf
        self._gen = 0
        self._fps_policy: float | None = None
        self._fps_sim: float | None = None
        self._sim_time: float | None = None
        self._last_tick_wall: float | None = None
        self.task: dict | None = None
        self.progress: dict | None = None
        self.rewards = MetricsRing()
        self.successes = MetricsRing()
        self.last: dict | None = None

    # -- run-side update points (existing sink hooks; runner_contract §3) --
    def note_tick(self, t: int, env=None) -> None:
        """Timing scalars only (tick-level cost budget: microseconds)."""
        now = time.time()
        prev = self._last_tick_wall
        self._last_tick_wall = now
        if prev is not None:
            dt = now - prev
            if 0.0 < dt < 1.0:
                inst = 1.0 / dt
                self._fps_policy = inst if self._fps_policy is None \
                    else _FPS_EMA_A * self._fps_policy + (1 - _FPS_EMA_A) * inst
                # decimation duck-type: isaaclab exposes `decimation`,
                # mujoco `_decimation` (telemetry-only, failure-tolerant)
                dec = getattr(env, "decimation", None)
                dec = getattr(env, "_decimation", None) if dec is None else dec
                self._fps_sim = self._fps_policy * dec \
                    if isinstance(dec, (int, float)) and dec else None
        step_dt = getattr(env, "step_dt", None)
        if isinstance(step_dt, (int, float)) and step_dt > 0:
            self._sim_time = float(t) * float(step_dt)
        self._write_scalars()

    def note_episode(self, env_ids, bundle, source=None) -> None:
        """Episode-level update (the engine's on_episode_end hook)."""
        steps = self._episode_steps(env_ids, bundle)
        scored = []
        if source is not None:
            try:
                scored = list(source.metrics_drain() or [])
            except Exception:               # noqa: BLE001 (telemetry only)
                scored = []
        reward = success = None
        directed = False
        for i, _ in enumerate(env_ids):
            tel = scored[i] if i < len(scored) else {}
            reward, success = tel.get("reward"), tel.get("success")
            directed = bool(tel.get("directed"))
            self.rewards.push(reward)
            self.successes.push(success)
        self.last = {"reward": reward, "success": success,
                     "directed": directed,
                     "steps": steps[-1] if steps else None}
        if source is not None:
            try:
                self.task = source.metrics_task() or self.task
                self.progress = source.metrics_progress()
            except Exception:               # noqa: BLE001 (telemetry only)
                pass
        self.set_document(self.document())

    # -- document --------------------------------------------------------------
    def document(self) -> dict:
        """Typed metrics document, runtime_contract §2 schema.

        Task identity + last episode {reward, success, steps} + the
        bounded reward/success rings + progress.
        """
        return {"task": self.task,
                "episode": {"last": self.last,
                            "rewards": self.rewards.to_list(),
                            "successes": self.successes.to_list()},
                "progress": self.progress}

    def timing(self) -> dict:
        """Live timing scalars (fps EMA / sim time), runtime_contract §2."""
        return {"fps_sim": self._fps_sim, "fps_policy": self._fps_policy,
                "sim_time": self._sim_time}

    def metrics_document(self, run_id: str, step: int, episodes: int) -> dict:
        """Full query answer: document + live status counters (cmd 10)."""
        doc = self.document()
        doc["run_id"] = run_id
        doc["timing"] = {"step": step, "episodes": episodes,
                         **self.timing()}
        return doc

    def set_document(self, doc: dict) -> None:
        """Rewrite the JSON document (episode end / progress changes)."""
        try:
            raw = json.dumps(doc).encode()
        except (TypeError, ValueError) as e:
            raise BridgeError(f"metrics document not serializable: {e}") from e
        if len(raw) > METRICS_JSON:      # bounded block: fail-fast, never clip
            raise BridgeError(f"metrics document {len(raw)}B exceeds fixed "
                              f"block {METRICS_JSON}B (ring cap exceeded?)")
        buf = self._buf
        seq = _u64(buf, METRICS_OFF + M_SEQ)
        _u64(buf, METRICS_OFF + M_SEQ, seq + 1)      # odd = writing
        _put_bytes(buf, METRICS_OFF + M_JSON, raw, METRICS_JSON)
        _f64(buf, METRICS_OFF + M_TS, time.time())
        _u64(buf, METRICS_OFF + M_CK, _payload_ck(raw))
        self._gen += 1
        _u64(buf, METRICS_OFF + M_GEN, self._gen)
        self._write_scalars()                        # same open window
        _u64(buf, METRICS_OFF + M_SEQ, seq + 2)      # even = stable

    # -- internals ----------------------------------------------------------------
    def _write_scalars(self) -> None:
        """Scalar prefix (tick path: own seqlock cycle)."""
        buf = self._buf
        seq = _u64(buf, METRICS_OFF + M_SEQ)
        _u64(buf, METRICS_OFF + M_SEQ, seq + 1)      # odd = writing
        self._put_scalars()
        _u64(buf, METRICS_OFF + M_SEQ, seq + 2)      # even = stable

    def _put_scalars(self) -> None:
        """Scalar bytes only (caller owns the seqlock window)."""
        _f64(buf := self._buf, METRICS_OFF + M_FPS_P,
             self._fps_policy if self._fps_policy is not None else 0.0)
        _f64(buf, METRICS_OFF + M_FPS_S,
             self._fps_sim if self._fps_sim is not None else 0.0)
        _f64(buf, METRICS_OFF + M_SIMT,
             self._sim_time if self._sim_time is not None else 0.0)
        _f64(buf, METRICS_OFF + M_TS, time.time())

    @staticmethod
    def _episode_steps(env_ids, bundle) -> list[int]:
        """Per-ended-env policy step count (time_in_episode / step_dt)."""
        extra = bundle.get("extra") if isinstance(bundle, dict) else None
        times = extra.get("time_in_episode") if extra is not None else None
        env = bundle.get("env") if isinstance(bundle, dict) else None
        step_dt = getattr(env, "step_dt", None)
        if times is None or not isinstance(step_dt, (int, float)) \
                or step_dt <= 0:
            return []
        try:
            return [int(round(float(times[int(ei)]) / float(step_dt)))
                    for ei in env_ids]
        except (TypeError, ValueError, IndexError):
            return []


def read_metrics_block(buf) -> dict:
    """Reader primitive: tear-free metrics block snapshot (any process).

    Returns the published scalars + parsed document ({} document when the
    writer has not published one yet or the JSON is momentarily invalid).
    """
    raw = _seqlock_read(buf, METRICS_OFF, M_JSON, METRICS_JSON)
    doc = {}
    if raw is not None:
        raw = raw.split(b"\x00", 1)[0]
        if raw:
            try:
                doc = json.loads(raw)
            except ValueError:
                doc = {}
    return {"gen": _u64(buf, METRICS_OFF + M_GEN),
            "fps_policy": _f64(buf, METRICS_OFF + M_FPS_P),
            "fps_sim": _f64(buf, METRICS_OFF + M_FPS_S),
            "sim_time": _f64(buf, METRICS_OFF + M_SIMT),
            "ts": _f64(buf, METRICS_OFF + M_TS),
            "document": doc}
