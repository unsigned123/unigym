"""watchers.timeline -- the unified event stream (design watchers.md §3).

``<out_dir>/events.jsonl`` is the run's single append-only event file;
every line ``{tick, ts_wall, ep, env, kind, name, level?, data}``.  Two
producers share the file (one writer, engine thread):

* :class:`TimelineSink` -- a regular sink in the FanOut chain (zero
  engine change): ``fall`` (newly terminated envs), ``first_contact``
  (rising edge per declared contact group/env), ``episode_end``,
  ``run_end``.
* :class:`~unigym.watchers.engine.WatcherEngine` -- ``watcher`` trigger
  events + ``watcher_error`` fail-safe events.

Consumers: ``GET /api/runs/<id>/events`` (daemon, file tail) and
``unigym timeline <run>``.  Sinks are read-only observers, so mounting
the timeline never perturbs recording or physics (no RNG consumption).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

__all__ = ["EventLog", "TimelineSink", "read_tail"]

MAX_DATA_BYTES = 8192  # per-line data guard (fail-safe truncation)


class EventLog:
    """Append-only JSONL writer + file-tail reader (single writer)."""

    def __init__(self, out_dir):
        self.path = Path(out_dir) / "events.jsonl" if out_dir else None
        self.count = 0

    def append(
        self,
        kind: str,
        name: str,
        *,
        tick: int,
        env=None,
        ep: int = 0,
        level: str | None = None,
        data: dict | None = None,
    ) -> None:
        if self.path is None:
            return
        line = {
            "tick": int(tick),
            "ts_wall": round(time.time(), 3),
            "ep": int(ep),
            "env": env,
            "kind": kind,
            "name": name,
        }
        if level is not None:
            line["level"] = level
        raw = json.dumps(data or {}, default=str)
        if len(raw) > MAX_DATA_BYTES:
            raw = json.dumps({"truncated": raw[:MAX_DATA_BYTES]})
        line["data"] = json.loads(raw)
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(line, default=str) + "\n")
            self.count += 1
        except OSError as e:  # additive face, never fatal
            print(f"[timeline] append failed: {e}", flush=True)


def read_tail(path, tail: int = 50, kind: str | None = None) -> list[dict]:
    """Last *tail* events (optionally one kind); missing file -> []."""
    p = Path(path)
    if not p.is_file():
        return []
    try:
        lines = p.read_text(errors="replace").splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-max(1, int(tail)) * (4 if kind else 1) :]:
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if kind and doc.get("kind") != kind:
            continue
        out.append(doc)
    return out[-int(tail) :]


class TimelineSink:
    """Regular sink: fall / first_contact / episode_end / run_end events.

    Args:
        event_log: the shared :class:`EventLog`.
        contact_keys: declared ``contact/<name>`` extra keys (empty when
            the config declares no contacts).
    """

    def __init__(self, event_log: EventLog, contact_keys=None):
        self.event_log = event_log
        self.contact_keys = list(contact_keys or [])
        self._prev_contact: dict[str, set] = {}
        self._prev_terminated: set = set()
        self._episodes = 0

    # -- sink protocol (runner_contract §3) --------------------------------- #
    def on_step(self, t, bundle) -> None:
        log = self.event_log
        if log is None or log.path is None:
            return
        extra = bundle.get("extra") or {}
        term = bundle.get("terminated")
        if term is not None:
            import torch

            ids = (
                set(int(i) for i in torch.nonzero(term).squeeze(-1).tolist())
                if getattr(term, "numel", lambda: 0)()
                else set()
            )
            fresh = ids - self._prev_terminated
            if fresh:
                log.append("fall", "terminated", tick=int(t), data={"envs": sorted(fresh)})
            self._prev_terminated = ids
        for key in self.contact_keys:
            val = extra.get(key)
            if val is None:
                continue
            try:
                cur = self._contact_envs(val)
            except (RuntimeError, ValueError, TypeError):
                continue
            prev = self._prev_contact.get(key, set())
            fresh = cur - prev
            if fresh:
                log.append(
                    "first_contact",
                    key[len("contact/") :],
                    tick=int(t),
                    data={"envs": sorted(fresh)},
                )
            self._prev_contact[key] = cur

    @staticmethod
    def _contact_envs(val) -> set:
        """Env-id set from a ``contact/<name>`` record value.

        Contact-manager reads are ``(N, ...)``: 1-D ``(N,)`` per-env bool
        or ``(N, B)`` group reads (B = sensor bodies / pairwise filters,
        contacts/base.py apply_format aggregate output).  Group reads
        reduce over the sensor axis with ``any`` so the event semantic
        stays "this env's group is in contact" (the 2-D bucketing fix: the old
        1-D-only path crashed on every multi-body group).
        """
        import numpy as np

        arr = val.detach().cpu().numpy() if hasattr(val, "detach") else np.asarray(val)
        if arr.ndim > 1:
            arr = arr.any(axis=-1)
        return {int(i) for i in np.asarray(arr).nonzero()[0].tolist()}

    def on_episode_end(self, env_ids, bundle) -> None:
        if self.event_log is None:
            return
        self._episodes += 1
        try:
            envs = [int(i) for i in env_ids.tolist()]
        except AttributeError:
            envs = []
        self.event_log.append(
            "episode_end",
            "episode",
            tick=int(bundle.get("t", -1) if isinstance(bundle, dict) else -1),
            data={"envs": envs, "episodes": self._episodes},
        )

    def on_step_end(self, t, bundle) -> None:
        pass

    def on_run_end(self) -> dict:
        if self.event_log is not None:
            self.event_log.append("run_end", "run", tick=-1, data={"episodes": self._episodes})
        return {"events_written": self.event_log.count if self.event_log is not None else 0}
