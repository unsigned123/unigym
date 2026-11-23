"""Pure episode buffering for the runner evaluation pipeline.

``EpisodeBuffer`` owns the per-env step slices, the open-episode rewind marks
and the per-step tick samples that an evaluation sink accumulates between
reset boundaries.  It slices, stores, takes, snapshots, restores and dumps --
it never invokes a benchmark task and imports no benchmark type.

The buffer is the single owner of open-episode history: the sink delegates
``on_step``/``on_rewind`` to it, the rewind provider snapshots/restores it, and
the episode-end path removes one env's history with :meth:`EpisodeBuffer.take`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from ..episodes import assemble_episode, slice_step_for_env

__all__ = ["EpisodeBuffer", "TakenEpisode", "TickSample"]

StepRecord = dict[str, object]
BufferState = dict[str, object]


@dataclass(frozen=True, slots=True)
class TickSample:
    """One env's per-task tick detection values for one policy tick."""

    metrics: Mapping[str, np.ndarray] = field(default_factory=dict)
    fired: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class TakenEpisode:
    """One env's buffered episode, removed from the buffer by ``take``."""

    steps: list[StepRecord] = field(default_factory=list)
    rewinds: list[dict[str, object]] = field(default_factory=list)
    ticks: list[dict[str, TickSample]] = field(default_factory=list)


class EpisodeBuffer:
    """Per-env open-episode history: step slices, rewind marks and tick samples."""

    def __init__(self, num_envs: int):
        if num_envs < 0:
            raise ValueError(f"episode buffer: num_envs must be >= 0, got {num_envs}")
        self.num_envs = int(num_envs)
        self.bufs: list[list[StepRecord]] = [[] for _ in range(self.num_envs)]
        self.rewinds: list[list[dict[str, object]]] = [[] for _ in range(self.num_envs)]
        self.ticks: list[list[dict[str, TickSample]]] = [[] for _ in range(self.num_envs)]

    def append_step(self, rec: Mapping[str, object]) -> None:
        """Slice one normalized per-step record into every env's episode."""
        step = dict(rec)
        for ei in range(self.num_envs):
            self.bufs[ei].append(slice_step_for_env(step, ei))
            self.ticks[ei].append({})

    def append_tick(self, env_id: int, task: str, sample: TickSample) -> None:
        """Attach one task's tick sample to the env's most recent step."""
        self.ticks[env_id][-1][task] = sample

    def mark_rewound(self, info: Mapping[str, object]) -> None:
        """Attach a rewind record to every open episode buffer."""
        for env_marks in self.rewinds:
            env_marks.append(dict(info))

    def take(self, env_id: int) -> TakenEpisode:
        """Remove and return one env's open episode history."""
        taken = TakenEpisode(
            steps=self.bufs[env_id],
            rewinds=self.rewinds[env_id],
            ticks=self.ticks[env_id],
        )
        self.bufs[env_id] = []
        self.rewinds[env_id] = []
        self.ticks[env_id] = []
        return taken

    def snapshot(self) -> BufferState:
        """Return a deep-copied state independent of the live buffers."""
        return {
            "bufs": [[_copy_slice(s) for s in buf] for buf in self.bufs],
            "marks": [[dict(m) for m in marks] for marks in self.rewinds],
            "ticks": [
                [{task: _copy_tick(sample) for task, sample in step.items()} for step in steps]
                for steps in self.ticks
            ],
            "episode_stamp": self.episode_stamp(),
        }

    def restore(self, state: Mapping[str, object]) -> None:
        """Write a snapshot back in place, preserving the outer list identity."""
        bufs_state = state.get("bufs")
        if not isinstance(bufs_state, list):
            raise ValueError(
                "episode buffer state: 'bufs' must be a list, " f"got {type(bufs_state).__name__}"
            )
        self.bufs[:] = [[_copy_slice(s) for s in buf] for buf in bufs_state]
        marks_state = state.get("marks")
        if marks_state is None:
            self.rewinds[:] = [[] for _ in range(self.num_envs)]
        else:
            if not isinstance(marks_state, list):
                raise ValueError(
                    "episode buffer state: 'marks' must be a list or None, "
                    f"got {type(marks_state).__name__}"
                )
            self.rewinds[:] = [[dict(m) for m in marks] for marks in marks_state]
        ticks_state = state.get("ticks")
        if ticks_state is None:
            self.ticks[:] = [[{} for _ in buf] for buf in self.bufs]
        else:
            if not isinstance(ticks_state, list):
                raise ValueError(
                    "episode buffer state: 'ticks' must be a list or None, "
                    f"got {type(ticks_state).__name__}"
                )
            self.ticks[:] = [
                [{task: _copy_tick(sample) for task, sample in step.items()} for step in steps]
                for steps in ticks_state
            ]

    def episode_stamp(self) -> int | None:
        """Episode id of the open buffer's last frame (first non-empty env)."""
        for buf in self.bufs:
            if buf:
                last = buf[-1].get("episode_id")
                if last is not None:
                    return int(np.asarray(last).reshape(-1)[0])
        return None

    def dump(self, out_dir: str, base: str) -> list[str]:
        """Write the current open buffers (steps + tick traces) to one npz."""
        path = os.path.join(out_dir, f"{base}.npz")
        arrays: dict[str, np.ndarray] = {}
        for ei, buf in enumerate(self.bufs):
            if not buf:
                continue
            assembled = assemble_episode(buf)
            for k, v in assembled.items():
                arr = v if isinstance(v, np.ndarray) else np.asarray(v)
                arrays[f"env{ei}/{k}"] = arr
        for ei, steps in enumerate(self.ticks):
            for si, step in enumerate(steps):
                for task, sample in step.items():
                    for k, v in sample.metrics.items():
                        arrays[f"env{ei}/ticks/{si}/{task}/{k}"] = np.asarray(v)
                    if sample.fired is not None:
                        arrays[f"env{ei}/ticks/{si}/{task}/fired"] = np.asarray(sample.fired)
        np.savez(path, **arrays)
        return [path]


def _copy_slice(s: Mapping[str, object]) -> StepRecord:
    """One buffered step slice, deep-copied (numpy arrays get fresh storage)."""
    out: StepRecord = {}
    for k, v in s.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif isinstance(v, list):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = dict(v)
        else:
            out[k] = v
    return out


def _copy_tick(sample: TickSample) -> TickSample:
    """One tick sample, deep-copied (metric and fired arrays are isolated)."""
    return TickSample(
        metrics={k: np.array(v, copy=True) for k, v in sample.metrics.items()},
        fired=None if sample.fired is None else np.array(sample.fired, copy=True),
    )
