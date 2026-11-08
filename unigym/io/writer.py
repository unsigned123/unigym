"""Sharded npz data writer.

Accumulates per-step (N, D) tensors in memory and flushes one
``shard_XXXX.npz`` every ``shard_steps`` policy steps. Floats are stored as
float32, bools stay bool, ints stay int. ``meta.json`` (config snapshot +
ordering info + field dims) is written alongside the first shard.

Undo-aware shard (design v0622-undo-aware-shard §2.1): the
recording cursor joins the rewind world.  ``commit_state`` distills the
committed on-disk state to two integers (full shards + the total row
count); ``restore_commit`` rolls the writer back to a commit state so the
shard directory is ALWAYS the winning timeline: post-checkpoint shard
files are deleted, the boundary shard is truncated back to the
checkpoint's rows (its head rows ARE the checkpoint's unflushed buffer --
the buffer is never cleared between capture and flush), and the in-memory
buffer is re-inflated from them so subsequent rows naturally overwrite
the abandoned branch's positions.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import numpy as np
import torch


class ShardWriter:
    def __init__(self, out_dir: str, shard_steps: int, meta: dict | None = None):
        self.out_dir = out_dir
        self.shard_steps = int(shard_steps)
        self.meta = dict(meta or {})
        os.makedirs(out_dir, exist_ok=True)
        self._buf: dict[str, list[np.ndarray]] = defaultdict(list)
        self._steps_in_shard = 0
        self._shard_idx = 0
        self._total_steps = 0
        self._meta_written = False

    @property
    def shard_count(self) -> int:
        return self._shard_idx

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @staticmethod
    def _to_numpy(v) -> np.ndarray:
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
            if v.dtype.kind == "f" and v.dtype != np.float32:
                v = v.astype(np.float32)
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
            v = np.array(v, dtype=object)
        else:
            v = np.asarray(v)
            if v.dtype.kind == "f" and v.dtype != np.float32:
                v = v.astype(np.float32)
        return v

    def append(self, step: dict) -> None:
        """Record one policy step; values are (N, D) tensors, bool allowed."""
        for k, v in step.items():
            self._buf[k].append(self._to_numpy(v))
        self._steps_in_shard += 1
        self._total_steps += 1
        if self._steps_in_shard >= self.shard_steps:
            self._flush()

    def append_episode(self, episode_data: dict) -> None:
        """Append a kept episode's step data (one env, already filtered).

        Values are ``(T_i, D)`` arrays.  They are expanded to ``(T_i, 1, D)``
        and then split into individual ``(1, D)`` step slices so that the
        internal buffer stays consistent with :meth:`append` — ``_flush``
        uses ``np.stack`` which requires uniform shapes.
        """
        T = None
        prepared = {}
        for k, v in episode_data.items():
            arr = np.atleast_1d(self._to_numpy(v))
            if T is None:
                T = arr.shape[0] if arr.ndim > 0 else 1
            if arr.dtype == object:
                arr = arr.reshape(T, 1)
            elif arr.ndim >= 1:
                arr = np.expand_dims(arr, axis=1)  # (T, 1, D) or (T, 1)
            prepared[k] = arr
        if T is None:
            return
        for k, arr in prepared.items():
            self._buf[k].extend(arr)  # adds T slices of shape (1, D)
        self._steps_in_shard += T
        self._total_steps += T
        if self._steps_in_shard >= self.shard_steps:
            self._flush()

    def _flush(self) -> None:
        if self._steps_in_shard == 0:
            return
        arrays = {k: np.stack(v, axis=0) for k, v in self._buf.items()}  # (T, N, D)
        path = os.path.join(self.out_dir, f"shard_{self._shard_idx:04d}.npz")
        np.savez(path, **arrays)
        if not self._meta_written:
            self.meta["field_dims"] = {k: list(a.shape[1:]) for k, a in arrays.items()}
            self.meta["shard_steps"] = self.shard_steps
            self.meta["flushed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
                json.dump(self.meta, f, indent=2, ensure_ascii=False)
            self._meta_written = True
        self._buf = defaultdict(list)
        self._steps_in_shard = 0
        self._shard_idx += 1

    def finalize(self) -> None:
        self._flush()

    # ------------------------------------------------------------------ #
    # undo-aware shard (design v0622-undo-aware-shard §2.1)
    # ------------------------------------------------------------------ #
    def commit_state(self) -> dict:
        """The recording cursor as a checkpoint payload entry: two
        integers only (design: light payload -- every freeze point
        snapshots it).  ``committed_shards`` counts the shards on disk
        (all full); ``committed_rows`` counts the timeline rows committed
        so far INCLUDING the rows still riding the in-memory buffer --
        whose values are recoverable at restore time from the boundary
        shard's head (the buffer is never cleared between capture and
        its flush)."""
        return {"committed_shards": int(self._shard_idx),
                "committed_rows": int(self._total_steps)}

    def restore_commit(self, state: dict) -> None:
        """Roll the writer back to *state* (an earlier ``commit_state``):
        delete post-checkpoint shard files, truncate the boundary shard
        back to the checkpoint's rows and re-inflate the in-memory buffer
        from it -- the next flush naturally rewrites the boundary shard
        with winning-branch rows (shard == winning timeline, always).

        Fail-fast on a corrupt state / boundary shard: raise BEFORE any
        mutation (the rewind safety net then rolls the whole world back
        -- never a partially truncated shard dir).  The validation is
        :func:`validate_commit_state` (module level, v063-rewind-preflight:
        the rewind pre-flight runs the SAME judgment read-only before the
        controller touches the world -- single source, no drift)."""
        n_shards, n_rows, r, head = validate_commit_state(state, self)
        # ① every shard file AFTER the boundary window is abandoned branch
        self._delete_shards_from(n_shards + 1 if r > 0 else n_shards)
        # ② boundary shard: its head r rows are the checkpoint's buffered
        # rows (the buffer rode straight into the flush) -- keep them,
        # rewrite the file, re-inflate the buffer from the same rows
        if r > 0:
            if head is not None:
                np.savez(os.path.join(self.out_dir,
                                      f"shard_{n_shards:04d}.npz"), **head)
                self._buf = defaultdict(list)
                for k, v in head.items():
                    self._buf[k].extend(v[i] for i in range(r))
            else:
                for k in self._buf:
                    del self._buf[k][r:]
            self._steps_in_shard = r
        else:
            self._buf = defaultdict(list)
            self._steps_in_shard = 0
        # ③ the cursor: next flush rewrites shard_N (or the tail), total
        # rows back to the checkpoint count
        self._shard_idx = n_shards
        self._total_steps = n_rows

    def _delete_shards_from(self, start: int) -> list:
        """Remove ``shard_XXXX.npz`` files with index >= *start*; returns
        the removed paths (contiguous rotation -- stop at the first
        hole)."""
        removed = []
        idx = start
        while True:
            path = os.path.join(self.out_dir, f"shard_{idx:04d}.npz")
            if not os.path.exists(path):
                break
            os.remove(path)
            removed.append(path)
            idx += 1
        return removed


def validate_commit_state(state: dict, writer: "ShardWriter") -> tuple:
    """Pure-read validation of a commit state against *writer*'s shard
    dir + buffer (v063-rewind-preflight: split out of
    ``ShardWriter.restore_commit`` so the rewind pre-flight can run the
    SAME judgment read-only before the controller touches the world --
    validate and apply single-source).  Returns ``(n_shards, n_rows, r,
    head)`` where ``r`` is the checkpoint row count riding the in-memory
    buffer and ``head`` the boundary shard's pre-read head rows (``None``
    when those rows live in the buffer).  Raises ValueError BEFORE any
    mutation -- texts and trigger conditions are the historical
    ``restore_commit`` ones, unchanged (us1a-us1e)."""
    n_shards = int(state["committed_shards"])
    n_rows = int(state["committed_rows"])
    ss = writer.shard_steps
    if not (n_shards >= 0 and n_shards * ss <= n_rows
            < (n_shards + 1) * ss):
        raise ValueError(
            f"shard commit state corrupt: committed_shards="
            f"{n_shards}, committed_rows={n_rows}, shard_steps={ss} "
            f"(need 0 <= N and N*ss <= M < (N+1)*ss)")
    r = n_rows - n_shards * ss     # checkpoint rows riding the buffer
    # validate the boundary shard BEFORE deleting anything (the whole
    # restore stays atomic on failure -- nothing half-truncated)
    head = None
    boundary = os.path.join(writer.out_dir, f"shard_{n_shards:04d}.npz")
    if r > 0 and os.path.exists(boundary):
        # allow_pickle: the writer itself may store object-dtype
        # rows (language string lists) -- self-produced files in
        # the run's own out_dir
        with np.load(boundary, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        counts = {int(v.shape[0]) for v in data.values()}
        if len(counts) != 1 or counts.pop() < r:
            raise ValueError(
                f"boundary shard_{n_shards:04d}.npz shorter than "
                f"the commit state ({r} rows needed) -- shard dir "
                f"and commit state disagree")
        head = {k: v[:r].copy() for k, v in data.items()}
    elif r > 0:
        # never flushed past the checkpoint: the rows must still live
        # in the buffer -- verify before any mutation
        lens = [len(v) for v in writer._buf.values()]
        if not lens or min(lens) < r:
            have = min(lens) if lens else 0
            raise ValueError(
                f"shard buffer holds fewer rows than the commit "
                f"state ({have} < {r}) and "
                f"shard_{n_shards:04d}.npz is absent -- commit "
                f"state unreachable")
    return n_shards, n_rows, r, head
