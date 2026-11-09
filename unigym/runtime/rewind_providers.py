"""Rewind state providers.

The sink facilities whose open-episode state a restored timeline must
agree with.

A rewind that restores only the engine + adapter leaves the SINKS marching.
On a forked timeline that mixed the abandoned branch's frames into the
open episode buffers -- Round A's "lift_dz baseline distortion (the
abandoned and continued fork branches mixing one buffer)"
-- and the episode ids read duplicated on the assembled episodes.  The
fix is the provider protocol (``runtime/rewind.py``): each provider's
``rewind_state()`` joins every checkpoint payload and is applied during
the restore, after the adapter, fail-fast.

Providers (v1 coverage, design §1.3 table):

* ``BenchmarkSinkProvider`` (name ``"benchmark"``) -- the open episode
  buffers (``ep_bufs``) + the per-env rewind marks riding them.  Counter
  policy (contract §4.1): scored-RESULT bookkeeping
  (``results`` / ``episode_docs`` / ``episodes`` / ``steps_total`` / the
  bridge metrics feed) is PROCESS TRUTH and is NOT restored -- it keeps
  marching monotonically, which de-duplicates scored docs on the
  abandoned branch.  The EPISODE FUSE counters (``n_episodes`` -- and
  with it the derived ``min_episodes`` stop latch) DO ride the fork:
  after an undo across an episode boundary the restored timeline
  re-earns its episode completions.  Without the restore, the
  abandoned branch's completion stayed counted, the rewound retry
  re-counted the same episode end, and ``min_episodes`` fired a
  premature run_end mid-strategy (the 2026-10-16 forensics: undo to
  a tick-1137 checkpoint, replay re-completed episode #1, n_episodes
  1->2, run ended at tick 1199 under the agent).  Scored docs of
  abandoned branches stay in the append-only lists (labelled
  ``rewound`` where applicable) -- nothing is erased.
* ``CollectSinkProvider`` (name ``episode_runner``) -- the collect
  writer's episode slicing state (``ep_bufs``), same buffer mechanism;
  its ``n_episodes`` fuse counter rides the fork for the same reason.
  The undo-aware shard design puts the
  SHARD RECORDING CURSOR on the same package (``shard_commit`` =
  ``{committed_shards, committed_rows}`` off the sink's
  :class:`~unigym.io.writer.ShardWriter`).  The restore deletes the
  post-checkpoint shard files, truncates the boundary shard back to the
  checkpoint's rows and re-inflates the writer's row buffer from it --
  the shard directory stays the WINNING timeline (monotonic (episode,
  tick), no abandoned-branch rows), so a run that contains undos replays
  bitwise end to end.  Runs recorded BEFORE that design mixed the
  abandoned branch's rows into the shard at every undo point -- replay
  of those is a qualitative reference only (runtime_contract §4.2).

Both providers also implement ``dump_attempt(dir, base)`` -- the §1.4
sidecar: before a rewind applies, the CURRENT (abandoned-branch) buffer
frames are written to ``<out_dir>/attempts/`` as npz (one key per env
field, ``assemble_episode`` stacked) so no data is ever discarded.  The
buffers are episode-scoped lists of per-step slice dicts (numpy/tensor
free after ``slice_step_for_env`` -- cpu clones), so the round trip is a
deep copy away from byte-exact.

pre-flight:
both providers implement ``validate_rewind_state(state)`` -- the SAME
refusal judgment their ``apply_rewind_state`` enforces, run read-only
by the controller BEFORE the sidecar dump / safety capture / restore
(single source: ``_guard_state`` here, ``validate_commit_state`` in
``unigym.io.writer``).  A refused rewind is a zero-change rejection;
the apply-face guards remain as the second line.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import numpy as np

from unigym.io.writer import validate_commit_state
from unigym.runtime.rewind import (RewindGateError, RewindStateProvider,
                                   SnapshotValue)

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from unigym.io.writer import ShardWriter

__all__ = ["BenchmarkSinkProvider", "CollectSinkProvider",
           "InitializerStateProvider",
           "build_rewind_providers"]


EpisodeBuffers: TypeAlias = list[list[dict[str, SnapshotValue]]] | None
MarksState: TypeAlias = list[list[dict[str, SnapshotValue]]] | None


class _ShardWriterFace(Protocol):
    """The shard recording-cursor face the collect provider restores."""

    def commit_state(self) -> SnapshotValue: ...

    def restore_commit(self, state: SnapshotValue) -> None: ...


class InitializerStateProvider:
    """The active initializer's sampling state in the rewind package."""

    name = "initializer"

    def __init__(self, initializer: object):
        self.initializer = initializer

    def rewind_state(self) -> SnapshotValue:
        """Return an isolated copy of the initializer's declared state."""
        snapshot = getattr(self.initializer, "snapshot_state")
        return copy.deepcopy(snapshot())

    def apply_rewind_state(self, state: SnapshotValue) -> None:
        """Restore from a private copy so the checkpoint remains reusable."""
        restore = getattr(self.initializer, "restore_state")
        restore(copy.deepcopy(state))

    def validate_rewind_state(self, state: SnapshotValue) -> None:
        """Reject malformed or unsupported snapshots before changing state."""
        if not callable(getattr(self.initializer, "restore_state", None)):
            raise ValueError("active initializer cannot restore rewind state")
        if state is None:
            raise ValueError("initializer rewind state is missing")


class _EpisodeBufferProvider:
    """Shared machinery for the sink-buffer providers.

    Subclasses name the facility and point ``_bufs``/``_marks`` at the
    sink's attributes.  The snapshot is a DEEP copy (fresh numpy arrays);
    the apply writes back in place (list identity preserved -- the sink
    holds the reference).  An optional ``episode_stamp`` (the episode id
    of the buffer's last frame) rides the state: a rewind whose
    checkpoint belongs to a DIFFERENT episode than the live one is
    refused fail-fast (the v1 boundary -- cross-episode rewind is out of
    scope, design §0).
    """

    name = ""

    def __init__(self, sink: object):
        self.sink = sink
        self.num_envs = int(getattr(sink, "num_envs", 1))

    # -- the sink attributes (subclass pins) -------------------------- #
    def _bufs(self) -> EpisodeBuffers:
        raise NotImplementedError

    def _marks(self) -> MarksState:
        return None

    def _fuse(self) -> dict[str, SnapshotValue]:
        """The facility's episode-fuse counters.

        They ride the fork so the restored timeline re-earns its
        episode completions (the premature-run_end forensics).
        Default: none.
        """
        return {}

    # -- provider protocol --------------------------------------------- #
    def rewind_state(self) -> dict[str, SnapshotValue]:
        """Deep-copied buffers + marks + episode stamp + fuse counters."""
        bufs = self._bufs()
        marks_state = self._marks()
        state: dict[str, SnapshotValue] = {
            "bufs": [[_copy_slice(s) for s in buf]
                     for buf in (bufs or [])],
            "marks": ([[dict(m) for m in marks]
                       for marks in marks_state]
                      if marks_state is not None else None),
            "episode_stamp": self._live_stamp(bufs),
            "fuse": self._fuse()}
        return state

    def validate_rewind_state(self, state: dict[str, SnapshotValue]) -> None:
        """Pre-flight.

        The SAME refusal judgment :meth:`apply_rewind_state` enforces,
        run read-only BEFORE the controller touches the world -- a
        refusal here is a zero-change rejection (no dump, no safety
        capture, no restore).  Single source: both phases call
        :meth:`_guard_state`.
        """
        self._guard_state(self._bufs(), state)

    def apply_rewind_state(self, state: dict[str, SnapshotValue]) -> None:
        """Write the snapshot back in place (list identity preserved).

        Raises:
            RuntimeError: Shape drift or a cross-episode rewind (v1
                boundary, refused fail-fast).
        """
        bufs = self._bufs()
        # the shared refusal judgment (single source with the
        # pre-flight validate_rewind_state) -- second line behind the
        # controller's pre-flight
        self._guard_state(bufs, state)
        if bufs is not None:
            bufs[:] = [[_copy_slice(s) for s in buf]
                       for buf in state["bufs"]]
        marks = self._marks()
        if marks is not None and state.get("marks") is not None:
            marks[:] = [[dict(m) for m in m_list]
                        for m_list in state["marks"]]
        fuse: Mapping[str, SnapshotValue] = state.get("fuse") or {}
        live_fuse = self._fuse()
        for k, v in fuse.items():
            if k in live_fuse:
                setattr(self.sink, k, v)

    def _guard_state(self, live_bufs: EpisodeBuffers,
                     state: dict[str, SnapshotValue]) -> None:
        """The shared refusal judgment (single source).

        Buffer shape drift + the cross-episode stamp check.  Raises
        :class:`RewindGateError` (a RuntimeError -- historical raise
        family and message texts preserved).
        """
        if live_bufs is None and state.get("bufs") is not None:
            # shape drift (checkpoint with buffers, live facility
            # without): refuse with the shape named instead of a
            # NoneType TypeError from the write-back
            raise RewindGateError(
                f"rewind provider '{self.name}': buffer shape drift "
                f"(checkpoint holds episode buffers but the live sink "
                f"has none) -- refusing rather than partially "
                f"restoring")
        live = self._live_stamp(live_bufs)
        stamp = state.get("episode_stamp")
        if stamp is not None and live is not None and int(stamp) != int(live):
            raise RewindGateError(
                f"rewind provider '{self.name}': cross-episode rewind "
                f"refused (checkpoint episode {int(stamp)} != live "
                f"episode {int(live)}) -- cross-set rewind is out of the "
                f"v1 scope (design v057 §0)")

    def episode_hint(self) -> int:
        """The episode ordinal for the sidecar filename (process truth)."""
        return int(getattr(self.sink, "n_episodes", 0))

    # -- internals ------------------------------------------------------ #
    def _live_stamp(self, bufs: EpisodeBuffers) -> int | None:
        """Episode id of the open buffer's last frame (None otherwise).

        None when the buffer is empty -- episode boundaries carry no
        frames -- or when the facility has no buffers at all: a plain
        collect run's ``ep_bufs`` is None by construction.
        """
        if bufs is None:
            return None
        for buf in bufs:
            if buf:
                last = buf[-1].get("episode_id")
                if last is not None:
                    return int(np.asarray(last).reshape(-1)[0])
        return None

    # -- §1.4 sidecar ----------------------------------------------------- #
    def dump_attempt(self, out_dir: str, base: str) -> list[str]:
        """The current (abandoned-branch) buffers -> one npz under *out_dir*.

        Returns the written paths (meta json is the controller's).  A
        buffer-LESS facility (a plain collect run's ``ep_bufs`` is None
        -- no filter chain, no open episode buffers) holds nothing to
        dump: the honest empty result, never a TypeError (the undo
        blocker -- the dump runs on EVERY rewind before the restore, so
        the unguarded enumerate(None) blocked undo for every plain
        collect run).
        """
        from unigym.runner.episodes import assemble_episode

        bufs = self._bufs()
        if bufs is None:
            return []
        path = f"{os.path.join(out_dir, base)}.npz"
        arrays: dict[str, Any] = {}
        for ei, buf in enumerate(bufs):
            if not buf:
                continue
            assembled: dict[str, SnapshotValue] = assemble_episode(buf)
            for k, v in assembled.items():
                arr = v if isinstance(v, np.ndarray) else np.asarray(v)
                arrays[f"env{ei}/{k}"] = arr
        np.savez(path, **arrays)
        return [path]


class BenchmarkSinkProvider(_EpisodeBufferProvider):
    """The benchmark sink's open episode buffers (name ``benchmark``)."""

    name = "benchmark"

    def __init__(self, sink):
        super().__init__(sink)
        # counter policy (contract §4.1): results /
        # episode_docs / episodes / steps_total / _metrics_pending are
        # deliberately OUT of rewind_state -- process truth marches on.
        # The FUSE counters below ride the fork: n_episodes is
        # the min_episodes stop gate's counter -- restoring it stops the
        # rewound retry from tripping a premature run_end on a count the
        # abandoned branch already spent.

    def _bufs(self) -> EpisodeBuffers:
        return getattr(self.sink, "ep_bufs", None)

    def _marks(self) -> MarksState:
        return getattr(self.sink, "_ep_rewinds", None)

    def _fuse(self) -> dict[str, SnapshotValue]:
        return {"n_episodes": int(getattr(self.sink, "n_episodes", 0)),
                "_stop": bool(getattr(self.sink, "_stop", False))}


class CollectSinkProvider(_EpisodeBufferProvider):
    """The collect writer's episode slicing state (name ``episode_runner``).

    Same buffer machinery; its episode fuse counter rides the fork.
    The undo-aware shard design puts the shard recording
    cursor (``ShardWriter.commit_state``, two integers) on the same
    package; its restore truncates the shard directory back to the
    committed winning timeline.
    """

    name = "episode_runner"

    def _bufs(self) -> EpisodeBuffers:
        return getattr(self.sink, "ep_bufs", None)

    def _marks(self) -> MarksState:
        return None

    def _fuse(self) -> dict[str, SnapshotValue]:
        return {"n_episodes": int(getattr(self.sink, "n_episodes", 0))} \
            if hasattr(self.sink, "n_episodes") else {}

    def _writer(self) -> ShardWriter | None:
        """The sink's shard writer (absent on provider-protocol doubles).

        The shard face simply does not ride those.
        """
        w = getattr(self.sink, "writer", None)
        return w if callable(getattr(w, "commit_state", None)) else None

    def rewind_state(self) -> dict[str, SnapshotValue]:
        """Base state + the shard recording cursor when a writer exists.

        Buffer-less collect runs (no filter chain) carry only the fuse
        counter + cursor -- the recursive provider walk made this shape
        reachable in production: plain collect runs now mount the
        provider too.
        """
        if self._bufs() is None:
            # non-filter collect run: no episode buffers exist -- the
            # shard cursor is the whole state (the recursive provider
            # walk made this shape reachable in production: plain
            # collect runs now mount the provider too)
            state: dict[str, SnapshotValue] = {
                "bufs": None, "marks": None, "episode_stamp": None,
                "fuse": self._fuse()}
        else:
            state = super().rewind_state()
        w = self._writer()
        if w is not None:
            state["shard_commit"] = w.commit_state()
        return state

    def validate_rewind_state(self, state: dict[str, SnapshotValue]) -> None:
        """Pre-flight the shard commit state (read-only, F4).

        Runs the base provider's judgment, then the SAME
        ``validate_commit_state`` check restore_commit runs (single
        source: unigym.io.writer.validate_commit_state) so a
        corrupt/unreachable commit state is refused before the world is
        touched, wrapped in :class:`RewindGateError` ({error, hint}).

        Args:
            state: The snapshot payload (``shard_commit`` optional).

        Raises:
            RewindGateError: The commit state fails validation.
        """
        super().validate_rewind_state(state)
        w = self._writer()
        commit = state.get("shard_commit")
        if w is not None and commit is not None:
            # pre-flight: the truncation surgery is judged
            # READ-ONLY before the controller touches the world -- a
            # corrupt/unreachable commit state can no longer surface
            # mid-apply with the shard files already cut.  The SAME
            # validator restore_commit runs (single source:
            # unigym.io.writer.validate_commit_state); wrapped in the
            # gate error so every pre-flight refusal speaks
            # {error, hint}.
            try:
                validate_commit_state(commit, w)
            except ValueError as e:
                raise RewindGateError(
                    f"rewind provider '{self.name}': {e}",
                    hint="the snapshot's shard commit state is "
                         "corrupt or unreachable against the live "
                         "shard dir; rewind refuses rather than "
                         "truncate") from e

    def apply_rewind_state(self, state: dict[str, SnapshotValue]) -> None:
        """Restore the buffers (base) and then the shard cursor.

        The shard cursor restore is plain file+cursor surgery -- a
        failure there raises through and the controller's safety net
        rolls the buffers back with the rest of the world.
        """
        if state.get("bufs") is None:
            # buffer-less shape: only the fuse counter rides the fork
            fuse: Mapping[str, SnapshotValue] = state.get("fuse") or {}
            for k, v in fuse.items():
                if hasattr(self.sink, k):
                    setattr(self.sink, k, v)
        else:
            # buffers first (the cross-episode refusal fires before ANY
            # mutation); the shard cursor restore is plain file+cursor
            # surgery -- a failure there raises through and the
            # controller's safety net rolls the buffers back with the
            # rest of the world
            super().apply_rewind_state(state)
        w = self._writer()
        commit = state.get("shard_commit")
        if w is not None and commit is not None:
            w.restore_commit(commit)


def build_rewind_providers(sink: object,
                           initializer: object | None = None
                           ) -> list[RewindStateProvider]:
    """One provider per provider-capable sink in the fan-out chain.

    collect_cli wiring; a plain sink or None -> whatever applies.

    The walk RECURSES into nested fan-outs -- production wraps
    the sink chain (``FanOutSink(FanOutSink(collect, timeline), ...)``
    etc.), and the one-level walk saw only the wrappers, so the
    providers silently never mounted in real runs (the mixed-shard
    undo artifacts are that hole's symptom).  Depth-first in
    construction order; identity-deduped against diamond re-visits.
    """
    out: list[RewindStateProvider] = []
    seen: list[object] = []

    def walk(s: object | None) -> None:
        if s is None or any(s is x for x in seen):
            return
        seen.append(s)
        fan = getattr(s, "sinks", None)
        if fan is not None:
            for child in fan:
                walk(child)
            return
        kind = type(s).__name__
        if kind == "BenchmarkSink" or hasattr(s, "_ep_rewinds"):
            out.append(BenchmarkSinkProvider(s))
        elif kind == "CollectSink" or (hasattr(s, "ep_bufs")
                                       and getattr(s, "ep_bufs", None)
                                       is not None):
            out.append(CollectSinkProvider(s))

    walk(sink)
    if (initializer is not None
            and callable(getattr(initializer, "snapshot_state", None))
            and callable(getattr(initializer, "restore_state", None))):
        out.append(InitializerStateProvider(initializer))
    return out


def _copy_slice(s: dict[str, SnapshotValue]) -> dict[str, SnapshotValue]:
    """One buffered step slice, deep-copied.

    numpy/tensor free per ``slice_step_for_env`` -- cpu clones; numpy
    copied, the rest passed by value.
    """
    out: dict[str, SnapshotValue] = {}
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
