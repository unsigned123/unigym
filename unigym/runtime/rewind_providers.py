"""Rewind state providers.

The sink facilities whose forked-timeline state a restored timeline must
agree with.

A rewind that restores only the engine + adapter leaves the SINKS marching.
The fix is the provider protocol (``runtime/rewind.py``): each provider's
``rewind_state()`` joins every checkpoint payload and is applied during
the restore, after the adapter, fail-fast.

Providers (v1 coverage, design §1.3 table):

* ``CollectSinkProvider`` (name ``episode_runner``) -- the collect
  writer's SHARD RECORDING CURSOR (``shard_commit`` =
  ``{committed_shards, committed_rows}`` off the sink's
  :class:`~unigym.io.writer.ShardWriter`) plus its ``n_episodes`` fuse
  counter, which rides the fork so the restored timeline re-earns its
  episode completions.  The restore deletes the
  post-checkpoint shard files, truncates the boundary shard back to the
  checkpoint's rows and re-inflates the writer's row buffer from it --
  the shard directory stays the WINNING timeline (monotonic (episode,
  tick), no abandoned-branch rows), so a run that contains undos replays
  bitwise end to end.  Runs recorded BEFORE that design mixed the
  abandoned branch's rows into the shard at every undo point -- replay
  of those is a qualitative reference only (runtime_contract §4.2).
* ``InitializerStateProvider`` -- the active initializer's sampling state.
* ``RunnerLifecycleProvider`` -- the runner's per-env open/closed lifecycle.

``CollectSinkProvider`` also implements ``dump_attempt(dir, base)`` -- the
§1.4 sidecar, run before a rewind applies.  A collect run keeps no episode
buffers, so it writes nothing (an empty result); the abandoned branch's
shard files are handled by its own restore path.

pre-flight:
each provider implements ``validate_rewind_state(state)`` -- the SAME
refusal judgment its ``apply_rewind_state`` enforces, run read-only
by the controller BEFORE the sidecar dump / safety capture / restore
(single source: ``validate_commit_state`` in ``unigym.io.writer`` for the
collect cursor).  A refused rewind is a zero-change rejection;
the apply-face guards remain as the second line.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, TypeGuard

import numpy as np

from unigym.io.writer import validate_commit_state
from unigym.runtime.rewind import RewindGateError, RewindStateProvider, SnapshotValue

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from unigym.io.writer import ShardWriter

__all__ = [
    "CollectSinkProvider",
    "InitializerStateProvider",
    "RunnerLifecycleProvider",
    "build_rewind_providers",
]


class _ShardWriterFace(Protocol):
    """The shard recording-cursor face the collect provider restores."""

    def commit_state(self) -> SnapshotValue: ...

    def restore_commit(self, state: SnapshotValue) -> None: ...


class _RunnerLifecycleFace(Protocol):
    """The runner-local lifecycle face the lifecycle provider restores.

    The runner owns this state; the provider is the checkpoint owner for it.
    ``_tick`` / ``_cmd`` / ``_obs_frame`` and the engine stay with the rewind
    controller.
    """

    _closed: list[bool]
    _episode_step: list[int]
    _reset_samples: list[dict[str, np.ndarray] | None]
    _reset_valid: list[bool | None]
    _reset_reason: list[str | None]
    _started: bool


class RunnerLifecycleProvider:
    """The runner's per-env open/closed lifecycle in the rewind package.

    Restores only runner-local lifecycle: the closed mask, episode-local step
    counters, reset samples/validity/reason and the started flag.  It never
    replays ``on_reset``; sink/pack state is restored by its own provider.
    """

    name = "runner_lifecycle"

    def __init__(self, runner: _RunnerLifecycleFace):
        self.runner = runner

    def rewind_state(self) -> dict[str, SnapshotValue]:
        """Return private copies of the full runner-local lifecycle state."""
        runner = self.runner
        return {
            "closed": list(runner._closed),
            "episode_step": list(runner._episode_step),
            "reset_samples": _copy_reset_samples(runner._reset_samples),
            "reset_valid": list(runner._reset_valid),
            "reset_reason": list(runner._reset_reason),
            "started": bool(runner._started),
        }

    def validate_rewind_state(self, state: dict[str, SnapshotValue]) -> None:
        """Refuse a malformed lifecycle payload before changing any state."""
        self._guard_state(state)

    def apply_rewind_state(self, state: dict[str, SnapshotValue]) -> None:
        """Write the snapshot back in place (list identity preserved)."""
        self._guard_state(state)
        runner = self.runner
        runner._closed[:] = [bool(value) for value in state["closed"]]
        runner._episode_step[:] = [int(value) for value in state["episode_step"]]
        runner._reset_samples[:] = _copy_reset_samples(state["reset_samples"])
        runner._reset_valid[:] = [
            None if value is None else bool(value) for value in state["reset_valid"]
        ]
        runner._reset_reason[:] = [
            None if value is None else str(value) for value in state["reset_reason"]
        ]
        runner._started = bool(state["started"])

    def _guard_state(self, state: object) -> None:
        """The shared refusal judgment for validate and apply."""
        if not isinstance(state, Mapping):
            raise RewindGateError(
                f"rewind provider '{self.name}': expected a lifecycle mapping, "
                f"got {type(state).__name__}"
            )
        expected = {
            "closed",
            "episode_step",
            "reset_samples",
            "reset_valid",
            "reset_reason",
            "started",
        }
        missing = sorted(expected - set(state))
        if missing:
            raise RewindGateError(
                f"rewind provider '{self.name}': missing lifecycle keys {missing}"
            )
        num_envs = len(self.runner._episode_step)
        closed = state["closed"]
        episode_step = state["episode_step"]
        reset_samples = state["reset_samples"]
        reset_valid = state["reset_valid"]
        reset_reason = state["reset_reason"]
        for key in ("closed", "episode_step", "reset_samples", "reset_valid", "reset_reason"):
            value = state[key]
            if not isinstance(value, list | tuple):
                raise RewindGateError(
                    f"rewind provider '{self.name}': {key} must be a sequence, "
                    f"got {type(value).__name__}"
                )
            if len(value) != num_envs:
                raise RewindGateError(
                    f"rewind provider '{self.name}': {key} length {len(value)} "
                    f"!= runner num_envs {num_envs}"
                )
        if any(not isinstance(value, bool) for value in closed):
            raise RewindGateError(f"rewind provider '{self.name}': closed mask values must be bool")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in episode_step
        ):
            raise RewindGateError(
                f"rewind provider '{self.name}': episode_step values must be " f"non-negative ints"
            )
        if any(value is not None and not isinstance(value, bool) for value in reset_valid):
            raise RewindGateError(
                f"rewind provider '{self.name}': reset_valid values must be bool or None"
            )
        if any(value is not None and not isinstance(value, str) for value in reset_reason):
            raise RewindGateError(
                f"rewind provider '{self.name}': reset_reason values must be str or None"
            )
        if any(value is not None and not isinstance(value, Mapping) for value in reset_samples):
            raise RewindGateError(
                f"rewind provider '{self.name}': reset_samples values must be " f"mappings or None"
            )
        if not isinstance(state["started"], bool):
            raise RewindGateError(f"rewind provider '{self.name}': started must be a bool")


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


class CollectSinkProvider:
    """The collect writer's shard cursor + episode fuse (name ``episode_runner``).

    A collect run keeps no episode buffers: the provider owns the shard
    recording cursor (``ShardWriter.commit_state``, two integers) and the
    ``n_episodes`` episode-fuse counter.  Its restore truncates the shard
    directory back to the committed winning timeline.
    """

    name = "episode_runner"

    def __init__(self, sink: object):
        self.sink = sink
        self.num_envs = int(getattr(sink, "num_envs", 1))

    def _fuse(self) -> dict[str, SnapshotValue]:
        """The facility's episode-fuse counters.

        They ride the fork so the restored timeline re-earns its episode
        completions.  Empty when the sink does not carry the counter.
        """
        return (
            {"n_episodes": int(getattr(self.sink, "n_episodes", 0))}
            if hasattr(self.sink, "n_episodes")
            else {}
        )

    def _writer(self) -> ShardWriter | None:
        """The sink's shard writer (absent on provider-protocol doubles).

        The shard face simply does not ride those.
        """
        w = getattr(self.sink, "writer", None)
        return w if callable(getattr(w, "commit_state", None)) else None

    def rewind_state(self) -> dict[str, SnapshotValue]:
        """The fuse counter + the shard recording cursor when a writer exists."""
        state: dict[str, SnapshotValue] = {"fuse": self._fuse()}
        w = self._writer()
        if w is not None:
            state["shard_commit"] = w.commit_state()
        return state

    def validate_rewind_state(self, state: dict[str, SnapshotValue]) -> None:
        """Pre-flight the fuse payload and the shard commit state (read-only).

        The SAME ``validate_commit_state`` check restore_commit runs (single
        source: unigym.io.writer.validate_commit_state) so a corrupt or
        unreachable commit state is refused before the world is touched,
        wrapped in :class:`RewindGateError` ({error, hint}).

        Args:
            state: The snapshot payload (``shard_commit`` optional).

        Raises:
            RewindGateError: The payload or commit state fails validation.
        """
        if not isinstance(state, Mapping):
            raise RewindGateError(
                f"rewind provider '{self.name}': expected a mapping, " f"got {type(state).__name__}"
            )
        fuse = state.get("fuse")
        if fuse is not None and not isinstance(fuse, Mapping):
            raise RewindGateError(
                f"rewind provider '{self.name}': fuse must be a mapping, "
                f"got {type(fuse).__name__}"
            )
        w = self._writer()
        commit = state.get("shard_commit")
        if w is not None and commit is not None:
            try:
                validate_commit_state(commit, w)
            except ValueError as e:
                raise RewindGateError(
                    f"rewind provider '{self.name}': {e}",
                    hint="the snapshot's shard commit state is "
                    "corrupt or unreachable against the live "
                    "shard dir; rewind refuses rather than "
                    "truncate",
                ) from e

    def apply_rewind_state(self, state: dict[str, SnapshotValue]) -> None:
        """Restore the episode fuse and then the shard cursor.

        The fuse counter is the whole sink state; the shard cursor restore
        is plain file+cursor surgery -- a failure there raises through to
        the controller's safety net.
        """
        self.validate_rewind_state(state)
        fuse: Mapping[str, SnapshotValue] = state.get("fuse") or {}
        for k, v in fuse.items():
            if hasattr(self.sink, k):
                setattr(self.sink, k, v)
        w = self._writer()
        commit = state.get("shard_commit")
        if w is not None and commit is not None:
            w.restore_commit(commit)

    def dump_attempt(self, out_dir: str, base: str) -> list[str]:
        """A collect run keeps no episode buffers: nothing to dump."""
        del out_dir, base
        return []


def _provider_problem(candidate: object) -> str | None:
    """Describe why *candidate* is not a rewind provider (None when valid).

    The structural provider shape is a non-empty string ``name`` plus callable
    ``rewind_state`` / ``apply_rewind_state``; ``validate_rewind_state`` is
    accepted when present and must then be callable.
    """
    problems: list[str] = []
    name = getattr(candidate, "name", None)
    if not isinstance(name, str) or not name:
        problems.append("name must be a non-empty string")
    for method in ("rewind_state", "apply_rewind_state"):
        if not callable(getattr(candidate, method, None)):
            problems.append(f"{method} must be callable")
    validate = getattr(candidate, "validate_rewind_state", None)
    if validate is not None and not callable(validate):
        problems.append("validate_rewind_state must be callable when present")
    return "; ".join(problems) if problems else None


def _is_rewind_provider(candidate: object) -> TypeGuard[RewindStateProvider]:
    """Structural provider check shared by discovery and supplier validation."""
    return _provider_problem(candidate) is None


def build_rewind_providers(
    sink: object, initializer: object | None = None
) -> list[RewindStateProvider]:
    """One provider per provider-capable sink, discovered structurally.

    collect_cli wiring; a plain sink or None -> whatever applies.

    Discovery is capability-based, never class-name or attribute sniffing:

    1. a node that itself satisfies the provider protocol is the provider;
    2. otherwise a ``rewind_provider`` attribute/method supplies one; a
       declared supplier that does not yield a valid provider fails fast;
    3. otherwise ``sinks`` children are walked.

    The walk RECURSES into nested fan-outs -- production wraps the sink chain
    (``FanOutSink(FanOutSink(collect, timeline), ...)`` etc.) -- depth-first in
    construction order, identity-deduped against diamond re-visits.  Provider
    names are checkpoint payload keys, so duplicates fail at build time; the
    rewind controller keeps its own guard as the second line.
    """
    out: list[RewindStateProvider] = []
    seen: list[object] = []

    def walk(node: object | None) -> None:
        if node is None or any(node is x for x in seen):
            return
        seen.append(node)
        if _is_rewind_provider(node):
            out.append(node)
            return
        supplier = getattr(node, "rewind_provider", None)
        if supplier is not None:
            supplied = supplier() if callable(supplier) else supplier
            if not _is_rewind_provider(supplied):
                problem = _provider_problem(supplied) or "not a provider"
                raise TypeError(
                    f"{type(node).__name__}.rewind_provider must supply a "
                    f"rewind provider (non-empty 'name' plus callable "
                    f"'rewind_state'/'apply_rewind_state'); got "
                    f"{type(supplied).__name__}: {problem}"
                )
            out.append(supplied)
            return
        children = getattr(node, "sinks", None)
        if children is not None:
            for child in children:
                walk(child)

    walk(sink)
    if (
        initializer is not None
        and callable(getattr(initializer, "snapshot_state", None))
        and callable(getattr(initializer, "restore_state", None))
    ):
        out.append(InitializerStateProvider(initializer))
    names = [provider.name for provider in out]
    if len(set(names)) != len(names):
        raise ValueError(
            f"duplicate rewind provider names: {names}; provider names are "
            "checkpoint payload keys and must be unique"
        )
    return out


def _copy_reset_samples(
    samples: list[dict[str, np.ndarray] | None],
) -> list[dict[str, np.ndarray] | None]:
    """Deep-copy per-env reset samples so the checkpoint never aliases live state."""
    return [
        None
        if sample is None
        else {str(key): copy.deepcopy(value) for key, value in sample.items()}
        for sample in samples
    ]
