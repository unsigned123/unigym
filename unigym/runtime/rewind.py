"""Rewind face: the snapshot/rewind/snapshots command handlers (Round B).

The time-travel superpower (design v052-round-b §1): a step boundary
snapshot of the WHOLE run state, and a restore that forks the timeline
back to it.  Everything the next ``adapter.act`` consumes is covered:

* engine  -- the backend's ``snapshot()``/``restore()`` (mujoco:
  mjSTATE_INTEGRATION full bits, warm-start re-pinned after the honest
  ``mj_forward``) plus the obs cache / obs frame / command / torch RNG of
  the snapshot boundary -- the exact iteration state the engine loop
  carries between ticks;
* adapter -- ``snapshot_internal()``/``restore_internal(state)``
  (SONIC state-history ring + mode/horizon pointers for agent_g1) and the
  ``on_rewind(restored_tick)`` resync hook (in-flight skill drop -> hold,
  EventCompiler seq rollback, TimeAccount/TimeControl resync, ledger
  rewind marker);
* providers -- :class:`RewindStateProvider` facilities (design
  v057-rewind-v2 §1): each provider's ``rewind_state()`` joins the SAME
  checkpoint payload (one freeze point, one package) and is applied
  after the adapter during the restore; any failure aborts the whole
  rewind (fail-fast, the safety net rolls engine + adapter + providers
  back).  The abandoned branch's provider buffers are dumped under
  ``<out_dir>/attempts/`` BEFORE the restore (dump, don't discard).

Pre-flight two-phase (design v063-rewind-preflight): the whole refusal
surface -- the controller-level cross-episode
guard, per-provider state presence, per-provider ``validate_rewind_state``
(the SAME judgment its apply enforces) -- runs READ-ONLY before the
sidecar dump / safety capture / restore, so a rejected rewind is a
ZERO-CHANGE rejection.  The safety net captures and rolls the adapter
too (env -> adapter -> providers), and its per-provider rollbacks are
individually guarded so a rollback failure can never mask the original
error.

Counter dichotomy (contract §4): engine tick / TimeAccount /
event seq ROLL BACK with the rewind; hb_seq / frame fid / episode_id are
process truth and NEVER rewind (the bridge heartbeat keeps bumping through
the park, frames keep their counter, the backend's episode_id sits outside
the mjSTATE payload on purpose).

Two-level capability gate (§1.4, fail-fast ``{error, hint}``): the backend
must declare the SNAPSHOT trait AND the bound adapter must declare
``rewindable = True`` with an ``on_rewind`` — explicit class attribute,
never hasattr.  Snapshot slots are an engine-side LRU ring capped at 8:
admitting a ninth evicts the OLDEST and the eviction is surfaced
explicitly in the ack (never silent); rewinding to an evicted id is an
explicit error.  v1 is memory-only (restart loses every snapshot — honest
note, no persistence).  The checkpoint ring: the slot cap is 16 (the
freeze paradigm spends slots — every freeze point auto-snapshots) and
each snapshot lands one append-only ``checkpoint`` line in the adapter
ledger (the undo anchor trail).

Commands land on the engine thread through the bridge command slot
(req-ack, pause/step family); a rewind auto-pauses.
Threading: everything here runs on the engine thread (command dispatch or
the parked service loop) — the same single-thread discipline as the rest
of the bridge.
"""
from __future__ import annotations

import collections
import copy
import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import (TYPE_CHECKING, Any, Protocol, TypeAlias, TypedDict,
                    runtime_checkable)

import numpy as np

from unigym._typing import JsonValue
from unigym.runtime.bridge import AckResult, RewindInfo

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from unigym.adapters.base import Adapter
    from unigym.runtime.bridge import Bridge

MAX_SNAPSHOTS = 64

#: Opaque engine / adapter / provider snapshot payloads: the runtime stores
#: and hands them back, never introspects them (arbitrary by construction).
SnapshotValue: TypeAlias = Any

#: The runner-loop seam is duck-typed (Runner protocol | SyncRunner); the
#: controller reads replay/sampler and jumps the loop via setattr.
RewindEngine: TypeAlias = Any


class RewindPayload(TypedDict):
    """One checkpoint's packaged run state (engine + adapter + providers)."""

    tick: int
    env: SnapshotValue
    obs_cache: dict[str, SnapshotValue]
    adapter: SnapshotValue
    providers: dict[str, SnapshotValue]
    episode_id: int | None
    obs_frame: SnapshotValue
    cmd: SnapshotValue
    torch_rng: SnapshotValue


class RewindRecord(TypedDict):
    """One live snapshot slot record."""

    snap_id: int
    tick: int
    label: str | None
    ts: float
    payload: RewindPayload


class _SnapshotBackend(Protocol):
    """The engine snapshot/restore face the rewind controller requires."""

    def snapshot(self) -> SnapshotValue: ...

    def restore(self, state: SnapshotValue) -> None: ...

__all__ = ["RewindStateProvider", "RewindController", "RewindGateError",
           "attach", "MAX_SNAPSHOTS"]


@runtime_checkable
class RewindStateProvider(Protocol):
    """One facility's rewindable state (design v057-rewind-v2 §1.1).

    The checkpoint ring snapshots the ENGINE + ADAPTER state; every other
    facility that accumulates state a restored timeline must agree with
    (the benchmark sink's open episode buffers, the collect sink's) joins
    through a provider: ``rewind_state()`` is captured into the SAME
    checkpoint payload as the engine snapshot (one freeze point, one
    package -- cross-facility consistency by construction), and
    ``apply_rewind_state`` runs during the restore, after the adapter's
    internal restore, BEFORE the engine boundary is armed.  ANY provider
    failure aborts the whole rewind (fail-fast, never a partial restore
    -- the safety net rolls the engine/adapter/providers back).

    Counter dichotomy carries over per facility (contract §4.1): process
    truth counters (episode ids, scored-episode counts, step totals) are
    NOT part of ``rewind_state`` and keep marching monotonically across
    rewinds.
    """

    name: str          # "benchmark" / "episode_runner" (payload key)

    def rewind_state(self) -> SnapshotValue:
        """The facility's necessary state at the freeze point."""

    def apply_rewind_state(self, state: SnapshotValue) -> None:
        """Restore to the snapshotted state (in place, same objects)."""

    def validate_rewind_state(self, state: SnapshotValue) -> None:
        """Pre-flight half of the two-phase rewind.

        Design v063-rewind-preflight: refuse a
        rewind this provider could not apply, READ-ONLY, before the
        controller touches the world -- a refusal here is a zero-change
        rejection (no sidecar dump, no safety capture, no restore).
        MUST embody the same judgment ``apply_rewind_state`` enforces
        (single source: the refusal logic lives in one place per
        provider and both phases call it).  Default: pass -- providers
        without extra refusal faces have nothing to pre-check; the
        controller duck-types the method, so protocol doubles without
        it keep working.  Refusals raise :class:`RewindGateError`.
        """
        return None


class RewindGateError(RuntimeError):
    """A rewind/snapshot request the run cannot honour.

    Two-level gate failure, unknown/evicted snap id, exhausted slots.
    Raised with a ``.hint`` so the ack carries ``{error, hint}``
    (director_contract §6 style).
    """

    def __init__(self, error: str, hint: str = ""):
        super().__init__(error)
        self.error = error
        self.hint = hint or "(none)"


class RewindController:
    """Snapshot/rewind handler trio mounted on one live bridge.

    Args:
        bridge: the run's Bridge (command table + gate + boundary slot).
        env: EnvBackend (must implement ``snapshot()``/``restore()``).
        adapter: Adapter (``rewindable`` + ``on_rewind`` required by
            the gate; ``snapshot_internal``/``restore_internal`` used when
            present).
        engine: SyncRunner (loop counter + iteration-state seam).
        backend_label: canonical backend name (registry trait lookup).
        providers: optional :class:`RewindStateProvider` list --
            each one's ``rewind_state()`` joins every checkpoint payload
            under its ``name`` and is applied during the restore, after
            the adapter (fail-fast: any failure aborts the whole rewind).
    """

    def __init__(self, bridge: Bridge, env: _SnapshotBackend,
                 adapter: Adapter, engine: RewindEngine,
                 backend_label: str,
                 providers: Sequence[RewindStateProvider] | None = None):
        self.bridge = bridge
        self.env = env
        self.adapter = adapter
        self.engine = engine
        self.backend_label = str(backend_label)
        self.providers: list[RewindStateProvider] = list(providers or [])
        names = [p.name for p in self.providers]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate rewind provider names: {names}")
        self._snaps: collections.OrderedDict[int, RewindRecord] = \
            collections.OrderedDict()
        self._next_id = 1
        bridge.register_command("snapshot", self.cmd_snapshot)
        bridge.register_command("rewind", self.cmd_rewind)
        bridge.register_command("snapshots", self.cmd_snapshots)

    # ---------------------------------------------------------------- #
    # two-level capability gate (§1.4)
    # ---------------------------------------------------------------- #
    def _gate(self) -> None:
        from ..backends.registry import Trait, backend_info

        info = backend_info(self.backend_label)
        if not info.has(Trait.SNAPSHOT):
            raise RewindGateError(
                f"backend '{self.backend_label}' does not implement the "
                f"snapshot capability (traits: "
                f"{sorted(t.value for t in info.traits)})",
                hint="rewind is mujoco-only in Round B -- run env.backend: "
                     "mujoco (isaaclab lands it with the isaaclab rewind "
                     "phase)")
        ad = self.adapter
        if (ad is None or not getattr(ad, "rewindable", False)
                or not callable(getattr(ad, "on_rewind", None))):
            name = type(ad).__name__ if ad is not None else "<none>"
            raise RewindGateError(
                f"adapter {name} is not rewindable (rewindable="
                f"{getattr(ad, 'rewindable', None)!r})",
                hint="rewind requires rewindable=True + on_rewind on the "
                     "bound adapter (agent_g1 implements it; the explicit "
                     "class attribute is the contract, not hasattr)")
        if getattr(self.engine, "replay", None) is not None:
            raise RewindGateError(
                "replay runs are not rewindable (trace injection vs a "
                "forked timeline have no defined composition)",
                hint="drop --replay or start a fresh run for the rewind "
                     "face")
        if getattr(self.engine, "sampler", None) is not None:
            # the gate's true invariant: no unsnapshotted sampler STATE
            # (a random resampler's RNG/timers live outside the
            # snapshot).  A sampler that declares itself rewind_inert
            # (explicit class attribute, the rewindable discipline) is
            # stateless by construction -- the checkpoint ring
            # needs it: agent_g1 session configs carry an IGNORED fixed
            # command sampler (the adapter never reads it) and would
            # otherwise lose every checkpoint.
            sampler: object | None = self.engine.sampler
            if not getattr(sampler, "rewind_inert", False):
                raise RewindGateError(
                    "sampler-based runs are not rewindable in v1 "
                    "(command resampler state lives outside the "
                    "snapshot)",
                    hint="self-commanding adapters (agent_g1, sonic_g1) "
                         "run without a sampler (or with an explicitly "
                         "rewind_inert one, e.g. fixed); rewind needs "
                         "one of those")

    # ---------------------------------------------------------------- #
    # command handlers (engine thread; errors land in the ack)
    # ---------------------------------------------------------------- #
    def cmd_snapshot(self, rid: int,
                     params: dict[str, JsonValue]) -> AckResult:
        """Cmd "snapshot": capture engine+adapter+providers into the ring.

        Raises:
            RewindGateError: Gate failure or an invalid label.
        """
        self._gate()
        label = params.get("label")
        if label is not None and not str(label).strip():
            raise RewindGateError("snapshot: 'label' must be a non-empty "
                                  "string when given")
        bundle = self.bridge._bundle or {}
        tick = int(getattr(self.bridge, "_last_step", 0) or 0)
        cache: Mapping[str, SnapshotValue] = getattr(
            self.env, "obs_cache", None) or {}
        payload: RewindPayload = {
            "tick": tick,
            "env": self.env.snapshot(),
            # obs cache: passthrough keys ("command"/"prev_action") survive
            # across steps by design, so the cache is part of the run state
            # the next compute_obs consumes -- branch-A writes must not leak
            "obs_cache": {k: _snap_val(v) for k, v in cache.items()},
            "adapter": (self.adapter.snapshot_internal()
                        if callable(getattr(self.adapter,
                                            "snapshot_internal", None))
                        else None),
            # provider states: the SAME freeze point packages the
            # engine + adapter + every facility's necessary state (the
            # consistency guarantee is constructional -- one package)
            "providers": {p.name: p.rewind_state()
                          for p in self.providers},
            # pre-flight face: the backend's episode counter at the
            # freeze point (env 0; process truth per contract §4.1 --
            # never rewinds).  The rewind pre-flight compares it against
            # the live bundle: the controller-level cross-episode guard
            # closes the bufferless blind spot (a plain collect run's
            # provider stamps are all None).  None (key absent -- older
            # payloads / doubles without ``extra``) degrades the guard
            # honestly, never false-kills.
            "episode_id": _episode_stamp(bundle),
            "obs_frame": _snap_val(bundle.get("obs_frame")),
            "cmd": _snap_val(bundle.get("cmd")),
            "torch_rng": _torch_rng_state(),
        }
        import time as _time
        rec: RewindRecord = {
            "snap_id": self._next_id, "tick": tick,
            "label": (str(label).strip() if label else None),
            "ts": _time.time(), "payload": payload}
        self._next_id += 1
        result: AckResult = {"snap_id": rec["snap_id"], "tick": tick,
                             "label": rec["label"]}
        if len(self._snaps) >= MAX_SNAPSHOTS:
            _, evicted = self._snaps.popitem(last=False)   # LRU: oldest
            evicted_doc: dict[str, JsonValue] = {
                "snap_id": evicted["snap_id"],
                "tick": evicted["tick"],
                "label": evicted["label"]}
            result["evicted"] = evicted_doc
            print(f"[rewind] LRU evicted snapshot "
                  f"{evicted['snap_id']} (tick {evicted['tick']}) -- slot "
                  f"cap {MAX_SNAPSHOTS} reached; rewinding to it now "
                  f"errors explicitly", flush=True)
        self._snaps[rec["snap_id"]] = rec
        # checkpoint ring: every snapshot is a freeze point's undo
        # anchor -- one append-only ledger line (the ledger lives on the
        # adapter; absent = bookkeeping-only mount, honest no-op)
        ledger = getattr(self.adapter, "_ledger", None)
        if ledger is not None:
            ledger.append("checkpoint",
                          {"label": rec["label"],
                           "snap_id": rec["snap_id"]},
                          {"status": "captured",
                           "message": (f"snapshot {rec['snap_id']} @ tick "
                                       f"{tick}"
                                       + (f" (evicted "
                                          f"{result['evicted']['snap_id']})"
                                          if "evicted" in result else ""))},
                          tick)
        return result

    def cmd_snapshots(self, rid: int,
                      params: dict[str, JsonValue]) -> AckResult:
        """Cmd "snapshots": the live slot list + the LRU cap."""
        self._gate()
        return {"snapshots": [{"snap_id": r["snap_id"], "tick": r["tick"],
                               "label": r["label"], "ts": r["ts"]}
                              for r in self._snaps.values()],
                "cap": MAX_SNAPSHOTS}

    def cmd_rewind(self, rid: int,
                   params: dict[str, JsonValue]) -> AckResult:
        """Cmd "rewind": fork the timeline back to a snapshot.

        Dumps the abandoned branch's provider buffers first, restores
        engine -> adapter -> providers (any failure rolls the pre-rewind
        state back and re-raises), then auto-pauses.

        Raises:
            RewindGateError: Gate failure / bad snap_id / unknown or
                evicted id.
        """
        self._gate()
        sid = params.get("snap_id")
        if isinstance(sid, bool) or not isinstance(sid, int):
            raise RewindGateError(
                f"rewind: 'snap_id' must be an integer, got {sid!r}",
                hint="GET snapshots (or `unigym snapshots <run>`) lists the "
                     "live snapshot slots")
        rec = self._snaps.get(sid)
        if rec is None:
            raise RewindGateError(
                f"rewind: unknown snap_id {sid}",
                hint=f"live snapshots: "
                     f"{[r['snap_id'] for r in self._snaps.values()]} "
                     f"(LRU cap {MAX_SNAPSHOTS}; evicted ids error "
                     f"explicitly; v1 snapshots are memory-only)")
        payload = rec["payload"]
        restored_tick = int(payload["tick"])

        # pre-flight two-phase (design v063-rewind-preflight): the WHOLE
        # refusal surface runs read-only
        # BEFORE the sidecar dump, the safety capture or any restore --
        # a rejected rewind is a ZERO-CHANGE rejection.  F4's root was
        # structural: the guards lived on the provider apply face, so
        # the collect provider's restore_commit had already truncated
        # the shard dir on disk by the time a later provider refused.
        self._preflight_rewind(sid, payload)

        # sidecar (design §1.4, "dump, don't discard"): BEFORE any
        # restore, the providers' current buffers land under
        # <out_dir>/attempts/ -- the abandoned branch's episode frames
        # are data, never waste.  A failed dump aborts the rewind
        # (pre-restore, so the world is untouched -- the no-data-loss
        # ruling is hard, not best-effort).
        self._dump_attempts(sid, restored_tick, rec.get("label"))

        # safety net: capture the pre-rewind engine state so a mid-flight
        # failure never leaves the run half-restored (the providers'
        # live states ride the same net -- a provider failing mid-apply
        # rolls the earlier ones back, never a partial restore)
        bundle = self.bridge._bundle or {}
        cache: Mapping[str, SnapshotValue] = getattr(
            self.env, "obs_cache", None) or {}
        safety: dict[str, SnapshotValue] = {
            "env": self.env.snapshot(),
            "obs_cache": {k: _snap_val(v) for k, v in cache.items()},
            "obs_frame": _snap_val(bundle.get("obs_frame")),
            "cmd": _snap_val(bundle.get("cmd")),
            "torch_rng": _torch_rng_state(),
            "tick": int(getattr(self.bridge, "_last_step", 0) or 0),
            # Adapter in the safety net: it rides the net too -- the
            # docstring always claimed "engine + adapter + providers";
            # the capture was missing the adapter
            "adapter": (self.adapter.snapshot_internal()
                        if callable(getattr(self.adapter,
                                            "snapshot_internal", None))
                        else None),
            "providers": {p.name: p.rewind_state()
                          for p in self.providers}}
        applied: list[str] = []
        try:
            self.env.restore(payload["env"])
            live_cache: dict[str, SnapshotValue] | None = getattr(
                self.env, "obs_cache", None)
            if live_cache is not None:
                live_cache.clear()
                live_cache.update({k: _snap_val(v)
                                   for k, v in payload["obs_cache"].items()})
            if payload.get("adapter") is not None and callable(
                    getattr(self.adapter, "restore_internal", None)):
                self.adapter.restore_internal(payload["adapter"])
            # restore order: engine -> adapter -> providers (design
            # §1.2); any provider failure raises through -- the except
            # below rolls EVERYTHING back (engine + adapter + providers).
            # State presence was pre-flighted above; each
            # provider's semantic guards stay on the apply face as the
            # second line (same judgment its validate runs).
            for p in self.providers:
                p.apply_rewind_state(payload["providers"][p.name])
                applied.append(p.name)
        except Exception:
            # catch-all baseline site (docs/audits/2026-10-12-except-audit.md):
            # the restore path spans engine + adapter + torch + providers,
            # so ANY failure type can leave the run half-restored -- the
            # safety net rolls the pre-rewind state back and RE-RAISES
            # (never swallows)
            self.env.restore(safety["env"])
            sc = getattr(self.env, "obs_cache", None)
            if sc is not None:
                sc.clear()
                sc.update({k: _snap_val(v)
                           for k, v in safety["obs_cache"].items()})
            _set_torch_rng(safety["torch_rng"])
            # Adapter rollback: restore order env -> adapter ->
            # providers mirroring the apply order -- a refused rewind no
            # longer leaves the adapter cursor at the checkpoint value
            # while the world marched on (the live misalignment)
            sad = safety.get("adapter")
            if sad is not None and callable(getattr(
                    self.adapter, "restore_internal", None)):
                try:
                    self.adapter.restore_internal(sad)
                except (ValueError, RuntimeError, OSError, KeyError,
                        TypeError, AttributeError) as rb:
                    print(f"[rewind] safety-net adapter rollback failed "
                          f"({type(rb).__name__}: {rb}) -- continuing; "
                          f"the original error below is authoritative",
                          flush=True)
            # Except-hardening: each provider rollback is
            # individually guarded with NARROWED types (no new catch-all
            # -- the baseline stays at 32); a rollback failure is
            # logged and must NEVER mask the original exception (the
            # Live case: the rollback's restore_commit ValueError
            # buried the cross-episode refusal)
            for p in self.providers:
                pre = safety["providers"].get(p.name)
                if pre is None:
                    continue
                try:
                    p.apply_rewind_state(pre)
                except (ValueError, RuntimeError, OSError, KeyError,
                        TypeError, AttributeError) as rb:
                    print(f"[rewind] safety-net rollback of provider "
                          f"'{p.name}' failed ({type(rb).__name__}: "
                          f"{rb}) -- continuing; the original error "
                          f"below is authoritative", flush=True)
            raise
        del safety
        # torch RNG: the abandoned branch's consumption must not leak into
        # the restored timeline (episode-duration sampling is the consumer)
        _set_torch_rng(payload["torch_rng"])
        # adapter resync hook AFTER the internal restore (it may consult
        # the restored state); in-flight skill drop -> hold lives here
        self.adapter.on_rewind(restored_tick)
        # engine seam: loop counter + the snapshot boundary's iteration
        # state (obs frame / command) -- adopted at the RewindBoundary
        self.engine._tick = restored_tick + 1
        self.engine._rewind_state = {"obs_frame": payload["obs_frame"],
                                     "cmd": payload["cmd"]}
        # process-truth record + one-shot boundary + auto-pause (§5-③)
        info: RewindInfo = {
            "tick": int(getattr(self.bridge, "_last_step", 0) or 0),
            "restored_tick": restored_tick, "snap_id": int(sid),
            "label": rec["label"]}
        self.bridge.rewind_log.append(info)
        self.bridge._rewind_next_tick = restored_tick + 1
        self.bridge._rewind_info = info
        self.bridge.pause_for_rewind()
        return {"restored_tick": restored_tick, "snap_id": int(sid),
                "paused": True,
                "providers_applied": applied,
                "attempts": getattr(self, "_last_attempt_paths", None),
                "hint": f"world restored to tick {restored_tick} and "
                        f"paused; resume/step to continue on the forked "
                        f"timeline"}

    # ---------------------------------------------------------------- #
    # pre-flight two-phase: zero-change rejection (design
    # v063-rewind-preflight)
    # ---------------------------------------------------------------- #
    def _preflight_rewind(self, sid: int, payload: RewindPayload) -> None:
        """The refusal surface, READ-ONLY, before dump/capture/restore.

        ① controller-level episode guard -- the snapshot's backend
        episode counter vs the live bundle's (both process truth,
        contract §4.1).  This closes the bufferless blind spot: a plain
        collect run's provider stamps are all None, so the provider
        face alone was structurally blind on exactly the repro
        shape.  Degradation: either side missing the key (older
        payloads, doubles without ``extra``) skips the guard, never
        false-kills -- the providers' stamp guards remain the second
        line.  ② every provider's payload state must exist (the
        mid-apply missing-state refusal, moved up here) and pass its
        ``validate_rewind_state`` -- the same judgment its apply
        enforces (single source).  Any refusal raises
        :class:`RewindGateError` with the world untouched.
        """
        bundle = self.bridge._bundle or {}
        snap_ep = payload.get("episode_id")
        live_ep = _episode_stamp(bundle)
        if (snap_ep is not None and live_ep is not None
                and int(snap_ep) != int(live_ep)):
            raise RewindGateError(
                f"cross-episode rewind refused (snapshot episode "
                f"{int(snap_ep)} != live episode {int(live_ep)})",
                hint="the checkpoint belongs to a different episode "
                     "than the live run -- cross-set rewind is out of "
                     "the v1 scope (design v057 §0); take a fresh "
                     "checkpoint in the live episode and rewind to it")
        for p in self.providers:
            state = (payload.get("providers") or {}).get(p.name)
            if state is None:
                raise RewindGateError(
                    f"provider '{p.name}' has no state in snapshot "
                    f"{sid}",
                    hint="the snapshot predates the provider mount or "
                         "the provider failed at capture; rewind "
                         "refuses rather than partially restore")
            fn = getattr(p, "validate_rewind_state", None)
            if callable(fn):
                fn(state)

    # ---------------------------------------------------------------- #
    # sidecar: the abandoned branch's buffers are data (§1.4)
    # ---------------------------------------------------------------- #
    def _dump_attempts(self, sid: int, restored_tick: int,
                       label: str | None) -> None:
        """Dump every provider's current buffers to the attempts sidecar.

        ``<out_dir>/attempts/rewind_{n}_ep{E}_tick{T}.npz`` + a meta json
        (span + the reason this branch is being abandoned).  Any dump
        failure aborts the rewind BEFORE the restore (fail-fast -- the
        no-data-loss ruling is not best-effort).
        """
        self._last_attempt_paths = []
        if not self.providers:
            return
        out_dir = getattr(self.bridge, "out_dir", None)
        if not out_dir:
            return                      # bookkeeping-only mount
        attempts = os.path.join(str(out_dir), "attempts")
        n = len(getattr(self.bridge, "rewind_log", [])) + 1
        tick = int(getattr(self.bridge, "_last_step", 0) or 0)
        for p in self.providers:
            fn = getattr(p, "dump_attempt", None)
            if not callable(fn):
                continue
            dump: Callable[[str, str], list[str]] = getattr(
                p, "dump_attempt")
            ep = getattr(p, "episode_hint", lambda: 0)()
            base = f"rewind_{n}_ep{int(ep)}_tick{tick}_{p.name}"
            os.makedirs(attempts, exist_ok=True)
            paths = dump(attempts, base)
            meta: dict[str, SnapshotValue] = {
                "rewind_ordinal": n, "snap_id": int(sid),
                "label": label, "provider": p.name,
                "tick": tick, "restored_tick": int(restored_tick),
                "reason": ("timeline forked back to the checkpoint "
                           "state; this branch's frames are kept "
                           "here (design v057 §1.4)"),
                "files": list(paths)}
            meta_path = os.path.join(attempts, base + ".meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=1, ensure_ascii=False)
            self._last_attempt_paths.append(meta_path)
        print(f"[rewind] attempt sidecar(s): {self._last_attempt_paths}",
              flush=True)


def attach(bridge: Bridge, env: _SnapshotBackend, adapter: Adapter,
           engine: RewindEngine, backend_label: str,
           providers: Sequence[RewindStateProvider] | None = None
           ) -> RewindController:
    """Mount the rewind face on a live bridge (collect_cli wiring)."""
    return RewindController(bridge, env=env, adapter=adapter, engine=engine,
                            backend_label=backend_label,
                            providers=providers)


# --------------------------------------------------------------------------- #
# helpers (torch kept lazy so the stdlib-only bridge import path stays clean)
# --------------------------------------------------------------------------- #
def _episode_stamp(bundle: Mapping[str, JsonValue] | None) -> int | None:
    """The bundle's backend episode counter, env 0 (pre-flight face).

    ``episode_id`` is process truth (contract §4.1 -- never rewinds),
    so snapshot-vs-live equality is exactly the same-episode test.  An
    absent key (older payloads, doubles without ``extra``) answers
    None: the controller guard degrades to skip, never false-kills.
    """
    extra = (bundle or {}).get("extra")
    eid = extra.get("episode_id") if isinstance(extra, Mapping) else None
    if eid is None:
        return None
    try:
        return int(np.asarray(eid).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None


def _clone(x: SnapshotValue) -> SnapshotValue:
    """Detach-clone a tensor iteration-state entry (None passes through)."""
    if x is None:
        return None
    return x.detach().clone()


def _snap_val(x: SnapshotValue) -> SnapshotValue:
    """Snapshot-safe copy of one obs_cache / obs-frame value.

    The cache is tensor-first (the engine's passthrough keys), but the
    The decision-package face legitimately parks structured payloads
    in it too (``contact_ops/<name>`` -> ``list[dict]``).  Tensors take
    the detach-clone path; numpy copies; JSON-safe structures deepcopy
    (snapshot AND restore must hand out private copies -- a rewind
    re-run or a second restore of the same payload must never alias
    the archived state.
    """
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().clone()
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, (dict, list, tuple)):
        return copy.deepcopy(x)
    return x


def _torch_rng_state() -> SnapshotValue:
    try:
        import torch
        return torch.get_rng_state().clone()
    except ImportError:                       # pragma: no cover
        return None


def _set_torch_rng(state: SnapshotValue) -> None:
    if state is None:
        return
    import torch
    torch.set_rng_state(state)
