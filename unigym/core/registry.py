"""core.registry -- the run instance view (registry v2).

docs/designs/v027-media-gateway.md §2: state lives in the instances --
each run writes its own serve registry v2 (/tmp/unigym-serve-<id>.json,
unigym/runtime/bridge.py):

    {schema: 2, run_id, pid, shm, output_dir, log_path, start_ts,
     cameras[], camera, config, gpu, obs_catalog_keys}

The port/auth/webui_url/curl_base/tunnel_hint fields are RETIRED (runs have
no HTTP port and no token -- the daemon media gateway terminates all
browser traffic; breaking change).  This module builds
the merged, heartbeat-verified view on demand:

    SHM scan (/dev/shm/unigym_*)  x  serve registries  ->  [RunInfo]

Liveness = SHM heartbeat freshness (bridge.hb_alive) -- the pid is display
and kill-target only, never a liveness signal.

The daemon's instance table is a rebuildable cache of exactly this
function -- kill -9 of the daemon loses nothing
(docs/designs/v026-unigymd.md §3: startup
rebuilds the view; manually nohup'd runs are equally visible because
discovery is registry/SHM based, never parent-child).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from unigym.runtime import bridge as _bridge


@dataclass
class RunInfo:
    """One run instance, merged view (registry v2)."""

    run_id: str
    pid: int
    shm: str
    output_dir: str
    log_path: str = ""
    phase: str = "?"
    cameras: list[str] = field(default_factory=list)
    registry_path: str = ""
    alive: bool = False
    # -- display extras (registry keys optional; missing keys read as "") --
    config: str = ""
    camera: str = ""
    step: int = 0
    episodes: int = 0
    start_ts: float = 0.0
    gpu: str | None = None
    # read-failure text for UNKNOWN rows ("" otherwise) -- a
    # segment that exists but cannot be read is actionable state, not
    # absence (design v049 §0/§1)
    error: str = ""

    def to_dict(self) -> dict:
        """JSON-ready dict with a stable key order (API + CLI tables)."""
        return {
            "run_id": self.run_id, "pid": self.pid, "shm": self.shm,
            "output_dir": self.output_dir, "log_path": self.log_path,
            "phase": self.phase, "cameras": self.cameras,
            "registry_path": self.registry_path, "alive": self.alive,
            "config": self.config, "camera": self.camera,
            "step": self.step, "episodes": self.episodes,
            "start_ts": self.start_ts, "gpu": self.gpu,
            "error": self.error,
        }


def read_entry(run_id: str, state_dir: str | None = None) -> dict:
    """Serve-registry entry of one run ({} when unknown/gone)."""
    for p in _bridge.serve_state_paths(state_dir):
        try:
            st = json.loads(Path(p).read_text())
        except (OSError, ValueError):
            continue
        if st.get("run_id") == run_id:
            return st
    return {}


def _from_shm(run_id: str, entry: dict) -> RunInfo | None:
    """SHM-anchored view (authoritative for phase/step/pid).

    A segment that EXISTS but cannot be opened/read surfaces as
    ``phase="UNKNOWN"`` with ``error`` = ``"<ExcType>: <text>"`` -- the
    actionable truth, never flattened into absence (a bare catch-all
    returning None made a live-but-unattachable run invisible).  Only a
    segment that VANISHED between the scan and the open (a race) returns
    None -- absence stays honest.  pid/output_dir for UNKNOWN rows come
    from the serve registry entry when it still exists; the daemon's
    rebuild additionally merges the run.json ledger when it does not.

    Liveness is the heartbeat on BOTH branches: readable segments carry
    it in their status read; UNKNOWN segments are probed raw (a
    pre-heartbeat layout reads 0.0 -> not alive -- the segment cannot
    prove it is ticking, so it is not).
    """
    try:
        with _bridge.ShmClient(run_id) as c:
            # the serve-registry camera rides in as the frameless
            # fallback: the entry IS the registry this scan just
            # parsed, so the per-segment registry re-scan inside
            # read_status would only re-list the same directories to
            # rediscover this very file
            s = c.read_status(
                camera_fallback=(str(entry["camera"])
                                 if entry.get("camera") is not None
                                 else None))
    except FileNotFoundError:
        return None                 # vanished mid-scan (race): true absence
    except Exception as e:          # noqa: BLE001 -- typed into UNKNOWN below,
        # never swallowed into None; the kill target comes from the
        # serve registry entry, else straight off the (magic-valid)
        # header -- an orphan segment stays actionable with no ledger
        pid = (int(entry.get("pid") or 0)
               or _bridge.read_header_pid(_bridge.SHM_PREFIX + run_id))
        return RunInfo(
            run_id=run_id, pid=pid,
            shm=_bridge.SHM_PREFIX + run_id,
            output_dir=str(entry.get("output_dir") or ""),
            log_path=str(entry.get("log_path") or ""),
            phase="UNKNOWN",
            cameras=list(entry.get("cameras") or []),
            registry_path=str(entry.get("registry_path") or ""),
            alive=_bridge.hb_alive(_bridge.probe_heartbeat(
                run_id=run_id, pid=pid or None)),
            config=str(entry.get("config") or ""),
            camera=str(entry.get("camera") or ""),
            start_ts=float(entry.get("start_ts") or 0.0),
            gpu=(str(entry["gpu"]) if entry.get("gpu") is not None else None),
            error=f"{type(e).__name__}: {e}",
        )
    pid = int(s.get("pid") or 0)
    return RunInfo(
        run_id=run_id, pid=pid,
        shm=_bridge.SHM_PREFIX + run_id,
        output_dir=str(entry.get("output_dir") or ""),
        log_path=str(entry.get("log_path") or ""),
        phase=str(s.get("phase") or "?"),
        cameras=list(entry.get("cameras") or []),
        registry_path=str(entry.get("registry_path") or ""),
        alive=_bridge.hb_alive(s.get("hb_ts") or 0.0),
        config=str(entry.get("config") or ""),
        camera=str(s.get("camera") or entry.get("camera") or ""),
        step=int(s.get("step") or 0), episodes=int(s.get("episodes") or 0),
        start_ts=float(entry.get("start_ts")
                       or s.get("started") or 0.0),
        gpu=(str(entry["gpu"]) if entry.get("gpu") is not None else None),
    )


def _from_registry(run_id: str, entry: dict, path: Path) -> RunInfo:
    """Registry-only view (the SHM scan found no segment for this entry).

    With no segment there is no heartbeat: alive = a raw heartbeat probe
    still finds one (the segment appeared between the scan and here --
    a spawn-window race).  A true remnant (worker died before/without a
    segment) reads 0.0 -> not alive, so the default view (include_dead
    False) drops it; `unigym clean` removes the stale files.
    """
    pid = int(entry.get("pid") or 0)
    return RunInfo(
        run_id=run_id, pid=pid,
        shm=_bridge.SHM_PREFIX + run_id,
        output_dir=str(entry.get("output_dir") or ""),
        log_path=str(entry.get("log_path") or ""),
        phase="starting",
        cameras=list(entry.get("cameras") or []),
        registry_path=str(path),
        alive=_bridge.hb_alive(_bridge.probe_heartbeat(run_id=run_id,
                                                       pid=pid or None)),
        config=str(entry.get("config") or ""),
        camera=str(entry.get("camera") or ""),
        start_ts=float(entry.get("start_ts") or 0.0),
        gpu=(str(entry["gpu"]) if entry.get("gpu") is not None else None),
    )


def list_runs(state_dir: str | None = None,
              include_dead: bool = False) -> list[RunInfo]:
    """Merged live view: SHM scan x serve registries, heartbeat-verified.

    Sorted by run_id (=`unigym runs` historical order, so 'latest' =
    list[-1]).  SHM-anchored instances always appear (`unigym
    runs` parity -- a DONE segment stays visible until `unigym clean`);
    registry-only entries (no segment -> no heartbeat) appear only when
    a probe still finds a heartbeat, unless ``include_dead`` (the daemon
    rebuild passes False).
    """
    entries: dict[str, tuple[dict, Path]] = {}
    for p in _bridge.serve_state_paths(state_dir):
        try:
            st = json.loads(Path(p).read_text())
        except (OSError, ValueError):
            continue
        rid = st.get("run_id")
        if rid:
            st.setdefault("registry_path", str(p))
            entries[rid] = (st, p)

    out: list[RunInfo] = []
    seen: set[str] = set()
    for rid in _bridge.active_run_ids():
        seen.add(rid)
        found = entries.get(rid)
        info = _from_shm(rid, found[0] if found else {})
        if info is not None:
            out.append(info)
    for rid, (entry, path) in entries.items():
        if rid in seen:
            continue
        info = _from_registry(rid, entry, path)
        if info.alive or include_dead:
            out.append(info)
    return sorted(out, key=lambda r: r.run_id)


def clean(state_dir: str | None = None, dry_run: bool = False,
          force: bool = False) -> list[str]:
    """Zombie SHM segments + stale registries hygiene (bridge primitive).

    ``force`` also removes the unattachable segments reported as
    version-mismatch / corrupt (design v049 §4)."""
    return _bridge.clean_zombies(state_dir=state_dir, dry_run=dry_run,
                                 force=force)
