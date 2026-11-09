"""Runtime bridge: SHM double buffer + step-boundary soft interrupt.

(feat/media-gateway) -- **run slimmed down to a pure simulation worker**
(design docs/designs/v027-media-gateway.md §2): the run-side HTTP face (bridge
server, token logic, webui static mount, MJPEG/WebRTC streaming) moved
WHOLESALE into the daemon media gateway (unigym/daemon/media.py +
webrtc.py).  What remains here -- and the ONLY channel between a run and
the world -- is the SHM segment:

* status block seqlock writes (step / episodes / last_metrics JSON /
  timestamps)
* **unconditional wall-scheduled frame publishing**: a run publishes
  the broadcasting camera's frame whenever the camera's render schedule
  comes due -- wall-clock target fps when resolvable (the scheduler's
  resolution lives in the camera manager; camera_contract §16), the legacy
  per-tick interval counter otherwise -- plus the FROZEN RENDER PUMP: the
  parked service loop keeps publishing on the same wall schedule while the
  world is frozen (drive gaps / born-paused / undo parked), so free-camera
  re-aims and pose writes land on the stream within one poll interval.  SHM
  layout and publish-pointer protocol unchanged.
* command-slot req-ack (pause/resume/set_props/get_view/shutdown/set_camera/
  step/inspect/save_npz -- the daemon media gateway and the CLI perform
  every operation through this; plus the typed queries metrics/config/report)
* registry v2 ({run_id, pid, shm, output_dir, log_path, start_ts,
  cameras[], ...}; the old-generation port/auth/webui_url/curl_base/
  tunnel_hint keys are retired -- a run has neither port nor token)

(feat/debug-console): the pause freeze loop is the
``unigym.console`` StepGate (paused + request_step(n) policy-step
budget); the dispatch table hosts the console handlers ``step`` /
``inspect`` / ``save_npz`` (logic lives in unigym/console/, this module
only hosts hooks).

``paused=True`` ctor arg (`unigym debug --pause`) -- the gate
is born PAUSED so the engine parks at the FIRST step boundary.

Foreground/backstage decoupling for long collection runs
(docs/contracts/runtime_contract.md is the contract twin of this module):

* The main thread (the SyncRunner step loop) is the ONLY sim visitor.
  The Bridge mounts as a Sink (via FanOutSink, zero engine changes); every
  step ``on_step -> tick``:
    ① status block seqlock write
    ② broadcast camera render schedule due -> copy the camera frame into
       the SHM frame slot (CameraManager.get_view; the slot write is an
       owned copy -- the ownership discipline)
    ③ read the command block: pause = StepGate docking (while docked,
       get_view/set_props/inspect keep being served); shutdown = set
       ``shutdown_requested`` (collect_cli triggers a normal break via the
       engine's existing ``stop_when`` seam -> on_run_end wrap-up)
* The daemon media gateway / CLI (a separate process) attach to the same
  SHM segment via :class:`ShmClient`: status / frame reads (seqlock atomic
  read of the latest frame) / pause / resume / camera / snapshot (a
  req_id-keyed command->ack native RPC, polling + timeout).

Three disciplines (owner-approved, blueprint §1):

1. **sim on the main thread only** -- camera frame reads all happen in the
   main-loop tick; readers only read SHM.
2. **Frames cross into SHM as one-way owned copies** -- every frame entering
   SHM is an owned copy (get_view already copies, the slot write copies
   again); readers get an in-slot snapshot, sharing no objects with the sim
   side.
3. **All sync primitives encapsulated in this module** -- single-writer/
   single-reader seqlock (sequence parity + double-read validation) plus an
   aligned u64 atomic publish pointer (frame_ptr), no mutex; the outside
   world only sees tick()/ShmClient.

Cross-process attach notes (measured): this box's Python 3.10.21
resource_tracker unlinks the shared segment when an attach-only client
exits (the unregister-prefix hack is unreliable; see
docs/contracts/runtime_contract.md §known limits) -- so :class:`ShmClient`
attaches via a direct /dev/shm open+mmap (stdlib, zero tracker
involvement); the owner side still uses multiprocessing.shared_memory
(create + unlink semantics are owner-exclusive).

Same-host pledge (docs/designs/v027-media-gateway.md red line): SHM never
crosses machines -- the daemon and all runs must be on one machine;
multi-machine = one daemon per machine. This module depends only on stdlib
+ numpy (lazy: the pure status/command paths do not trigger the numpy
import).
"""
from __future__ import annotations

import io
import json
import mmap
import os
import re
import struct
import threading
import time
import uuid
import zlib
from collections.abc import Iterable, Mapping, Sequence
from glob import glob
from multiprocessing import shared_memory
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Protocol, TypedDict,
                    TypeAlias, overload, runtime_checkable)

from unigym import runtime_dirs
from unigym._typing import JsonValue

if TYPE_CHECKING:  # pragma: no cover - typing-only
    import torch
    from numpy.typing import NDArray

    from unigym.cameras.manager import CameraManager
    from unigym.cameras.base import CameraSpec

__all__ = ["Bridge", "ShmClient", "BridgeError",
           "BridgeTimeout", "make_run_id", "active_run_ids",
           "clean_zombies", "serve_state_paths", "CMD_NAMES",
           "SHM_PREFIX", "MAGIC", "VERSION",
           "HB_TIMEOUT_S", "probe_heartbeat", "hb_alive",
           "METRICS_OFF", "METRICS_JSON", "METRICS_SIZE"]

# --------------------------------------------------------------------------- #
# protocol constants (docs/contracts/runtime_contract.md §2/§3)
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[2]
SHM_PREFIX = "unigym_"
MAGIC = b"UGYM"
VERSION = 5                             # header heartbeat (hb_seq/hb_ts); the
                                        # layout table lives in
                                        # docs/contracts/runtime_contract.md §2

PHASE_RUNNING, PHASE_PAUSED, PHASE_DONE = 0, 1, 2
PHASE_NAMES = {PHASE_RUNNING: "RUNNING", PHASE_PAUSED: "PAUSED",
               PHASE_DONE: "DONE"}

CMD_NONE = 0
CMD_PAUSE = 1
CMD_RESUME = 2
CMD_SET_PROPS = 3
CMD_GET_VIEW = 4
CMD_SHUTDOWN = 5
CMD_SET_CAMERA = 6              # S2: stream_bind producer rebind
CMD_STEP = 7                    # console: paused step/fast-forward
CMD_INSPECT = 8                 # console: summary/obs/extra/eval
CMD_SAVE_NPZ = 9                # console: npz dump
CMD_METRICS = 10                # typed query: metrics document
CMD_CONFIG = 11                 # typed query: effective config
CMD_REPORT = 12                 # typed query: benchmark draft
CMD_GET_POSE = 13               # freecam: world-pose readback
CMD_DIRECT = 14                 # director: direct.* op family
CMD_WATCH = 15                  # watchers: add/list/clear/update/test
CMD_SCENE = 16                  # typed query: curated scene face
CMD_SNAPSHOT = 17               # rewind: capture an engine snapshot {label?}
CMD_REWIND = 18                 # rewind: restore a snapshot (auto-pauses)
CMD_SNAPSHOTS = 19              # rewind: list live snapshot slots
CMD_NAMES = {CMD_PAUSE: "pause", CMD_RESUME: "resume",
             CMD_SET_PROPS: "set_props", CMD_GET_VIEW: "get_view",
             CMD_SHUTDOWN: "shutdown", CMD_SET_CAMERA: "set_camera",
             CMD_STEP: "step", CMD_INSPECT: "inspect",
             CMD_SAVE_NPZ: "save_npz", CMD_METRICS: "metrics",
             CMD_CONFIG: "config", CMD_REPORT: "report",
             CMD_GET_POSE: "get_world_pose",
             CMD_DIRECT: "direct", CMD_WATCH: "watch",
             CMD_SCENE: "scene", CMD_SNAPSHOT: "snapshot",
             CMD_REWIND: "rewind", CMD_SNAPSHOTS: "snapshots"}
CMD_IDS = {v: k for k, v in CMD_NAMES.items()}
_CONSOLE_CMDS = frozenset((CMD_STEP, CMD_INSPECT, CMD_SAVE_NPZ))
_REWIND_CMDS = frozenset((CMD_SNAPSHOT, CMD_REWIND, CMD_SNAPSHOTS))

ACK_OK, ACK_ERR = 0, 1

FMT_RAW, FMT_JPEG = 0, 1
DTYPE_UINT8, DTYPE_FLOAT32 = 0, 1

# 16KB query snapshots: the curated scene face grew
# past 8KB -- tree (joint-limit rows) + state ~8.4KB honest doc
PAYLOAD_JSON = 16384
STATUS_JSON = 8192
DEFAULT_FRAME_CAPACITY = 4 * 1024 * 1024   # per slot; fits 1280x800x3 ego (2.93MB) with headroom
CAMERA_NAME_LEN = 64
MAX_DIMS = 8

# ---- SHM layout (all little-endian; offsets are absolute) ------------------
# The authoritative layout table is docs/contracts/runtime_contract.md §2.
# The heartbeat (hb_seq/hb_ts) is the worker liveness signal: EVERY engine
# loop iteration bumps both -- the parked gate loop included -- and every
# reader judges liveness by hb_ts freshness (HB_TIMEOUT_S).  The header is
# the only liveness authority; no pid probing anywhere on the judgment path.
# H_VIEWER (viewer gating) and H_PORT (run HTTP port) are retired
# fields -- the owner writes 0 and no code reads them for behaviour any
# more (media lives in the daemon; frames publish unconditionally at the
# camera render interval).  The segment is laid out by SLOT_HDR, so
# foreign-layout readers cannot misparse a newer slot (they are rejected
# by the VERSION gate at attach).
HDR_SIZE = 64
H_MAGIC, H_VERSION = 0, 4
H_PHASE, H_PAUSED, H_VIEWER, H_FMT = 6, 7, 8, 9
H_CAP = 12                       # u32 frame slot payload capacity
H_FPTR = 16                      # u64 publish pointer: (frame_id << 8) | slot
H_STARTED = 24                   # f64 unix ts
H_PID = 32                       # u64 owner pid (display / kill target)
H_PORT = 40                      # u16 retired (always 0; runs have no HTTP port)
H_HB_SEQ = 48                    # u64 heartbeat counter (loop iterations)
H_HB_TS = 56                     # f64 unix ts of the last heartbeat bump

CMD_OFF = 64
C_REQ, C_CMD, C_JSON = 0, 8, 16
C_CK = 16 + PAYLOAD_JSON
CMD_SIZE = 16416                 # 16 hdr + 16384 payload + 8 ck (16-aligned)

ACK_OFF = CMD_OFF + CMD_SIZE
A_REQ, A_STATUS, A_JSON = 0, 8, 16
A_CK = 16 + PAYLOAD_JSON
ACK_SIZE = 16416

STATUS_OFF = ACK_OFF + ACK_SIZE
S_SEQ, S_STEP, S_EP, S_JSON = 0, 8, 16, 24
S_TS = 24 + STATUS_JSON
S_CK = S_TS + 8
STATUS_SIZE = 8240               # 8 seq + 8+8 + 8192 json + 8 ts + 8 ck

# ---- metrics block (docs/contracts/runtime_contract.md §2) --------
# seqlock-guarded like status: scalar prefix (timing, updated per tick) +
# JSON document (task/episode/progress, updated at episode end).  Fixed
# capacity -- rings live in the document with a hard cap, never growing
# with run time.
METRICS_OFF = STATUS_OFF + STATUS_SIZE
M_SEQ, M_GEN, M_FPS_P, M_FPS_S, M_SIMT, M_TS = 0, 8, 16, 24, 32, 40
M_JSON = 48
METRICS_JSON = 8192              # document budget (fail-fast, never clipped)
M_CK = M_JSON + METRICS_JSON     # 8240
METRICS_SIZE = 8256              # 6 scalars + 8192 json + ts + ck (16-align)

SLOT_OFF = METRICS_OFF + METRICS_SIZE
SLOT_HDR = 192                   # 176 + step u64 + ts u64 (16-aligned)
L_SEQ, L_FID, L_NAME = 0, 8, 16
L_NDIM, L_DTYPE, L_SHAPE = 80, 84, 88
L_LEN, L_FMT = 152, 160
L_STEP, L_TS = 168, 176          # publish-time policy step + monotonic
L_DATA = 192

# registry schema version (serve-registry v2 keys; see Bridge).
REGISTRY_SCHEMA = 2

#: A control-plane response body: handler-defined, JSON-ish but not
#: guaranteed homogeneous (arbitrary ack shape by contract).
AckResult: TypeAlias = dict[str, Any]

#: A status document (step/episodes/metrics + identity/telemetry counters):
#: scalar and nested values with no single value type.
StatusDoc: TypeAlias = dict[str, Any]

#: The SHM buffer shapes the struct helpers touch (read + slice-assignable).
Buf: TypeAlias = bytearray | memoryview | mmap.mmap

#: ``fn(rid, params) -> result`` -- the runtime command handler contract.
CommandHandler: TypeAlias = Callable[[int, dict[str, JsonValue]], AckResult]


@runtime_checkable
class MetricsSource(Protocol):
    """The duck-typed benchmark telemetry feed (BenchmarkSink is ref impl)."""

    def metrics_drain(self) -> Sequence[Mapping[str, JsonValue]]: ...

    def metrics_task(self) -> Mapping[str, JsonValue] | None: ...

    def metrics_progress(self) -> Mapping[str, JsonValue] | None: ...

    def metrics_report(self) -> JsonValue: ...


@runtime_checkable
class ConsoleFace(Protocol):
    """The debug-console face mounted via :meth:`Bridge._install_console`."""

    env: object | None

    def cmd_step(self, rid: int,
                 params: dict[str, JsonValue]) -> AckResult: ...

    def cmd_inspect(self, rid: int,
                    params: dict[str, JsonValue]) -> AckResult: ...

    def cmd_save_npz(self, rid: int,
                     params: dict[str, JsonValue]) -> AckResult: ...


@runtime_checkable
class WatcherFace(Protocol):
    """The watcher engine face mounted on ``bridge._watchers``."""

    def on_tick(self, t: int, bundle: Mapping[str, JsonValue]) -> None: ...

    def on_reset(self, env_ids: object) -> None: ...

    def status(self) -> dict[str, JsonValue]: ...

    def status_brief(self) -> dict[str, JsonValue]: ...


class FrameSlot(TypedDict):
    """One published frame-slot snapshot (None when unreadable)."""

    frame_id: int
    camera: str
    dtype: int
    shape: list[int]
    format: int
    step: int
    ts: int
    data: bytes


class RewindInfo(TypedDict):
    """Process-truth record of one applied rewind."""

    tick: int
    restored_tick: int
    snap_id: int
    label: str | None


@overload
def _u16(buf: Buf, off: int) -> int: ...
@overload
def _u16(buf: Buf, off: int, v: int) -> None: ...
def _u16(buf: Buf, off: int, v: int | None = None) -> int | None:
    if v is None:
        return struct.unpack_from("<H", buf, off)[0]
    struct.pack_into("<H", buf, off, v)
    return None


@overload
def _u32(buf: Buf, off: int) -> int: ...
@overload
def _u32(buf: Buf, off: int, v: int) -> None: ...
def _u32(buf: Buf, off: int, v: int | None = None) -> int | None:
    if v is None:
        return struct.unpack_from("<I", buf, off)[0]
    struct.pack_into("<I", buf, off, v)
    return None


@overload
def _u64(buf: Buf, off: int) -> int: ...
@overload
def _u64(buf: Buf, off: int, v: int) -> None: ...
def _u64(buf: Buf, off: int, v: int | None = None) -> int | None:
    if v is None:
        return struct.unpack_from("<Q", buf, off)[0]
    struct.pack_into("<Q", buf, off, v)
    return None


@overload
def _f64(buf: Buf, off: int) -> float: ...
@overload
def _f64(buf: Buf, off: int, v: float) -> None: ...
def _f64(buf: Buf, off: int, v: float | None = None) -> float | None:
    if v is None:
        return struct.unpack_from("<d", buf, off)[0]
    struct.pack_into("<d", buf, off, v)
    return None


def _put_bytes(buf: Buf, off: int, raw: bytes, region: int) -> None:
    """Write raw into buf[off:off+region] and NUL-pad the remainder."""
    if len(raw) > region:
        raise BridgeError(f"payload {len(raw)}B exceeds fixed slot {region}B")
    buf[off:off + len(raw)] = raw
    buf[off + len(raw):off + region] = _ZEROS[:region - len(raw)]


_ZEROS = bytes(max(PAYLOAD_JSON, STATUS_JSON, METRICS_JSON))
# >= largest padded region (the 16KB command/ack JSON outgrew
# the status JSON block that used to be the largest)


def _as_str(value: JsonValue) -> str | None:
    """Narrow one wire payload field to a string (None otherwise)."""
    return value if isinstance(value, str) else None


def _as_int(value: JsonValue) -> int | None:
    """Narrow one wire payload field to an int (None otherwise; no bools)."""
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _as_mapping(value: JsonValue) -> dict[str, JsonValue]:
    """Narrow one wire payload field to a JSON object ({} otherwise)."""
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {}


def _payload_ck(raw: bytes) -> int:
    return zlib.adler32(raw) & 0xFFFFFFFF


def _write_command(buf: Buf, cmd_id: int,
                   payload: dict[str, JsonValue]) -> int:
    """Publish one command into the SHM command block (req_id LAST).

    The one wire discipline shared by every writer (CLI, daemon media
    gateway, WebRTC DataChannel relay).
    """
    rid = time.time_ns()
    raw = json.dumps(payload or {}).encode()
    if len(raw) > PAYLOAD_JSON:
        raise BridgeError(f"command payload {len(raw)}B > {PAYLOAD_JSON}B")
    _u16(buf, CMD_OFF + C_CMD, int(cmd_id))
    _put_bytes(buf, CMD_OFF + C_JSON, raw, PAYLOAD_JSON)
    _u64(buf, CMD_OFF + C_CK, _payload_ck(raw))
    _u64(buf, CMD_OFF + C_REQ, rid)               # publish LAST
    return rid


def _poll_ack(buf: Buf, rid: int, deadline: float,
              phase_fn: Callable[[], int]
              ) -> tuple[int, dict[str, Any]] | None:
    """Poll the ack block for ``rid`` until deadline; None on timeout.

    ``phase_fn`` is a zero-arg callable returning the current phase int
    (DONE short-circuits: nobody will ever serve the command).
    """
    while True:
        a = _u64(buf, ACK_OFF + A_REQ)
        if a == rid:
            status = _u16(buf, ACK_OFF + A_STATUS)
            raw = bytes(buf[ACK_OFF + A_JSON:
                            ACK_OFF + A_JSON + PAYLOAD_JSON])
            raw = raw.split(b"\x00", 1)[0]
            try:
                result: dict[str, Any] = json.loads(raw)
            except ValueError:
                result = {"error": "unparseable ack payload"}
            return status, result
        if phase_fn() == PHASE_DONE:
            raise BridgeError(
                f"run finished before ack for "
                f"'{CMD_NAMES.get(_u16(buf, CMD_OFF + C_CMD), '?')}'")
        if time.time() >= deadline:
            return None
        time.sleep(0.005)


# --------------------------------------------------------------------------- #
# seqlock primitives (single writer / single reader, no locks)
# --------------------------------------------------------------------------- #
def _seqlock_read(buf: Buf, base: int, start: int, length: int,
                  retries: int = 2000) -> bytes | None:
    """Tear-free read of buf[base+start : base+start+length].

    Writer marks seq odd during write and even after; a stable even seq
    across the copy means the snapshot is consistent.  Bounded retries so a
    perpetually-writing peer cannot hang the reader (practically never:
    writes are µs-scale).
    """
    for _ in range(retries):
        s1 = _u64(buf, base)
        if not s1 & 1:
            data = bytes(buf[base + start:base + start + length])
            if _u64(buf, base) == s1:
                return data
        time.sleep(0.0005)
    return None


def _read_status_block(buf: Buf) -> StatusDoc:
    data = _seqlock_read(buf, STATUS_OFF, 8, STATUS_SIZE - 16)
    if data is None:
        raise BridgeError("status seqlock unreadable (writer starving?)")
    step, ep = struct.unpack_from("<QQ", data, 0)
    raw = data[16:16 + STATUS_JSON].split(b"\x00", 1)[0]
    ck = struct.unpack_from("<Q", data, 16 + STATUS_JSON + 8)[0]
    metrics: JsonValue = {}
    if raw and _payload_ck(raw) == ck:
        try:
            metrics = json.loads(raw)
        except ValueError:
            metrics = {}
    ts = struct.unpack_from("<d", data, 16 + STATUS_JSON)[0]
    return {"step": step, "episodes": ep, "ts": ts, "metrics": metrics}


def _frame_cap(buf: Buf) -> int:
    return _u32(buf, H_CAP)


def _slot_base(buf: Buf, slot: int) -> int:
    return SLOT_OFF + slot * (SLOT_HDR + _frame_cap(buf))


def _read_frame_slot(buf: Buf, base: int) -> FrameSlot | None:
    """One tear-free snapshot of a frame slot (header + payload bytes)."""
    # seqlock region excludes the seq counter itself (view starts at L_FID;
    # every field offset below is L_<x> - 8)
    hdr = _seqlock_read(buf, base, 8, SLOT_HDR - 8)
    if hdr is None:
        return None
    fid = struct.unpack_from("<Q", hdr, L_FID - 8)[0]
    name = hdr[L_NAME - 8:L_NAME - 8 + CAMERA_NAME_LEN].split(
        b"\x00", 1)[0].decode("utf-8", "replace")
    ndim, dtype = struct.unpack_from("<II", hdr, L_NDIM - 8)
    shape = list(struct.unpack_from(f"<{MAX_DIMS}Q", hdr,
                                    L_SHAPE - 8))[:ndim]
    n, fmt = struct.unpack_from("<QI", hdr, L_LEN - 8)
    if fid == 0 or n == 0 or n > _frame_cap(buf):
        return None
    # publish-time policy step + monotonic stamp (the seqlock
    # region above covers the whole header, so these are tear-free)
    step, ts = struct.unpack_from("<QQ", hdr, L_STEP - 8)
    data = bytes(buf[base + L_DATA:base + L_DATA + n])
    return {"frame_id": fid, "camera": name, "dtype": dtype, "shape": shape,
            "format": fmt, "step": step, "ts": ts, "data": data}


def _read_latest_frame(buf: Buf,
                       retries: int = 200) -> dict[str, Any] | None:
    """Reader entry point: publish pointer -> slot seqlock read.

    The publish pointer (header, aligned u64) selects the slot AFTER the
    writer fully closed that slot's seqlock, so the double-check below
    (slot seq + pointer unchanged) rejects any torn/overwritten copy.
    """
    for _ in range(retries):
        f1 = _u64(buf, H_FPTR)
        fid, slot = f1 >> 8, f1 & 0xFF
        if fid == 0:
            return None
        base = _slot_base(buf, slot)
        s1 = _u64(buf, base)
        if s1 & 1:
            time.sleep(0.0005)
            continue
        frame = _read_frame_slot(buf, base)
        f2 = _u64(buf, H_FPTR)
        s2 = _u64(buf, base)
        if (frame is not None and f1 == f2 and s1 == s2
                and frame.get("frame_id") == fid):
            return dict(frame)
        time.sleep(0.0005)
    return None


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class BridgeError(RuntimeError):
    """Bridge protocol / lifecycle error."""


class BridgeTimeout(BridgeError):
    """Command was not served within the poll timeout."""


class RewindBoundary(Exception):
    """Internal control flow: a rewind was applied at this step boundary.

    Raised by ``Bridge.tick`` AFTER the rewind ack is written, so it
    unwinds out of ``sink.on_step`` and the engine abandons the rest of
    the current iteration (episode-end / reset chain belong to the
    abandoned timeline branch).  The engine catches it, restores the
    iteration state (obs frame / command) the snapshot carries, jumps the
    loop counter to ``next_tick`` and continues — never the caller.
    ``info`` is the rewind record (process-truth, also appended to
    ``bridge.rewind_log``) fanned out to sinks via ``on_rewind``.
    """

    def __init__(self, next_tick: int, info: RewindInfo | None = None):
        super().__init__(next_tick)
        self.next_tick = int(next_tick)
        self.info = dict(info or {})


# --------------------------------------------------------------------------- #
# helpers shared by Bridge / CLI / daemon
# --------------------------------------------------------------------------- #
def make_run_id(run_name: str) -> str:
    """`<stamp>-<safe_run_name>-<pid>-<rand>` — unique, shm-name safe.

    The random suffix keeps two same-named runs in the same process/second
    from colliding on the SHM segment name (a real FileExistsError seen
    when two ``Run`` objects carry the same ``run_name``).
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_name)).strip("_") or "run"
    return (f"{time.strftime('%Y%m%d-%H%M%S')}-{safe}"
            f"-{os.getpid() % 100000}-{uuid.uuid4().hex[:6]}")


def active_run_ids() -> list[str]:
    """Run ids of live `unigym_*` segments present in /dev/shm."""
    return sorted(p[len(SHM_PREFIX):] for p in
                  (os.path.basename(x) for x in glob(f"/dev/shm/{SHM_PREFIX}*")))


def serve_state_paths(state_dir: str | os.PathLike[str] | None = None) -> list[Path]:
    """Serve-registry json paths across the scanned directories.

    Layout (designs/v068-runtime-layout.md §3): the ONE scan lives
    in :func:`unigym.runtime_dirs.serve_registry_paths` -- scan set
    [explicit state_dir, UNIGYM_SERVE_DIR, runtime serve dir, repo root
    (docker registries mount the repo)]; no /tmp-root fallback.
    """
    return runtime_dirs.serve_registry_paths(state_dir, extra_dirs=[
        str(_REPO_ROOT)])



def _pid_alive(pid: int) -> bool:
    """Process signalability (kill -0), NOT run liveness.

    The run liveness judgment everywhere is the heartbeat (hb_alive).
    This helper remains only for process-file hygiene: a serve-registry
    FILE has no segment to carry a heartbeat, so its owner pid is the
    only staleness signal available.
    """
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True


# -- heartbeat (the run liveness signal) -------------------------------------
# The worker bumps hb_seq/hb_ts on EVERY engine-loop iteration (the parked
# gate loop included); readers judge liveness purely by hb_ts freshness.
HB_TIMEOUT_S = 10.0              # liveness window (worker loop period <= 1s)


def hb_alive(hb_ts: float, now: float | None = None) -> bool:
    """Run liveness = heartbeat freshness.

    ``hb_ts == 0`` (segment absent, unreadable, or a pre-heartbeat
    layout) is never alive.
    """
    if not hb_ts:
        return False
    return (time.time() if now is None else now) - hb_ts < HB_TIMEOUT_S


def read_header_heartbeat(name: str) -> float:
    """hb_ts of one /dev/shm segment WITHOUT attaching (64B header read).

    0.0 when the segment is absent/short/foreign-magic -- callers judge
    with :func:`hb_alive`, so absence and expiry collapse into "not
    alive" (the honest reading: nothing is ticking).
    """
    try:
        with open(f"/dev/shm/{name}", "rb") as f:
            head = f.read(HDR_SIZE)
        if len(head) == HDR_SIZE and head[:4] == MAGIC:
            return struct.unpack_from("<d", head, H_HB_TS)[0]
    except OSError:
        pass
    return 0.0


def read_header_pid(name: str) -> int:
    """Owner pid of one /dev/shm segment without attaching (64B read).

    0 when absent/foreign-magic (a corrupt header's pid is garbage, not
    truth).  Gives UNKNOWN rows (unreadable but magic-valid segments --
    the orphan shape) their kill target straight from the header, even
    when no serve registry / ledger survives.
    """
    try:
        with open(f"/dev/shm/{name}", "rb") as f:
            head = f.read(HDR_SIZE)
        if len(head) == HDR_SIZE and head[:4] == MAGIC:
            return int(struct.unpack_from("<Q", head, H_PID)[0])
    except OSError:
        pass
    return 0


def probe_heartbeat(run_id: str | None = None,
                    pid: int | None = None) -> float:
    """The heartbeat readable for a run without a full attach.

    By segment name (``unigym_<run_id>``) when the run_id is known; the
    pid scan (owner match over active segments) covers the spawn window
    where a ledger exists but the worker has not named itself yet.
    0.0 = no heartbeat found (absent/foreign segment).
    """
    if run_id:
        return read_header_heartbeat(SHM_PREFIX + str(run_id))
    if pid:
        want = int(pid)
        for rid in active_run_ids():
            try:
                with open(f"/dev/shm/{SHM_PREFIX}{rid}", "rb") as f:
                    head = f.read(HDR_SIZE)
            except OSError:
                continue
            if len(head) == HDR_SIZE and head[:4] == MAGIC \
                    and struct.unpack_from("<Q", head, H_PID)[0] == want:
                return struct.unpack_from("<d", head, H_HB_TS)[0]
    return 0.0


def clean_zombies(state_dir: str | os.PathLike[str] | None = None,
                  dry_run: bool = False,
                  force: bool = False) -> list[str]:
    """Remove zombie `unigym_*` segments (+ their dead-pid serve-state files).

    Zombie = header phase DONE (run finished but segment leaked) or
    heartbeat expired (the worker stopped ticking -- killed, crashed, or
    wedged past HB_TIMEOUT_S).  Unattachable segments are REPORTED by
    class and only removed with ``force`` (design v049 §4 -- the
    classification shares the UNKNOWN probe semantics): a wrong VERSION
    reads "version-mismatch" (the orphan shape), garbage magic reads
    "corrupt".  A heartbeat-fresh segment ("leaked-live") is never
    garbage -- force does not touch it.  Returns the removed run ids.
    """
    removed: list[str] = []
    for run_id in active_run_ids():
        name = SHM_PREFIX + run_id
        path = Path("/dev/shm") / name
        verdict = "unknown"
        try:
            with open(path, "rb") as f:
                head = f.read(HDR_SIZE)
            if len(head) == HDR_SIZE and head[:4] == MAGIC:
                version = struct.unpack_from("<H", head, H_VERSION)[0]
                if version != VERSION:
                    # exists but unattachable (UNKNOWN state's probe twin)
                    verdict = "version-mismatch"
                else:
                    phase, pid = head[6], struct.unpack_from("<Q", head,
                                                             H_PID)[0]
                    if phase == PHASE_DONE:
                        verdict = "done"
                    else:
                        hb = struct.unpack_from("<d", head, H_HB_TS)[0]
                        if hb_alive(hb):
                            verdict = "leaked-live"
                        else:
                            age = ("" if not hb else
                                   f", hb {time.time() - hb:.0f}s stale")
                            verdict = f"dead(pid {pid}{age})"
            else:
                verdict = "corrupt"
        except OSError as e:
            verdict = f"unreadable({e})"
        removable = (verdict == "done" or verdict.startswith("dead(")
                     or (force and verdict in ("version-mismatch",
                                               "corrupt")))
        if removable:
            if not dry_run:
                try:
                    os.unlink(path)
                except OSError as e:
                    print(f"[clean] {run_id}: unlink failed: {e}")
                    continue
            removed.append(run_id)
            print(f"[clean] removed shm {name} ({verdict})")
        else:
            # only the force-removable classes get the hint -- a
            # leaked-live segment is never garbage, force or not
            hint = " (--force removes it)" \
                if force and verdict in ("version-mismatch", "corrupt") \
                else ""
            print(f"[clean] keep {name} ({verdict}){hint}")
    for p in serve_state_paths(state_dir):
        try:
            st = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        pid = st.get("pid")
        if pid and not _pid_alive(pid):
            if not dry_run:
                p.unlink(missing_ok=True)
            verb = "removed" if not dry_run else "(dry-run) would remove"
            print(f"[clean] {verb} stale serve-state {p.name} "
                  f"(pid {pid} dead)")
    return removed


def _attach_segment(name: str) -> mmap.mmap:
    """Attach to an existing /dev/shm segment WITHOUT the resource tracker.

    (Tracker rationale in the module docstring; verified 2026-09-14 on this
    host's Python 3.10.21: attach-only clients get their segments unlinked
    at exit even with a correct unregister() call.)
    """
    fd = os.open(f"/dev/shm/{name}", os.O_RDWR)
    try:
        return mmap.mmap(fd, os.fstat(fd).st_size)
    finally:
        os.close(fd)


def _encode_jpeg(arr: NDArray[Any]) -> bytes:
    """JPEG-encode an (H,W,3) uint8 array (float32 depth: min-max norm)."""
    import numpy as np
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        if not np.isfinite(a).all() or hi <= lo:
            a = np.zeros(a.shape, dtype=np.uint8)
        else:
            a = ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)
    try:
        from PIL import Image
    except ImportError as exc:
        from unigym._optional import MissingExtraError

        raise MissingExtraError(
            "PIL.Image", "record", "JPEG capture") from exc
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=90)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Bridge (owner side, main-thread sink) -- the pure simulation worker face
# --------------------------------------------------------------------------- #
class Bridge:
    """SHM runtime worker mounted as a Sink (engine-zero-change wiring).

    the run owns NO HTTP.  Everything a remote client needs is
    either read from this SHM segment (status, frames) or executed through
    the command slot (req-ack); the daemon media gateway terminates all
    browser traffic.

    Args:
        run_id: unique run identifier (shm name `unigym_<run_id>`).
        camera_manager: CameraManager or None (no-camera run: broadcast /
            get_view report an error instead of frames).
        camera: broadcast (publishing) camera name (default: first
            instantiated).  Re-bindable live via the ``set_camera`` command.
        frame_capacity: per-slot payload capacity in bytes (default 4MB).
            An oversized frame is NOT copied: dropped (counted + warned
            once) -- the publish path must never take down the sim (the
            remedy is a larger capacity or a smaller camera resolution).
        poll_interval: parked StepGate polling period (s).
        state_dir: serve-registry json directory (None = UNIGYM_SERVE_DIR
            or /tmp).  The registry file is removed on clean close.
        obs_catalog: the run's obs catalog key list (carried into the
            registry for the daemon's /info face).
        paused: start with the step-gate PAUSED (`unigym debug
            --pause`): the engine parks at the FIRST step boundary and
            never advances until resume/step (status phase/gate = PAUSED
            from construction, SHM H_PAUSED=1).  Default False: the
            historical start-RUNNING behavior.
        log_path: the run's own log file (spawn builder knows it); carried
            in registry v2 so the daemon view survives daemon restarts.
        config_doc: the EFFECTIVE run config (post CLI overrides).  Captured
            once into the ``config`` typed query (cmd 11) as a JSON-safe
            snapshot; oversize fails fast HERE (startup assert), not at
            query time (design §2).
        metrics_source: optional duck-typed benchmark telemetry feed
            (BenchmarkSink is the reference impl; runner_contract §3.1):
            metrics_drain() / metrics_task() / metrics_progress() /
            metrics_report().  Feeds the metrics block at episode end and
            the ``report`` query; None = collect-only run (null report).
    """

    # Wall-clock floor of the FROZEN RENDER PUMP for counter-mode
    # managers (no wall fps resolvable -- test doubles / probe runs).
    # Managers WITH a wall fps (the wall scheduler) pump at the camera's
    # own target period instead (§16); a property write always publishes
    # on the next pump beat via the force flag.  0.04s = 25Hz, the
    # historical parked cadence.
    PARKED_PUBLISH_PERIOD = 0.04

    @property
    def buf(self) -> memoryview:
        """The SHM buffer as a non-Optional hot-path accessor.

        ``multiprocessing.shared_memory.SharedMemory.buf`` is typed
        ``memoryview | None`` (it becomes None after teardown); the bridge
        owns the segment for its whole life, so the Optional is contained
        here instead of leaking into every seqlock read/write.
        """
        buf = self.shm.buf
        if buf is None:
            raise BridgeError("shared memory segment is closed")
        return buf

    def __init__(self, run_id: str, camera_manager: CameraManager | None = None,
                 *, camera: str | None = None,
                 frame_capacity: int = DEFAULT_FRAME_CAPACITY,
                 poll_interval: float = 0.02, config: str | None = None,
                 out_dir: str | None = None, state_dir: str | None = None,
                 obs_catalog: Iterable[str] | None = None,
                 paused: bool = False,
                 log_path: str | None = None,
                 config_doc: Mapping[str, JsonValue] | None = None,
                 metrics_source: MetricsSource | None = None):
        self.run_id = str(run_id)
        self.shm_name = SHM_PREFIX + self.run_id
        self.camera_manager = camera_manager
        names = list(camera_manager.names) if camera_manager is not None else []
        self.camera_names = names
        self.camera = camera or (names[0] if names else None)
        if camera is not None and names and camera not in names:
            raise BridgeError(f"camera {camera!r} not instantiated "
                              f"(cameras: {names})")
        # the run's obs catalog key list (carried in registry v2)
        self.obs_catalog_keys = (sorted(obs_catalog)
                                 if obs_catalog is not None else [])
        self.frame_capacity = int(frame_capacity)
        self.poll_interval = float(poll_interval)
        try:
            self.shm = shared_memory.SharedMemory(
                name=self.shm_name, create=True,
                size=SLOT_OFF
                + 2 * (SLOT_HDR + self.frame_capacity))
        except FileExistsError as e:
            raise BridgeError(
                f"shm {self.shm_name} already exists (stale segment? run "
                f"`unigym clean`)") from e
        buf = self.buf
        # fresh POSIX shm is zero-filled; write the header fields explicitly
        struct.pack_into("<4sH", buf, H_MAGIC, MAGIC, VERSION)
        _u32(buf, H_CAP, self.frame_capacity)
        _u64(buf, H_FPTR, 0)
        _f64(buf, H_STARTED, time.time())
        _u64(buf, H_PID, os.getpid())
        _u64(buf, H_HB_SEQ, 0)
        _f64(buf, H_HB_TS, time.time())   # born alive: the segment carries a
        # fresh heartbeat from construction, so the spawn-window faces
        # (registry wait / ledger rows) read it as alive before tick 1
        # phase/paused/viewer/fmt/port stay 0 (RUNNING / not paused /
        # retired fields) -- unless paused=True (`unigym debug
        # --pause`): the gate is born PAUSED so the engine parks at the
        # first step boundary and /status reports phase PAUSED from the
        # start.
        try:
            # SharedMemory hardcodes mode 0600; a containerized owner (root)
            # would otherwise lock the host CLI out of the shared segment
            os.chmod(f"/dev/shm/{self.shm_name}", 0o666)
        except OSError:  # best effort: chmod is a convenience, never a gate
            pass

        self._tick_count = 0
        self._last_publish = 0.0
        self._last_served = 0
        self._frame_count = 0
        self._episodes = 0
        self._frames_dropped = 0
        self._n_served = 0
        self._oversize_warned = False
        self._frozen = False
        self._closed = False
        self.shutdown_requested = False
        # wall scheduler: a camera property write (set_props /
        # set_camera) invalidates the render schedule -- the next pump
        # / tick publishes IMMEDIATELY (the aim→new-frame ≤50ms beat:
        # the parked service loop iterates at poll_interval, so a pose
        # change is visible on the stream within one poll + one render)
        self._publish_now = False
        # console hooks: StepGate replaces the inline freeze loop;
        # handlers (step/inspect/save_npz) register via _install_console.
        # Lazy import: stepping is stdlib-only, so the stdlib-only CLI
        # import path stays numpy-free.
        from ..console.stepping import StepGate
        self._gate = StepGate(paused=bool(paused))
        if paused:
            buf[H_PAUSED] = 1
            buf[H_PHASE] = PHASE_PAUSED
        self.console: ConsoleFace | None = None
        # runtime extension face: ``fn(rid, params) -> result dict``
        self._handlers: dict[int, CommandHandler] = {}
        self.out_dir = out_dir          # console npz dump root (data_out/<run>)
        self._bundle: Mapping[str, JsonValue] | None = None
        # current step bundle (inspector view)
        self._watchers: WatcherFace | None = None  # watcher engine (attach face)
        # the last per-tick metrics dict (§8.3: re-published with fresh
        # flags when a live apply lands while the gate is PARKED -- the
        # parked loop serves commands but runs no ticks)
        self._last_metrics: dict[str, JsonValue] = {}
        # rewind face (Round B): the controller registers the snapshot/
        # rewind/snapshots handlers; the boundary slot arms the one-shot
        # RewindBoundary raise at the end of the tick that applied a
        # rewind; rewind_log is the process-truth record of rewinds
        # (monotonic append-only — the counter dichotomy's ledger).
        self._rewind_next_tick: int | None = None
        self._rewind_info: RewindInfo | None = None
        self.rewind_log: list[RewindInfo] = []

        self._cmd_lock = threading.Lock()   # serializes in-process writers
        self._write_status_locked(step=0, metrics={})

        # ---- metrics block (runtime_contract §2) -----------------
        # bounded structured telemetry: timing scalars per tick, typed
        # document at episode end, typed queries (cmd 10-12) for the daemon.
        from .metrics import MetricsWriter
        self._metrics = MetricsWriter(buf)
        self._metrics_source = metrics_source
        self._config_doc = None
        if config_doc is not None:
            try:                            # JSON-safe snapshot, captured ONCE
                self._config_doc = json.loads(
                    json.dumps(config_doc, default=str))
            except (TypeError, ValueError) as e:
                raise BridgeError(f"config snapshot not serializable: {e}")
            if len(json.dumps(self._config_doc).encode()) > PAYLOAD_JSON:
                raise BridgeError(
                    f"effective config snapshot exceeds the {PAYLOAD_JSON}B "
                    f"query payload -- shrink the config (startup fail-fast, "
                    f"design §2)")

        # ---- serve registry v2 (breaking schema change; design §2) ------------
        # {run_id, pid, shm, output_dir, log_path, start_ts, cameras[]} is
        # the v2 core; config/camera/gpu/obs_catalog ride along as the
        # daemon's display/preflight face.  port/auth/webui_url/curl_base/
        # tunnel_hint are RETIRED (the run has no port and no token).
        self._state_path = None
        state = {"schema": REGISTRY_SCHEMA,
                 "run_id": self.run_id, "pid": os.getpid(),
                 "shm": self.shm_name,
                 "output_dir": out_dir, "log_path": log_path,
                 "start_ts": time.time(), "cameras": names,
                 "camera": self.camera, "config": config,
                 "gpu": os.environ.get("CUDA_VISIBLE_DEVICES") or None,
                 "obs_catalog_keys": self.obs_catalog_keys,
                 "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        # default registry home is <runtime_dir>/serve
        # (designs/v068-runtime-layout.md §3); the explicit --serve_state_dir
        # and the UNIGYM_SERVE_DIR legacy entry keep winning.
        sdir = Path(state_dir or os.environ.get("UNIGYM_SERVE_DIR")
                    or runtime_dirs.serve_dir())
        try:
            sdir.mkdir(parents=True, exist_ok=True)
            self._state_path = sdir / f"unigym-serve-{self.run_id}.json"
            self._state_path.write_text(json.dumps(state, indent=1))
        except OSError as e:
            print(f"[bridge] serve-state write skipped: {e}", flush=True)
        print(f"[bridge] worker ready run_id={self.run_id} "
              f"shm={self.shm_name} (camera: {self.camera}, "
              f"registry: {self._state_path})", flush=True)

    # -- sink protocol (docs/contracts/runner_contract.md §3) -------------------------
    def on_step(self, t: int, bundle: Mapping[str, JsonValue]) -> None:
        """Sink hook: the engine loop's per-step entry (= :meth:`tick`)."""
        self.tick(t, bundle)

    def on_episode_end(self, env_ids: torch.Tensor,
                       bundle: Mapping[str, JsonValue]) -> None:
        """Sink hook: episode counters + metrics document + watchers."""
        self._episodes += int(env_ids.numel())
        # metrics feed: bounded rings + document (existing sink
        # hook -- no new engine plumbing; benchmark facts come from the
        # duck-typed source scored BEFORE this callback in the fan-out).
        self._metrics.note_episode(env_ids, bundle,
                                   source=self._metrics_source)
        # watchers: callable on_reset lifecycle hooks (§8.3); never fatal
        if self._watchers is not None:
            try:
                self._watchers.on_reset(env_ids)
            except Exception as e:             # noqa: BLE001
                print(f"[watcher] on_reset hook skipped: {e}", flush=True)

    def on_step_end(self, t: int, bundle: Mapping[str, JsonValue]) -> None:
        """Sink hook: end-of-step (unused by this bridge)."""
        pass

    def on_run_end(self) -> dict[str, JsonValue]:
        """Sink hook: final summary + :meth:`close` (idempotent)."""
        summary = {"run_id": self.run_id, "shm": self.shm_name,
                   "frames_published": self._frame_count,
                   "frames_dropped": self._frames_dropped,
                   "commands_served": self._n_served}
        self.close()
        return summary

    # -- per-step tick (the whole foreground cost) --------------------------
    def tick(self, t: int, bundle: Mapping[str, JsonValue]) -> None:
        """The whole foreground cost, per engine tick.

        ① heartbeat bump ② status seqlock write ③ watcher evaluation
        ④ step-gate boundary ⑤ scheduled publish (wall fps;
        legacy interval counter without a policy clock) ⑥ command serving.

        The heartbeat leads: every loop iteration is a liveness proof,
        before anything else can stall.  The watcher hook runs BEFORE the
        gate: a trigger that asks for pause parks the SAME tick (design
        watchers.md §2.1).
        """
        self._bump_heartbeat()
        metrics: dict[str, JsonValue] = {"t": int(t)}
        tm = bundle.get("term_metrics")
        if isinstance(tm, dict):
            metrics.update(tm)
        if self._watchers is not None:
            self._watchers.on_tick(int(t), bundle)
            metrics["watchers"] = self._watchers.status_brief()
        # the live render state rides the status JSON block so
        # cross-process readers (daemon status/run-info, webui badge)
        # see the live toggle without a command round-trip; the capture
        # backend face rides too (webui auto-disable signal)
        state = self._render_state()
        if state is not None:
            metrics["render_source"] = state["render_source"]
        # ALWAYS carried (null = no capture face): the webui auto-disable
        # signal must distinguish "isaaclab" from "field absent"
        metrics["capture_backend"] = getattr(self.camera_manager,
                                             "capture_backend", None)
        self._last_metrics = metrics
        self._write_status_locked(step=int(t), metrics=metrics)
        # tick-level metrics cost = timing scalars only (µs)
        self._metrics.note_tick(int(t), bundle.get("env")
                                if isinstance(bundle, dict) else None)
        self._bundle = bundle              # inspector view of this step
        if self._gate.arrive():            # paused + budget exhausted -> park
            self._gate_park()
        self._tick_count += 1
        if self._publish_due():            # unconditional, interval-
            self._copy_broadcast_frame()   # paced (no viewer gate anywhere)
        self._serve_commands()
        if self._rewind_next_tick is not None:
            # a rewind was applied while serving this boundary (running
            # path or the parked service loop -- every command lands
            # inside a tick() frame).  The ack is already written; unwind
            # past sink.on_step so the engine abandons the rest of the
            # iteration that belongs to the abandoned timeline branch.
            nxt, self._rewind_next_tick = self._rewind_next_tick, None
            info, self._rewind_info = self._rewind_info, None
            raise RewindBoundary(nxt, info=info)

    def pause_for_watcher(self) -> None:
        """Watcher-triggered pause.

        Identical state to CMD_PAUSE (the engine thread parks at the
        gate.arrive() check right after).
        """
        self._enter_paused()

    def pause_for_rewind(self) -> None:
        """Rewind auto-pause (owner ruling §5-③).

        Same state as CMD_PAUSE -- the world parks restored,
        resume/step drive it.
        """
        self._enter_paused()

    def pause_for_halt(self) -> None:
        """Freeze paradigm: skill halt -> world freeze.

        Called by the adapter on the ENGINE thread at the halt point,
        so the gate parks at the halt tick's own step boundary -- zero
        poll-latency gap, no daemon round trip (the HTTP pause path
        would deadlock here: the engine thread blocks on an ack only it
        can write).  Same state as CMD_PAUSE.
        """
        self._enter_paused()

    def _enter_paused(self) -> None:
        self.buf[H_PAUSED] = 1
        self.buf[H_PHASE] = PHASE_PAUSED
        self._gate.pause()

    def _bump_heartbeat(self) -> None:
        """One loop iteration = one liveness proof (header hb_seq/hb_ts).

        Called from tick() AND from the parked gate's service loop -- a
        PAUSED run keeps its heartbeat (the parked loop iterates just
        like the running one), so liveness is about the loop being
        serviced, never about the sim advancing.
        """
        buf = self.buf
        _u64(buf, H_HB_SEQ, _u64(buf, H_HB_SEQ) + 1)
        _f64(buf, H_HB_TS, time.time())

    def _write_status_locked(self, step: int,
                             metrics: Mapping[str, JsonValue]) -> None:
        self._last_step = int(step)   # frame header step source
        buf = self.buf
        seq = _u64(buf, STATUS_OFF + S_SEQ)
        _u64(buf, STATUS_OFF + S_SEQ, seq + 1)          # odd = writing
        try:
            raw = json.dumps(metrics).encode()
        except (TypeError, ValueError):
            raw = b"{}"
        if len(raw) > STATUS_JSON:
            raw = json.dumps({"t": metrics.get("t")}).encode()
        _u64(buf, STATUS_OFF + S_STEP, step)
        _u64(buf, STATUS_OFF + S_EP, self._episodes)
        _put_bytes(buf, STATUS_OFF + S_JSON, raw, STATUS_JSON)
        _f64(buf, STATUS_OFF + S_TS, time.time())
        _u64(buf, STATUS_OFF + S_CK, _payload_ck(raw))
        _u64(buf, STATUS_OFF + S_SEQ, seq + 2)          # even = stable

    # -- frame slots ---------------------------------------------------------
    def _write_frame(self, arr: NDArray[Any] | bytes | bytearray,
                     camera: str, fmt: int = FMT_RAW) -> int | None:
        """Copy one frame into the next double-buffer slot; publish id."""
        import numpy as np
        if isinstance(arr, (bytes, bytearray)):     # opaque payload (jpeg)
            a = np.frombuffer(bytes(arr), dtype=np.uint8)
            code = DTYPE_UINT8
        else:
            a = np.asarray(arr)
            if a.dtype == np.float32:
                code = DTYPE_FLOAT32
            elif a.dtype == np.uint8:
                code = DTYPE_UINT8
            else:
                a = a.astype(np.uint8)
                code = DTYPE_UINT8
        if not a.flags["C_CONTIGUOUS"]:
            a = np.ascontiguousarray(a)
        n = a.nbytes
        if n > self.frame_capacity:
            self._frames_dropped += 1
            if not self._oversize_warned:
                self._oversize_warned = True
                print(f"[bridge] frame of camera '{camera}' is {n}B > slot "
                      f"capacity {self.frame_capacity}B -> dropped "
                      f"(counted; raise --serve_frame_mb or shrink the "
                      f"camera resolution)", flush=True)
            return None
        if len(a.shape) > MAX_DIMS:
            raise BridgeError(f"frame rank {len(a.shape)} > {MAX_DIMS}")
        buf = self.buf
        slot = self._frame_count & 1
        base = _slot_base(buf, slot)
        seq = _u64(buf, base)
        _u64(buf, base, seq + 1)                        # slot seq odd
        name = camera.encode()[:CAMERA_NAME_LEN - 1]
        struct.pack_into("<Q", buf, base + L_FID, self._frame_count + 1)
        # NUL-fill the whole name field: a shorter name after a longer one
        # must not leak stale bytes (stream_bind-era camera
        # switching exposed this latent bug: "chase"->"ego" read
        # back as "egose")
        buf[base + L_NAME:base + L_NAME + CAMERA_NAME_LEN] = \
            name.ljust(CAMERA_NAME_LEN, b"\x00")
        struct.pack_into("<II", buf, base + L_NDIM, len(a.shape), code)
        shape = list(a.shape) + [0] * (MAX_DIMS - len(a.shape))
        struct.pack_into(f"<{MAX_DIMS}Q", buf, base + L_SHAPE, *shape)
        struct.pack_into("<QI", buf, base + L_LEN, n, fmt)
        # publish-time telemetry, same tick as the status block
        # (step) + monotonic write stamp (ts) -- inside the seqlock window
        struct.pack_into("<QQ", buf, base + L_STEP, self._last_step,
                         time.monotonic_ns())
        buf[base + L_DATA:base + L_DATA + n] = a.data.cast("B")
        _u64(buf, base, seq + 2)                        # slot seq even
        # publish LAST (aligned u64 store): (frame_id << 8) | slot
        _u64(buf, H_FPTR, ((self._frame_count + 1) << 8) | slot)
        self._frame_count += 1
        return self._frame_count

    def _specs(self) -> Sequence[CameraSpec]:
        """The instantiated camera specs (empty without a manager)."""
        manager = self.camera_manager
        return tuple(manager.specs) if manager is not None else ()

    def _camera_interval(self) -> int:
        """The broadcast camera's legacy render interval (ticks/frame).

        Rig spec.interval; the counter path of the wall scheduler —
        only used when no wall fps is resolvable (managers without a
        policy clock: test doubles / probe runs).  Duck-typed: managers
        without specs publish every tick.
        """
        for s in self._specs():
            if s.name == self.camera:
                try:
                    return max(1, int(getattr(s, "interval", 1)))
                except (TypeError, ValueError):
                    return 1
        return 1

    def _camera_fps(self) -> float | None:
        """The broadcast camera's effective wall fps (§16.1).

        Duck-asks the manager's ``effective_fps`` face (the ONE
        resolution point: target_fps > interval fold > tier default);
        managers without the face fall back to an explicit
        ``spec.target_fps`` attribute only.  ``None`` = counter mode
        (the legacy per-tick interval pacing).
        """
        if not self.camera:
            return None
        fn: Any = getattr(self.camera_manager, "effective_fps", None)
        if callable(fn):
            try:
                fps: Any = fn(self.camera)
                return float(fps) if fps else None
            except Exception:              # noqa: BLE001 (status face)
                return None
        for s in self._specs():
            if s.name == self.camera:
                tf = getattr(s, "target_fps", None)
                if tf:
                    return float(tf)
        return None

    def _publish_due(self) -> bool:
        """True this tick at the camera's render schedule.

        Force flag (a property write just landed) > wall fps due >
        legacy interval counter.  Frames publish UNCONDITIONALLY
        (viewer gating state machine deleted; design §2).
        """
        if self._publish_now:
            return True
        fps = self._camera_fps()
        if fps is not None:
            return (time.time() - self._last_publish) >= (1.0 / fps)
        return self._tick_count % self._camera_interval() == 0

    def _pump_due(self) -> bool:
        """The frozen-pump due check (§2.2; parked service loop).

        Same schedule as :meth:`_publish_due` except the counter-mode
        fallback uses the wall ``PARKED_PUBLISH_PERIOD`` floor (a parked
        run advances no ticks -- the pump is the only publisher).
        """
        if self._publish_now:
            return True
        fps = self._camera_fps()
        period = (1.0 / fps) if fps is not None else self.PARKED_PUBLISH_PERIOD
        return (time.time() - self._last_publish) >= period

    def _render_state(self) -> dict[str, JsonValue] | None:
        """The publishing camera's live render state (``render``).

        ``None`` = the plain publish path / no such face.
        """
        return self._render_state_named(self.camera)

    def _render_state_named(self, name: str | None) \
            -> dict[str, JsonValue] | None:
        getter = getattr(self.camera_manager, "render_state", None)
        if getter is None or not name:
            return None
        try:
            return getter(name)
        except Exception:                     # noqa: BLE001 (status face)
            return None

    def _refresh_status_block(self) -> None:
        """Re-publish the per-tick status JSON with fresh render state.

        A live apply while PARKED must be visible cross-process
        immediately -- the parked loop serves commands but runs no
        ticks.  Best effort: the next tick rewrites it anyway.
        """
        try:
            metrics = dict(self._last_metrics or {})
            state = self._render_state()
            if state is not None:
                metrics["render_source"] = state["render_source"]
            else:
                metrics.pop("render_source", None)
            metrics["capture_backend"] = getattr(self.camera_manager,
                                                 "capture_backend", None)
            self._write_status_locked(step=self._last_step, metrics=metrics)
        except Exception:                     # noqa: BLE001 (status face)
            pass

    def _copy_broadcast_frame(self) -> None:
        """Camera copy; never raises into the sim loop.

        §8.3: the publish loop reads the camera's live render
        STATE (the ``render`` live key); a non-default state renders
        through the RenderSpec capture face with that state (WYSIWYG --
        the stream shows the toggled render), the default (``None``)
        keeps the plain path bit-for-bit.  The capture face itself stays
        stateless.  The publish path consumes the force flag (a property write
        landed -- the schedule restarts from HERE).
        """
        self._last_publish = time.time()
        self._publish_now = False
        if self.camera_manager is None or not self.camera:
            return
        try:
            state = self._render_state()
            if state is not None:
                doc: dict[str, Any] = self.camera_manager.capture(
                    self.camera, None, render=state["render"])
                self._write_frame(doc["rgb"], self.camera, FMT_RAW)
                return
            frame = self.camera_manager.get_view(self.camera, None)
            self._write_frame(frame, self.camera, FMT_RAW)
        except Exception as e:                    # noqa: BLE001
            print(f"[bridge] broadcast frame skipped: {e}", flush=True)

    # -- command block -------------------------------------------------------
    def _serve_commands(self) -> None:
        buf = self.buf
        rid = _u64(buf, CMD_OFF + C_REQ)
        if rid == 0 or rid == self._last_served:
            return
        payload = b""
        for _ in range(2):                    # ck re-read for torn writes
            cmd = _u16(buf, CMD_OFF + C_CMD)
            raw = bytes(buf[CMD_OFF + C_JSON:CMD_OFF + C_JSON + PAYLOAD_JSON])
            payload = raw.split(b"\x00", 1)[0]
            if _payload_ck(payload) == _u64(buf, CMD_OFF + C_CK):
                break
            time.sleep(0.001)
        else:
            return                            # still torn; retry next tick
        # claim BEFORE dispatch: the pause freeze loop serves commands
        # re-entrantly, so the in-flight command must never re-fire
        self._last_served = rid
        self._dispatch(rid, cmd, payload)

    def _dispatch(self, rid: int, cmd: int, payload: bytes) -> None:
        status, result = ACK_OK, AckResult()
        try:
            params: dict[str, JsonValue] = (
                json.loads(payload) if payload else {})
            handler = self._handlers.get(cmd)
            if handler is not None:
                result = handler(rid, params)
            elif cmd == CMD_PAUSE:
                was = self.buf[H_PAUSED]
                self.buf[H_PAUSED] = 1
                self.buf[H_PHASE] = PHASE_PAUSED
                self._gate.pause()         # pause wins over a step budget
                result = {"paused": True, "was_paused": bool(was)}
            elif cmd == CMD_RESUME:
                self._gate.resume()
                self.buf[H_PAUSED] = 0
                self.buf[H_PHASE] = PHASE_RUNNING
                result = {"paused": False}
            elif cmd == CMD_SET_PROPS:
                name = _as_str(params.get("camera")) or self.camera
                if self.camera_manager is None:
                    raise BridgeError("no cameras in this run")
                if name is None:
                    raise BridgeError("no camera selected for set_props")
                self.camera_manager.apply_properties(
                    name, _as_mapping(params.get("params")),
                    _as_int(params.get("env_id")))
                result = {"applied": True, "camera": name}
                # the ack echoes the resulting live render state
                # (the applied source label; the full spec rides when
                # non-default), and the status JSON block is refreshed
                # NOW -- a parked run runs no ticks, so the
                # cross-process status would otherwise show a stale
                # render_source until resume
                state = self._render_state_named(name)
                if state is not None:
                    result["render_source"] = state["render_source"]
                    result["effective_render"] = state["render"]
                else:
                    result["render_source"] = "default"
                self._refresh_status_block()
                # the render schedule restarts NOW -- the next
                # tick / pump beat publishes the new state immediately
                # (the aim -> new-frame <=50ms beat, §16.4)
                self._publish_now = True
            elif cmd == CMD_GET_POSE:
                # freecam face: fixed-rig world-pose readback
                name = _as_str(params.get("camera")) or self.camera
                if self.camera_manager is None:
                    raise BridgeError("no cameras in this run")
                if name is None:
                    raise BridgeError("no camera selected for get_world_pose")
                result = self.camera_manager.get_world_pose(name)
            elif cmd == CMD_GET_VIEW:
                result = self._cmd_get_view(params)
            elif cmd == CMD_SHUTDOWN:
                self.shutdown_requested = True
                result = {"shutdown": True}
            elif cmd == CMD_SET_CAMERA:
                # stream_bind (L2): rebind the publishing producer
                name = _as_str(params.get("camera"))
                if name is None or name not in self.camera_names:
                    raise BridgeError(f"unknown camera {name!r} "
                                      f"(cameras: {self.camera_names})")
                was = self.camera
                self.camera = name
                self._copy_broadcast_frame()       # publish a first frame
                result = {"camera": name, "was": was}
            elif cmd == CMD_METRICS:
                # typed query: document + live status counters
                result = self._metrics.metrics_document(
                    self.run_id, int(_u64(self.buf,
                                          STATUS_OFF + S_STEP)),
                    self._episodes)
            elif cmd == CMD_CONFIG:
                # typed query: effective config (startup snapshot)
                result = {"config": self._config_doc}
            elif cmd == CMD_REPORT:
                # typed query: benchmark draft (None w/o benchmark)
                src = self._metrics_source
                result = {"report": src.metrics_report()
                          if src is not None else None}
            else:
                if cmd in _CONSOLE_CMDS:
                    raise BridgeError(
                        "console not attached to this bridge "
                        "(the run process wires unigym.console at start)")
                if cmd == CMD_SCENE:
                    raise BridgeError(
                        "scene face not mounted on this run (a scene-enabled "
                        "required; older builds answer no typed scene query)")
                if cmd in _REWIND_CMDS:
                    raise BridgeError(
                        "rewind face not mounted on this run (the run "
                        "process wires unigym.runtime.rewind at start; "
                        "needs a snapshot-capable backend + rewindable "
                        "adapter)")
                status = ACK_ERR
                result = {"error": f"unknown command id {cmd}"}
        except Exception as e:                # noqa: BLE001 -> ack, not crash
            # handlers may attach a next-action hint (director_contract
            # §6 error face: {error, hint}).  The ack
            # is the error transport -- prefix the type so a handler bug
            # (TypeError etc.) is visible to the caller, not just its
            # message text.
            hint = getattr(e, "hint", None)
            status = ACK_ERR
            result = {"error": f"{type(e).__name__}: {e}"}
            if hint:
                result["hint"] = str(hint)
        self._write_ack(rid, status, result)
        self._n_served += 1
        # pause freeze happens AFTER the pause ack (CLI unblocks, sim frozen)
        if cmd == CMD_PAUSE and not self._frozen:
            self._frozen = True
            try:
                self._gate_park()
            finally:
                self._frozen = False

    # -- step-gate park (replaces the old inline freeze loop) ---------
    def _gate_park(self) -> None:
        """Park the engine in the StepGate until resume / step budget.

        While parked the bridge runs the FROZEN RENDER PUMP (§2.2):
        the world is frozen (no step), so a periodic wall-clock check
        re-renders the broadcasting camera from the stable mjData and
        publishes to SHM -- camera moves (aim / pose writes) and style
        changes show up on the stream within one poll interval, and the
        parked feed holds the camera's wall fps (the orbit cruise of a
        frozen scene is a smooth video at zero tick cost).  Rendering
        runs on THIS thread -- the engine's EGL-bound thread (the
        copy-render path; no new render thread / GL context anywhere).
        Commands (get_view / set_props / inspect / step / ...) keep
        being served between pump beats.
        """
        env = self.console.env if self.console is not None else None

        def service():
            self._bump_heartbeat()     # parked != dead: the loop iterates
            if self._pump_due():       # the frozen render pump
                self._copy_broadcast_frame()
            self._serve_commands()

        self._gate.wait(env, service,
                        stop_fn=lambda: self._closed
                        or self.shutdown_requested,
                        poll_interval=self.poll_interval)

    def _cmd_get_view(self, params: dict[str, JsonValue]) -> AckResult:
        if self.camera_manager is None:
            raise BridgeError("no cameras in this run")
        pos, lookat = params.get("pos"), params.get("lookat")
        posed = pos is not None or lookat is not None
        # ux-7: a posed snapshot defaults to the debug fly camera "free"
        name = _as_str(params.get("camera")) or \
            ("free" if posed else self.camera)
        if name is None:
            raise BridgeError(
                "no broadcast camera in this run; pass an explicit camera")
        if "flags" in params:
            # owner ruling: the flags key is GONE at the bridge
            # layer (no deprecation window; the CLI's --flags is the
            # only surviving alias and it expands BEFORE the wire).  The
            # RenderSpec key is `render`; both keys together fail-fast
            # here by construction.
            raise BridgeError(
                "'flags' was removed in v054 (RenderSpec face): send "
                "render={'view': name} or a RenderSpec dict -- got "
                f"flags={params['flags']!r}"
                + (" TOGETHER with render= (the two keys are mutually "
                   "exclusive)" if "render" in params else ""))
        if "render" in params:
            # RenderSpec capture face (camera_contract §14): the spec is
            # resolved by the manager (view registry + aspect-merge
            # compiler, complete-or-error).  The rgb frame rides the
            # frame slot; every SECONDARY artifact (orbit GIF, named
            # stills, seg PNG) lands as files under <out_dir>/capture/
            # (the frame slot is a single-latest transport: a second
            # same-command frame would always be superseded for the
            # reader); the legend JSON + effective_render/render_source
            # ride this ack.
            doc: dict[str, Any] = self.camera_manager.capture(
                name, _as_int(params.get("env_id")),
                pos=pos, lookat=lookat, render=params["render"])
            result = {"frame_id": self._write_frame(doc["rgb"], name,
                                                    FMT_RAW),
                      "camera": name, "format": FMT_RAW,
                      "effective_render": doc["effective_render"],
                      "render_source": doc["render_source"]}
            if "gif" in doc:
                gif_path = self._write_capture_artifact(
                    name, result["frame_id"], ".gif", doc["gif"])
                if gif_path is not None:
                    result["gif_path"] = gif_path
                else:
                    result["gif_error"] = (
                        "gif not written (no out_dir on this bridge)")
            if doc.get("frames"):
                paths = {label: p for label, p in
                         self._write_capture_artifacts(
                             name, result["frame_id"], doc["frames"],
                             ".png").items() if p is not None}
                if paths:
                    result["frame_paths"] = paths
            if "seg" in doc:
                seg_path = self._write_capture_artifact(
                    name, result["frame_id"], "_seg.png", doc["seg"])
                if seg_path is not None:
                    result["seg_path"] = seg_path
                else:
                    result["seg_error"] = (
                        "seg not written (no out_dir on this bridge)")
            if "legend" in doc:
                result["legend"] = doc["legend"]
        else:
            if posed:
                frame = self.camera_manager.get_view(
                    name, _as_int(params.get("env_id")),
                    pos=pos, lookat=lookat)
            else:
                frame = self.camera_manager.get_view(
                    name, _as_int(params.get("env_id")))
            if params.get("jpeg"):
                jpg = _encode_jpeg(frame)
                result = {"frame_id": self._write_frame(jpg, name, FMT_JPEG),
                          "camera": name, "format": FMT_JPEG,
                          "bytes": len(jpg)}
            else:
                result = {"frame_id": self._write_frame(frame, name, FMT_RAW),
                          "camera": name, "format": FMT_RAW}
        intr = self._camera_intrinsics(name)
        if intr is not None:
            result["intrinsics"] = intr
        return result

    def _camera_intrinsics(self, name: str) -> dict[str, JsonValue] | None:
        """Pinhole face of one instantiated camera (annotate projection).

        Specs without geometry (test doubles) honestly report none.
        """
        for s in self._specs():
            if s.name == name:
                w, h, f = (getattr(s, "width", None),
                           getattr(s, "height", None),
                           getattr(s, "hfov_deg", None))
                if w is None or h is None or f is None:
                    return None
                return {"width": int(w), "height": int(h),
                        "hfov_deg": float(f)}
        return None

    def _capture_dir(self) -> str | None:
        """``<out_dir>/capture/`` (created) or None without an out_dir."""
        if not self.out_dir:
            return None
        d = os.path.join(os.path.abspath(str(self.out_dir)), "capture")
        os.makedirs(d, exist_ok=True)
        return d

    def _write_capture_artifact(self, name: str, fid: int, suffix: str,
                                payload: NDArray[Any] | bytes
                                | bytearray) -> str | None:
        """One run-side artifact under ``<out_dir>/capture/``.

        ``payload``: GIF/PNG bytes, or an array (2-D int id buffer is
        colorized first).  The fid disambiguates same-second captures.
        Returns None when the bridge has no out_dir (an honest ack field
        instead of a lie -- the recorder/ledger convention).
        """
        d = self._capture_dir()
        if d is None:
            return None
        import numpy as np

        path = os.path.join(
            d, f"{time.strftime('%Y%m%d_%H%M%S')}_{name}_{fid}{suffix}")
        if isinstance(payload, (bytes, bytearray)):
            with open(path, "wb") as f:
                f.write(payload)
            return path
        try:
            from PIL import Image
        except ImportError as exc:
            from unigym._optional import MissingExtraError

            raise MissingExtraError(
                "PIL.Image", "record", "image capture") from exc
        arr = np.asarray(payload)
        if arr.ndim == 2:                     # (H, W) int id buffer safety
            from unigym.cameras.geomflags import colorize_seg
            arr = colorize_seg(arr)
        Image.fromarray(np.ascontiguousarray(arr)).save(path)
        return path

    def _write_capture_artifacts(self, name: str, fid: int,
                                 frames: Mapping[str, NDArray[Any] | bytes],
                                 suffix: str) -> dict[str, str | None]:
        """Named stills -> ``<ts>_<name>_<fid>_<label>.png`` files.

        Labels are sanitized to filename-safe characters; a failed write
        answers None for that label (the ack carries only what landed).
        """
        out: dict[str, str | None] = {}
        for label, frame in frames.items():
            safe = "".join(c if c.isalnum() else "_" for c in str(label))
            out[str(label)] = self._write_capture_artifact(
                name, fid, f"_{safe}{suffix}", frame)
        return out

    @staticmethod
    def _decode_frame(frame: Mapping[str, Any]) -> NDArray[Any]:
        import numpy as np
        dt = np.uint8 if frame["dtype"] == DTYPE_UINT8 else np.float32
        return np.frombuffer(frame["data"], dtype=dt).reshape(frame["shape"])

    # -- debug console face (unigym/console/) ----------------------
    @property
    def done(self) -> bool:
        """True once the run finished (or the bridge is closing)."""
        return self._closed or self.buf[H_PHASE] == PHASE_DONE

    def register_command(self, name: str, fn: CommandHandler) -> None:
        """Register a runtime command handler (extension face).

        ``fn(rid, params) -> result dict``; exceptions land in the ack,
        never the engine loop.  ``name`` is a CMD_NAMES string.
        """
        cmd_id = CMD_IDS.get(name) if isinstance(name, str) else int(name)
        if cmd_id is None:
            raise BridgeError(f"unknown command name {name!r} "
                              f"(known: {sorted(CMD_NAMES.values())})")
        self._handlers[cmd_id] = fn

    def _install_console(self, console: ConsoleFace) -> None:
        """Wire the debug console (unigym.console.attach calls this)."""
        self.console = console
        self.register_command("step", console.cmd_step)
        self.register_command("inspect", console.cmd_inspect)
        self.register_command("save_npz", console.cmd_save_npz)

    def _write_ack(self, rid: int, status: int, result: AckResult) -> None:
        buf = self.buf
        try:
            raw = json.dumps(result).encode()
        except (TypeError, ValueError):
            raw = json.dumps({"error": "unserializable result"}).encode()
        if len(raw) > PAYLOAD_JSON:
            raw = json.dumps({"error": "result too large"}).encode()[
                :PAYLOAD_JSON - 1]
        _u16(buf, ACK_OFF + A_STATUS, status)
        _put_bytes(buf, ACK_OFF + A_JSON, raw, PAYLOAD_JSON)
        _u64(buf, ACK_OFF + A_CK, _payload_ck(raw))
        _u64(buf, ACK_OFF + A_REQ, rid)               # publish LAST

    # -- status / teardown ----------------------------------------------------
    def read_status(self) -> StatusDoc:
        """Owner-side status view (SHM status + bridge-private fields)."""
        buf = self.buf
        st = _read_status_block(buf)
        st.update({"run_id": self.run_id, "shm": self.shm_name,
                   "phase": PHASE_NAMES.get(buf[H_PHASE], "?"),
                   "paused": bool(buf[H_PAUSED]),
                   "gate": self._gate.state,
                   "camera": self.camera,
                   "cameras": list[JsonValue](self.camera_names),
                   "obs_catalog_keys": list[JsonValue](
                       self.obs_catalog_keys),
                   "pid": os.getpid(),
                   "frame_id": _u64(buf, H_FPTR) >> 8,
                   "frames_dropped": self._frames_dropped,
                   "rewinds": len(self.rewind_log)})
        # the owner-side status carries the live render state
        # (the applied source label; null = plain path) + the capture
        # backend face (the webui auto-disable signal) directly; the
        # cross-process copies ride the status JSON block
        state = self._render_state()
        if state is not None:
            st["render_source"] = state["render_source"]
        st["capture_backend"] = getattr(self.camera_manager,
                                        "capture_backend", None)
        # the full watcher status face rides only the owner-side view;
        # cross-process readers get the per-tick status_brief via the
        # status JSON block (metrics["watchers"])
        if self._watchers is not None:
            try:
                st["watchers"] = self._watchers.status()
            except Exception:                  # noqa: BLE001
                st["watchers"] = None
        return st

    def close(self) -> None:
        """Idempotent teardown: phase DONE -> registry removed -> shm unlinked."""
        if self._closed:
            return
        self._closed = True
        try:
            self.buf[H_PHASE] = PHASE_DONE
        except (ValueError, OSError):  # best effort: idempotent close
            pass
        if self._state_path is not None:
            try:
                self._state_path.unlink(missing_ok=True)
            except OSError:  # best effort: state file may already be gone
                pass
        try:
            self.shm.close()
            self.shm.unlink()
        except (FileNotFoundError, BufferError, OSError):  # idempotent teardown
            pass


# --------------------------------------------------------------------------- #
# ShmClient (reader/commander side, cross-process, stdlib-only runtime)
# --------------------------------------------------------------------------- #
class ShmClient:
    """Attach to a live run's SHM and speak the command->ack protocol.

    Read side is seqlock-guarded snapshots; commands are fire-and-poll
    (req_id one-to-one ack, bounded timeout).  One outstanding command at
    a time (single-controller discipline, docs/contracts/runtime_contract.md §3).
    """

    def __init__(self, run_id: str, name: str | None = None):
        self.run_id = run_id
        self.shm_name = name or (SHM_PREFIX + run_id)
        self._buf: mmap.mmap | None = _attach_segment(self.shm_name)
        if bytes(self.buf[H_MAGIC:H_MAGIC + 4]) != MAGIC:
            self.close()
            raise BridgeError(f"{self.shm_name}: bad magic "
                              "(not a unigym bridge segment)")
        ver = _u16(self.buf, H_VERSION)
        if ver != VERSION:
            self.close()
            if ver < VERSION:
                # foreign-layout segment: zero fallback reading -- it shows
                # up as an UNKNOWN row, visible / killable / deletable
                raise BridgeError(
                    f"{self.shm_name}: pre-v0.50 segment (no heartbeat; "
                    f"spawn/upgrade to v0.50+)")
            raise BridgeError(
                f"{self.shm_name}: segment version {ver} > supported "
                f"{VERSION} (upgrade unigym)")

    @property
    def buf(self) -> mmap.mmap:
        """Attached segment (non-Optional while the client is open)."""
        buf = self._buf
        if buf is None:
            raise BridgeError("segment is closed")
        return buf

    # -- header --------------------------------------------------------------
    @property
    def phase(self) -> str:
        """The run's lifecycle phase name ("RUNNING"/"PAUSED"/"DONE")."""
        return PHASE_NAMES.get(self.buf[H_PHASE], "?")

    @property
    def paused(self) -> bool:
        """True while the paused gate holds (header byte)."""
        return bool(self.buf[H_PAUSED])

    @property
    def port(self) -> int:
        """Retired field : runs have no HTTP port; always 0."""
        return _u16(self.buf, H_PORT)

    def close(self) -> None:
        """Detach the segment (idempotent; never raises into callers)."""
        buf = self._buf
        if buf is not None:
            try:
                buf.close()
            except (BufferError, ValueError):  # best effort: export in flight
                pass
            self._buf = None

    def __enter__(self):
        return self

    def __exit__(self, *exc: object):
        self.close()

    # -- status / frames -------------------------------------------------------
    def read_status(self, camera_fallback: str | None = None) \
            -> StatusDoc:
        """Tear-free SHM status + header identity counters (any process).

        ``camera_fallback`` names what the caller already knows from the
        run's serve registry (list_runs has the parsed entry in hand);
        it replaces the registry re-scan in the frameless camera
        lookup, which otherwise costs a full serve-dir listing per
        segment per rebuild.
        """
        buf = self.buf
        st = _read_status_block(buf)
        fptr = _u64(buf, H_FPTR)
        st.update({"run_id": self.run_id, "shm": self.shm_name,
                   "phase": self.phase, "paused": self.paused,
                   "camera": self._camera_name(camera_fallback),
                   "pid": _u64(buf, H_PID), "frame_id": fptr >> 8,
                   "started": _f64(buf, H_STARTED),
                   "hb_seq": _u64(buf, H_HB_SEQ),
                   "hb_ts": _f64(buf, H_HB_TS)})
        return st

    @property
    def camera_name(self) -> str:
        """The publishing camera's name (frame header, else serve registry)."""
        return self._camera_name(None)

    def _camera_name(self, fallback: str | None) -> str:
        # the publishing camera is not in the header; recover from the
        # latest frame when available, else from the caller-provided
        # serve-registry camera, else from the serve-state files
        frame = _read_latest_frame(self.buf, retries=4)
        if frame:
            return frame["camera"]
        if fallback is not None:
            return fallback
        try:
            for p in serve_state_paths():
                st = json.loads(p.read_text())
                if st.get("run_id") == self.run_id:
                    return st.get("camera") or ""
        except (OSError, ValueError):  # best effort: registry gone/garbled
            pass
        return ""

    def read_frame(self, want_fid: int | None = None,
                   timeout: float = 5.0) -> dict[str, Any] | None:
        """Latest frame snapshot; with want_fid, wait until published.

        None on timeout.
        """
        deadline = time.time() + timeout
        while True:
            frame = _read_latest_frame(self.buf, retries=20)
            if frame is not None and (want_fid is None
                                      or frame["frame_id"] >= want_fid):
                return frame
            if time.time() > deadline:
                return None
            time.sleep(0.02)

    # -- commands ------------------------------------------------------------
    def send(self, cmd: str | int,
             payload: dict[str, JsonValue] | None = None,
             timeout: float = 10.0) -> tuple[int, dict[str, Any]]:
        """Write a command, poll the ack.  Returns (status, result)."""
        cmd_id = CMD_IDS[cmd] if isinstance(cmd, str) else int(cmd)
        if self.phase == PHASE_DONE:
            raise BridgeError(
                f"run {self.run_id} is DONE — nothing left to serve "
                f"'{CMD_NAMES.get(cmd_id, cmd_id)}'")
        rid = _write_command(self.buf, cmd_id, payload or {})
        got = _poll_ack(self.buf, rid, time.time() + timeout,
                        lambda: self.buf[H_PHASE])
        if got is None:
            raise BridgeTimeout(
                f"no ack within {timeout}s for "
                f"'{CMD_NAMES.get(cmd_id, cmd_id)}' (req_id {rid}); is the run "
                f"alive and stepping? (phase={self.phase})")
        return got
