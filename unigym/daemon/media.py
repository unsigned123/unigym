"""daemon.media -- the unigymd media gateway (design §3).

The run-side HTTP face moved WHOLESALE here (this
file -- together with webrtc.py -- is deliberately self-contained so the
whole media service can be lifted out later without touching anything
else).  The daemon terminates all browser traffic; runs are pure SHM
workers:

* :class:`RunFrames` -- the shared SHM frame reader (ONE attach per run,
  shared by every MJPEG/WebRTC/status consumer): seqlock latest-frame
  atomic read (reuses the bridge primitives verbatim -- nothing
  re-implemented), want_fid new-frame waiting, and fid/timestamp
  staleness judgement (the zombie-run red flag: a RUNNING run whose
  frame id and status timestamp both stopped advancing).
 * :class:`MediaGateway` -- per-run mux (run x camera) over the readers
   plus the req-ack command faces: MJPEG multipart streaming (viewer
   disconnect detection = write failure -> session cleanup, the bridge
   streaming pattern), single-frame JPEG (broadcast camera straight from the
   slots; any other instantiated camera through the command slot),
   console/inspect/save_npz passthrough (the console req-ack path, via
   unigym.core.session), the control schema face, /info and /status.

Routes (mounted by daemon.portal on 8600, the single browser source):

    GET  /run/<id>/cam/<name>.mjpg        live MJPEG (publishing camera;
                                          drop-oldest + fid dedup)
    GET  /run/<id>/cam/<name>/latest.jpg  latest frame, pure SHM read
                                          (publishing camera; q/scale tiers)
    GET  /run/<id>/cam/<name>/frame.jpg   one JPEG (any instantiated cam)
    POST /run/<id>/console|inspect|save_npz   console req-ack (tokened)
    POST /run/<id>/control                control schema req-ack (tokened)
    GET  /run/<id>/info | /status         self-description / live status
    (m-3 adds POST /run/<id>/offer -- aiortc WebRTC signalling)

Auth tri-state (single point now: the DAEMON, unigym.yaml
``webui_token`` -- semantics unchanged since introduction): ``none`` = open intranet
posture, ``auto`` = one random token per daemon boot, ``fixed`` = the
configured token checked per request.  The GET face stays open in every
mode; portal pages embed the token automatically.

Same-host promise: this module reads /dev/shm directly -- the daemon and
every run MUST be on one machine (multi-machine = one daemon per machine).
"""
from __future__ import annotations

import io
import threading
import time
import urllib.parse
from typing import Any, Protocol

from unigym.console import parse_console, ConsoleError
from unigym.core import registry as core_registry
from unigym.core.session import SessionError, attach
from unigym.runtime import bridge as _bridge
from unigym.runtime.bridge import (DTYPE_UINT8, FMT_JPEG,
                                   _read_latest_frame, _u64, H_FPTR, H_PHASE)
from unigym.runtime.schemas import (CONSOLE_SCHEMA, CONTROL_SCHEMA,
                                    INSPECT_SCHEMA, SAVE_NPZ_SCHEMA,
                                    SCHEMA_VERSION)
from unigym.stream import ControlError, parse_control_json, \
    to_bridge_command

__all__ = ["RunFrames", "MediaGateway", "TierError", "parse_tier",
           "run_info_document", "route_run_get", "route_run_post"]

STALE_AFTER_S = 5.0          # RUNNING run with no fid/status progress
MJPEG_FPS = 25.0             # v069 (design §3.3): ~4-5 Mbps @ 480p
FIRST_FRAME_GRACE_S = 5.0
# QoS (designs/v048-stream-qos §4-a): SIOCOUTQ drop-oldest gate.
# A consumer whose kernel send queue holds more unacked bytes than this
# gets its frames SKIPPED (never queued) -- the backlog stays bounded and
# staleness converges to <= threshold / consumer rate (~0.2-0.5s) instead
# of the measured unbounded 17s-at-30s lag.  ~2 chase frames.
MJPEG_BACKLOG_BYTES = 50_000
JPEG_QUALITY = 90            # default tier == the original wire format


class _JsonHandler(Protocol):
    """The HTTP handler shape the media gateway calls into.

    The daemon's :class:`~unigym.daemon.portal.PortalHandler` supplies these;
    typing the gateway against the protocol (instead of the concrete handler)
    keeps the media face decoupled and lets ``_send`` / ``_json_send`` carry
    their real contracts.
    """

    headers: Any
    wfile: Any

    def send_response(self, code: int, message: str | None = ...) -> None: ...

    def send_header(self, keyword: str, value: str) -> None: ...

    def end_headers(self) -> None: ...

    def _send(self, code: int, raw: bytes, ctype: str,
              cache: str = ...) -> None: ...

    def _json_send(self, code: int, obj: dict) -> None: ...


class TierError(ValueError):
    """Invalid ?q=/scale= media tier parameter."""


def parse_tier(query: str | None) -> tuple[int, float]:
    """Media tier from a query string -> ``(quality, scale)``.

    Absent parameters keep the default wire format (q90, full size);
    non-numeric values raise TierError; out-of-range values clamp into
    the tier table (q 10-95, scale 0.25-1.0).
    """
    quality, scale = JPEG_QUALITY, 1.0
    qs = urllib.parse.parse_qs(query or "")
    try:
        if "q" in qs:
            quality = int(qs["q"][0])
        if "scale" in qs:
            scale = float(qs["scale"][0])
    except ValueError as e:
        raise TierError(f"invalid media tier: {e}") from e
    return max(10, min(95, quality)), max(0.25, min(1.0, scale))


def _encode_tiered(frame: dict, quality: int, scale: float) -> bytes:
    """One SHM frame snapshot -> JPEG bytes at a q/scale tier.

    The default tier (q90, scale 1.0) is the original path verbatim
    (JPEG-slot passthrough / bridge encoder) -- byte-identical output.
    Any other tier re-encodes in the daemon (PIL resize + save); depth
    snapshots keep the bridge min-max normalization first.
    """
    if quality == JPEG_QUALITY and scale == 1.0:
        if frame["format"] == FMT_JPEG:
            return frame["data"]
        return _bridge._encode_jpeg(_bridge.Bridge._decode_frame(frame))
    from PIL import Image
    if frame["format"] == FMT_JPEG:
        img = Image.open(io.BytesIO(frame["data"]))
    elif frame["dtype"] == DTYPE_UINT8:
        img = Image.fromarray(_bridge.Bridge._decode_frame(frame))
    else:                      # depth: min-max norm first (bridge semantics)
        img = Image.open(io.BytesIO(
            _bridge._encode_jpeg(_bridge.Bridge._decode_frame(frame))))
    if scale != 1.0:
        img = img.resize((max(1, int(img.width * scale)),
                          max(1, int(img.height * scale))))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def _backlog_ok(sock) -> bool:
    """True unless the socket's send backlog is over the cap.

    The unacked-send queue (Linux ``SIOCOUTQ``, exposed by CPython's
    termios as ``TIOCOUTQ``) is compared against
    :data:`MJPEG_BACKLOG_BYTES`.  Any failure -- non-Linux (neither
    constant exists), sandboxed ioctl, closed socket -- disables the
    gate so the fast path keeps its byte-identical behavior (design
    §4-a fallback).
    """
    if sock is None:
        return True
    try:
        import fcntl
        import struct
        import termios
        outq = struct.unpack("i", fcntl.ioctl(
            sock, termios.TIOCOUTQ, b"\x00" * 4))[0]
        return outq <= MJPEG_BACKLOG_BYTES
    except Exception:                      # noqa: BLE001 (gate is optional)
        return True


class RunFrames:
    """Shared SHM frame reader for one run (ONE attach, many consumers).

    Staleness judgement (design §3 staleness rule): a RUNNING run is stale when
    BOTH its frame-id and its status timestamp stopped advancing for
    ``stale_after`` seconds -- a SIGSTOP'd or wedged engine shows up red
    in the portal while a DONE run simply reads as finished.
    """

    def __init__(self, run_id: str, stale_after: float = STALE_AFTER_S):
        self.run_id = run_id
        self.stale_after = float(stale_after)
        self.client = _bridge.ShmClient(run_id)
        self.refs = 0                     # MediaGateway acquire/release
        self._last_fid = 0
        self._last_progress = time.time()
        # frame telemetry (design §3): every new fid is observed
        # exactly once even with several consumers polling concurrently
        from .media_telemetry import FrameTelemetry
        self.telemetry = FrameTelemetry()
        self._fid_lock = threading.Lock()

    # -- frames --------------------------------------------------------------
    def latest(self) -> dict | None:
        """Tear-free latest-frame snapshot; tracks fid progress."""
        frame = _read_latest_frame(self.client.buf)
        if frame is not None and frame["frame_id"] > self._last_fid:
            fid = frame["frame_id"]
            with self._fid_lock:
                if fid > self._last_fid:   # re-check: one observe per fid
                    self.telemetry.observe(fid)
                    self._last_fid = fid
                    self._last_progress = time.time()
        return frame

    def wait_frame(self, want_fid: int | None = None, timeout: float = 5.0,
                   camera: str | None = None) -> dict | None:
        """Latest frame (>= want_fid) or None; used by exact-frame fetches."""
        deadline = time.time() + timeout
        while True:
            frame = self.latest()
            if frame is not None and frame["camera"] == camera \
                    and (want_fid is None
                         or frame["frame_id"] >= want_fid):
                return frame
            if time.time() > deadline:
                return None
            time.sleep(0.02)

    # -- status / staleness ----------------------------------------------------
    def raw_status(self) -> dict:
        """The raw SHM status document (no media-face additions)."""
        return self.client.read_status()

    def status(self) -> dict:
        """SHM status + the media face (contract streams §3.5 shape).

        Adds the stale marker, the publishing camera, and the v054
        live render-state mirror from the status JSON block.
        """
        st = self.raw_status()
        st["camera"] = self.client.camera_name
        st["stale"] = self.stale(raw=st)
        st["frame_id"] = int(_u64(self.client.buf, H_FPTR) >> 8)
        metrics = st.get("metrics") or {}
        if "render_source" in metrics:
            st["render_source"] = metrics["render_source"]
        if "capture_backend" in metrics:   # null = no capture face
            st["capture_backend"] = metrics["capture_backend"]
        return st

    def stale(self, raw: dict | None = None) -> bool:
        """Staleness judgment (design §3 rule, see the class docstring)."""
        try:
            st = raw or self.raw_status()
            phase = int(self.client.buf[H_PHASE])
            if phase == _bridge.PHASE_DONE:
                return False
            status_ts = float(st.get("ts") or 0.0)
            newest = max(self._last_progress, status_ts)
            return (time.time() - newest) > self.stale_after
        except Exception:                      # noqa: BLE001 (unreadable)
            return True

    def media(self) -> dict:
        """The frame-telemetry section (design §3).

        Publish/consumed rates, gap counters, last observed fid,
        staleness.
        """
        tel = self.telemetry.snapshot()
        tel["last_fid"] = self._last_fid
        tel["stale"] = self.stale()
        return tel

    @property
    def phase(self) -> str:
        """The run's lifecycle phase: RUNNING / PAUSED / DONE."""
        return self.client.phase

    def close(self) -> None:
        """Detach the SHM client (idempotent via the bridge)."""
        self.client.close()


class MediaGateway:
    """Per-run reader mux + the req-ack command faces (design §3)."""

    def __init__(self, auth_mode: str = "none", control_token: str | None = None,
                 stale_after: float = STALE_AFTER_S):
        self.readers: dict[str, RunFrames] = {}
        self.stale_after = float(stale_after)
        self.lock = threading.Lock()
        self.auth_mode = str(auth_mode or "none")
        self.control_token = control_token if self.auth_mode != "none" \
            else None

    # -- auth (single point; original daemon semantics) ------------------------------
    def token_ok(self, query: str | None, headers=None) -> bool:
        """Single-point auth check: ?token= query or X-Unigym-Token header."""
        if self.control_token is None:
            return True
        import hmac
        given = ""
        if query:
            given = urllib.parse.parse_qs(query).get("token", [""])[0]
        if not given and headers is not None:
            given = headers.get("X-Unigym-Token", "")
        return bool(hmac.compare_digest(given, self.control_token))

    # -- readers ---------------------------------------------------------------
    def reader(self, run_id: str, create: bool = True) -> RunFrames | None:
        """Check the shared reader OUT for one consumer (refs +1).

        Consumers: MJPEG stream, frame fetch, status/info request,
        webrtc pump.  EVERY caller must hand it back with
        :meth:`release` in a ``finally`` -- a reader is closed only by
        that path (last consumer out of a finished run), never under a
        live user (the viewer refresh race: the old viewer's DONE-
        cleanup used to close the reader the freshly-reloaded stream
        was already using).  :meth:`drop` stays the explicit teardown
        for shutdown/tests.
        """
        with self.lock:
            rd = self.readers.get(run_id)
            if rd is None and create:
                try:
                    rd = RunFrames(run_id, stale_after=self.stale_after)
                except (_bridge.BridgeError, FileNotFoundError, OSError):
                    return None
                self.readers[run_id] = rd
            if rd is not None:
                rd.refs += 1
            return rd

    def release(self, run_id: str) -> None:
        """Hand one reader reference back.

        Closing happens when the LAST consumer leaves AND the run is
        finished (or its SHM is already unreadable); live runs keep the
        cached reader for the next viewer (attach is expensive,
        mappings are cheap).
        """
        with self.lock:
            rd = self.readers.get(run_id)
            if rd is None:
                return
            rd.refs = max(0, rd.refs - 1)
            if rd.refs > 0:
                return
            try:
                finished = rd.phase == "DONE"
            except Exception:              # noqa: BLE001 (closed/unreadable)
                finished = True
            if not finished:
                return
            self.readers.pop(run_id, None)
        rd.close()

    def drop(self, run_id: str) -> None:
        """Explicit teardown of one reader (shutdown/tests)."""
        with self.lock:
            rd = self.readers.pop(run_id, None)
        if rd is not None:
            rd.close()

    def close(self) -> None:
        """Close every cached reader (daemon shutdown)."""
        with self.lock:
            readers = list(self.readers.values())
            self.readers.clear()
        for rd in readers:
            rd.close()

    # -- MJPEG (multipart; disconnect = write failure -> cleanup) -------------
    def mjpeg(self, handler: _JsonHandler, run_id: str,
              camera: str, query: str = "") -> None:
        """Stream GET /run/<id>/cam/<name>.mjpg (multipart MJPEG).

        Publishing camera only; disconnect = write failure -> the
        reader reference is released in ``finally``.
        """
        rd = self.reader(run_id)
        if rd is None:
            return handler._json_send(404, {
                "ok": False, "error": f"no such run: {run_id}",
                "hint": "GET / (portal) lists live runs, or "
                        "GET /api/runs on the control API"})
        try:
            self._mjpeg_stream(handler, run_id, camera, rd, query)
        except (BrokenPipeError, ConnectionResetError):
            pass                          # viewer left: session cleaned up
        finally:
            # the drop-on-DONE now lives in release(): the reader closes
            # when the LAST consumer leaves a finished run -- never under
            # a concurrent stream (the viewer refresh race)
            self.release(run_id)

    def _mjpeg_stream(self, handler: _JsonHandler, run_id: str,
                      camera: str, rd: RunFrames, query: str = "") -> None:
        publishing = rd.client.camera_name
        if not publishing or camera != publishing:
            cams = _registry_cameras(run_id)
            return handler._json_send(404, {
                "ok": False,
                "error": f"camera {camera!r} is not the publishing camera "
                         f"({publishing!r}); MJPEG streams the publishing "
                         f"camera only",
                "hint": (f"switch cameras from the view page (stream_bind) "
                         f"or use /run/{run_id}/cam/{camera}/frame.jpg"
                         if camera in cams else
                         f"instantiated cameras: {cams or 'none'}; "
                         f"config-declared cameras missing from the run "
                          f"usually need --enable_cameras at start")})
        try:
            quality, scale = parse_tier(query)
        except TierError as e:
            return handler._json_send(400, {
                "ok": False, "error": str(e),
                "hint": "media tiers: ?q=<10-95 JPEG quality> and "
                        "?scale=<0.25-1.0 resize factor>, both optional "
                        "(default: q90 full size)"})
        handler.send_response(200)
        handler.send_header("Content-Type",
                            "multipart/x-mixed-replace; "
                            "boundary=unigymframe")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        sock = getattr(handler, "connection", None)
        last_fid = -1
        first_deadline = time.time() + FIRST_FRAME_GRACE_S
        while True:
            frame = self._next_frame(rd, first_deadline)
            if frame is None or frame["camera"] != camera:
                break
            fid = frame["frame_id"]
            # drop-oldest + fid dedup: a slow consumer's kernel
            # send queue grows past MJPEG_BACKLOG_BYTES -> the stale
            # frame is skipped (never queued), so the client degrades to
            # "realtime with frame drops" instead of unbounded delayed
            # replay; the fid gate never re-sends a frame the publish
            # side has not advanced past.  Fast clients see a zero gate.
            if fid > last_fid and _backlog_ok(sock):
                t0 = time.monotonic()
                jpg = _encode_tiered(frame, quality, scale)
                part = (b"--unigymframe\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpg)).encode()
                        + b"\r\n\r\n")
                handler.wfile.write(part + jpg + b"\r\n")
                last_fid = fid
                # v069 pacing: the idle sleep shrinks by the work the
                # iteration already spent (encode + write), so the wire
                # cadence stays ~MJPEG_FPS under GIL contention instead
                # of decaying to 1/(period + work).
                time.sleep(max(0.001, 1.0 / MJPEG_FPS
                               - (time.monotonic() - t0)))
            else:
                time.sleep(1.0 / MJPEG_FPS)
            if rd.phase == "DONE":        # finished: end the stream here;
                break                     # the browser keeps the last frame
            first_deadline = 0.0     # subsequent frames: no grace

    def _next_frame(self, rd: RunFrames, first_deadline: float) -> dict | None:
        while True:
            frame = rd.latest()
            if frame is not None:
                return frame
            if rd.phase == "DONE" or time.time() > first_deadline:
                return None
            time.sleep(0.05)

    # -- single frames -------------------------------------------------------
    def frame_jpeg(self, handler: _JsonHandler, run_id: str,
                   camera: str) -> None:
        """Serve one JPEG from any instantiated camera.

        Broadcast camera: straight from the SHM slots (no command).
        Any other instantiated camera: command-slot get_view req-ack
        (the ux-6 path, now via unigym.core.session).
        """
        rd = self.reader(run_id)
        if rd is None:
            return handler._json_send(404, {
                "ok": False, "error": f"no such run: {run_id}",
                "hint": "GET / (portal) lists live runs"})
        try:
            self._frame_jpeg(handler, run_id, camera, rd)
        finally:
            self.release(run_id)

    def _frame_jpeg(self, handler: _JsonHandler, run_id: str,
                    camera: str, rd: RunFrames) -> None:
        cams = _registry_cameras(run_id)
        if camera not in cams:
            return handler._json_send(404, {
                "ok": False,
                "error": f"camera {camera!r} is not instantiated in this "
                         f"run (instantiated: {cams or 'none'})",
                "hint": "config-declared cameras missing from this list "
                        "usually mean the run started without "
                        "--enable_cameras (isaaclab); restart the run "
                        f"with it, or use one of: {cams or 'none'}"})
        if camera == rd.client.camera_name:
            frame = self._next_frame(rd, time.time() + FIRST_FRAME_GRACE_S)
            if frame is not None and frame["camera"] == camera:
                jpg = frame["data"] if frame["format"] == FMT_JPEG else \
                    _bridge._encode_jpeg(_bridge.Bridge._decode_frame(frame))
                return handler._send(200, jpg, "image/jpeg")
            return handler._json_send(404, {
                "ok": False, "error": "no frame yet",
                "hint": "the worker publishes on the camera render "
                        "interval; check /run/<id>/status (DONE / stale?) "
                        "and retry"})
        # non-publishing camera -> command slot
        try:
            with attach(run_id, timeout_s=10.0) as s:
                ack = s._send("get_view", {"camera": camera, "jpeg": True},
                              timeout_s=10.0)
        except SessionError as e:
            return handler._json_send(400, {"ok": False, "error": str(e)})
        jpg = self._fetch_exact_jpeg(rd, camera, ack["frame_id"])
        if jpg is not None:
            return handler._send(200, jpg, "image/jpeg")
        handler._json_send(202, {
            "ok": None, "type": "frame", "result": ack,
            "hint": "frame not retrievable (raced or engine stalled); "
                    "retry"})

    def _fetch_exact_jpeg(self, rd: RunFrames, camera: str,
                          want_fid: int) -> bytes | None:
        """frame_id-exact retrieval with bounded re-issue (pattern)."""
        frame = rd.wait_frame(want_fid=want_fid, timeout=2.0, camera=camera)
        if frame is not None and frame["format"] == FMT_JPEG:
            return frame["data"]
        return None

    # -- latest frame (pure SHM read, zero command slot) ---------------
    def latest_jpeg(self, handler: _JsonHandler, run_id: str,
                    camera: str, query: str = "") -> None:
        """Serve GET /run/<id>/cam/<name>/latest.jpg.

        The newest frame of the PUBLISHING camera, read straight from
        the SHM slots (never the command slot; a non-publishing camera
        is an explicit 404, it must not wake the engine).
        ``X-Frame-Id`` carries the source frame id; ?q=&scale= tiers
        apply (design v048 §4-b/c).
        """
        rd = self.reader(run_id)
        if rd is None:
            return handler._json_send(404, {
                "ok": False, "error": f"no such run: {run_id}",
                "hint": "GET / (portal) lists live runs"})
        try:
            if camera != rd.client.camera_name:
                cams = _registry_cameras(run_id)
                return handler._json_send(404, {
                    "ok": False,
                    "error": f"camera {camera!r} has no SHM latest frame "
                             f"(publishing camera: {rd.client.camera_name!r})",
                    "hint": ("latest.jpg serves the broadcasting camera "
                             "only and never wakes the engine; use "
                             f"/run/{run_id}/cam/{camera}/frame.jpg "
                             "(command slot get_view)"
                             if camera in cams else
                             f"instantiated cameras: {cams or 'none'}")})
            try:
                quality, scale = parse_tier(query)
            except TierError as e:
                return handler._json_send(400, {
                    "ok": False, "error": str(e),
                    "hint": "media tiers: ?q=<10-95 JPEG quality> and "
                            "?scale=<0.25-1.0 resize factor>, both "
                            "optional (default: q90 full size)"})
            frame = self._next_frame(rd, time.time() + FIRST_FRAME_GRACE_S)
            if frame is None or frame["camera"] != camera:
                return handler._json_send(404, {
                    "ok": False, "error": "no frame yet",
                    "hint": "the worker publishes on the camera render "
                            "interval; check /run/<id>/status (DONE / "
                            "stale?) and retry"})
            jpg = _encode_tiered(frame, quality, scale)
            handler.send_response(200)
            handler.send_header("Content-Type", "image/jpeg")
            handler.send_header("Cache-Control", "no-store")
            handler.send_header("X-Frame-Id", str(frame["frame_id"]))
            handler.send_header("Content-Length", str(len(jpg)))
            handler.end_headers()
            handler.wfile.write(jpg)
        finally:
            self.release(run_id)

    # -- console / control (req-ack via core.session) --------------------------
    def console(self, handler: _JsonHandler, run_id: str,
                obj: dict, timeout: float) -> None:
        """The console faces: POST /run/<id>/console plus inspect / save_npz.

        Console schema -> command slot req-ack via core.session.
        """
        try:
            cmd, payload = parse_console(obj)
        except (ConsoleError, ValueError) as e:
            return handler._json_send(400, {
                "ok": False, "error": str(e),
                "hint": "GET /run/<id>/info -> endpoints -> body.examples"})
        try:
            with attach(run_id, timeout_s=timeout) as s:
                status, result = s.client.send(cmd, payload, timeout=timeout)
        except SessionError as e:
            msg = str(e)
            code = 404 if ("no live bridge run" in msg
                           or "cannot attach" in msg) else 400
            return handler._json_send(code, {"ok": False, "error": msg})
        except _bridge.BridgeTimeout as e:
            return handler._json_send(202, {
                "ok": None, "req_id": None, "type": obj.get("type"),
                "hint": str(e)})
        ok = status == _bridge.ACK_OK
        handler._json_send(200, {"ok": ok, "type": cmd, "result": result})

    def control(self, handler: _JsonHandler, run_id: str,
                body, timeout: float) -> None:
        """POST /run/<id>/control.

        Control schema (stream/__init__) -> command slot.  Drives the
        view page (pause/resume/stream_bind camera switch/snapshot).
        """
        from unigym.stream import parse_control
        try:
            if isinstance(body, dict):
                ctrl = parse_control(body)
            else:
                ctrl = parse_control_json(
                    body if isinstance(body, (bytes, str)) else "")
            cmd, payload = to_bridge_command(ctrl)
        except ControlError as e:
            return handler._json_send(400, {
                "ok": False, "error": str(e),
                "hint": "GET /run/<id>/info -> endpoints -> body.examples"})
        try:
            with attach(run_id, timeout_s=timeout) as s:
                status, result = s.client.send(cmd, payload, timeout=timeout)
        except SessionError as e:
            return handler._json_send(400, {"ok": False, "error": str(e)})
        except _bridge.BridgeTimeout as e:
            return handler._json_send(202, {
                "ok": None, "type": ctrl["type"], "hint": str(e)})
        handler._json_send(200, {"ok": status == _bridge.ACK_OK,
                                 "type": ctrl["type"], "result": result})


# --------------------------------------------------------------------------- #
# /run/<id>/info document (daemon face of the old run-bridge /info)
# --------------------------------------------------------------------------- #
def _media_endpoints(tokened: bool) -> list[dict]:
    return [
        {"method": "GET", "path": "/run/{run_id}/status", "auth": False,
         "desc": "live status JSON (step/phase/stale/cameras/metrics)"},
        {"method": "GET", "path": "/run/{run_id}/info", "auth": False,
         "desc": "this self-description -- start here"},
        {"method": "GET", "path": "/run/{run_id}/cam/{camera}.mjpg",
         "auth": False,
         "desc": "MJPEG live stream of the PUBLISHING camera"},
        {"method": "GET", "path": "/run/{run_id}/cam/{camera}/frame.jpg",
         "auth": False,
         "desc": "single JPEG; ANY instantiated camera (publishing camera "
                 "serves from the SHM slots, others via the command slot)"},
        {"method": "GET",
         "path": "/run/{run_id}/cam/{camera}/latest.jpg", "auth": False,
         "desc": "latest-frame JPEG of the PUBLISHING camera only (pure "
                 "SHM read, never the command slot); tiers ?q=10-95 "
                 "?scale=0.25-1.0; X-Frame-Id response header = source "
                 "frame id"},
        {"method": "POST", "path": "/run/{run_id}/console", "auth": tokened,
         "desc": "debug console envelope (step/inspect/eval/save_npz)"},
        {"method": "POST", "path": "/run/{run_id}/inspect", "auth": tokened,
         "desc": "shorthand: bare console-inspect payload"},
        {"method": "POST", "path": "/run/{run_id}/save_npz", "auth": tokened,
         "desc": "shorthand: bare console save_npz payload"},
        {"method": "POST", "path": "/run/{run_id}/control", "auth": tokened,
         "desc": "control schema (pause/resume/snapshot/stream_bind)"},
        {"method": "POST", "path": "/run/{run_id}/offer", "auth": tokened,
         "desc": "WebRTC signalling (aiortc answer; m-3)"},
    ]


_INFO_CONSOLE_BODY = {
    "shape": "one JSON object: {type: <name>, ...type-specific keys}",
    "types": ["step", "inspect", "eval", "save_npz"],
    "examples": [
        {"type": "step", "n": 1},
        {"type": "inspect", "target": "summary"},
        {"type": "inspect", "target": "obs", "key": "<obs-key>"},
        {"type": "eval", "expr": "np.arange(4).sum()"},
        {"type": "save_npz", "source": "all"},
    ],
}

_INFO_CONTROL_BODY = {
    "shape": "one JSON object: {type: <name>, ...type-specific keys}",
    "types": ["pause", "resume", "shutdown", "snapshot",
              "camera_set_properties", "stream_bind"],
    "examples": [
        {"type": "pause"},
        {"type": "stream_bind", "camera": "<camera-name>"},
        {"type": "snapshot", "camera": "free",
         "pos": [2.0, -2.0, 1.5], "lookat": [0.0, 0.0, 0.8]},
    ],
}


def run_info_document(run_id: str, gateway: MediaGateway | None = None) \
        -> dict:
    """One run's self-description from SHM + registry v2 (design §3)."""
    entry = core_registry.read_entry(run_id)
    rd = gateway.reader(run_id) if gateway is not None else None
    if rd is not None:
        try:
            st = rd.status()
        finally:
            if gateway is not None:
                gateway.release(run_id)
    else:
        try:
            rd = RunFrames(run_id)         # one-shot attach (not cached)
        except (_bridge.BridgeError, FileNotFoundError, OSError) as e:
            raise SessionError(f"cannot attach to '{run_id}': {e}") from e
        try:
            st = rd.status()
        finally:
            rd.close()
    tokened = bool(gateway and gateway.control_token is not None)
    return {
        "run_id": run_id,
        "shm": _bridge.SHM_PREFIX + run_id,
        "pid": int(st.get("pid") or 0),
        "phase": st.get("phase"),
        "gate": "PAUSED" if st.get("paused") else "RUNNING",
        "step": st.get("step"),
        "episodes": st.get("episodes"),
        "stale": st.get("stale"),
        "frame_id": st.get("frame_id"),
        "auth": {"mode": (gateway.auth_mode if gateway else "none"),
                 "tokened": tokened,
                 "token_header": "X-Unigym-Token (or ?token= query)"},
        "cameras": {"broadcast": st.get("camera"),
                    "instantiated": list(entry.get("cameras") or [])},
        "render_source": st.get("render_source"),
        "capture_backend": st.get("capture_backend"),
        "obs_catalog_keys": list(entry.get("obs_catalog_keys") or []),
        "output_dir": entry.get("output_dir"),
        "log_path": entry.get("log_path"),
        "registry": entry.get("registry_path"),
        "endpoints": _media_endpoints(tokened),
        "schemas": {"version": SCHEMA_VERSION,
                    "control": CONTROL_SCHEMA,
                    "console": CONSOLE_SCHEMA,
                    "inspect": INSPECT_SCHEMA,
                    "save_npz": SAVE_NPZ_SCHEMA},
    }


def _registry_cameras(run_id: str) -> list[str]:
    entry = core_registry.read_entry(run_id)
    return list(entry.get("cameras") or [])


# --------------------------------------------------------------------------- #
# portal-side route adapters (the 8600 handler delegates /run/<id>/... here)
# --------------------------------------------------------------------------- #
def _split_run_path(path: str) -> tuple[str, str] | None:
    """/run/<run_id>/<rest> -> (run_id, rest) or None (not a run route)."""
    parts = urllib.parse.unquote(path).split("/", 3)[1:]
    if len(parts) == 3 and parts[0] == "run":
        return parts[1], parts[2]
    return None


def route_run_get(handler: _JsonHandler, path: str, query: str,
                  svc) -> bool:
    """Serve GET /run/<id>/... ; returns False when not a run route."""
    parsed = _split_run_path(path)
    if parsed is None:
        return False
    run_id, rest = parsed
    gw: MediaGateway = svc.media
    if rest == "status":
        rd = gw.reader(run_id)
        if rd is None:
            handler._json_send(404, {"ok": False,
                                     "error": f"no such run: {run_id}"})
        else:
            try:
                handler._json_send(200, rd.status())
            finally:
                gw.release(run_id)
    elif rest == "info":
        try:
            handler._json_send(200, run_info_document(run_id, gw))
        except SessionError as e:
            handler._json_send(404, {"ok": False, "error": str(e)})
    elif rest.startswith("cam/"):
        route = rest[len("cam/"):]
        if route.endswith(".mjpg"):
            gw.mjpeg(handler, run_id, urllib.parse.unquote(route[:-5]),
                     query)
        elif route.endswith("/latest.jpg"):
            gw.latest_jpeg(handler, run_id,
                           urllib.parse.unquote(route[:-11]), query)
        elif route.endswith("/frame.jpg") or route.endswith("/frame.jpeg"):
            gw.frame_jpeg(handler, run_id,
                          urllib.parse.unquote(route[:route.rfind("/frame")]))
        else:
            handler._json_send(404, {
                "ok": False, "error": f"bad camera route: /cam/{route}",
                "hint": "use /run/<id>/cam/<name>.mjpg (live), "
                        "/run/<id>/cam/<name>/latest.jpg (latest frame) or "
                        "/run/<id>/cam/<name>/frame.jpg (single frame)"})
    else:
        handler._json_send(404, {
            "ok": False, "error": f"no such endpoint: /run/{run_id}/{rest}",
            "hint": "GET /run/<id>/info lists every media endpoint"})
    return True


def route_run_post(handler: _JsonHandler, path: str, query: str,
                   body, svc) -> bool:
    """Serve POST /run/<id>/... (tokened faces); False when not a run."""
    parsed = _split_run_path(path)
    if parsed is None:
        return False
    run_id, rest = parsed
    gw: MediaGateway = svc.media
    if rest not in ("console", "inspect", "save_npz", "control", "offer"):
        handler._json_send(404, {
            "ok": False, "error": f"no such endpoint: /run/{run_id}/{rest}",
            "hint": "console | inspect | save_npz | control | offer"})
        return True
    if not gw.token_ok(query, handler.headers):
        handler._json_send(401, {
            "ok": False,
            "error": "invalid or missing token (append ?token=... or send "
                     "the X-Unigym-Token header)",
            "hint": f"auth={gw.auth_mode}: read the token from the daemon "
                    f"log [webui] line, then retry with ?token=<token>"})
        return True
    try:
        timeout = float(urllib.parse.parse_qs(query)
                        .get("timeout", ["8"])[0])
    except ValueError:
        timeout = 8.0
    timeout = max(0.2, min(15.0, timeout))
    obj = body
    if rest in ("inspect", "save_npz") and isinstance(obj, dict):
        obj = dict(obj, type=rest)
    if rest == "control":
        gw.control(handler, run_id, body, timeout)
        return True
    if rest == "offer":
        # WebRTC signalling (m-3): webrtc.py (and thus aiortc) imports
        # LAZILY here so a daemon without it keeps every other feature
        # (design §3).  Any import failure -- missing aiortc OR a
        # deliberately removed/blocked webrtc module -- degrades to 501.
        if not isinstance(body, dict) or not body.get("sdp"):
            handler._json_send(400, {
                "ok": False, "error": "body needs {'sdp': <offer sdp>}",
                "hint": "create an RTCPeerConnection offer and POST its "
                        "JSON {sdp, type:'offer'}"})
            return True
        try:
            answer = svc.webrtc_gateway().offer(run_id, str(body["sdp"]))
        except ImportError as e:
            handler._json_send(501, {
                "ok": False,
                "error": f"WebRTC unavailable (aiortc/webrtc import "
                         f"failed: {e})",
                "hint": "install aiortc in the daemon venv, or use the "
                        "MJPEG stream: /run/<id>/cam/<name>.mjpg"})
            return True
        except KeyError:
            handler._json_send(404, {
                "ok": False, "error": f"no such run: {run_id}",
                "hint": "GET / (portal) lists live runs"})
            return True
        except Exception as e:                 # noqa: BLE001
            handler._json_send(500, {"ok": False, "error": str(e)})
            return True
        handler._json_send(200, {"sdp": answer, "type": "answer"})
        return True
    gw.console(handler, run_id, obj, timeout)
    return True
