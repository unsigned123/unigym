"""daemon.webrtc -- aiortc WebRTC face of the media gateway .

This file is deliberately SELF-CONTAINED (together
with media.py) so the whole WebRTC service can be lifted out later
without touching anything else.  It is the only module importing aiortc;
the daemon loads it LAZILY (on the first ``/run/<id>/offer`` request) --
without aiortc installed, every other daemon/media feature keeps
working.

Ported from unigym/stream/media_webrtc.py (S2, now deleted --
the standalone media process is obsolete: media terminates in the
daemon):

* **Video** -- a pump thread polls the shared frame readers
  (:class:`unigym.daemon.media.RunFrames`, frame-id dedup) and converts
  snapshots into ``av.VideoFrame``: raw uint8 RGB -> ``rgb24``; raw
  float32 depth -> min-max normalized ``gray``; JPEG snapshots ->
  decoded through PyAV.  aiortc encodes H.264 internally (libx264,
  ``tune=zerolatency`` + ultrafast preset, default 1 Mbps; browsers can
  raise it via ``b=AS:`` in the offer).  ENCODING HAPPENS HERE -- the
  run pays nothing beyond its unconditional SHM publishing.
* **Per-PC independent track** -- every peer connection gets its OWN
  :class:`FrameRelayTrack` (its own asyncio.Event + encoder).  The
  ported lesson is preserved: a SHARED event across PCs starves (one
  awaiting recv() eats the set() and the others never wake).
* **Signalling** -- ``POST /run/<id>/offer`` with an SDP offer JSON body
  returns the SDP answer inline (1:1, no signalling server; CORS open
  so the portal-served view page can call it).
* **Control plane** -- the offerer may create a DataChannel named
  ``control``; JSON messages use the SAME unigym.stream control schema
  as POST /run/<id>/control and are executed against the run's command
  slot via unigym.core.session (single-controller discipline applies:
  this is one more writer).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from unigym._optional import require
from unigym.runtime import bridge as _bridge
from unigym.stream import ControlError, parse_control_json, to_bridge_command

if TYPE_CHECKING:  # pragma: no cover - optional media extra (lazy at runtime)
    import av
    from aiortc.mediastreams import MediaStreamError, VideoStreamTrack
else:
    av = require("av", "media", feature="the WebRTC media gateway")
    _mediastreams = require("aiortc.mediastreams", "media", feature="the WebRTC media gateway")
    MediaStreamError = _mediastreams.MediaStreamError
    VideoStreamTrack = _mediastreams.VideoStreamTrack

__all__ = ["FrameRelayTrack", "WebRtcGateway"]

PUMP_HZ = 60.0
MAX_WIDTH = 1280  # downscale guard (16-aligned below)

# pts pacing: frame timestamps follow the WALL arrival
# spacing of the SHM pump, clamped to the pump ceiling (60 fps) -- the
# fixed 1/30 next_timestamp() step capped delivery at 30 fps and lied
# about arrival jitter.  Presentation rate = source publish rate; the
# CONTENT rate still equals the run's render/tick rate (decoupling
# smooths presentation, it does not create information -- honest note
# in camera_contract §16.5).
PTS_CLOCK = 90_000
PTS_TIME_BASE = Fraction(1, PTS_CLOCK)
MIN_FRAME_S = 1.0 / PUMP_HZ


# --------------------------------------------------------------------------- #
# frame conversion (unit-testable, no event loop)
# --------------------------------------------------------------------------- #
def snapshot_to_video_frame(frame: dict) -> av.VideoFrame:
    """One SHM frame-slot snapshot -> av.VideoFrame (owned copy)."""
    fmt = frame["format"]
    if fmt == _bridge.FMT_JPEG:
        codec = av.CodecContext.create("mjpeg", "r")  # stateless per snap
        dec = codec.decode(av.Packet(frame["data"]))
        if not dec:
            raise ValueError("empty mjpeg decode")
        return dec[0].reformat(format="yuv420p")
    dt = np.uint8 if frame["dtype"] == 0 else np.float32
    arr = np.frombuffer(frame["data"], dtype=dt).reshape(frame["shape"])
    if arr.dtype == np.float32:  # metric depth -> gray
        a = arr.squeeze()
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        if not np.isfinite(a).all() or hi <= lo:
            g = np.zeros(a.shape, dtype=np.uint8)
        else:
            g = ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)
        return av.VideoFrame.from_ndarray(g, format="gray")
    if arr.ndim == 2:  # single-channel u8
        return av.VideoFrame.from_ndarray(arr, format="gray")
    return av.VideoFrame.from_ndarray(arr[..., :3], format="rgb24")


def downscale(frame: av.VideoFrame, max_width: int = MAX_WIDTH) -> av.VideoFrame:
    """Width-guard reformat (camera resolutions above 1280 waste uplink)."""
    if frame.width <= max_width:
        return frame
    h = max(2, int(frame.height * max_width / frame.width) // 2 * 2)
    return frame.reformat(width=max_width // 2 * 2, height=h, format="yuv420p")


# --------------------------------------------------------------------------- #
# relay track (one per PC -- the shared-Event starvation lesson)
# --------------------------------------------------------------------------- #
class FrameRelayTrack(asyncio.Event, VideoStreamTrack):
    """SHM frame slot -> WebRTC track (new-frame-id pacing, wall pts).

    ``feed()`` is called from the pump thread; ``recv()`` resolves with
    the newest frame (duplicate frame ids are skipped).  pts follows the
    WALL spacing between emits, clamped below by
    ``MIN_FRAME_S`` -- pts never runs ahead of the pump ceiling and never
    regresses (the fixed 1/30 ``next_timestamp()`` step capped delivery
    at 30 fps regardless of arrival rate).  KEEPALIVE (the docstring
    promise, kept): the LAST SENT frame is kept -- a 1 s stall with no
    new frame re-emits it under a fresh (wall-advanced) pts, so the
    encoder/browser clock never dries up (a frozen world / parked loop
    still streams a steady picture instead of starving).  ``freeze()``
    (the run reached DONE) stops accepting new frames but keeps the
    keepalive loop -- the track holds its last picture, aligned with
    the MJPEG face's DONE behavior (no black screen).
    """

    KEEPALIVE_S = 1.0

    def __init__(self):
        asyncio.Event.__init__(self)
        VideoStreamTrack.__init__(self)
        self._latest: av.VideoFrame | None = None
        self._last_sent: av.VideoFrame | None = None
        self._fid = 0
        self._lock = threading.Lock()
        self.stopped = False
        self.frozen = False
        # wall-pts bookkeeping (v069): monotonic stamp + running pts of
        # the previous emit (recv is a single async consumer -- the
        # aiortc pull loop awaits sequentially, no lock needed)
        self._last_emit_mono: float | None = None
        self._last_pts = 0

    def feed(self, fid: int, frame: av.VideoFrame) -> None:
        """Accept one new frame from the pump (deduped by fid)."""
        if self.frozen or self.stopped:
            return
        with self._lock:
            if fid <= self._fid:
                return  # duplicate slot read
            self._latest = downscale(frame)
            self._fid = fid
        self.set()

    def freeze(self):
        """DONE alignment: no new frames accepted.

        recv() keeps re-emitting the last sent one (MJPEG parity --
        the stream holds its final picture instead of ending in a
        decoder error).
        """
        self.frozen = True
        self.set()

    def _next_pts(self) -> int:
        """Wall-arrival pts (90 kHz), floor-clamped at the pump ceiling.

        pts advances by the measured emit-to-emit wall delta (>=
        ``MIN_FRAME_S``), so a 50 fps source delivers ~50 fps of
        timestamps instead of the old flat 1/30 step; a burst is never
        timestamped backwards.
        """
        now = time.monotonic()
        if self._last_emit_mono is None:
            pts = 0
        else:
            delta = now - self._last_emit_mono
            if delta < MIN_FRAME_S:
                delta = MIN_FRAME_S
            pts = self._last_pts + int(round(delta * PTS_CLOCK))
        self._last_emit_mono = now
        self._last_pts = pts
        return pts

    def _emit_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        if self.readyState != "live":
            raise MediaStreamError("track stopped")
        frame.pts = self._next_pts()
        frame.time_base = PTS_TIME_BASE
        return frame

    async def recv(self):
        """Next video frame for the encoder (aiortc pull loop).

        Emits fresh frames when fed; on a 1 s stall / DONE re-emits the
        last sent frame under a fresh (wall-advanced) pts (keepalive,
        see class docstring); raises MediaStreamError once stopped.
        """
        stall = False
        while True:
            with self._lock:
                frame, self._latest = self._latest, None
                self.clear()
            if frame is not None:
                out = self._emit_frame(frame)
                with self._lock:
                    self._last_sent = out
                stall = False
                return out
            if self.stopped:
                raise MediaStreamError("track stopped")
            last = self._last_sent
            if last is not None and (stall or self.frozen):
                # 1 s stall / DONE: re-emit the last sent frame (the
                # same decoded picture, a fresh wall-advanced pts --
                # the frame pacing IS the arrival pacing, no synthetic
                # sleep anymore)
                return self._emit_frame(last)
            if last is not None:
                stall = True
            try:
                await asyncio.wait_for(self.wait(), timeout=self.KEEPALIVE_S)
            except asyncio.TimeoutError:
                pass

    def stop(self):
        """Stop the track (recv() raises MediaStreamError afterwards)."""
        self.stopped = True
        self.set()
        super().stop()


# --------------------------------------------------------------------------- #
# WebRTC gateway (event loop + pump, all runs)
# --------------------------------------------------------------------------- #
class WebRtcGateway:
    """aiortc face over the media gateway's shared readers (design §3).

    One event-loop thread (PC/SDP work) + one pump thread (SHM reads +
    frame conversion).  ``offer(run_id, sdp)`` is the synchronous entry
    the portal route calls.
    """

    def __init__(self, media, stun_urls: tuple = ()):
        self.media = media  # MediaGateway (readers mux)
        self.stun_urls = tuple(stun_urls)
        self.tracks: dict[str, list[FrameRelayTrack]] = {}
        self.pcs: list = []
        self.frames_pumped = 0
        self._stop = threading.Event()
        self.loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, name="webrtc-loop", daemon=True)
        self._pump = threading.Thread(target=self._pump_loop, name="webrtc-pump", daemon=True)

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        """Start the event-loop and pump threads.

        One-shot (the lazy gateway is created at most once per daemon).
        """
        self._loop_thread.start()
        self._pump.start()

    def close(self):
        """Stop the pump, close every PC/track, join the loop thread."""
        self._stop.set()
        for tracks in self.tracks.values():
            for t in tracks:
                t.stop()
        for pc in list(self.pcs):
            try:
                self.loop.call_soon_threadsafe(pc.close)
            except RuntimeError:
                pass
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except RuntimeError:
            pass
        self._loop_thread.join(timeout=3)

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # -- pump (SHM readers -> per-PC tracks) ---------------------------------
    def _pump_loop(self):
        while not self._stop.is_set():
            fed_any = False
            for run_id, tracks in list(self.tracks.items()):
                if not tracks:
                    continue
                rd = self.media.reader(run_id, create=False)
                if rd is None:
                    for t in tracks:
                        t.stop()
                    continue
                try:
                    if rd.phase == "DONE":
                        # DONE alignment (MJPEG parity): the tracks hold
                        # their last picture (keepalive re-emits) instead
                        # of stopping into a decoder error / black screen
                        for t in tracks:
                            t.freeze()
                        continue
                    frame = rd.latest()
                    if frame is None:
                        continue
                    vf = snapshot_to_video_frame(frame)
                except Exception:  # noqa: BLE001 (skip bad tick --
                    continue  # never kill the whole pump)
                finally:
                    self.media.release(run_id)
                fid = frame["frame_id"]
                for t in list(tracks):  # every connected PC gets it
                    self.loop.call_soon_threadsafe(t.feed, fid, vf)
                fed_any = True
            if fed_any:
                self.frames_pumped += 1
            if not self.tracks:
                self._stop.wait(0.05)
            else:
                self._stop.wait(1.0 / PUMP_HZ)

    # -- signalling / datachannel ------------------------------------------
    async def handle_offer(self, run_id: str, sdp_text: str) -> str:
        """One RTCPeerConnection: set remote offer, answer, return sdp.

        Raises KeyError when the run has no readable reader.
        """
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

        rd = self.media.reader(run_id)
        if rd is None:
            raise KeyError(run_id)
        self.media.release(run_id)  # existence check only; long usage refs belong to the pump
        # STUN ruling (camera_contract §16.6 / design §3.1): the
        # iceServers list is passed EXPLICITLY, empty by default.
        # configuration=None makes aiortc fall back to its default
        # stun:stun.l.google.com:19302 -- unreachable from this network,
        # so aioice gathers candidates against a 5 s timeout and EVERY
        # answer (and thus first frame) paid a flat ~5 s (perf probe
        # p5: t_signal_post 5026 ms, decomposed to the gather window).
        # Empty list = host candidates only -> ~150 ms.  Deployments
        # that need NAT traversal set ``webrtc.ice_servers`` in
        # unigym.yaml (list of stun:/turn: URLs).
        ice = [RTCIceServer(urls=u) for u in self.stun_urls]
        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice))
        track = FrameRelayTrack()  # one encoder + one event per PC
        self.tracks.setdefault(run_id, []).append(track)
        self.pcs.append(pc)
        pc.addTrack(track)

        @pc.on("datachannel")
        def on_channel(channel):
            print(f"[webrtc] datachannel '{channel.label}' open " f"(run {run_id})", flush=True)

            @channel.on("message")
            def on_message(message):
                reply = self._control(run_id, message)
                channel.send(json.dumps(reply))

        @pc.on("connectionstatechange")
        def on_state():
            print(f"[webrtc] pc state: {pc.connectionState}", flush=True)
            if pc.connectionState in ("failed", "closed"):
                track.stop()
                lst = self.tracks.get(run_id) or []
                if track in lst:
                    lst.remove(track)
                if pc in self.pcs:
                    self.pcs.remove(pc)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp_text, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return pc.localDescription.sdp

    def offer(self, run_id: str, sdp_text: str) -> str:
        """Synchronous signalling entry: run handle_offer on the loop.

        Args:
            run_id: Target run (its reader must exist).
            sdp_text: The browser's offer SDP text.

        Returns:
            The answer SDP text.

        Raises:
            KeyError: Unknown/unreadable run.
            Exception: Propagated from the coroutine (15 s await cap).
        """
        fut = asyncio.run_coroutine_threadsafe(self.handle_offer(run_id, sdp_text), self.loop)
        return fut.result(timeout=15)

    def _control(self, run_id: str, message) -> dict:
        """Datachannel JSON -> control schema -> run command slot."""
        from unigym.core.session import attach

        try:
            ctrl = parse_control_json(message if isinstance(message, (bytes, str)) else "")
            cmd, payload = to_bridge_command(ctrl)
        except ControlError as e:
            return {"ok": False, "error": str(e)}
        try:
            with attach(run_id, timeout_s=8.0) as s:
                status, result = s.client.send(cmd, payload, timeout=8.0)
        except Exception as e:  # noqa: BLE001 (timeout/DONE;
            # audit #14: the error text carries the type so a handler
            # bug shows up in the browser console instead of vanishing)
            return {"ok": None, "error": f"{type(e).__name__}: {e}"}
        return {"ok": status == _bridge.ACK_OK, "result": result, "type": ctrl["type"]}
