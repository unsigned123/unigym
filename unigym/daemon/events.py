"""daemon.events -- the control-plane EventBus + the SSE stream face.

Daemon state changes are
fanned out to in-process subscribers as tiny notifications; the SSE
handler turns one subscription into one HTTP stream on /api/events
(both faces: the 8670 control API + the 8600 portal mirror).  Events
say "changed" and never carry the state itself -- clients pull details
through the existing queries (push notifies, pull confirms; the sampling
sampling discipline holds).

* :class:`EventBus` -- pure threads/queues: :meth:`~EventBus.publish`
  appends to a bounded ring (256 entries, monotonic seq) and fans out
  to every subscriber's bounded queue (drop-oldest + a ``dropped``
  marker -- a slow consumer can never block a publisher).
* :func:`serve_sse` -- the wire face: ``id``/``event``/``data`` frames,
  15s ``: ping`` keepalives (they double as dead-connection probes),
  Last-Event-ID replay from the ring, ``reset`` on ring overflow,
  ``types`` family filter + ``run_id`` filter.

The bus lives on the service (``DaemonService.events``); both HTTP
handlers delegate here, keeping routing module-level per api.py's
single-sourcing discipline.  docs/contracts/daemon_api_contract.md
§1.22 is the wire contract twin.
"""

from __future__ import annotations

import contextlib
import json
import queue
import socket
import threading
import time
from collections import deque

RING_SIZE = 256  # replayed history (design §2 self-healing)
QUEUE_SIZE = 256  # per-subscriber buffer (drop-oldest)
PING_S = 15.0  # ": ping" keepalive cadence
# thread convergence: a client that vanished without a TCP reset
# (half-open) never triggers a write error -- the heartbeat gains the
# same disconnect detection the MJPEG stream has (media._backlog_ok
# pattern): a write timeout bounds any blocked write, and the socket's
# unacked-send queue (TIOCOUTQ) must DRAIN between heartbeats -- a queue
# that stays non-draining for SSE_STALL_CHECKS consecutive probes is a
# dead peer and the handler thread exits.
SSE_WRITE_TIMEOUT_S = 30.0
SSE_BACKLOG_BYTES = 262_144  # hard cap on unacked send bytes
SSE_STALL_CHECKS = 20  # non-draining heartbeats before exit
# (20 x PING_S = 5 min to declare dead)

DROPPED = "dropped"  # control events (contract §1.22)
RESET = "reset"


def _matches(types, etype: str) -> bool:
    """Filter semantics for the ``types`` query parameter.

    A family ("run") matches its children ("run.phase"); an exact
    type matches itself.
    """
    return not types or any(etype == t or etype.startswith(t + ".") for t in types)


class Subscription:
    """One subscriber: its queue plus the filters it asked for."""

    __slots__ = ("q", "types", "run_id")

    def __init__(self, types=None, run_id: str | None = None):
        self.q: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.types = tuple(types) if types else None
        self.run_id = run_id

    def wants(self, ev: dict) -> bool:
        """True when the event passes both the run_id and type filters."""
        if self.run_id and ev.get("run_id") != self.run_id:
            return False
        return _matches(self.types, ev["type"])


class EventBus:
    """Fan-out bus: ring + monotonic seq + per-subscriber queues."""

    def __init__(self, ring: int = RING_SIZE):
        self._lock = threading.Lock()
        self._seq = 0
        self._ring: deque = deque(maxlen=int(ring))
        self._subs: list[Subscription] = []

    # -- publishing ---------------------------------------------------------- #
    def publish(self, etype: str, run_id: str | None = None, data: dict | None = None) -> dict:
        """Append to the ring + fan out.

        NEVER blocks the publisher and never raises on subscriber
        backpressure (drop-oldest + marker: the two oldest frames give
        way to a ``dropped`` marker plus this event, so the stream
        stays bounded AND self-describing).

        Returns:
            The published event dict (seq/ts/type/run_id/data).
        """
        with self._lock:
            self._seq += 1
            ev = {
                "seq": self._seq,
                "ts": round(time.time(), 3),
                "type": etype,
                "run_id": run_id,
                "data": dict(data or {}),
            }
            self._ring.append(ev)
            for sub in self._subs:
                if not sub.wants(ev):
                    continue
                try:
                    sub.q.put_nowait(ev)
                except queue.Full:
                    for _ in range(2):
                        with contextlib.suppress(queue.Empty):
                            sub.q.get_nowait()
                    with contextlib.suppress(queue.Full):
                        sub.q.put_nowait(
                            {
                                "seq": ev["seq"],
                                "ts": ev["ts"],
                                "type": DROPPED,
                                "run_id": None,
                                "data": {"lost": 2},
                            }
                        )
                        sub.q.put_nowait(ev)
            return ev

    # -- subscribing --------------------------------------------------------- #
    def subscribe(self, types=None, run_id: str | None = None) -> tuple[Subscription, list[dict]]:
        """Register a subscriber.

        Returns:
            (sub, ring replay already filtered the way the live
            stream will be).
        """
        sub = Subscription(types=types, run_id=run_id)
        with self._lock:
            self._subs.append(sub)
            replay = [e for e in self._ring if sub.wants(e)]
        return sub, replay

    def unsubscribe(self, sub: Subscription) -> None:
        """Remove a subscriber (unknown sub is silently ignored)."""
        with self._lock:
            with contextlib.suppress(ValueError):
                self._subs.remove(sub)

    # -- replay-window arithmetic (consumed by serve_sse) ---------------- #
    @property
    def seq(self) -> int:
        """Monotonic sequence number of the most recently published event."""
        with self._lock:
            return self._seq

    @property
    def subscriber_count(self) -> int:
        """Live subscriptions (the /api/status SSE observation face)."""
        with self._lock:
            return len(self._subs)

    @property
    def oldest_seq(self) -> int | None:
        """Seq of the oldest ringed event (None while the ring is empty).

        A Last-Event-ID below ``oldest_seq - 1`` means the client is
        past the window -> the wire face answers ``reset``.
        """
        with self._lock:
            return self._ring[0]["seq"] if self._ring else None


# --------------------------------------------------------------------------- #
# the SSE wire face
# --------------------------------------------------------------------------- #
def sse_frame(ev: dict) -> bytes:
    """One text/event-stream frame: id/event/data + blank line."""
    data = {"ts": ev["ts"], "run_id": ev["run_id"], **ev["data"]}
    return (
        f"id: {ev['seq']}\nevent: {ev['type']}\n" f"data: {json.dumps(data, default=str)}\n\n"
    ).encode()


def _last_event_id(handler, qs: dict) -> int | None:
    """Resolve the client's replay cursor.

    Last-Event-ID header first (the EventSource reconnect channel),
    then the query aliases (curl/agent convenience).  Garbage -> None
    (a fresh live-only subscription, never a 400).
    """
    raw = (
        handler.headers.get("Last-Event-ID")
        or (qs.get("lastEventId") or qs.get("last_event_id") or [None])[0]
    )
    try:
        return int(raw) if raw not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _send_backlog(sock) -> int | None:
    """Unacked-send bytes of one socket (TIOCOUTQ), None if unavailable.

    Same probe as the MJPEG stream's ``media._backlog_ok`` gate -- the
    except set is deliberately NARROWED (import/ioctl/unpack failures
    only) so the control-plane catch-all baseline stays untouched; any
    other bug must surface.
    """
    if sock is None:
        return None
    try:
        import fcntl
        import struct
        import termios

        return struct.unpack("i", fcntl.ioctl(sock, termios.TIOCOUTQ, b"\x00" * 4))[0]
    except (ImportError, OSError, ValueError):  # probe optional (v068)
        return None


def serve_sse(handler, svc, qs: dict) -> None:
    """GET /api/events -- stream the bus to one client.

    Auth is the CALLER's posture (8670 open per contract §0; the 8600
    mirror gates the subscription like a POST face -- EventSource cannot
    send headers, so the token rides the query).

    Thread convergence: a client disconnect surfaces
    as (a) a failed write (broken pipe / reset), (b) a write blocked
    past ``SSE_WRITE_TIMEOUT_S`` (socket timeout), or (c) an unacked
    send queue that stops draining across ``SSE_STALL_CHECKS``
    heartbeats (half-open peer -- writes "succeed" into the void
    forever, exactly the lingering-``queue.get`` shape py-spy caught).
    Any of the three exits the handler thread; ``finally`` unsubscribes.
    """
    bus = svc.events
    types = [t.strip() for t in (qs.get("types") or [""])[0].split(",") if t.strip()] or None
    run_id = (qs.get("run_id") or [""])[0].strip() or None
    last = _last_event_id(handler, qs)
    sub, replay = bus.subscribe(types=types, run_id=run_id)
    sock = getattr(handler, "connection", None)
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.settimeout(SSE_WRITE_TIMEOUT_S)
        # TCP keepalive bounds the idle half-open case (peer machine
        # vanished, no FIN/RST): the OS probes and errors the socket,
        # so the next heartbeat write fails and the thread exits.
        with contextlib.suppress(OSError, AttributeError):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for opt, val in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 10), ("TCP_KEEPCNT", 6)):
            if hasattr(socket, opt):
                with contextlib.suppress(OSError):
                    sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), val)
    try:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        w = handler.wfile
        oldest = bus.oldest_seq
        if last is not None and oldest is not None and last < oldest - 1:
            # ring overflow: the client is past the replay window --
            # tell it to re-pull everything, then keep streaming live
            w.write(
                sse_frame(
                    {
                        "seq": bus.seq,
                        "ts": time.time(),
                        "type": RESET,
                        "run_id": None,
                        "data": {"last": last, "oldest": oldest},
                    }
                )
            )
        elif last is not None:
            for ev in replay:  # incremental replay (design §2)
                if ev["seq"] > last:
                    w.write(sse_frame(ev))
        w.flush()
        stall = 0
        prev_outq: int | None = None
        while True:
            try:
                ev = sub.q.get(timeout=PING_S)
                payload = sse_frame(ev)
            except queue.Empty:
                payload = b": ping\n\n"  # proxy keepalive + liveness probe
            w.write(payload)
            w.flush()
            outq = _send_backlog(sock)
            if outq is None:  # probe unavailable: error path only
                stall = 0
                prev_outq = None
            else:
                if outq == 0 or (prev_outq is not None and outq < prev_outq):
                    stall = 0  # drained (or draining): alive
                else:
                    stall += 1  # unacked bytes never leave
                prev_outq = outq
                if outq > SSE_BACKLOG_BYTES:
                    raise OSError("sse send backlog over cap (dead peer)")
            if stall >= SSE_STALL_CHECKS:
                raise OSError(
                    "sse send queue stalled across " f"{SSE_STALL_CHECKS} heartbeats (dead peer)"
                )
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
        pass  # client hung up
    finally:
        bus.unsubscribe(sub)
