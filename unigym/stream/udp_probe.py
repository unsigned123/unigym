"""UDP reachability/loss probe (S2) -- dual-end, stdlib only.

Decides whether WebRTC (ICE over UDP) can work across the campus/dorm
path, or whether the MJPEG-over-ssh-tunnel must stay the primary route.

Usage (server end, on the sim host -- one-shot: replies + exits)::

    python -m unigym.stream.udp_probe --listen 9700

Usage (client end, from the remote site)::

    python -m unigym.stream.udp_probe --send <host>:9700 --n 200 --rate 50

Packet wire format (16B header + padding, little-endian)::

    magic "UGPR" | seq u32 (0xFFFFFFFF = finish marker) | send_ts f64 | pad

The server replies with a REPORT packet (seq 0xFFFFFFFE: recv_count /
first_seq / last_seq / out-of-order count), sent 3x for reliability.
Honesty note: loss is measured WITHIN the observed sequence window
(losses before the first arriving packet are invisible).  Client exit
code 0 = report received (UDP path works), 1 = no report (blocked or
lossy -- keep the MJPEG primary route).
"""
from __future__ import annotations

import argparse
import socket
import struct
import time

MAGIC = b"UGPR"
SEQ_FINISH = 0xFFFFFFFF
SEQ_REPORT = 0xFFFFFFFE
HDR = struct.Struct("<4sIf8x")       # magic, seq, ts
REPORT = struct.Struct("<4sIIIII")   # magic, SEQ_REPORT, recv, first, last, ooo

__all__ = ["serve", "probe", "main"]


def _pkt(seq: int, size: int = 64) -> bytes:
    return HDR.pack(MAGIC, seq, time.time()) + b"\x00" * max(0, size - 16)


def serve(sock: socket.socket, seconds: float | None = None,
          reply_reps: int = 3) -> dict:
    """Receive probe packets; on the finish marker reply with a REPORT.

    Returns the server-side stats dict (also encoded into the report).
    With ``seconds`` set, keeps serving (repeating probe server mode);
    otherwise exits after the first finished probe.
    """
    recv = first = last = ooo = 0
    expect = 0
    started = time.time()
    while True:
        if seconds is not None and time.time() - started > seconds:
            break
        try:
            data, addr = sock.recvfrom(65536)
        except socket.timeout:
            if seconds is None:
                break
            continue
        if len(data) < 16 or data[:4] != MAGIC:
            continue
        seq, = struct.unpack_from("<I", data, 4)
        if seq == SEQ_FINISH:
            report = REPORT.pack(MAGIC, SEQ_REPORT, recv,
                                 first, last, ooo)
            for _ in range(reply_reps):
                sock.sendto(report, addr)
            stats = {"received": recv, "first_seq": first,
                     "last_seq": last, "out_of_order": ooo}
            print(f"[udp-probe] report -> {addr[0]}:{addr[1]} {stats}",
                  flush=True)
            if seconds is None:
                return stats
            recv = first = last = ooo = 0
            continue
        if seq == SEQ_REPORT:
            continue                    # our own report loopback
        recv += 1
        if first == 0:
            first, expect = seq, seq + 1
        else:
            if seq < expect:
                ooo += 1
            expect = max(expect, seq + 1)
        last = seq
        if recv % 50 == 0:
            print(f"[udp-probe] {recv} packets "
                  f"(ooo={ooo})", flush=True)
    return {"received": recv, "first_seq": first, "last_seq": last,
            "out_of_order": ooo}


def probe(host: str, port: int, n: int = 200, rate: float = 50.0,
          size: int = 1000, timeout: float = 5.0) -> dict:
    """Send n probe packets, await the REPORT; returns client stats."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.5)
    interval = 1.0 / rate if rate > 0 else 0.0
    sent, t0 = 0, time.time()
    while sent < n:
        sock.sendto(_pkt(sent + 1, size), (host, port))
        sent += 1
        if interval:
            # catch up in bulk after a slow sendto instead of drifting
            target = t0 + sent * interval
            pause = target - time.time()
            if pause > 0:
                time.sleep(pause)
    print(f"[udp-probe] sent {sent} -> {host}:{port}, awaiting report",
          flush=True)
    sock.sendto(_pkt(SEQ_FINISH, size), (host, port))
    deadline = time.time() + timeout
    while time.time() < deadline:
        sock.settimeout(max(0.1, deadline - time.time()))
        try:
            data, _addr = sock.recvfrom(65536)
        except socket.timeout:
            break
        if len(data) >= REPORT.size and data[:4] == MAGIC:
            _m, seq, recv, first, last, ooo = REPORT.unpack(data[:REPORT.size])
            if seq == SEQ_REPORT:
                elapsed = time.time() - t0
                window = (last - first + 1) if first else 0
                loss = (100.0 * (window - recv) / window) if window else 0.0
                return {"sent": sent, "elapsed_s": round(elapsed, 2),
                        "received": recv, "first_seq": first,
                        "last_seq": last, "window": window,
                        "loss_pct": round(loss, 2),
                        "out_of_order": ooo, "report_rtt_s": None,
                        "reachable": True}
    return {"sent": sent, "reachable": False,
            "hint": f"no report within {timeout}s -- UDP path blocked or "
                    f"heavily lossy; keep MJPEG (ssh tunnel) primary"}


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m unigym.stream.udp_probe",
        description="UDP reachability/loss probe (dual-end, stdlib only)")
    p.add_argument("--listen", type=int, default=None, metavar="PORT",
                   help="server end: listen on this UDP port (one-shot)")
    p.add_argument("--serve-seconds", type=float, default=None,
                   dest="serve_seconds",
                   help="listen mode: keep serving for N s instead of "
                        "one-shot (repeating probe server)")
    p.add_argument("--send", default=None, metavar="HOST:PORT",
                   help="client end: target host:port")
    p.add_argument("--n", type=int, default=200, help="packets to send")
    p.add_argument("--rate", type=float, default=50.0,
                   help="packets per second (default 50)")
    p.add_argument("--size", type=int, default=1000,
                   help="packet payload bytes (default 1000)")
    p.add_argument("--timeout", type=float, default=5.0,
                   help="report wait (default 5 s)")
    a = p.parse_args()
    if a.listen:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(("0.0.0.0", a.listen))
        print(f"[udp-probe] listening on udp/{a.listen} "
              f"({'one-shot' if a.serve_seconds is None else 'repeat'})",
              flush=True)
        serve(sock, a.serve_seconds)
    elif a.send:
        host, _, port = a.send.partition(":")
        if not port:
            p.error("--send needs HOST:PORT")
        stats = probe(host, int(port), a.n, a.rate, a.size, a.timeout)
        print("[udp-probe] client stats:", json_dumps(stats))
        raise SystemExit(0 if stats.get("reachable") else 1)
    else:
        p.error("one of --listen / --send is required")


def json_dumps(obj: dict) -> str:
    import json
    return json.dumps(obj)


if __name__ == "__main__":
    main()
