"""core.alloc -- port/GPU allocation for parallel runs .

The daemon allocates before every
spawn so parallel runs never collide on a bridge HTTP port and land on
the freest GPUs.  Pure functions, explicit arguments, no global state
(daemon/tests/`python -c` equally callable).  Minimal implementation:
port = bind-probe over the bridge scan window; GPU = nvidia-smi free
memory order minus the GPUs occupied by live registry runs.
"""

from __future__ import annotations

import socket
import subprocess
from collections.abc import Iterable
import shutil


# Same window as the bridge auto-scan (unigym/runtime/bridge.py).
PORT_LO, PORT_HI = 8600, 8659


def _bindable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", int(port)))
            return True
        except OSError:
            return False


def pick_port(preferred: int = 0, exclude: Iterable[int] = ()) -> int | None:
    """First bindable port in 8600-8659 (``preferred`` tried first).

    Returns None when the window is exhausted (caller decides fail vs
    fallback).  ``exclude`` reserves ports already handed out by this
    process (in-flight spawns whose bridge has not bound yet).
    """
    ex = {int(p) for p in exclude}
    cands = ([int(preferred)] if preferred else []) + list(range(PORT_LO, PORT_HI + 1))
    for p in cands:
        if p in ex:
            continue
        if _bindable(p):
            return p
    return None


def gpu_inventory() -> list[tuple[int, int]]:
    """[(index, mem_used_mib)] from nvidia-smi, ascending by mem_used.

    [] when nvidia-smi is absent/unreadable or there is no GPU -- the
    caller treats that as "no allocation information" (spawn without a
    CUDA_VISIBLE_DEVICES override).
    """
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    inv: list[tuple[int, int]] = []
    for line in r.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            inv.append((int(parts[0]), int(float(parts[1]))))
        except ValueError:
            continue
    return sorted(inv, key=lambda x: (x[1], x[0]))


def pick_gpu(exclude: Iterable[int] = ()) -> int | None:
    """Freest GPU not in ``exclude``; None = no info or nothing free.

    ``exclude`` = indices occupied by live runs (registry ``gpu`` field,
    computed by the caller from its instance view) -- the minimal
    "registry occupation + nvidia-smi free-memory order" allocation of
    the design (§2/implementation notes).
    """
    ex = {int(g) for g in exclude}
    for idx, _used in gpu_inventory():
        if idx not in ex:
            return idx
    return None
