"""Runtime layer: foreground/backstage decoupling (SHM bridge).

``bridge.py`` is deliberately the single module owning every shared-memory
layout constant and sync primitive.  It only needs
stdlib + numpy (lazy), so CLI tooling can import it without torch /
mujoco / isaaclab.

the run-side HTTP face moved to the daemon media gateway
(unigym/daemon/media.py + webrtc.py) -- BridgeServer is gone; the run is
a pure SHM worker.
"""
from __future__ import annotations

from .bridge import (Bridge, BridgeError, BridgeTimeout,
                     ShmClient, active_run_ids, clean_zombies, make_run_id,
                     serve_state_paths)

__all__ = ["Bridge", "BridgeError", "BridgeTimeout",
           "ShmClient", "active_run_ids", "clean_zombies", "make_run_id",
           "serve_state_paths"]
