"""core.spawn -- core-layer access to the ONE spawn command builder.

The byte-exact truth for docker/mujoco command synthesis (and the
completion criteria / container teardown primitives) stays in
``unigym.scripts.spawn_spec`` -- pinned by tests/test_launcher.py L1
byte-exact snapshots (docs/contracts/launcher_contract.md §1 guardrail).
This facade is the ONLY access point for the core/daemon layers,
so no second synthesis path can appear.  probe/bash/pytest exec bodies are re-exported here as well.
"""

from __future__ import annotations

from unigym.scripts.spawn_spec import (  # noqa: F401 (facade)
    BACKENDS,
    SpawnSpec,
    build_spawn,
    cmd_bash,
    cmd_probe,
    cmd_pytest,
    container_state,
    is_complete,
    load_cfg,
    log_tail,
    out_dir_for,
    resolve_backend,
    runs_root,
    teardown_container,
)
