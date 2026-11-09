"""unigym.core -- the pure business layer .

Shell/core split: everything that
talks to a run instance (SHM attach, spawn, registry discovery,
port/GPU allocation) lives here as explicit-argument functions with
dataclass returns -- no argparse types, no HTTP serving, no global
state.  Callers: the unigymd daemon, tests, and `python -c` equally.

**The CLI does not import this package** (the CLI is a
thin daemon API client; the d-1
transitional direct calls in unigym/scripts/client.py are marked
`SCAFFOLDING(d-1)` and removed in d-2).

Modules:
    registry  RunInfo + list_runs()/read_entry() (SHM scan + serve
              registry + PID liveness, merged view)
    session   attach(run_id) -> Session (pause/resume/step/inspect/
              snapshot/capture/camera/shutdown/status)
    run_ops   RunSpec/RunHandle + start_run() (setsid detached spawn)
              + wait_once() (the `unigym wait` primitive)
    spawn     facade over the ONE spawn command builder
              (unigym.scripts.spawn_spec, byte-pinned by L1)
    alloc     pick_port()/pick_gpu() (parallel-run allocation)
"""

from __future__ import annotations
