"""`python -m unigym.daemon` -- the unigymd entry point .

Defaults come from unigym.yaml: ``daemon_api_port`` (8670) for the
control API and ``webui_port`` (8600) for the portal.  Run detached via
`unigym daemon start` (nohup + state file), under systemd with
`unigym daemon start --foreground`, or directly for debugging:

    python -m unigym.daemon [--api-port N] [--webui-port M]
                            [--runs-root R] [--poll S] [--rebuild S]
"""

from __future__ import annotations

import os

# -- daemon-side torch/OMP pool cap (docs/usage/daemon.md §10) -------------- #
# Measured mechanism: /api/start -> RunSpec.validate -> adapter_params
# validation -> import unigym.adapters eager-imports torch, whose OpenMP
# pools otherwise initialize at machine width -- 48 intra-op + internals
# = ~127 idle OS threads resident in the daemon on this 48-core box
# (128-thread mystery).  The setdefault runs at MODULE LEVEL, before any
# other unigym import (nothing but stdlib above), so torch initializes
# its pools small.  Scope = the daemon process ONLY: the marker var
# tells SpawnSpec.full_env to strip the key again when building CHILD
# env, so runs keep their shell env contract (a pre-existing user value
# is never overridden and arms no marker; an explicit per-run value via
# RunSpec.env rides env_force and wins regardless).
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["_UNIGYM_DAEMON_OMP_INJECTED"] = "1"

import argparse  # noqa: E402  (stdlib imports stay above the env pin)
import sys  # noqa: E402

from .service import DaemonService  # noqa: E402  (after the env pin)

_CODE_PACKS_LOADED = False


def _load_code_packs() -> None:
    """Import mounted code packs once, before the control plane starts."""
    global _CODE_PACKS_LOADED
    if _CODE_PACKS_LOADED:
        return
    from unigym.packs.load import load_mounted_code

    load_mounted_code()
    _CODE_PACKS_LOADED = True


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args, print the code-origin line, and run DaemonService.

    Args:
        argv: Argument vector for tests; None reads sys.argv.

    Returns:
        0 on a clean stop, 1 when the daemon failed to bind its ports.
    """
    from unigym.core.spawn import load_cfg

    cfg = load_cfg()
    p = argparse.ArgumentParser(
        prog="unigym.daemon",
        description="unigymd -- resident daemon: webui portal + control "
        "API + spawn scheduling (runs outlive it by design)",
    )
    p.add_argument(
        "--api-port",
        type=int,
        dest="api_port",
        default=int(cfg.get("daemon_api_port") or 8670),
        help="control API port (default: unigym.yaml " "daemon_api_port or 8670)",
    )
    p.add_argument(
        "--webui-port",
        type=int,
        dest="webui_port",
        default=int(cfg.get("webui_port") or 8600),
        help="portal port (default: unigym.yaml webui_port "
        "or 8600; busy -> scans upward with a note)",
    )
    p.add_argument(
        "--runs-root",
        default=None,
        help="job world root (default: unigym.yaml runs_root " "or <repo>/runs)",
    )
    p.add_argument(
        "--output-root",
        dest="output_root",
        default=None,
        help="ledger scan root for the history face + record "
        "delete (default: unigym.yaml output_root, "
        "relative to the repo)",
    )
    p.add_argument(
        "--poll", type=float, default=2.0, help="pending-job scan interval seconds (default 2)"
    )
    p.add_argument(
        "--rebuild",
        type=float,
        dest="rebuild",
        default=2.0,
        help="instance-table rebuild interval seconds " "(default 2)",
    )
    p.add_argument(
        "--no-state-file",
        action="store_true",
        dest="no_state_file",
        help="do not write /tmp/unigym-daemon.json (tests)",
    )
    a = p.parse_args(argv)
    # the code-origin line (cwd footgun made visible): WHICH checkout
    # and version this daemon serves -- a wrong cwd silently runs
    # another worktree's stale code (an entire debugging session was
    # spent on that before it became this line)
    import unigym

    print(f"[unigymd] code {unigym.__file__} version " f"{unigym.__version__}", flush=True)
    _load_code_packs()
    svc = DaemonService(
        api_port=a.api_port,
        webui_port=a.webui_port,
        runs_root=a.runs_root,
        poll=a.poll,
        rebuild_s=a.rebuild,
        output_root=a.output_root,
        write_state=not a.no_state_file,
    )
    try:
        svc.run()
    except OSError as e:
        print(f"[unigymd] failed to bind: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
