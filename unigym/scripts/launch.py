#!/usr/bin/env python
"""unigym launcher -- one CLI for both backends (thin dispatch shell).

(docs/designs/v026-unigymd.md): thin-CLI architecture -- every
business subcommand is a unigymd CONTROL-API request + formatting
(zero unigym.core imports; `unigym daemon start|stop|status` is the
only local process command).  This file remains ONLY argparse wiring
(user-facing surface stable since the first releases; the daemon
subcommand is the unigymd-era addition).  The implementation roles:

    unigymd       resident daemon (unigym/daemon/): control API + webui
                  portal + instance view + spawn scheduling
    unigym.core   business layer (registry/session/run_ops/spawn/alloc);
                  the CLI does NOT import it -- the daemon does
    spawn_spec.py the ONE spawn command builder (byte-pinned by L1)
    client.py     CLI command layer: API requests + output formatting
    (server.py    REMOVED -- the job supervisor world is gone)

Subcommands:
    run CONFIG... [-b BACKEND] [--bg] [--serve] [collect passthrough...]
        mujoco   -> host venv python -m unigym.scripts.collect_cli (MUJOCO_GL=egl)
        isaaclab -> synthesized `docker run` (defaults from unigym.yaml at repo root)
    wait  -- poll completion criteria ([collect] summary: + artifacts) and
             TEARDOWN the container (docker rm -f) once met.  --once for
             single-shot polling (exit 0 = done, 1 = pending).
    probe NAME [--backend ...] [probe args...]   -> probe_<name> module
    bash  -> interactive container (same mounts)
    pytest [args...] -> host venv pytest at repo root
Benchmark packs (docs/contracts/benchmark_contract.md §10):
    benchmarks list [--pack X] -> tasks grouped by pack namespace
    benchmarks validate <dir>  -> aggregated pack validation (exit 0/1)
Contact groups (docs/contracts/contacts_contract.md):
    yaml `contacts:` -> contact/<name> record keys (both backends; probe_contacts)
Object state (slimmed): every prop records
    object/<name>/{pos,quat} extras (props/object_state).  The auto
    held_<name>[__<group>] generation was RETIRED -- hand
    contact criteria are DECLARED contact groups now (see below).
Declarative contacts (docs/contracts/contacts_contract.md):
    yaml `contacts:` groups may address robot body-parts groups
    (sensor_bodies: {body_parts: grasp.left}) and props (filter_bodies:
    {prop: cracker}) by name -- zero hardcoded bodies; the retired
    auto-held chain is expressed as two declared pairwise groups.
usd props : type: usd -> isaaclab UsdFileCfg direct load
    (materials in file; usd_prep_prop.py sim-ready prep); mujoco
    declares its representation per prop (mjcf_file / box_texture /
    box -- fail-fast when missing).
Hand contact fix : isaaclab contact sensor_bodies entries may
    carry "scope/body_name" nested paths -- segment-wise ContactSensor
    matching never reached the nested finger prims through flat names
    (probe: probe_hand_contacts -- retired in git history, v063 cleanup).
Server world (REMOVED): `unigym submit` / `unigym server` and the
runs/ pending/running/done protocol are gone (zero users; batch =
`unigym run a.yaml b.yaml ...`).  Run verdicts now live in the run's
own output_dir (`run.json` ledger; designs/v043-run-record.md):
    status           -> live SHM runs + spawn deaths
Runtime bridge (docs/contracts/runtime_contract.md -- run with --serve):
    runs     -> list live bridge runs (SHM scan + serve registry)
    attach   -> live progress line for one run
    cam      -> print the MJPEG / snapshot URLs (open in a browser)
    snapshot -> save one camera frame as JPEG (ux-7: optional
                --pos/--lookat transient world-pose override)
    pause/resume -> freeze/continue the rollout (req-ack over SHM)
    rewind/snapshots -> rewind face (Round B B1): capture engine snapshots
                 (--label), restore one (auto-pauses; time-line fork),
                 list the live slots (mujoco + rewindable adapter only)
    camera   -> live property apply (set) / frame fetch (get)
    capture  -> one command -> one artifact absolute path on stdout
                (jpg frame / npz dump; ux-5, agent-facing)
    clean    -> remove zombie `unigym_*` shm segments + stale registries
 Debug console (docs/contracts/console_contract.md -- live on --serve runs):
     inspect  -> eval expr in {np torch obs_cache extra bundle adapter env}
                 / fixed obs|extra key query / run summary
     step     -> advance n policy steps while paused (auto-pauses; the run
                 records every stepped step exactly like a live rollout)
 Agent session (ux-12, S3):
     debug    -> preflight (--enable_cameras / port / registry conflicts,
                 fail-fast) + run(--serve --webui --bg) + bridge wait +
                 /info summary + console/capture cheatsheet
                 (--pause: the step-gate starts PAUSED; release via
                 `unigym resume|step`)

Dispatch is table-driven: ``_COMMANDS`` maps a subcommand to its owning
module + registration function.  ``--help`` renders from the table
without importing any command module, and dispatch imports only the
module that owns the requested command (the optional extras therefore
stay lazy: a base install can still list and run base commands).
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from unigym._optional import MissingExtraError

_REPO = Path(__file__).resolve().parents[2]
_UNIGYM_YAML = _REPO / "unigym.yaml"

_SELF = __name__

#: name -> (module, register function, one-line description).  The
#: description is the top-level help line; the module is imported only
#: when that command is dispatched.
_COMMANDS: dict[str, tuple[str, str, str]] = {
    "run": (_SELF, "_register_run", "run one or more collect configs"),
    "wait": (_SELF, "_register_run",
             "poll completion criteria; teardown container"),
    "probe-mod": (_SELF, "_register_run",
                  "run unigym.scripts.probe_<name> (module probes; renamed "
                  "from renamed -- probe is now the e2e one-command probe)"),
    "bash": (_SELF, "_register_run",
             "interactive container with unigym mounts"),
    "pytest": (_SELF, "_register_run", "host venv pytest at repo root"),
    "probe": ("unigym.scripts.probe_cli", "add_probe_subparser",
              "one-command e2e probe: launch -> wait -> gate scorecard; "
              "spawn via the daemon run face"),
    "benchmarks": ("unigym.scripts.benchmarks_cli",
                   "add_benchmarks_subparser",
                   "list / validate benchmark packs"),
    "submit": (_SELF, "_register_status",
               "REMOVED: use `unigym run CONFIG... --bg`"),
    "status": (_SELF, "_register_status",
               "live SHM runs + deaths (submit queue removed)"),
    "runs": (_SELF, "_register_status",
             "list live bridge runs (SHM scan)"),
    "attach": (_SELF, "_register_status",
               "live progress line for one run"),
    "cam": (_SELF, "_register_status",
            "print MJPEG / snapshot URLs for one run"),
    "snapshot": (_SELF, "_register_status",
                 "save one camera frame as JPEG"),
    "capture": (_SELF, "_register_status",
                "one command -> one artifact path on stdout (ux-5: jpg "
                "frame | npz dump)"),
    "pause": (_SELF, "_register_status",
              "freeze the rollout at the step boundary"),
    "resume": (_SELF, "_register_status",
               "continue a paused rollout"),
    "inspect": (_SELF, "_register_status",
                "debug console: eval expr / obs|extra query / summary"),
    "step": (_SELF, "_register_status",
             "advance n policy steps while paused (auto-pauses; stays "
             "paused)"),
    "rewind": (_SELF, "_register_status",
               "rewind face: --label captures a snapshot; SNAP_ID restores "
               "it (auto-pauses; mujoco + rewindable adapter runs only)"),
    "snapshots": (_SELF, "_register_status",
                  "list the live rewind snapshot slots of one run (LRU cap "
                  "16 since the v056 checkpoint ring, memory-only)"),
    "direct": (_SELF, "_register_status",
               "director commands (only on runs started with --direct; else "
               "rejected with a restart hint)"),
    "camera": (_SELF, "_register_status",
               "live camera property apply / frame fetch / inspect (debug "
               "toolkit)"),
    "watch": (_SELF, "_register_status",
              "programmable observers: add/list/clear/update/test/events "
              "(design watchers.md §8)"),
    "timeline": (_SELF, "_register_status",
                 "post-hoc event viewer for a run (events.jsonl tail; "
                 "watchers.md §3)"),
    "clean": (_SELF, "_register_status",
              "remove zombie unigym_* shm segments"),
    "scene": (_SELF, "_register_status",
              "scene tree + live state (curated: robot/props/contacts/"
              "cameras; PARKED-capable -- agent face of the scene query)"),
    "describe": ("unigym.scripts.describe_cli", "add_describe_subparser",
                 "print a config's assembled ModelIR in-process (summary / "
                 "joints / bodies / cameras / geoms / sites / actuators; "
                 "--json / --compiled)"),
    "debug": (_SELF, "_register_status",
              "ux-12 agent session: preflight -> run(--serve --webui --bg) "
              "-> wait for the bridge -> /info summary + console/capture "
              "cheatsheet (collect passthrough allowed, e.g. "
              "--enable_cameras)"),
    "trace": ("unigym.scripts.trace_cli", "add_trace_subparsers",
              "debug trace toolkit: record / diff / import-shard / "
              "diverge (docs/contracts/trace_contract.md)"),
    "doctor": ("unigym.scripts.doctor_cli", "add_doctor_subparser",
               "health check / env discovery / local-overlay writer"),
    "pack": ("unigym.packs.cli", "add_pack_subparser",
             "pack registry: list / doctor / path"),
    "asset": ("unigym.commands.asset_cli", "add_asset_subparser",
              "L0 asset primitive: the derived catalog, payload paths and "
              "the selection locks"),
    "analyze": ("unigym.scripts.analyze_cli", "add_analyze_subparser",
                "behavior report over existing run shards (zero collection; "
                "docs/contracts/analyze_contract.md)"),
    "vdiff": ("unigym.scripts.vdiff_cli", "add_vdiff_subparser",
              "compare two images: side-by-side + diff heatmap + histogram "
              "stats + one-line verdict"),
    "orbit": ("unigym.scripts.orbit_cli", "add_orbit_subparser",
              "circle a target with the free camera and capture N "
              "sidecar-tagged frames + manifest"),
    "daemon": (_SELF, "_register_daemon",
               "unigymd resident daemon (webui portal + control API + spawn "
               "scheduling; runs outlive it)"),
}


# --------------------------------------------------------------------------- #
# registration groups (imports stay inside: no command module loads at
# module scope, so --help works in a base install)
# --------------------------------------------------------------------------- #
def _register_run(sub) -> None:
    """Register run/wait/probe-mod/bash/pytest (client + spawn_spec)."""
    from unigym.scripts import client as cli
    from unigym.scripts import spawn_spec as ss

    r = sub.add_parser("run", help="run one or more collect configs")
    r.add_argument("config", nargs="+", help="run yaml(s); multiple run in order")
    r.add_argument("-b", "--backend", choices=["isaaclab", "mujoco"], default=None,
                   help="override yaml env.backend (default: read from yaml)")
    r.add_argument("--bg", action="store_true",
                   help="background: nohup + independent log + state file")
    r.add_argument("--serve", action="store_true",
                   help="runtime bridge: SHM telemetry/control + MJPEG live "
                        "camera (then: unigym runs/attach/cam/pause/...)")
    r.add_argument("--webui", action="store_true", dest="webui",
                   help="serve the streaming WebUI on the bridge HTTP port "
                        "(POST /control + /webui/*; access via ssh -L "
                        "tunnel; token = unigym.yaml webui_token tri-state: "
                        "unset/empty = no auth, 'auto' = per-run random, "
                        "non-empty = fixed; implies --serve)")
    r.add_argument("--serve-port", type=int, default=0, dest="serve_port",
                   help="bridge HTTP port (default: auto from 8600; "
                        "--webui default: unigym.yaml webui_port)")
    r.add_argument("--serve-camera", default=None, dest="serve_camera",
                   help="broadcast camera (default: first instantiated)")
    r.add_argument("--pause", action="store_true", dest="pause",
                   help="born-paused spawn: the step-gate starts "
                        "PAUSED (collect_cli --serve_pause); release with "
                        "`unigym resume latest` or `unigym step latest N`")
    r.add_argument("--watch", default=None, dest="watch", metavar="YAML",
                   help="arm watchers from a yaml pack at start "
                        "(born-armed; watchers_contract.md §6)")
    r.add_argument("--json", action="store_true",
                   help="single JSON document on stdout (errors too); "
                        "multi-config accumulates under 'launched'")
    r.set_defaults(fn=cli.cmd_run)

    w = sub.add_parser("wait", help="poll completion criteria; teardown container")
    w.add_argument("--state", default=None, help="state json from `run --bg`")
    w.add_argument("--container", default=None)
    w.add_argument("--log", default=None)
    w.add_argument("--output-dir", default=None)
    w.add_argument("--timeout", type=float, default=None)
    w.add_argument("--poll", type=float, default=None)
    w.add_argument("--once", action="store_true", help="single check, no loop")
    w.add_argument("--json", action="store_true",
                   help="single verdict document at the end "
                        "(complete|pending|container_vanished|timeout; "
                        "exit code unchanged)")
    w.set_defaults(fn=lambda a, pt: cli.cmd_wait(a))

    pm = sub.add_parser("probe-mod", help="run unigym.scripts.probe_<name> "
                                          "(module probes; renamed from "
                                          "renamed -- probe is "
                                          "now the e2e one-command probe)")
    pm.add_argument("name")
    pm.add_argument("-b", "--backend", choices=["isaaclab", "mujoco"], default="isaaclab")
    pm.set_defaults(fn=lambda a, pt: ss.cmd_probe(a, pt))

    b = sub.add_parser("bash", help="interactive container with unigym mounts")
    b.set_defaults(fn=lambda a, pt: ss.cmd_bash(a, pt))

    t = sub.add_parser("pytest", help="host venv pytest at repo root")
    t.set_defaults(fn=lambda a, pt: ss.cmd_pytest(a, pt))


def _register_status(sub) -> None:
    """Register the bridge/console/director command surface (client)."""
    from unigym.scripts import client as cli

    # ---- server world ---------------------------------------------------------
    # REMOVAL (designs/v043-run-record.md §2): the submit queue is
    # gone (zero users; batch = `unigym run a.yaml b.yaml ...`).  The
    # `submit` word stays as an argparse stub so old muscle memory gets
    # the migration hint, not "invalid choice".
    su = sub.add_parser("submit", help="REMOVED: use "
                                       "`unigym run CONFIG... --bg`")
    su.add_argument("config", nargs="*", help=argparse.SUPPRESS)
    su.add_argument("-b", "--backend", default=None, help=argparse.SUPPRESS)
    su.add_argument("-o", "--override", action="append", default=None,
                    dest="override", help=argparse.SUPPRESS)
    su.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    su.add_argument("--serve-port", type=int, default=0, dest="serve_port",
                    help=argparse.SUPPRESS)
    su.add_argument("--serve-camera", default=None, dest="serve_camera",
                    help=argparse.SUPPRESS)
    su.add_argument("--pause", action="store_true", dest="pause",
                    help=argparse.SUPPRESS)
    su.add_argument("--runs-root", default=None, help=argparse.SUPPRESS)
    su.add_argument("--json", action="store_true",
                    help="single JSON document on stdout (errors too)")
    su.set_defaults(fn=lambda a, pt: cli.cmd_submit(a, pt))

    st = sub.add_parser("status", help="live SHM runs + deaths "
                                       "(submit queue removed)")
    st.add_argument("run", nargs="?", default=None,
                    help="run id (or 'latest'): with --metrics/--config/"
                         "--report, answer that run's typed query ("
                         "same API face as the webui)")
    st.add_argument("--json", action="store_true", help="machine-readable output")
    st.add_argument("--no-shm", action="store_true", dest="no_shm",
                    help="skip the live SHM scan")
    st.add_argument("--metrics", action="store_true", dest="metrics",
                    help="with `run`: typed metrics query (step/fps/"
                         "episodes/reward rings/benchmark progress)")
    st.add_argument("--config", action="store_true", dest="config",
                    help="with `run`: effective config snapshot")
    st.add_argument("--report", action="store_true", dest="report",
                    help="with `run`: benchmark report draft (null when "
                         "the run has no benchmark)")
    st.set_defaults(fn=lambda a, pt: cli.cmd_status(a, pt))

    # ---- runtime bridge  -----------------------------------------
    def bridge_parser(name, help_text):
        b = sub.add_parser(name, help=help_text)
        b.add_argument("run", help="run id (see `unigym runs`; 'latest' = newest)")
        b.add_argument("--timeout", type=float, default=10.0,
                       help="command->ack timeout in seconds")
        return b

    rr = sub.add_parser("runs", help="list live bridge runs (SHM scan)")
    rr.add_argument("--history", action="store_true", dest="history",
                    help="list the run.json ledger instead "
                         "(<output_root>/*/run.json; name/id/state/exit/"
                         "duration; DONE/DEAD + RUNNING(stale) rows)")
    rr.add_argument("--limit", type=int, default=20,
                    help="history: how many recent runs (default 20)")
    rr.add_argument("--json", action="store_true",
                    help="single JSON document on stdout")
    rr.set_defaults(fn=lambda a, pt: cli.cmd_runs(a, pt))

    at = bridge_parser("attach", "live progress line for one run")
    at.set_defaults(fn=lambda a, pt: cli.cmd_attach(a, pt))

    cm = bridge_parser("cam", "print MJPEG / snapshot URLs for one run")
    cm.add_argument("--camera", default=None, help="camera name (default: broadcast)")
    cm.add_argument("--json", action="store_true",
                    help="single JSON document on stdout (errors too)")
    cm.set_defaults(fn=lambda a, pt: cli.cmd_cam(a, pt))

    sn = bridge_parser("snapshot", "save one camera frame as JPEG")
    sn.add_argument("--camera", default=None)
    sn.add_argument("--env", type=int, default=None, help="env_id (per_env cameras)")
    sn.add_argument("--pos", default=None, metavar="X,Y,Z",
                    help="ux-7 transient world-pose override (with "
                         "--lookat; fixed/free rigs only; restored after "
                         "the frame)")
    sn.add_argument("--lookat", default=None, metavar="X,Y,Z",
                    help="aim point for --pos (world frame)")
    sn.add_argument("--out", default=None, help="output jpg path")
    sn.add_argument("--json", action="store_true",
                    help="single JSON document on stdout (errors too)")
    sn.set_defaults(fn=lambda a, pt: cli.cmd_snapshot(a, pt))

    cp = bridge_parser("capture", "one command -> one artifact path on "
                                  "stdout (ux-5: jpg frame | npz dump)")
    cp.add_argument("--camera", default=None,
                    help="camera name (default: broadcast; jpg mode only)")
    cp.add_argument("--env", type=int, default=None, help="env_id (jpg mode only)")
    cp.add_argument("--npz", choices=["obs", "extra", "all"], default=None,
                    help="capture a console save_npz dump instead of a "
                         "frame (source filter)")
    cp.add_argument("--out", default=None,
                    help="output path (default: <output_dir>/capture/"
                         "<ts>_<camera>.jpg; jpg mode only)")
    cp.add_argument("--json", action="store_true",
                    help="single JSON document on stdout instead of the "
                         "bare path line")
    cp.set_defaults(fn=lambda a, pt: cli.cmd_capture(a, pt))

    pz = bridge_parser("pause", "freeze the rollout at the step boundary")
    pz.add_argument("--json", action="store_true",
                    help="single JSON document on stdout (errors too)")
    pz.set_defaults(fn=lambda a, pt: cli.cmd_pause(a, pt))
    rs = bridge_parser("resume", "continue a paused rollout")
    rs.add_argument("--json", action="store_true",
                    help="single JSON document on stdout (errors too)")
    rs.set_defaults(fn=lambda a, pt: cli.cmd_resume(a, pt))

    # ---- debug console (docs/contracts/console_contract.md) ------------------
    ins = sub.add_parser("inspect",
                         help="debug console: eval expr / obs|extra query / "
                              "summary")
    ins.add_argument("run", help="run id (see `unigym runs`; 'latest' = newest)")
    ins.add_argument("expr", nargs="?", default=None,
                     help="python expression or statement (pass-through "
                          "eval: full builtins/imports/assignments; "
                          "injected: np torch obs_cache extra bundle "
                          "adapter env)")
    ins.add_argument("--target", choices=["summary", "obs", "extra"],
                     default=None, help="fixed query (default: summary)")
    ins.add_argument("--key", default=None, help="obs/extra key (with --target)")
    ins.add_argument("--timeout", type=float, default=10.0)
    ins.add_argument("--json", action="store_true",
                     help="single JSON document on stdout (errors too)")
    ins.set_defaults(fn=lambda a, pt: cli.cmd_inspect(a, pt))

    sp = sub.add_parser("step", help="advance n policy steps while paused "
                                     "(auto-pauses; stays paused)")
    sp.add_argument("run", help="run id (or 'latest')")
    sp.add_argument("n", type=int, nargs="?", default=1,
                    help="policy steps (1..100000, default 1)")
    sp.add_argument("--timeout", type=float, default=10.0)
    sp.add_argument("--json", action="store_true",
                    help="single JSON document on stdout (errors too)")
    sp.set_defaults(fn=lambda a, pt: cli.cmd_step(a, pt))

    # ---- rewind face (Round B B1; runtime_contract §4 cmd 17-19) -------
    rw = sub.add_parser("rewind", help="rewind face: --label captures a "
                        "snapshot; SNAP_ID restores it (auto-pauses; "
                        "mujoco + rewindable adapter runs only)")
    rw.add_argument("run", help="run id (or 'latest')")
    rw.add_argument("snap_id", type=int, nargs="?", default=None,
                    help="snapshot id to restore (omit = capture a new "
                         "snapshot)")
    rw.add_argument("--label", default=None,
                    help="snapshot label (capture mode)")
    rw.add_argument("--timeout", type=float, default=30.0,
                    help="command->ack timeout in seconds")
    rw.add_argument("--json", action="store_true",
                    help="single JSON document on stdout (errors too)")
    rw.set_defaults(fn=cli.cmd_rewind)

    snl = sub.add_parser("snapshots", help="list the live rewind snapshot "
                         "slots of one run (LRU cap 16 since the v056 "
                         "checkpoint ring, memory-only)")
    snl.add_argument("run", help="run id (or 'latest')")
    snl.add_argument("--timeout", type=float, default=10.0)
    snl.add_argument("--json", action="store_true",
                     help="single JSON document on stdout (errors too)")
    snl.set_defaults(fn=cli.cmd_snapshots)

    # ---- director (docs/contracts/director_contract.md §5) --------
    dr = sub.add_parser("direct", help="director commands (only on runs "
                                       "started with --direct; else "
                                       "rejected with a restart hint)")
    dr.add_argument("sub", choices=["teleport", "setvel", "contact",
                                    "attach", "detach", "pin", "follow",
                                    "release", "push", "joints", "gripper",
                                    "status"])
    dr.add_argument("run", help="run id (or 'latest')")
    dr.add_argument("args", nargs="*",
                    help="contact: GROUP on|off|clear; gripper: "
                         "move|grasp|release|where")
    dr.add_argument("--name", default=None, help="prop name (teleport/"
                                                   "setvel/pin/release) or "
                                                   "'robot' (push)")
    dr.add_argument("--obj", default=None, help="prop name (attach/detach/"
                                                "follow/gripper grasp)")
    dr.add_argument("--host", default=None,
                    help="'gripper' | 'hand' | prop name | prop:NAME | "
                         "body link name (attach/follow)")
    dr.add_argument("--mode", default=None, choices=["kin", "slave",
                                                     "weld", "set", "add"],
                    help="attach mode (default: kin; weld = mujoco "
                         "opt-in) | push mode (default: add)")
    dr.add_argument("--pos", default=None, metavar="X,Y,Z")
    dr.add_argument("--quat", default=None, metavar="W,X,Y,Z")
    dr.add_argument("--lin", default=None, metavar="X,Y,Z")
    dr.add_argument("--ang", default=None, metavar="X,Y,Z")
    dr.add_argument("--rel", action="store_true",
                    help="gripper move: --pos is a world-frame delta")
    dr.add_argument("--reset-vel", action="store_true", dest="reset_vel",
                    help="teleport: keep velocities zeroed (default true; "
                         "flag is documentary)")
    dr.add_argument("--joints", default=None, metavar="JSON",
                    help='teleport: {"joint": rad} articulation writes')
    dr.add_argument("--joint", action="append", default=None,
                    metavar="NAME=VALUE",
                    help="joints op: robot joint write NAME=rad "
                         "(repeatable)")
    dr.add_argument("--zero-vel", action="store_true", dest="zero_vel",
                    help="joints op: also zero the written joints' "
                         "velocity")
    dr.add_argument("--timeout", type=float, default=10.0)
    dr.add_argument("--json", action="store_true",
                    help="single JSON document on stdout (errors too)")
    dr.set_defaults(fn=lambda a, pt: cli.cmd_direct(a, pt))

    ca = sub.add_parser("camera", help="live camera property apply / frame "
                                       "fetch / inspect (debug toolkit)")
    ca.add_argument("op", choices=["set", "get", "inspect"])
    ca.add_argument("run", help="run id (or 'latest')")
    ca.add_argument("camera_name", nargs="?", default=None,
                    help="camera name; inspect: omit for every "
                         "instantiated rig")
    ca.add_argument("kv", nargs="*", help="set only: k=v pairs (JSON values)")
    ca.add_argument("--env", type=int, default=None)
    ca.add_argument("--out", default=None)
    ca.add_argument("--view", default=None,
                    help="get only: registered RenderSpec view name "
                         "(camera_contract §14; visual | "
                         "clean_collision | palm_pose_occl | "
                         "palm_pose_xray; absent = visual)")
    ca.add_argument("--spec", default=None, metavar="JSON",
                    help="get only: inline RenderSpec JSON ({'view': "
                         "name} + overrides, or a full spec; "
                         "camera_contract §14)")
    ca.add_argument("--flags", default=None,
                    help="DEPRECATED (v053 alias): capture flags JSON, "
                         "mechanically expanded to a RenderSpec -- use "
                         "--view/--spec")
    ca.add_argument("--timeout", type=float, default=10.0)
    ca.add_argument("--json", action="store_true",
                    help="single JSON document on stdout (set/get/inspect; "
                         "errors too)")
    ca.set_defaults(fn=lambda a, pt: cli.cmd_camera(a, pt))

    # ---- watchers (docs/contracts/watchers_contract.md §7) ----------
    wt = sub.add_parser("watch", help="programmable observers: add/list/"
                                      "clear/update/test/events (design "
                                      "watchers.md §8)")
    wt.add_argument("watch_cmd", choices=["add", "list", "clear", "update",
                                          "test", "events"])
    wt.add_argument("run", help="run id (or 'latest')")
    wt.add_argument("name", nargs="?", default=None,
                    help="watch name (add/update/clear)")
    wt.add_argument("--template", default=None,
                    help="predicate template (key_compare, proximity, "
                         "contact_state, key_window, timeout, log, tilt, "
                         "height, joint_limit)")
    wt.add_argument("--param", action="append", default=None,
                    dest="param", metavar="KEY=VALUE",
                    help="template parameter, e.g. --param key=extra/"
                         "base_pos_w --param op=abs_gt --param value=0.2 "
                         "(repeatable; JSON values)")
    wt.add_argument("--expr", default=None,
                    help="python predicate expression (console eval "
                         "pass-through semantics; bound: t step sim_time "
                         "episode env_id np key() action() contact() "
                         "pose() dist() obs_cache extra bundle adapter env)")
    wt.add_argument("--callable", default=None, dest="callable",
                    metavar="PKG.MOD:Class | /abs/file.py:Class",
                    help="external watcher class (explicit importlib/"
                         "file path, zero intrusion; the RUN process "
                         "resolves module paths -- use the absolute file "
                         "form for out-of-tree modules; "
                         "check(ctx)->WatchResult)")
    wt.add_argument("--on", default=None,
                    help="action list: pause,capture,dump,mark,notify"
                         "[,direct] (default: mark,notify)")
    wt.add_argument("--mode", default=None, choices=["edge", "level"],
                    help="trigger mode (default edge = rising edge)")
    wt.add_argument("--hold", type=int, default=None,
                    help="level mode: true for N consecutive evaluated "
                         "ticks (default 1)")
    wt.add_argument("--once", action="store_true", default=None,
                    help="auto-disarm after the first fire")
    wt.add_argument("--cooldown", type=int, default=None,
                    help="ticks between fires (default 50)")
    wt.add_argument("--every", type=int, default=None,
                    help="evaluate every K ticks (tier-2 default 5)")
    wt.add_argument("--tier", type=int, default=None, choices=[1, 2],
                    help="evaluation budget tier (default: per template; "
                         "expr/callable = 2, cap 8 active)")
    wt.add_argument("--guard", default=None,
                    help="tier-1 armed-window expr (tier-2 only evaluates "
                         "when this is true)")
    wt.add_argument("--note", default=None, help="free-text annotation")
    wt.add_argument("--all", action="store_true", dest="all",
                    help="clear: remove every watcher")
    wt.add_argument("--enabled", default=None,
                    help="update: rearm/disable (true|false)")
    wt.add_argument("--tail", type=int, default=50,
                    help="events: last N lines (default 50)")
    wt.add_argument("--kind", default=None,
                    help="events: filter by kind (watcher|watcher_error|"
                         "fall|first_contact|episode_end|run_end)")
    wt.add_argument("--timeout", type=float, default=10.0)
    wt.add_argument("--json", action="store_true",
                    help="single JSON document on stdout (errors too)")
    wt.set_defaults(fn=cli.cmd_watch)

    tl = sub.add_parser("timeline", help="post-hoc event viewer for a run "
                                         "(events.jsonl tail; watchers.md "
                                         "§3)")
    tl.add_argument("run", help="run id (or 'latest')")
    tl.add_argument("--tail", type=int, default=50)
    tl.add_argument("--kind", default=None)
    tl.add_argument("--json", action="store_true")
    tl.set_defaults(fn=lambda a, pt: cli.cmd_timeline(a, pt),
                    watch_cmd="events")

    cl = sub.add_parser("clean", help="remove zombie unigym_* shm segments")
    cl.add_argument("--dry-run", action="store_true", dest="dry_run")
    cl.add_argument("--force", action="store_true", dest="force",
                    help="also remove unattachable segments ("
                         "version-mismatch / corrupt; reported in the "
                         "dry-run class list)")
    cl.add_argument("--json", action="store_true",
                    help="single JSON document on stdout (errors too)")
    cl.set_defaults(fn=lambda a, pt: cli.cmd_clean(a, pt))

    sc = sub.add_parser("scene", help="scene tree + live state "
                        "(curated: robot/props/contacts/cameras; "
                        "PARKED-capable -- agent face of the "
                        "scene query)")
    sc.add_argument("run", help="run id (or 'latest')")
    sc.add_argument("--node", default=None, dest="node",
                    help="one entity: robot | props/<name> | <name> | "
                         "contacts/<group> | cameras")
    sc.add_argument("--full", action="store_true", dest="full",
                    help="raw full tree (escape hatch; UI never consumes)")
    sc.add_argument("--json", action="store_true", dest="json",
                    help="raw scene document (machine face)")
    sc.add_argument("--timeout", type=float, default=15.0)
    sc.set_defaults(fn=lambda a, pt: cli.cmd_scene(a, pt))
    dbg = sub.add_parser("debug", help="ux-12 agent session: preflight -> "
                         "run(--serve --webui --bg) -> wait for the bridge "
                         "-> /info summary + console/capture cheatsheet "
                         "(collect passthrough allowed, e.g. "
                         "--enable_cameras)")
    dbg.add_argument("config", help="run yaml (single)")
    dbg.add_argument("-b", "--backend", choices=["isaaclab", "mujoco"],
                     default=None,
                     help="override yaml env.backend (default: read from yaml)")
    dbg.add_argument("--serve-port", type=int, default=0, dest="serve_port",
                     help="bridge HTTP port (default: unigym.yaml "
                          "webui_port, then auto-scan)")
    dbg.add_argument("--pause", action="store_true", dest="pause",
                     help="start the run step-gate PAUSED: the engine "
                          "does not step until you release it with "
                          "`unigym resume <run>` or `unigym step <run> N` "
                          "(/status shows phase PAUSED from the start)")
    dbg.add_argument("--wait-timeout", type=float, default=180.0,
                     dest="wait_timeout",
                     help="seconds to wait for the bridge to come up")
    dbg.add_argument("--poll", type=float, default=1.0,
                     help="registry poll interval seconds")
    dbg.set_defaults(fn=cli.cmd_debug)


def _register_daemon(sub) -> None:
    """Register the unigymd command group (daemon imported at dispatch)."""
    dm = sub.add_parser("daemon", help="unigymd resident daemon (webui "
                                       "portal + control API + spawn "
                                       "scheduling; runs outlive it)")
    dms = dm.add_subparsers(dest="daemon_cmd", required=True)
    dstart = dms.add_parser("start", help="start unigymd (nohup detached; "
                                          "--foreground for systemd)")
    dstart.add_argument("--api-port", type=int, default=None,
                        dest="api_port",
                        help="control API port (default: unigym.yaml "
                             "daemon_api_port or 8670)")
    dstart.add_argument("--webui-port", type=int, default=None,
                        dest="webui_port",
                        help="portal port (default: unigym.yaml "
                             "webui_port or 8600)")
    dstart.add_argument("--foreground", action="store_true",
                        dest="foreground",
                        help="run unigymd in THIS process (systemd "
                             "user-unit mode)")
    dstop = dms.add_parser("stop", help="stop unigymd (TERM, then KILL)")
    dstop.add_argument("--timeout", type=float, default=15.0,
                       dest="timeout",
                       help="seconds to wait for a clean stop (default 15)")
    dstat = dms.add_parser("status", help="daemon identity + health")
    dstat.add_argument("--json", action="store_true",
                       help="single JSON document on stdout")
    dm.set_defaults(fn=None)                    # dispatched below


# --------------------------------------------------------------------------- #
# lazy parser construction + dispatch
# --------------------------------------------------------------------------- #
def _new_parser() -> tuple[argparse.ArgumentParser, argparse._SubParsersAction]:
    """Build the root parser with only the version action wired."""
    p = argparse.ArgumentParser(prog="unigym",
                                description="UniGym data-collection launcher")
    p.add_argument("--version", action="version",
                   version=f"unigym launcher (unigym.yaml at {_UNIGYM_YAML})")
    return p, p.add_subparsers(dest="cmd", required=True)


def _build_help_parser() -> argparse.ArgumentParser:
    """Root parser whose subcommands are table placeholders.

    This is the --help path: no command module is imported, so a base
    install (or a missing extra) can still enumerate the surface.
    """
    p, sub = _new_parser()
    for name, (_module, _register, description) in _COMMANDS.items():
        sub.add_parser(name, help=description)
    return p


def _load_register(name: str):
    """Resolve one table entry to its registration function."""
    module_name, register_name, _description = _COMMANDS[name]
    if module_name == _SELF:
        return globals()[register_name]
    module = importlib.import_module(module_name)
    return getattr(module, register_name)


def _build_command_parser(name: str) -> argparse.ArgumentParser:
    """Root parser with only the requested command group registered."""
    p, sub = _new_parser()
    _load_register(name)(sub)
    return p


def _command_from_argv(argv: list[str]) -> str | None:
    """First positional token when it names a known command, else None."""
    for token in argv:
        if token == "--":
            break
        if token in ("-h", "--help", "--version"):
            continue
        if token.startswith("-"):
            continue
        return token if token in _COMMANDS else None
    return None


def main(argv: list[str] | None = None) -> None:
    """Parse and dispatch one command; missing extras exit 2 with one line."""
    argv = list(sys.argv[1:] if argv is None else argv)
    command = _command_from_argv(argv)
    try:
        parser = (_build_command_parser(command) if command is not None
                  else _build_help_parser())
        a, extra = parser.parse_known_args(argv)
        if a.cmd == "pytest" and not extra:
            extra = ["-q"]
        from unigym.packs.load import load_mounted_code
        load_mounted_code()
        if a.cmd == "daemon":
            from unigym.daemon import manage as daemon_manage
            sys.exit(daemon_manage.cmd_daemon(a))
        sys.exit(a.fn(a, extra))
    except MissingExtraError as exc:
        print(f"[unigym] {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
