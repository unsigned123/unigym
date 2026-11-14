"""unigym main entry: read the yaml, assemble the env backend + adapter +
sampler + runner; SyncRunner drives the rollout, CollectSink consumes
and persists it.

Generic usage:
    python -m unigym.scripts.collect_cli --config path/to/run.yaml

This script is a thin CLI: yaml parsing + builder wiring (robot profile /
adapter / env backend / sampler / filters / terminations / recorder /
language / writer / logger / sink / runner).  The rollout control flow lives
in ``unigym.runner.sync.SyncRunner``; everything consumer-side
lives in sinks -- ``runner.sinks.collect.CollectSink`` (recording,
filtering, video, logging) and ``runner.sinks.benchmark.BenchmarkSink``
(episode scoring, docs/contracts/benchmark_contract.md) -- see docs/contracts/runner_contract.md.

The yaml ``mode`` key selects the consumer wiring (default ``"collect"``):

* ``collect``   -- CollectSink only (the original behavior, bit-exact).
* ``benchmark`` -- BenchmarkSink only: tasks score completed episodes,
  ``benchmark_report.json`` lands in the output dir, the loop stops early
  once ``benchmark.min_episodes`` episodes completed (and never exceeds
  ``benchmark.max_steps``).
* ``both``      -- CollectSink + BenchmarkSink behind a FanOutSink; the
  collect side stays bit-exact while episodes are scored in parallel.

Rollout order per policy step (engine):
    act -> postprocess -> env.step -> sampler.update -> compute_obs ->
    termination check -> sink.on_step -> [episode clock/reset policy] ->
    sink.on_episode_end -> env.reset_envs -> adapter.on_reset ->
    sampler.reset -> sink.on_step_end
unless the adapter declares otherwise (see the adapter contract for the
recorded-obs semantics of history-managing adapters).
Episode resets are per-env: after recording the final step of an env's episode
(record BEFORE reset) the env teleports that subset back to spawn and the
adapter history / command sampler are reset for the same ids.

Reset policy (yaml ``runner:`` section, optional):
    runner.reset_on_termination: false   # true -> terminated envs join the
                                         # due mask and reset the same step
    runner.reset_mask_fn: (reserved)     # not deserialized yet; the engine
                                         # accepts a callable injection seam

Backend dispatch: the yaml ``env.backend`` key selects the collection backend
(default ``"isaaclab"``); the backend is built through
``unigym.env.backends.build_env_backend(...)`` (now called from
``unigym.run.build.build_instance`` -- the one construction point) and only
talks to the engine
through the EnvBackend contract (docs/contracts/backend_contract.md).  All isaaclab
imports stay inside the isaaclab branch so other backends never touch the kit.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import yaml

from unigym.backends.registry import backend_info

parser = argparse.ArgumentParser(description="General robot locomotion data collection")
parser.add_argument("--config", type=str, required=True, help="path to run yaml")
# flag aliases: every underscore flag carries its hyphen spelling
# (dest is ALWAYS the underscore form -- docs and CLI stay interchangeable)
parser.add_argument(
    "--num_envs",
    "--num-envs",
    dest="num_envs",
    type=int,
    default=None,
    help="override env.num_envs",
)
parser.add_argument("--steps", type=int, default=None, help="override collect.steps")
parser.add_argument(
    "--episode_length_s",
    "--episode-length-s",
    dest="episode_length_s",
    type=float,
    default=None,
    help="override env.episode_length_s (long-single-episode video runs)",
)
parser.add_argument(
    "--output_dir",
    "--output-dir",
    dest="output_dir",
    type=str,
    default=None,
    help="override output dir",
)
# per-run scene artifacts: ON by default; the flag turns
# them off.  Six files land under <run output dir>/scene/.
parser.add_argument(
    "--no-artifacts",
    dest="no_artifacts",
    action="store_true",
    help="disable the per-run scene artifacts "
    "(draw/resolution/scene/payload_index/composed/"
    "compile; default ON)",
)
parser.add_argument(
    "--ckpt_dir",
    "--ckpt-dir",
    dest="ckpt_dir",
    type=str,
    default=None,
    help="override adapter ckpt dir",
)
parser.add_argument(
    "--instruction", type=str, default=None, help="override adapter.instruction (language adapters)"
)
parser.add_argument("--seed", type=int, default=None)
# config overrides: dot-path rewrites applied after the yaml
# load, before the engine is built; effective diff -> meta.json
parser.add_argument(
    "-s",
    "--set",
    dest="set",
    action="append",
    default=None,
    metavar="KEY=VALUE",
    help="config override, e.g. -s manual.control_dt=0.02 "
    "-s run_name=x31tmp (repeatable; type-inferred: "
    "int/float/bool/JSON)",
)
# spawn adapter params (most-specific overlay of the yaml adapter section;
# scalars / scalar lists only -- the channel cannot touch any other
# config section, and 'name' may be overridden to switch the adapter,
# e.g. manual: {"name": "manual", "control_dt": 0.02} )
parser.add_argument(
    "--adapter_params",
    "--adapter-params",
    dest="adapter_params",
    type=str,
    default=None,
    metavar="JSON",
    help="spawn overlay of the yaml adapter section as a "
    "JSON object, e.g. --adapter-params "
    '\'{"name": "manual", "control_dt": '
    '"data_out/<run>"}\' (shallow merge; scalars / '
    "scalar lists only; unknown keys fail fast)",
)
# run label: display metadata only -- the WebUI ledger shows
# label || run_name || run_id; it never enters any filesystem path.
parser.add_argument(
    "--label",
    dest="label",
    type=str,
    default=None,
    metavar="TEXT",
    help="display-only run label (<= 80 chars, WebUI "
    "ledger face); pure metadata -- run_id/out_dir "
    "are never derived from it",
)
# runtime worker (bridge; run-slimming -- the run is a PURE
# SHM worker: every run publishes status/frames and serves the command slot;
# media/console live in the daemon media gateway, not here)
parser.add_argument(
    "--serve",
    action="store_true",
    help="ACCEPTED AS NO-OP: every run is a SHM "
    "worker now (SHM telemetry + command slot are "
    "always on; media/console live in the unigymd "
    "portal)",
)
parser.add_argument(
    "--serve_port",
    "--serve-port",
    dest="serve_port",
    type=int,
    default=0,
    help="ACCEPTED AS NO-OP (runs have no HTTP " "port; the daemon portal serves all media)",
)
parser.add_argument(
    "--serve_camera",
    "--serve-camera",
    dest="serve_camera",
    type=str,
    default=None,
    help="publishing (broadcast) camera for the SHM frame " "slots (default: first instantiated)",
)
parser.add_argument(
    "--serve_state_dir",
    "--serve-state-dir",
    dest="serve_state_dir",
    type=str,
    default=None,
    help="serve-registry json directory (default " "UNIGYM_SERVE_DIR or /tmp)",
)
parser.add_argument(
    "--run_log",
    "--run-log",
    dest="run_log",
    type=str,
    default=None,
    help="this run's log file (carried into serve registry " "v2 by the spawn builder)",
)
parser.add_argument(
    "--serve_pause",
    "--serve-pause",
    dest="serve_pause",
    action="store_true",
    help="start with the debug step-gate PAUSED: "
    "the engine parks at the FIRST step boundary and "
    "does not advance until `unigym resume` or "
    "`unigym step` (release: `unigym debug --pause` "
    "sets this)",
)
# trace replay (debug toolkit B; docs/contracts/trace_contract.md §7)
parser.add_argument(
    "--replay",
    type=str,
    default=None,
    help="replay a trace bundle into this run: the recorded "
    "action sequence drives the rollout (toolkit B; "
    "contract §7).  Combine with the trace record CLI "
    "for the determinism closed loop",
)
parser.add_argument(
    "--replay_mode",
    "--replay-mode",
    dest="replay_mode",
    choices=["action", "obs"],
    default="action",
    help="action (default): recorded raw replaces the "
    "adapter output before postprocess, physics runs "
    "live | obs: recorded policy frame feeds act, the "
    "LIVE raw is compared against the trace "
    "(downstream regression; physics bypassed in the "
    "information loop)",
)
parser.add_argument(
    "--replay_key",
    "--replay-key",
    dest="replay_key",
    type=str,
    default=None,
    help="trace segment feeding the raw face (default "
    "action/raw; e.g. action/wbc_raw for imported HA "
    "traces)",
)
parser.add_argument(
    "--replay_strict",
    "--replay-strict",
    dest="replay_strict",
    action="store_true",
    help="fail fast on in-stride holds (record stride > 1) " "instead of WARN",
)
parser.add_argument(
    "--replay_freeze_rng",
    "--replay-freeze-rng",
    dest="replay_freeze_rng",
    action="store_true",
    help="re-seed the global rng faces from the trace "
    "manifest seed (opt-in: adapter rng is outside "
    "the determinism guarantee face by default, "
    "contract §7.4)",
)
# director mode (docs/contracts/director_contract.md) -- the ONLY
# opt-in face: without it no Director exists, direct.* commands are
# rejected fail-fast, and the model is byte-identical (contract §0.3)
parser.add_argument(
    "--direct",
    action="store_true",
    help="mount the director (direct.* command family): "
    "teleport/setvel/contact-override/attach/virtual "
    "gripper.  Benchmark-integrity guard: runs started "
    "WITHOUT --direct reject director commands and "
    "directed episodes are marked in the report",
)
# watchers (docs/contracts/watchers_contract.md): born-armed
# predicate pack -- every run also streams events.jsonl regardless.
parser.add_argument(
    "--watch",
    type=str,
    default=None,
    help="arm watchers from a yaml pack at start "
    "(born-armed; docs/designs/watchers.md §8.6).  "
    "Live management: unigym watch <run> add|list|...",
)
parser.add_argument(
    "--serve_frame_mb",
    "--serve-frame-mb",
    dest="serve_frame_mb",
    type=float,
    default=4.0,
    help="per-slot frame capacity in MB (default 4)",
)
# retired streaming flags (no-op, soft landing)
parser.add_argument(
    "--serve_webui",
    "--serve-webui",
    dest="serve_webui",
    action="store_true",
    help="ACCEPTED AS NO-OP (the WebUI moved to " "the unigymd portal on 8600)",
)
parser.add_argument(
    "--serve_token",
    "--serve-token",
    dest="serve_token",
    type=str,
    default=None,
    help="ACCEPTED AS NO-OP (auth moved to the " "unigymd portal; runs have no token)",
)


def _peek_backend(config_path: str) -> str:
    """Resolve ``env.backend`` from the run yaml WITHOUT isaaclab.

    Fail-fast contract (backend_contract §1; audit 2026-10 P0-2): an
    unreadable config or a yaml syntax error dies HERE with an actionable
    {error, hint} message -- never a silent fall-back to the default
    backend (a typo'd path on a mujoco config would otherwise ride the
    whole isaaclab/kit launch chain before dying in ``main()``).  A config
    that parses but omits ``env.backend`` keeps the documented schema
    default ``"isaaclab"``.
    """
    try:
        with open(config_path) as f:
            peek = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise SystemExit(
            f"[collect] error: cannot peek env.backend from config "
            f"{config_path}: {e!r} "
            f"| hint: fix the config path / yaml syntax first -- the "
            f"backend decides the launch chain, no silent default is "
            f"taken"
        ) from e
    if not isinstance(peek, dict):
        raise SystemExit(
            f"[collect] error: config {config_path} must be a yaml mapping "
            f"(got {type(peek).__name__}) "
            f"| hint: see backend_contract §1 (env.backend selects the "
            f"backend; default isaaclab)"
        )
    env = peek.get("env") or {}
    if not isinstance(env, dict):
        raise SystemExit(
            f"[collect] error: config {config_path}: 'env' must be a "
            f"mapping (got {type(env).__name__}) "
            f"| hint: see backend_contract §1 (env.backend selects the "
            f"backend; default isaaclab)"
        )
    return str(env.get("backend", "isaaclab"))


backend_label = _peek_backend(parser.parse_known_args()[0].config)


def _warn_retired_flags(args_cli) -> None:
    """Soft landing (design §2): the retired serve/webui/token flags
    are accepted but do nothing -- one line each, then move on."""
    if getattr(args_cli, "serve", False):
        print(
            "[collect] NOTE: --serve is a no-op -- every run "
            "is a SHM worker (media/console live in the unigymd portal)",
            flush=True,
        )
    if getattr(args_cli, "serve_port", 0):
        print("[collect] NOTE: --serve_port is a no-op (runs " "have no HTTP port)", flush=True)
    if getattr(args_cli, "serve_webui", False):
        print(
            "[collect] NOTE: --serve_webui is a no-op (WebUI "
            "moved to the unigymd portal on 8600)",
            flush=True,
        )
    if getattr(args_cli, "serve_token", None):
        print(
            "[collect] NOTE: --serve_token is a no-op (auth " "moved to the unigymd portal)",
            flush=True,
        )


simulation_app = None
_kit_launcher = backend_info(backend_label).kit_launcher
if _kit_launcher is not None:
    # kit backends: full AppLauncher CLI (add_app_launcher_args) and kit
    # launch BEFORE any kit-dependent builder runs (robot profiles import
    # isaaclab_assets, which requires the kit up).  The launcher lives in
    # the backend package; this script only asks the registry face.
    simulation_app, args_cli = _kit_launcher()(parser, warn=_warn_retired_flags)
else:
    # non-kit backends: keep the core CLI surface (--device/--headless/
    # --enable_cameras/--livestream accepted) but skip the kit launch; values
    # are consumed by main() as before, the launch itself is ignored.
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device used by the backend simulation " "(non-isaaclab backends default to cpu)",
    )
    parser.add_argument(
        "--headless", action="store_true", help="accepted for CLI parity; headless is always forced"
    )
    parser.add_argument(
        "--enable_cameras",
        "--enable-cameras",
        dest="enable_cameras",
        action="store_true",
        help="accepted for CLI parity; ignored without a " "camera-capable backend",
    )
    parser.add_argument(
        "--livestream",
        type=int,
        default=-1,
        help="accepted for CLI parity; ignored without isaaclab",
    )
    args_cli = parser.parse_args()
    args_cli.headless = True
    _warn_retired_flags(args_cli)


def _resolve_artifacts_dir(cfg, args_cli, out_dir) -> Path | None:
    """Resolve the per-run scene artifact directory.

    Args:
        cfg: The effective run config (kept for call-site parity).
        args_cli: Parsed CLI args; ``--no-artifacts`` turns the face off.
        out_dir: The resolved run output directory.

    Returns:
        ``<out_dir>/scene``, or ``None`` when artifacts are disabled.
    """
    if args_cli.no_artifacts:
        return None
    return Path(out_dir) / "scene"


def main():
    with open(args_cli.config) as f:
        cfg = yaml.safe_load(f)
    # Code packs activate through the explicit runtime at this composition
    # boundary; build_spec receives the same runtime (no module global).  A
    # daemon-spawned worker is a fresh process, so scenes, benchmarks and asset
    # resources must be discovered here before spec construction.
    from unigym.pack_api import discover

    runtime = discover(entry_points=True)
    runtime.activate_all()
    # v068 runtime layout: the worker ensures the artifact tree too
    # (idempotent; design §2 -- the daemon side also mkdir -p's at
    # start, this covers direct `collect_cli` invocations)
    from unigym import runtime_dirs
    from unigym.core.spawn import load_cfg as _launch_cfg

    runtime_dirs.ensure_tree(_launch_cfg())
    # normalization / scene preset / placeholder expansion / CLI overrides:
    # the pure-data half of the old inline path (unigym/run/spec.py).
    from unigym.run.spec import build_spec

    spec = build_spec(cfg, args_cli, runtime=runtime)

    # the run output dir resolves BEFORE the backend build: the recorded
    # scene artifacts are written during env construction
    # and land under <run output dir>/scene/.  The daemon drops the run.json
    # ledger (as run.json.pending-<mypid>) into out_dir BEFORE the worker
    # starts, so "exists"/"non-empty" is no longer the collision signal --
    # a dir holding ONLY our pending ledger is OURS (the daemon created it
    # moments ago for this very run); any real data (shards/meta/summary/
    # videos or a previous run's ledger) rotates to the timestamped sibling.
    # The pid tag means a shared-name spawn can never clobber the previous
    # run's ledger.
    out_dir = spec.output.out_dir
    if out_dir is None:
        raise SystemExit("[collect] error: run output dir is unresolved")
    spawn_dir = out_dir
    own_pending = f"run.json.pending-{os.getpid()}"
    if os.path.exists(out_dir) and set(os.listdir(out_dir)) - {"run.json", own_pending}:
        out_dir = f"{out_dir}_{time.strftime('%Y%m%d_%H%M%S')}"
    # ADOPT: the ledger must live where the data lives (file move only --
    # content stays daemon-authored; the daemon follows via pid-keyed
    # lookups).  No rotation -> adopt in place; rotation -> adopt into the
    # rotated dir.
    if os.path.isfile(os.path.join(spawn_dir, own_pending)):
        try:
            os.makedirs(out_dir, exist_ok=True)
            os.replace(os.path.join(spawn_dir, own_pending), os.path.join(out_dir, "run.json"))
        except OSError:
            pass  # ledger stays behind: the scan face
            # keeps listing the run (pending files are read too); the
            # daemon follows when the worker names itself
    artifacts_dir = _resolve_artifacts_dir(spec.cfg, args_cli, out_dir)
    from dataclasses import replace

    spec = replace(
        spec,
        output=replace(
            spec.output,
            out_dir=out_dir,
            artifacts_dir=(str(artifacts_dir) if artifacts_dir is not None else None),
        ),
    )

    # the only construction point: spec -> live resources
    from unigym.run.build import build_instance

    instance = build_instance(spec)

    from unigym.runner import make_runner

    engine = make_runner(instance)
    # rewind face (Round B B1): snapshot/rewind/snapshots handlers on the
    # bridge dispatch table; the two-level gate (backend SNAPSHOT trait +
    # adapter rewindable) fails fast at command time with {error, hint}.
    # v057 providers: the sink facilities' open-episode state joins every
    # checkpoint payload and is restored fail-fast with the engine+adapter
    from unigym.runtime.rewind import attach as attach_rewind

    attach_rewind(
        instance.bridge,
        env=instance.env,
        adapter=instance.adapter,
        engine=engine,
        backend_label=spec.backend,
        providers=instance.rewind_providers,
    )
    try:
        engine.run(stop_when=instance.stop_when)
        if instance.replay is not None:
            for line in instance.replay.report_lines():
                print(line, flush=True)
    finally:
        # normal path: on_run_end already closed the worker (idempotent);
        # exception path: still DONE + unlink so no zombie segment is left
        instance.close()


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()
