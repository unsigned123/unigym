#!/usr/bin/env python
"""spawn_spec.py -- the ONE spawn command builder .

Every path that launches a collect run -- the direct `unigym run` world and
the `unigym server` supervisor -- goes through :func:`build_spawn`.  This is
the anti-drift guard: there is no second place where a docker/mujoco command
line is synthesized (docs/contracts/launcher_contract.md §3).

Synthesis rules are byte-for-byte the pre-split launch.py ones (locked by
tests/test_launcher.py snapshot tests):

    mujoco   -> host venv python -m unigym.scripts.collect_cli (MUJOCO_GL=egl
                via setdefault; foreground exec or background Popen + log)
    isaaclab -> synthesized `docker run` (defaults from unigym.yaml at repo root;
                background = nohup + log; --serve adds `--ipc host`)

backend and launch shape are DECOUPLED -- build_spawn dispatches
through the executor profile table (unigym/executors.py; the docker
synthesis moved there bit-for-bit).  With no executors/backends sections
the built-in profile reproduces the two branches above byte-for-byte
(L1 snapshots are the judge).  `unigym.local.yaml` (gitignored) deep-
merges over unigym.yaml so machine paths stay out of the repo.

Also hosts the run-lifecycle primitives shared by `unigym wait` (client)
and the server supervisor: completion criteria (summary marker + artifacts)
and container teardown.

Stdlib (argparse/subprocess) + yaml only; importable without
isaaclab/mujoco/torch.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from unigym import runtime_dirs
from unigym.executors import (          # noqa: F401 (docker synthesis home)
    Executor,
    collect_cli_in_container,
    container_state,
    containerize,
    docker_base_args,
    executor_for,
    resolve_executors,
    teardown_container,
    translate_path,
)

_REPO = Path(__file__).resolve().parents[2]
_UNIGYM_YAML = _REPO / "unigym.yaml"
_LOCAL_YAML = _REPO / "unigym.local.yaml"      # gitignored overlay

_DEFAULTS = {
    "image": "isaac-lab-2.3.2-ort:uni",
    # repo-relative / env-driven: no machine-private paths in the product.
    "host_repo": str(_REPO),
    "container_repo": "/workspace/unigym",
    "host_ro_prefix": str(_REPO.parent),
    "workdir": "/workspace/unigym",
    "venv_python": os.environ.get("UNIGYM_VENV_PYTHON", sys.executable),
    "gpus": "all",
    "network": "host",
    "device": "cpu",
    "headless": True,
    "output_root": "data_out",
    # v068 runtime layout: spawn artifacts live under ONE root
    # (design docs/designs/v068-runtime-layout.md §2); log_dir/state_dir/
    # serve_dir derive from it in load_cfg (explicit yaml keys win --
    # unigym.local.yaml's legacy log_dir stays honored).
    "runtime_dir": runtime_dirs.DEFAULT_RUNTIME_DIR,
    "state_prefix": "unigym-run",
    "wait_timeout": 7200,
    "wait_poll": 15,
    "runs_root": str(_REPO / "runs"),
}

_SUMMARY_MARKER = "[collect] summary:"
BACKENDS = ("isaaclab", "mujoco")


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursive merge: dicts merge deep, scalars overwrite, lists replace."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_cfg() -> dict:
    """Launcher cfg: _DEFAULTS < unigym.yaml < unigym.local.yaml.

    The local layer (gitignored) carries machine-specific values (venv /
    host_repo / executors / backends bindings) so the committed yaml stays
    portable -- unigym doctor --suggest --write produces it.

    v068: the artifact family derives from ``runtime_dir``
    (runtime_dirs.derive_cfg_defaults) -- explicit yaml keys still win.
    """
    cfg = dict(_DEFAULTS)
    if _UNIGYM_YAML.exists():
        cfg.update(yaml.safe_load(_UNIGYM_YAML.read_text()) or {})
    if _LOCAL_YAML.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(_LOCAL_YAML.read_text()) or {})
    return runtime_dirs.derive_cfg_defaults(cfg)


def runs_root(cfg: dict | None = None) -> Path:
    cfg = cfg or load_cfg()
    r = Path(cfg.get("runs_root") or _DEFAULTS["runs_root"])
    return r if r.is_absolute() else _REPO / r


def resolve_backend(cfg_path: Path, override: str | None) -> str:
    if override:
        return override
    with open(cfg_path) as f:
        y = yaml.safe_load(f) or {}
    return (y.get("env") or {}).get("backend", "isaaclab")


def out_dir_for(cfg: dict, cfg_path: Path, passthrough: list[str]) -> Path:
    if "--output_dir" in passthrough:
        return Path(passthrough[passthrough.index("--output_dir") + 1])
    if "--output-dir" in passthrough:
        return Path(passthrough[passthrough.index("--output-dir") + 1])
    with open(cfg_path) as f:
        y = yaml.safe_load(f) or {}
    run_name = y.get("run_name", cfg_path.stem)
    root = y.get("output_root", cfg["output_root"])
    root = Path(root) if Path(root).is_absolute() else _REPO / root
    return root / run_name


def _state_root(cfg: dict) -> Path:
    """Spawn-state json home.

    v068: state_dir; legacy cfg dicts that only carry log_dir keep
    working; bare dicts derive from runtime_dir.
    """
    raw = cfg.get("state_dir") or cfg.get("log_dir")
    return Path(raw) if raw else runtime_dirs.state_dir(cfg)


def write_state(cfg: dict, backend: str, container: str | None, log: Path,
                cfg_path: Path, out_dir: Path,
                paused: bool = False, direct: bool = False) -> Path:
    state = {
        "backend": backend, "container": container, "log": str(log),
        "config": str(cfg_path), "output_dir": str(out_dir),
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    if paused:                     # born-paused audit trail (absent == false)
        state["paused"] = True
    if direct:                     # director-mode audit trail (absent == false)
        state["direct"] = True
    root = _state_root(cfg)
    root.mkdir(parents=True, exist_ok=True)    # idempotent (design §2)
    path = root / f"{cfg['state_prefix']}-{int(time.time())}.json"
    path.write_text(json.dumps(state, indent=1))
    return path


def newest_state(cfg: dict) -> Path:
    """Newest spawn-state json.

    No /tmp-root fallback: readers see only the dedicated dirs --
    pre-layout files age out naturally.
    """
    root = _state_root(cfg)
    cands = sorted(root.glob(f"{cfg['state_prefix']}-*.json"))
    if not cands:
        sys.exit("[unigym wait] no run state found (start with: unigym run CONFIG --bg)")
    return cands[-1]


def worker_args(serve_camera: str | None, state_dir: str,
                log_path: Path | None) -> list[str]:
    """collect_cli flags for the ALWAYS-ON SHM worker .

    The old ``--serve``/``--serve_port``/``--serve_webui``/``--serve_token``
    spawn flags are gone: every run is a pure SHM worker (design
    docs/designs/v027-media-gateway.md §2), so only the registry directory,
    the run's own log path (registry v2 carries it) and the publishing
    camera remain selectable at spawn time.
    """
    args = ["--serve_state_dir", state_dir]
    if log_path is not None:
        args += ["--run_log", str(log_path)]
    if serve_camera:
        args += ["--serve_camera", serve_camera]
    return args


# --------------------------------------------------------------------------- #
# SpawnSpec: structured description of one launch + .command()/.spawn()/.exec()
# --------------------------------------------------------------------------- #
@dataclass
class SpawnSpec:
    backend: str                       # isaaclab | mujoco
    argv: list[str]                    # full host-view command (nohup included)
    config: Path
    out_dir: Path
    bg: bool
    container: str | None = None       # isaaclab container name (bg only)
    log_path: Path | None = None       # bg only (foreground inherits stdio)
    state_path: Path | None = None     # launcher state json (bg only)
    env_delta: dict = field(default_factory=dict)
    env_setdefault: bool = False       # True: apply env_delta via setdefault
    use_nohup: bool = False            # isaaclab bg wraps argv with nohup
    # (docs/designs/v026-unigymd.md §9.2): HARD-set env (daemon GPU
    # alloc -- CUDA_VISIBLE_DEVICES must not inherit the daemon's own env,
    # so setdefault semantics are wrong here).  Empty by default: L1
    # snapshots and the historical env behavior are untouched.
    env_force: dict = field(default_factory=dict)

    def command(self) -> str:
        """Render the command line (shell-quoted, host view)."""
        return shlex.join(self.argv)

    def full_env(self) -> dict:
        env = dict(os.environ)
        # daemon torch-pool cap scope guard (unigym/daemon/__main__.py):
        # the OMP value the daemon pinned for ITSELF must not leak into
        # run children -- runs keep their shell env contract.  Only the
        # daemon-injected pair is stripped (a pre-existing user value
        # carries no marker and passes through); env_force below still
        # wins for an explicit per-run OMP_NUM_THREADS.
        if env.pop("_UNIGYM_DAEMON_OMP_INJECTED", None):
            env.pop("OMP_NUM_THREADS", None)
        for k, v in self.env_delta.items():
            if self.env_setdefault:
                env.setdefault(k, v)
            else:
                env[k] = v
        for k, v in self.env_force.items():
            env[k] = v
        return env

    def spawn(self) -> subprocess.Popen:
        """Background launch: independent log + detached session (bg only)."""
        if not self.bg:
            raise RuntimeError("spawn() is for bg specs; use exec()")
        assert self.log_path is not None, "bg specs carry a log path"
        argv = (["nohup"] + self.argv) if self.use_nohup else self.argv
        with open(self.log_path, "w") as f:
            return subprocess.Popen(argv, stdout=f, stderr=f,
                                    start_new_session=True,
                                    env=self.full_env())

    def exec(self):                        # noqa: A003 (foreground exec semantics)
        """Foreground launch: replace the current process (never returns)."""
        if self.bg:
            raise RuntimeError("exec() is for fg specs; use spawn()")
        # execvpe resolves the program against the SAME env we hand to the
        # child (per-run env deltas stay visible to the search path).
        os.execvpe(self.argv[0], self.argv, self.full_env())


def build_spawn(cfg: dict, config_path: Path, *, backend: str | None = None,
                passthrough: list[str] | None = None,
                serve_camera: str | None = None,
                bg: bool = True, multi: bool = False,
                track_state: bool = True,
                paused: bool = False, direct: bool = False) -> SpawnSpec:
    """The one command builder -- direct-run and server both call this.

    no serve/webui/port/token knobs -- every spawned run is a
    pure SHM worker (media lives in the daemon; design §2/§5).

    ``paused``: born-paused spawn -- appends the collect_cli
    ``--serve_pause`` flag (same name on both backends; the docker wrap
    carries it inside the argv) so the step-gate is PAUSED at birth
    (`unigym resume latest` releases).  Absent/false: argv is unchanged.

    ``direct``: director mode -- appends the collect_cli
    ``--direct`` flag (the ONLY opt-in face of the direct.* command
    family; without it commands are rejected fail-fast).
    """
    passthrough = list(passthrough or [])
    backend = resolve_backend(config_path, backend)
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend: {backend}")

    # bg runs get their log up front so the SHM worker can carry the path
    # into registry v2 (the daemon view survives daemon restarts).
    # v068: log_dir resolves to <runtime_dir>/logs (explicit yaml wins).
    log_path = None
    if bg:
        log_root = Path(cfg.get("log_dir")
                        or runtime_dirs.logs_dir(cfg))
        log_root.mkdir(parents=True, exist_ok=True)   # idempotent (§2)
        log_path = log_root / f"{cfg['state_prefix']}-{int(time.time())}.log"

    # backend -> executor profile (builtin when no executors/backends
    # sections -- byte-identical to the two historical branches).
    ex = executor_for(cfg, backend)
    serve_root = cfg.get("serve_dir") or str(runtime_dirs.serve_dir(cfg))
    argv = (["-m", "unigym.scripts.collect_cli", "--config", str(config_path)]
            + passthrough
            + worker_args(serve_camera, ex.state_dir(serve_root), log_path))
    if paused:
        argv.append("--serve_pause")
    if direct:
        argv.append("--direct")
    if ex.kind == "docker":
        name = ex.container_name(bg=bg, multi=multi,
                                 config_path=str(config_path))
        spec = SpawnSpec(backend=backend, argv=ex.wrap(argv, bg=bg, name=name),
                         config=config_path,
                         out_dir=out_dir_for(cfg, config_path, passthrough),
                         bg=bg, container=name, use_nohup=bg)
    else:
        spec = SpawnSpec(backend=backend, argv=ex.wrap(argv),
                         config=config_path,
                         out_dir=out_dir_for(cfg, config_path, passthrough),
                         bg=bg,
                         env_delta={"MUJOCO_GL": "egl"}, env_setdefault=True)

    if bg and log_path is not None:
        spec.log_path = log_path
        if track_state:
            spec.state_path = write_state(cfg, spec.backend, spec.container,
                                          log_path, config_path,
                                          spec.out_dir, paused=paused,
                                          direct=direct)
    return spec


# --------------------------------------------------------------------------- #
# run-lifecycle primitives shared by `unigym wait` (client) and server.py
# (container_state / teardown_container live in unigym/executors.py)
# --------------------------------------------------------------------------- #
def is_complete(log: Path, out_dir: Path) -> bool:
    """Completion criteria: [collect] summary marker + artifacts on disk."""
    txt = log.read_text(errors="replace") if log.exists() else ""
    marker = _SUMMARY_MARKER in txt
    art = out_dir.exists() and (out_dir / "meta.json").exists() and (
        (out_dir / "summary.json").exists()
        or (out_dir / "benchmark_report.json").exists())
    return marker and art


def log_tail(log: Path, n: int = 1500) -> str:
    txt = log.read_text(errors="replace") if log.exists() else ""
    return txt[-n:]


# --------------------------------------------------------------------------- #
# auxiliary spawns (probe / bash / pytest) -- same synthesis discipline
# --------------------------------------------------------------------------- #
def cmd_probe(a, extra: list[str]) -> int:
    cfg = load_cfg()
    module = f"unigym.scripts.probe_{a.name}"
    from unigym.backends.registry import backend_info

    info = backend_info(a.backend)
    if info.probe_in_host:
        cmd = [cfg["venv_python"], "-m", module] + extra
        env = dict(os.environ)
        for key, value in info.probe_env:
            env.setdefault(key, value)
        os.execvpe(cmd[0], cmd, env)
    docker = docker_base_args(cfg, None) + ["--entrypoint", "/isaac-sim/python.sh",
                                            cfg["image"], "-m", module] + extra
    os.execvp(docker[0], docker)


def cmd_bash(a, extra: list[str]) -> int:
    cfg = load_cfg()
    docker = docker_base_args(cfg, None, interactive=True) + ["--entrypoint", "bash",
                                                              cfg["image"]] + extra
    os.execvp(docker[0], docker)


def cmd_pytest(a, extra: list[str]) -> int:
    cfg = load_cfg()
    os.chdir(_REPO)
    os.execvp(cfg["venv_python"], [cfg["venv_python"], "-m", "pytest"] + extra)
