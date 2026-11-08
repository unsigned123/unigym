"""unigym.runtime_dirs -- the runtime artifact tree (v068).

Every spawn-side artifact lives under ONE root, ``runtime_dir``
(default ``/tmp/unigym``; unigym.yaml / unigym.local.yaml key
``runtime_dir``), instead of littering the /tmp root (design
docs/designs/v068-runtime-layout.md §2):

    <runtime_dir>/logs/               unigym-run-<ts>.log
    <runtime_dir>/state/              unigym-run-<ts>.json  (spawn state)
    <runtime_dir>/serve/              unigym-serve-<run_id>.json
    <runtime_dir>/deaths/             unigym-death-<pid>.json
    <runtime_dir>/watch/<run_id>/     watcher artifacts (watch/ + watchers.json)

Readers see ONLY the dedicated subdirectories -- there is NO /tmp-root
fallback (owner ruling: debt over compat; pre-layout files age out
naturally, the /tmp-root archive lives in /tmp/unigym-archive/).
Existing files of live runs are never migrated (their paths are held in
the run's own spec/state).

stdlib only -- importable from spawn_spec (no yaml/torch/isaaclab).
"""
from __future__ import annotations

import glob as _glob
import os
from pathlib import Path

#: the single knob's default (unigym.yaml ``runtime_dir`` overrides)
DEFAULT_RUNTIME_DIR = "/tmp/unigym"
#: the five subdirectories (design §2)
SUBDIRS = ("logs", "state", "serve", "deaths", "watch")
#: deaths' legacy entry point (kept honored; else runtime_dir/deaths)
SERVE_DIR_ENV = "UNIGYM_SERVE_DIR"
SERVE_GLOB = "unigym-serve-*.json"


def runtime_root(cfg: dict | None = None) -> Path:
    """The artifact root: cfg['runtime_dir'] or the default."""
    raw = (cfg or {}).get("runtime_dir") or DEFAULT_RUNTIME_DIR
    return Path(str(raw)).expanduser()


def logs_dir(cfg: dict | None = None) -> Path:
    """Spawn run-log home (``unigym-run-<ts>.log``)."""
    return runtime_root(cfg) / "logs"


def state_dir(cfg: dict | None = None) -> Path:
    """Spawn state-json home (``unigym-run-<ts>.json``)."""
    return runtime_root(cfg) / "state"


def serve_dir(cfg: dict | None = None) -> Path:
    """Serve-registry home (``unigym-serve-<run_id>.json``)."""
    return runtime_root(cfg) / "serve"


def deaths_dir(cfg: dict | None = None) -> Path:
    """Death-record home (UNIGYM_SERVE_DIR legacy entry wins)."""
    env = os.environ.get(SERVE_DIR_ENV)
    return Path(env) if env else runtime_root(cfg) / "deaths"


def watch_dir(run_id: str, cfg: dict | None = None) -> Path:
    """One run's watcher-artifact home (watch/ + watchers.json)."""
    return runtime_root(cfg) / "watch" / str(run_id)


def ensure_tree(cfg: dict | None = None) -> Path:
    """Mkdir -p the whole tree (idempotent; daemon + CLI startup call).

    Returns the root.  Failures surface as OSError -- callers at process
    startup fail fast; per-writer call sites re-run the one-level mkdir
    so a runtime_dir that vanished mid-flight never kills a spawn.
    """
    root = runtime_root(cfg)
    for d in [root] + [root / s for s in SUBDIRS]:
        d.mkdir(parents=True, exist_ok=True)
    return root


def ensure_tree_safe(cfg: dict | None = None, log=print) -> Path:
    """ensure_tree that degrades to a log line (daemon startup face).

    The daemon must come up even when the tree cannot be created
    (unwritable root); writers re-run their own mkdir per file.
    """
    try:
        return ensure_tree(cfg)
    except OSError as e:
        log(f"runtime tree mkdir failed ({e!r}) -- writers will "
            f"retry per file")
        return runtime_root(cfg)


def derive_cfg_defaults(cfg: dict) -> dict:
    """Fill the derived launch-cfg family (spawn_spec.load_cfg tail).

    ``runtime_dir`` (explicit) -> ``log_dir``/``state_dir``/``serve_dir``
    defaults; keys already present (unigym.yaml / unigym.local.yaml)
    win -- the legacy single-root ``log_dir`` stays honored when a local
    overlay sets it explicitly.
    """
    root = runtime_root(cfg)
    cfg.setdefault("log_dir", str(root / "logs"))
    cfg.setdefault("state_dir", str(root / "state"))
    cfg.setdefault("serve_dir", str(root / "serve"))
    return cfg


# -- the ONE serve-registry scan (bridge / client / probe consumers) ------ #
def serve_registry_trace(rid: str) -> bool:
    """Cheap trace check: does any scanned dir hold a serve registry for ``rid``?

    The cache-miss fall-through for the daemon op
    pre-check -- a cache miss is inconclusive (the rebuild loop may
    lag the serve registry), so the op path checks for a trace before
    paying one fresh rebuild for the honest classification.
    """
    for d in (os.environ.get(SERVE_DIR_ENV), str(serve_dir(cfg=None))):
        if d and Path(d, f"unigym-serve-{rid}.json").is_file():
            return True
    return False


def serve_registry_paths(state_dir: str | os.PathLike | None = None,
                         extra_dirs: list[str] | None = None) -> list[Path]:
    """Serve-registry json paths across the scanned directories.

    Scan set (design §3): [explicit state_dir, UNIGYM_SERVE_DIR,
    runtime serve dir, *extra_dirs] -- the repo root rides as extra_dirs
    for callers that need it (docker registries mount the repo).  No
    /tmp-root fallback: readers see only the dedicated dirs (pre-layout
    files age out naturally).  Sorted by file name; duplicates by path
    removed.
    """
    dirs = [str(state_dir) if state_dir else None,
            os.environ.get(SERVE_DIR_ENV),
            str(serve_dir(cfg=None)),
            *(str(d) for d in (extra_dirs or []))]
    out: dict[str, Path] = {}
    for d in filter(None, dict.fromkeys(dirs)):
        for p in sorted(_glob.glob(str(Path(d) / SERVE_GLOB))):
            out[str(p)] = Path(p)
    return sorted(out.values(), key=lambda p: p.name)
