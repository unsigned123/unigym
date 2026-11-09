#!/usr/bin/env python
"""executors.py -- launch-shape profiles, decoupled from backends.

backend = engine code path; executor = how one python process is
materialized.  An Executor wraps the
collect_cli primitive argv ["-m", "unigym.scripts.collect_cli", ...] into
the concrete command via wrap(), maps host->env paths via map_path(), and
reports doctor diagnostics via check().  Four kinds (whitelist-structured,
fail-fast, no eval):

    host    [python] + argv                venv / conda env prefix / uv python
    prefix  prefix + python + argv         conda run / uv run / apptainer
    script  [wrapper (python)?] + argv     HPC `module load ...; exec "$@"`
    docker  `docker run ...` synthesis     moved here BIT-FOR-BIT from
                                           spawn_spec (teardown lifecycle too)

Fallback invariant (top acceptance gate): with NO `executors:`/`backends:`
sections the built-in profile {mujoco: host(venv_python), isaaclab:
docker(top-level keys)} reproduces the two historical spawn branches
byte-for-byte -- tests/test_launcher.py L1 snapshots judge this.
Validation errors point at the yaml field (`executors.<name>.python: ...`).
Stdlib only; importable without isaaclab/mujoco/torch.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import NoReturn
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# docker synthesis -- moved verbatim from spawn_spec (byte-locked by L1)
# --------------------------------------------------------------------------- #
def translate_path(path: str, cfg: dict) -> str:
    """host repo prefix -> container mount point (host_ro_prefix covers the rest)."""
    p = str(Path(path).resolve())
    host_repo, c_repo = cfg["host_repo"], cfg["container_repo"]
    if p == host_repo:
        return c_repo
    if p.startswith(host_repo + os.sep):
        return c_repo + p[len(host_repo):]
    return p


def containerize(cli_args: list[str], cfg: dict, to_container: bool) -> list[str]:
    """Rewrite --config/--output_dir values between host and container views.

    Both spellings of the output-dir flag are translated (the hyphen
    alias: ``--output-dir`` rides the same path as ``--output_dir``).
    """
    if not to_container:
        return cli_args
    out = []
    args = list(cli_args)
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--config", "--output_dir", "--output-dir") and \
                i + 1 < len(args):
            out += [a, translate_path(args[i + 1], cfg)]
            i += 2
        elif a.startswith("--config=") or \
                a.startswith("--output_dir=") or a.startswith("--output-dir="):
            k, v = a.split("=", 1)
            out.append(f"{k}={translate_path(v, cfg)}")
            i += 1
        else:
            out.append(a)
            i += 1
    return out


def docker_base_args(cfg: dict, name: str | None, interactive: bool = False) -> list[str]:
    args = ["docker", "run"]
    if name:
        args += ["--name", name]
    else:
        args.append("--rm")
    if interactive:
        args.append("-it")
    if cfg.get("gpus"):
        args += ["--gpus", str(cfg["gpus"])]
    if cfg.get("network"):
        args += ["--network", str(cfg["network"])]
    args += [
        "-e", "ACCEPT_EULA=Y",
        "-v", f"{cfg['host_repo']}:{cfg['container_repo']}",
        "-v", f"{cfg['host_ro_prefix']}:{cfg['host_ro_prefix']}:ro",
        "-w", cfg["workdir"],
    ]
    # forward a launcher-side CUDA_VISIBLE_DEVICES into the
    # container (GPU pinning face -- e.g. CUDA_VISIBLE_DEVICES=2 makes the
    # container's ORT see physical GPU 2 as cuda:0).  Unset -> no -e flag,
    # behavior byte-identical to before.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        args += ["-e", f"CUDA_VISIBLE_DEVICES={cvd}"]
    return args


def collect_cli_in_container(cfg: dict) -> str:
    return f"{cfg['container_repo']}/unigym/scripts/collect_cli.py"


def container_state(name: str) -> str | None:
    """'running' | 'exited' | ... | None (absent)."""
    r = subprocess.run(["docker", "ps", "-a", "--filter", f"name=^{name}$",
                        "--format", "{{.State}}"], capture_output=True, text=True)
    return r.stdout.strip() or None


def teardown_container(name: str | None) -> str:
    """Best-effort teardown of a finished run's container; returns an action."""
    if not name:
        return "none"
    s = container_state(name)
    if s == "running":
        subprocess.run(["docker", "rm", "-f", name], check=False)
        return f"TEARDOWN: docker rm -f {name}"
    if s is not None:
        subprocess.run(["docker", "rm", name], check=False)
        return f"container {name} already exited; removed"
    return f"container {name} absent"


# --------------------------------------------------------------------------- #
# Executor: one launch shape + wrap/map_path/check
# --------------------------------------------------------------------------- #
@dataclass
class Diag:
    """One doctor finding (hint style aligned with select_backend)."""
    name: str
    status: str                       # "ok" | "warn" | "fail"
    detail: str = ""
    hint: str = ""


@dataclass
class Executor:
    name: str
    kind: str                         # host | prefix | script | docker
    params: dict = field(default_factory=dict)   # validated spec keys (no kind)
    cfg: dict = field(default_factory=dict)      # launcher cfg (docker fallbacks)

    def wrap(self, argv: list[str], *, bg: bool = False,
             name: str | None = None) -> list[str]:
        """Primitive collect_cli argv -> executable command (host view)."""
        if self.kind == "docker":
            return self._wrap_docker(argv, bg=bg, name=name)
        return self._launch_head() + list(argv)

    def _launch_head(self) -> list[str]:
        p = self.params
        if self.kind == "host":
            return [p["python"]]
        if self.kind == "prefix":
            return list(p["prefix"]) + [p["python"]]
        if self.kind == "script":
            return [p["wrapper"]] + ([p["python"]] if p.get("python") else [])
        raise ValueError(f"executors.{self.name}.kind: unknown kind {self.kind!r}")

    def _wrap_docker(self, argv, *, bg, name) -> list[str]:
        cv = self.docker_cfg()
        cli = containerize(list(argv), cv, True)
        entry = None
        if cli[:2] == ["-m", "unigym.scripts.collect_cli"]:
            entry = collect_cli_in_container(cv)   # the run face
            cli = cli[2:]
            if cv.get("headless") and not any(x.startswith("--headless") for x in cli):
                cli.append("--headless")
            if not any(x == "--device" or x.startswith("--device=") for x in cli):
                cli += ["--device", str(cv["device"])]
        # the IPC namespace is ALWAYS shared -- every run is a SHM worker and
        # the host-side daemon reads its frames/commands (same-host promise).
        return (docker_base_args(cv, name if bg else None) + ["--ipc", "host"]
                + ["--entrypoint", "/isaac-sim/python.sh", cv["image"]]
                + ([entry] if entry else []) + cli)

    def container_name(self, *, bg: bool, multi: bool, config_path: str) -> str | None:
        """Container name for bg docker runs (teardown handle); else None."""
        if self.kind != "docker" or not bg:
            return None
        ts = time.strftime("%Y%m%d-%H%M%S")
        return f"unigym-{ts}" + (f"-{Path(config_path).stem}" if multi else "")

    def docker_cfg(self) -> dict:
        """Docker key view: inline params override top-level cfg keys."""
        cv = dict(self.cfg)
        cv.update({k: v for k, v in self.params.items() if k in _DOCKER_KEYS})
        return cv

    def map_path(self, path: str) -> str:
        """Host path -> execution-env path (host/prefix/script: identity)."""
        if self.kind == "docker":
            return translate_path(path, self.docker_cfg())
        return path

    def state_dir(self, default: str) -> str:
        """--serve_state_dir as the run process sees it (docker: rw repo)."""
        if self.kind == "docker":
            return str(self.docker_cfg()["container_repo"])
        return default

    def check(self) -> list[Diag]:
        """Static doctor items; import probes run through wrap() in doctor."""
        n = f"executors.{self.name}"
        if self.kind == "docker":
            return self._check_docker(n)
        out = []
        py = self.params.get("python")
        if py and os.path.isabs(py):
            out.append(_file_diag(f"{n}.python", py, "interpreter"))
        if self.kind == "script":
            out.append(_file_diag(f"{n}.wrapper", self.params["wrapper"],
                                  'wrapper (contract: must exec "$@")'))
        if self.kind == "prefix":
            head = self.params["prefix"][0]
            ok = (Path(head).exists() if os.sep in head
                  else shutil.which(head) is not None)
            out.append(Diag(f"{n}.prefix", "ok" if ok else "fail", head,
                            "" if ok else f"prefix head not resolvable: {head}"))
        return out

    def _check_docker(self, n: str) -> list[Diag]:
        cv = self.docker_cfg()
        img = str(cv.get("image"))
        try:
            r = subprocess.run(["docker", "image", "inspect", img],
                               capture_output=True, text=True, timeout=30)
            rc, why = r.returncode, (r.stderr or "").strip().splitlines()[:1]
        except (OSError, subprocess.SubprocessError) as e:
            rc, why = None, [str(e)]
        hint = ("docker unreachable: " + (why[0] if why else img) if rc is None
                else f"image not found: {why[0] if why else img}")
        out = [Diag(f"{n}.image", "ok" if rc == 0 else "fail", img,
                    "" if rc == 0 else hint)]
        for key in ("host_repo", "host_ro_prefix"):
            src = str(cv.get(key) or "")
            ok = bool(src) and Path(src).exists()
            out.append(Diag(f"{n}.{key}", "ok" if ok else "fail", src,
                            "" if ok else f"mount source missing: {src}"))
        return out


def _file_diag(name: str, path: str, what: str) -> Diag:
    ok = os.path.isfile(path) and os.access(path, os.X_OK)
    return Diag(name, "ok" if ok else "fail", path,
                "" if ok else f"{what} missing or not executable: {path}")


# --------------------------------------------------------------------------- #
# Resolution + fail-fast validation (errors point at yaml fields)
# --------------------------------------------------------------------------- #
_KINDS = ("host", "prefix", "script", "docker")
_DOCKER_KEYS = frozenset((
    "image", "host_repo", "container_repo", "host_ro_prefix", "workdir",
    "gpus", "network", "headless", "device"))


def _fail(field: str, msg: str) -> NoReturn:
    raise ValueError(f"{field}: {msg}")


def builtin_executors(cfg: dict) -> dict[str, Executor]:
    """The builtin fallback profile: mujoco=host(venv_python), isaaclab=docker(top keys)."""
    return {"mujoco": Executor("mujoco", "host",
                               {"python": cfg.get("venv_python")}, cfg),
            "isaaclab": Executor("isaaclab", "docker", {}, cfg)}


def _parse_executor(name: str, raw, cfg: dict) -> Executor:
    f = f"executors.{name}"
    if not isinstance(raw, dict) or not raw.get("kind"):
        _fail(f, "needs a 'kind:' key (host | prefix | script | docker)")
    kind = raw["kind"]
    if kind not in _KINDS:
        _fail(f"{f}.kind", f"unknown kind {kind!r} "
                           f"(expected one of: {', '.join(_KINDS)})")
    p = {k: v for k, v in raw.items() if k != "kind"}
    py = p.get("python")
    if kind == "host":
        if not isinstance(py, str) or not py:
            _fail(f"{f}.python", "required (absolute interpreter path)")
        if not os.path.isabs(py):
            _fail(f"{f}.python", f"must be an absolute path, got {py!r}")
    elif kind == "prefix":
        pre = p.get("prefix")
        if (not isinstance(pre, list) or not pre
                or not all(isinstance(x, str) and x for x in pre)):
            _fail(f"{f}.prefix", "must be a non-empty list of strings")
        if not isinstance(py, str) or not py:
            _fail(f"{f}.python", "required (bare name allowed -- resolved "
                                 "inside the prefix context, e.g. conda run)")
    elif kind == "script":
        w = p.get("wrapper")
        if not isinstance(w, str) or not w or not os.path.isabs(w):
            _fail(f"{f}.wrapper", f"must be an absolute path, got {w!r}")
        if py is not None and (not isinstance(py, str) or not os.path.isabs(py)):
            _fail(f"{f}.python", f"optional, but must be an absolute path, got {py!r}")
    else:
        unknown = set(p) - _DOCKER_KEYS
        if unknown:
            _fail(f, f"unknown docker keys: {', '.join(sorted(unknown))} "
                     f"(allowed: {', '.join(sorted(_DOCKER_KEYS))})")
    return Executor(name, kind, p, cfg)


def resolve_executors(cfg: dict) -> tuple[dict[str, Executor], dict[str, str]]:
    """(executor table, backend -> executor name); fail-fast on bad specs.

    No `executors:`/`backends:` sections -> the BUILT-IN profile (fallback
    gate).  Declared executors REQUIRE a `backends:` binding (no silent
    defaults); `backends.<b>` accepts an executor name or {executor: name}.
    """
    declared = cfg.get("executors")
    binding_raw = cfg.get("backends")
    if not declared and not binding_raw:
        return builtin_executors(cfg), {"mujoco": "mujoco", "isaaclab": "isaaclab"}
    if declared and not binding_raw:
        _fail("backends", "required when executors: is declared "
                          "(bind every backend to an executor name)")
    table = ({k: _parse_executor(k, v, cfg) for k, v in declared.items()}
             if declared else builtin_executors(cfg))
    binding: dict[str, str] = {}
    for backend, tgt in (binding_raw or {}).items():
        name = tgt.get("executor") if isinstance(tgt, dict) else tgt
        if not isinstance(name, str) or name not in table:
            _fail(f"backends.{backend}",
                  f"unknown executor {name!r} (declared: "
                  f"{', '.join(sorted(table)) or 'none'})")
        binding[backend] = name
    return table, binding


def executor_for(cfg: dict, backend: str) -> Executor:
    """The executor a backend's runs go through (build_spawn dispatch)."""
    table, binding = resolve_executors(cfg)
    if backend not in binding:
        _fail("backends", f"no executor bound for backend {backend!r} "
                          f"(bound: {', '.join(sorted(binding)) or 'none'})")
    return table[binding[backend]]
