#!/usr/bin/env python
"""doctor_cli.py -- `unigym doctor`: health check / discovery / overlay
writer.

    unigym doctor                    # exam: executors + platform (exit 1 = FAIL)
    unigym doctor --suggest          # print a unigym.local.yaml fragment
    unigym doctor --suggest --write  # write it (refuses to overwrite; --force);
                                     # fragment must pass a re-check BEFORE

Import probes run THROUGH the executor (wrap() -> `-c "import unigym;
import <mod>"`, 60s cap) -- one probe code covers venv / conda / docker;
doctor understands no specific environments.  Discovery renders a PURE
function of probe findings (unit tests never touch the real box).  The
GPU item shares probe_cli.nvidia_rows (single source).  Stdlib + yaml.
"""

from __future__ import annotations

import datetime
import fnmatch
import glob
import importlib.util
import json
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

from unigym import executors as ex
from unigym.backends.registry import MJWARP_BACKEND
from unigym.scripts import spawn_spec as ss

_PROBE_TIMEOUT = 60.0  # isaaclab lazy import must not wake Kit
_WHITELIST = frozenset({"unigym", "mujoco", "isaaclab", "torch", "warp", "mujoco_warp"})
_BACKEND_MODULES = {"mujoco": "mujoco", "mjwarp": "mujoco_warp", "isaaclab": "isaaclab"}
#: extra -> its runtime import names (find_spec only -- doctor imports
#: nothing).  webui repeats daemon on purpose: it is a meta-extra that
#: pulls the same runtime (``unigym[webui]`` == portal + daemon).
_EXTRA_MODULES = {
    "daemon": ("zmq", "requests"),
    "webui": ("zmq", "requests"),
    "record": ("PIL", "imageio"),
    "agents": ("scipy", "pydantic", "onnxruntime"),
    "media": ("av", "aiortc"),
    "agentmcp": ("mcp",),
    "usd": ("pxr",),
    "warp": ("mujoco_warp", "warp"),
    "dev": ("pytest", "ruff", "pre_commit", "importlinter", "radon"),
}


def _venv_globs() -> tuple[str, ...]:
    """Discovery globs: home-relative plus the configured dependency root.

    The machine-private prefix is never a literal; it comes from the
    product's own deps anchor (env ``UNIGYM_DEPS_ROOT`` / yaml).  When the
    anchor is unset the home-relative patterns still work.
    """
    globs = ["~/.venv*/bin/python", "~/venv*/bin/python", "~/*/venv*/bin/python"]
    from unigym.errors import ConfigError

    try:
        from unigym.core.asset_root import deps_root

        root = deps_root()
    except ConfigError:
        root = None
    if root:
        globs += [
            f"{root}/*/.venv*/bin/python",
            f"{root}/*/*/.venv*/bin/python",
            f"{root}/*/*/*/.venv*/bin/python",
        ]
    return tuple(globs)


# --------------------------------------------------------------------------- #
# probes (through the executor)
# --------------------------------------------------------------------------- #
def probe_code(modules: list[str]) -> str:
    """Fixed template; whitelist-only names.  The trailing __file__ print is
    the REAL-import criterion: a stray cwd dir (e.g. /tmp/isaaclab/)
    satisfies `import isaaclab` as an empty namespace package (loader
    None, __file__ None) -- that is NOT an install."""
    mods = ["unigym"] + [m for m in modules if m != "unigym"]
    bad = sorted(set(mods) - _WHITELIST)
    if bad:
        raise ValueError(f"probe modules not in whitelist: {bad}")
    return "; ".join(f"import {m}" for m in mods) + f"; print({mods[-1]}.__file__)"


def _real_module_file(stdout: str) -> str:
    """Last stdout line unless it's a namespace-package None/empty."""
    lines = [ln for ln in (stdout or "").strip().splitlines() if ln.strip()]
    return lines[-1] if lines and lines[-1] != "None" else ""


def probe_argv(e: ex.Executor, modules: list[str]) -> list[str]:
    """The probe rides the executor's own wrap() -- docker/conda/venv alike."""
    return e.wrap(["-c", probe_code(modules)])


def run_probe(
    e: ex.Executor, modules: list[str], timeout: float = _PROBE_TIMEOUT
) -> tuple[bool, str]:
    """(ok, detail) -- detail = module __file__ when ok, error tail when not.
    cwd=/tmp: `import unigym` must resolve via the env's own editable/.pth,
    never via the checkout the doctor happens to run from."""
    try:
        r = subprocess.run(
            probe_argv(e, modules), capture_output=True, text=True, timeout=timeout, cwd="/tmp"
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout:.0f}s"
    except OSError as err:
        return False, str(err)
    if r.returncode == 0:
        return True, _real_module_file(r.stdout)
    tail = [ln for ln in (r.stderr or "").strip().splitlines() if ln.strip()][-2:]
    return False, " | ".join(tail) or f"exit {r.returncode}"


def _warp_lane_diag() -> ex.Diag:
    """The ``backends.mjwarp`` row for a lane no executor is bound to.

    A bound lane is already covered by :func:`executor_diags` (the real
    import probe there).  The unbound row reports the registry probe's
    version, or the fail-fast reason + install hint when the stack is
    absent (never a silent fallback).
    """
    from unigym.backends.registry import (
        BackendSelectionError,
        mujoco_version,
        mujoco_warp_version,
        select_backend,
    )

    try:
        select_backend("mjwarp")
    except BackendSelectionError as err:
        return ex.Diag("backends.mjwarp", "warn", err.error, err.hint)
    mw = mujoco_warp_version()
    mj = mujoco_version()
    detail = f"mujoco-warp {mw[0]}.{mw[1]}" if mw else "mujoco-warp"
    if mj:
        detail += f", mujoco {mj[0]}.{mj[1]}"
    return ex.Diag("backends.mjwarp", "ok", detail)


def executor_diags(cfg: dict, run_probe=run_probe) -> list[ex.Diag]:
    """Static checks + import probes for every executor, per bound backend."""
    table, binding = ex.resolve_executors(cfg)
    out: list[ex.Diag] = []
    for name in sorted(table):
        e = table[name]
        out += e.check()
        ok, why = run_probe(e, [])
        out.append(
            ex.Diag(
                f"executors.{name}.import_unigym",
                "ok" if ok else "fail",
                why,
                ""
                if ok
                else f"fix the env (editable/.pth) or rebind "
                f"executors.{name} (unigym doctor --suggest)",
            )
        )
    for backend in sorted(binding):
        mod = _BACKEND_MODULES.get(backend)
        if not mod:
            continue
        e = table[binding[backend]]
        ok, why = run_probe(e, [mod])
        out.append(
            ex.Diag(
                f"backends.{backend}",
                "ok" if ok else "fail",
                f"via {binding[backend]}: import {mod}" + (f" ({why})" if why else ""),
                ""
                if ok
                else f"install {mod} into the executor env, or rebind "
                f"backends.{backend} (unigym doctor --suggest)",
            )
        )
    if MJWARP_BACKEND not in binding:
        out.append(_warp_lane_diag())
    return out


# --------------------------------------------------------------------------- #
# platform items
# --------------------------------------------------------------------------- #
def _writable(p: Path) -> tuple[bool, str]:
    try:
        p.mkdir(parents=True, exist_ok=True)
        (p / ".doctor_write_probe").write_text("x")
        (p / ".doctor_write_probe").unlink()
        return True, ""
    except OSError as e:
        return False, str(e)


def _port_state(port: int) -> str:
    """'free' | 'unigymd' (healthy /api/health) | 'busy' (stranger)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as r:
            json.loads(r.read())
        return "unigymd"
    except (OSError, ValueError):
        pass
    s = socket.socket()
    s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", port))
        return "busy"
    except OSError:
        return "free"
    finally:
        s.close()


def _asset_face_diag(asset_pack_root) -> ex.Diag:
    """One ``platform.asset_face`` item: the face carries ``generators/``.

    The content pack's ``assets`` directory IS the only asset root, so the
    probe is a plain presence test: a face without ``generators/`` serves
    nothing and fails closed with the closure-tool hint.  An undeclared
    anchor is a failure too -- there is no second rung to fall back to.
    """
    try:
        root = Path(asset_pack_root())
    except OSError as err:  # AssetPathError is a FileNotFound
        return ex.Diag(
            "platform.asset_face",
            "fail",
            str(err),
            "declare the asset face (unigym.yaml "
            "`unigym_asset_pack_root`) or set UNIGYM_ASSET_PACK",
        )
    generators = root / "generators"
    if generators.is_dir():
        return ex.Diag("platform.asset_face", "ok", str(generators))
    return ex.Diag(
        "platform.asset_face",
        "fail",
        f"{generators} (absent)",
        "fold the missing payload: python tools/pack_asset_closure.py",
    )


def extras_diags(find_spec=importlib.util.find_spec) -> list[ex.Diag]:
    """One ``extras.<name>`` row per optional capability.

    ``find_spec`` is called with the declared import names only -- a probe
    must never import the optional module (an absent extra answers the
    teaching line, not a traceback).  A missing extra is a WARN: the base
    install is still healthy, and doctor's exit code only tracks failures.

    Parameters:
        find_spec: import-name -> spec oracle (injectable for tests).

    Returns:
        The extras rows, in the fixed ``_EXTRA_MODULES`` order.
    """
    out: list[ex.Diag] = []
    for extra, modules in _EXTRA_MODULES.items():
        missing = [m for m in modules if find_spec(m) is None]
        if missing:
            out.append(
                ex.Diag(
                    f"extras.{extra}",
                    "warn",
                    "missing: " + ", ".join(missing),
                    f"pip install 'unigym[{extra}]' (or 'unigym[all]')",
                )
            )
        else:
            out.append(ex.Diag(f"extras.{extra}", "ok", ", ".join(modules)))
    return out


def platform_diags(cfg: dict) -> list[ex.Diag]:
    from unigym.core.asset_root import (  # lazy (thin-CLI door)
        asset_pack_root,
        asset_root,
    )

    a = asset_root()
    out = [
        ex.Diag(
            "platform.asset_root",
            "ok" if os.path.isdir(a) else "fail",
            a,
            "" if os.path.isdir(a) else "set UNIGYM_ASSET_ROOT (unigym/core/asset_root.py)",
        )
    ]
    out.append(_asset_face_diag(asset_pack_root))
    for label, p in (
        ("runs_root", ss.runs_root(cfg)),
        ("data_out", Path(cfg.get("output_root") or "data_out")),
    ):
        if not p.is_absolute():
            p = ss._REPO / p
        ok, why = _writable(p)
        out.append(ex.Diag(f"platform.{label}", "ok" if ok else "fail", str(p), "" if ok else why))
    for label, key, dflt in (
        ("api_port", "daemon_api_port", 8670),
        ("webui_port", "webui_port", 8600),
    ):
        port = int(cfg.get(key) or dflt)
        st = _port_state(port)
        out.append(
            ex.Diag(
                f"platform.{label}",
                "warn" if st == "busy" else "ok",
                f"{port} ({st})",
                "occupied by a non-unigym process -- free it or change the port"
                if st == "busy"
                else "",
            )
        )
    from unigym.scripts.probe_cli import nvidia_rows  # single source

    rows, err = nvidia_rows()
    if err:
        missing = "No such file" in err
        out.append(
            ex.Diag(
                "platform.gpu",
                "warn" if missing else "fail",
                f"nvidia-smi: {err}",
                "" if missing else "check the driver",
            )
        )
    else:
        out.append(ex.Diag("platform.gpu", "ok", " | ".join(rows) or "none"))
    return out


# --------------------------------------------------------------------------- #
# discovery + suggestion (pure rendering, unit-testable)
# --------------------------------------------------------------------------- #
def _run(args: list[str]) -> str | None:
    """stdout of one probe command, or None (missing tool / failure)."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _import_ok(python: str, module: str, timeout: float = 30.0) -> bool:
    try:
        r = subprocess.run(
            [python, "-c", f"import {module}; print({module}.__file__)"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp",
        )
        # namespace-package guard: a stray cwd dir makes `import X` "pass"
        # with __file__ None -- that is not an install.
        return r.returncode == 0 and _real_module_file(r.stdout) != ""
    except (OSError, subprocess.SubprocessError):
        return False


def discover(cfg: dict) -> dict:
    """Probe the box -> findings {pythons: [...], images: [...]} (pure input
    for render_suggestion)."""
    pythons: dict[str, dict] = {}

    def add(path: str, source: str) -> None:
        # abspath, NOT realpath: a venv's bin/python symlinks to the base
        # interpreter -- realpath would collapse every venv built on the
        # same base into one bare interpreter (venv identity destroyed,
        # import unigym fails on the base's site-packages).
        ap = os.path.abspath(os.path.expanduser(path))
        if ap not in pythons and os.path.isfile(ap):
            pythons[ap] = {"python": ap, "source": source, "has": {}}

    for ln in (_run(["conda", "env", "list"]) or "").splitlines():
        p = ln.split()[-1] if ln.split() else ""
        if p.startswith("/"):
            add(f"{p}/bin/python", "conda")
    if u := _run(["uv", "python", "find"]):
        add(u.strip(), "uv")
    for g in _venv_globs():
        for p in glob.glob(os.path.expanduser(g)):
            add(p, "venv")
    for info in pythons.values():
        for m in ("unigym", "mujoco", "isaaclab"):  # carrier-decisive first
            info["has"][m] = _import_ok(info["python"], m)
        # torch is informational (scoring only) -- skip the ~5s import on
        # envs that cannot be carriers anyway (no unigym).
        info["has"]["torch"] = (
            _import_ok(info["python"], "torch") if info["has"]["unigym"] else False
        )
        # the MuJoCo Warp backend carrier: probe only where the
        # base mujoco stack exists (mujoco_warp imports mujoco + warp).
        info["has"]["mujoco_warp"] = (
            _import_ok(info["python"], "mujoco_warp")
            if info["has"].get("unigym") and info["has"].get("mujoco")
            else False
        )
    images = sorted(
        {
            ln.strip()
            for ln in (
                _run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"]) or ""
            ).splitlines()
            if ln.strip()
            and "<none>" not in ln
            and any(fnmatch.fnmatch(ln.strip(), p) for p in ("isaac-lab*", "*uni*"))
        }
    )
    return {"pythons": sorted(pythons.values(), key=lambda x: x["python"]), "images": images}


def render_suggestion(cfg: dict, findings: dict) -> str:
    """findings -> yaml fragment (PURE; the isaaclab carrier is a host env
    ONLY if it actually imports isaaclab -- otherwise the docker image)."""
    images = list(findings["images"])
    if cfg.get("image") in images:  # continuity: keep the pinned image
        images = [cfg["image"]] + [i for i in images if i != cfg["image"]]

    def qualified(needs: tuple[str, ...]) -> list[dict]:
        cands = [p for p in findings["pythons"] if all(p["has"].get(m) for m in needs)]
        cands.sort(key=lambda p: (-sum(p["has"].values()), p["python"]))
        cands.sort(key=lambda c: c["python"] != cfg.get("venv_python"))
        return cands

    def ename(python: str) -> str:
        base = os.path.basename(os.path.dirname(os.path.dirname(python)))
        return "host_" + "".join(ch if ch.isalnum() else "_" for ch in base).lstrip("_")

    lines = [
        "# unigym doctor --suggest -- machine-local overlay fragment",
        "# (review, then: unigym doctor --suggest --write)",
        "executors:",
    ]
    backends: list[tuple[str, str, str]] = []

    def host(needs: tuple[str, ...], backend: str, why: str) -> None:
        cands = qualified(needs)
        if not cands:
            lines.append(
                f"  # hint: no python with {'+'.join(needs)} found" " -- install them, then rerun"
            )
            return
        p = cands[0]
        n = ename(p["python"])
        if f"  {n}:" not in "\n".join(lines):
            lines.append(f"  {n}: {{kind: host, python: {p['python']}}}")
        backends.append((backend, n, f"{why} ({p['source']})"))

    host(("unigym", "mujoco"), "mujoco", "imports unigym+mujoco")
    host(("unigym", "mujoco", "mujoco_warp"), "mjwarp", "imports unigym+mujoco_warp")
    if qualified(("unigym", "isaaclab")):
        host(("unigym", "isaaclab"), "isaaclab", "imports unigym+isaaclab")
    elif images:
        lines.append(f"  docker_il: {{kind: docker, image: {images[0]}}}")
        n_probed = sum(1 for p in findings["pythons"] if p["has"].get("unigym"))
        backends.append(
            (
                "isaaclab",
                "docker_il",
                f"docker image (probed {n_probed} unigym env(s), " "none imports isaaclab)",
            )
        )
    else:
        lines.append(
            "  # hint: no isaaclab carrier -- no env imports it and" " no isaac-lab* image"
        )
    if backends:
        lines.append("backends:")
        lines += [f"  {b}: {n}   # {why}" for b, n, why in backends]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def write_local(cfg: dict, fragment: str, force: bool) -> int:
    """Write unigym.local.yaml -- only after the fragment passes a re-check."""
    path = ss._LOCAL_YAML
    if path.exists() and not force:
        print(
            f"[unigym doctor] REFUSED: {path} already exists " "(--force overrides)",
            file=sys.stderr,
        )
        return 1
    merged = ss._deep_merge(cfg, yaml.safe_load(fragment) or {})
    fails = [d for d in executor_diags(merged) if d.status == "fail"]
    for d in fails:
        print(f"[unigym doctor] FAIL {d.name}: {d.detail}\n" f"    hint: {d.hint}", file=sys.stderr)
    if fails:
        print(
            "[unigym doctor] NOT written: suggested profile must be green " "first", file=sys.stderr
        )
        return 1
    path.write_text(
        "# unigym.local.yaml -- written by `unigym doctor --suggest --write` "
        f"({datetime.datetime.now():%Y-%m-%d %H:%M}); machine overlay, "
        "gitignored\n" + fragment
    )
    print(f"[unigym doctor] wrote {path} (checks green; " "run `unigym doctor` for the full exam)")
    return 0


def cmd_doctor(a, extra: list[str]) -> int:
    cfg = ss.load_cfg()
    if a.suggest:
        frag = render_suggestion(cfg, discover(cfg))
        print(json.dumps({"suggestion": frag}) if a.json else frag, end="" if not a.json else "\n")
        return write_local(cfg, frag, a.force) if a.write else 0
    diags = executor_diags(cfg) + platform_diags(cfg) + extras_diags()
    n = {s: sum(1 for d in diags if d.status == s) for s in ("ok", "warn", "fail")}
    if a.json:
        print(json.dumps({"ok": n["fail"] == 0, "checks": [vars(d) for d in diags]}))
    else:
        for d in diags:
            tag = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}[d.status]
            print(
                f"{tag} {d.name}: {d.detail}".rstrip() + (f"\n    hint: {d.hint}" if d.hint else "")
            )
        print(f"[unigym doctor] {n['ok']} ok / {n['warn']} warn / " f"{n['fail']} fail")
    return 1 if n["fail"] else 0


def add_doctor_subparser(sub) -> None:
    d = sub.add_parser("doctor", help="health check / env discovery / " "local-overlay writer")
    d.add_argument(
        "--suggest",
        action="store_true",
        help="discover pythons + docker images, print a " "unigym.local.yaml fragment",
    )
    d.add_argument(
        "--write",
        action="store_true",
        help="with --suggest: write unigym.local.yaml (refuses " "to overwrite; --force)",
    )
    d.add_argument("--force", action="store_true", help="overwrite an existing unigym.local.yaml")
    d.add_argument(
        "--json", action="store_true", help="single JSON document on stdout (errors too)"
    )
    d.set_defaults(fn=cmd_doctor)
