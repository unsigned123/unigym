"""core.run_ops -- run lifecycle: spawn + completion wait .

docs/designs/v026-unigymd.md §2/§6: launch.py `cmd_run`'s body (and the
`unigym wait` primitive) as explicit-argument functions.  start_run()
ALWAYS spawns a detached session (subprocess.Popen(start_new_session=
True) via SpawnSpec.spawn; no PDEATHSIG anywhere) -- the daemon calls
this for /api/start, and a daemon restart/upgrade can never take a
benchmark down (§1(1)/(2)).  The old foreground exec is gone by
design (§9.7): the CLI has zero local process operations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from unigym.core import spawn as cspawn

#: run label soft cap (display metadata only -- never a path component);
#: over-cap spawns are rejected with a teaching error, not truncated.
LABEL_MAX = 80


def check_label(label) -> str | None:
    """Validate a run label; returns the teaching error or None.

    Shared by RunSpec.validate (daemon faces) and collect_cli --label
    (direct CLI) so both faces reject identically.  ``None``/empty is
    legal (label is opt-in display metadata)."""
    if label is None or label == "":
        return None
    if not isinstance(label, str):
        return f"'label' must be a string, got {type(label).__name__}"
    if len(label) > LABEL_MAX:
        return (f"'label' is display metadata, keep it <= {LABEL_MAX} "
                f"chars (got {len(label)}: {label[:60]!r}...) -- it never "
                "enters the filesystem paths, so brevity is the whole point")
    return None


@dataclass
class RunSpec:
    """One run request (daemon /api/start body; explicit, json-able).

    (registry v2 / run-slimming, design §2/§5): the serve family
    of spawn knobs is GONE -- every run is a pure SHM worker (media and
    console terminate in the daemon media gateway).  ``serve_camera``
    only picks the publishing camera.
    """

    config: str                                  # collect yaml path
    backend: str | None = None                   # None = read from yaml
    passthrough: list[str] = field(default_factory=list)
    serve_camera: str | None = None              # publishing camera choice
    env: dict = field(default_factory=dict)      # HARD env, e.g. CUDA_VISIBLE_DEVICES
    multi: bool = False                          # multi-config naming suffix
    # daemon scheduling: "auto" -> alloc.pick_gpu (registry+free
    # memory) and a HARD CUDA_VISIBLE_DEVICES; anything else = no GPU
    # allocation (the old default; new knob, off unless requested).
    gpu: str | None = None
    # born-paused spawn: structured passthrough of collect_cli
    # --serve_pause -- the run parks at the FIRST step boundary
    # (`unigym resume latest` releases).  Off unless requested.
    paused: bool = False
    # director mode: structured passthrough of collect_cli
    # --direct -- mounts the director (the direct.* command family).
    # Without it direct commands are rejected fail-fast.  Off unless
    # requested.
    direct: bool = False
    # spawn adapter params (structured overlay of the yaml adapter
    # section; serialized onto --adapter-params at spawn).  Values are
    # scalars / scalar lists only; the channel cannot touch any config
    # section beyond 'adapter'.
    adapter_params: dict = field(default_factory=dict)
    # display-only run label (designs/v0611): shown by the WebUI ledger as
    # label || run_name || run_id.  Pure metadata -- NEVER a path
    # component (run_id/out_dir derivation is untouched); soft-capped at
    # LABEL_MAX chars.  Serialized onto --label at spawn.
    label: str | None = None

    def validate(self) -> list[str]:
        """Cheap structural validation (config existence is NOT checked
        here -- the daemon fail-fasts on the resolved absolute path)."""
        errs = []
        if not self.config or not isinstance(self.config, str):
            errs.append("'config' must be a non-empty path string")
        if self.backend is not None and self.backend not in cspawn.BACKENDS:
            errs.append(f"backend must be one of {cspawn.BACKENDS}, "
                        f"got {self.backend!r}")
        if any(not isinstance(x, str) for x in self.passthrough):
            errs.append("'passthrough' must be a list of strings")
        if not isinstance(self.paused, bool):
            errs.append("'paused' must be a boolean")
        if not isinstance(self.direct, bool):
            errs.append("'direct' must be a boolean")
        if not isinstance(self.adapter_params, dict):
            errs.append("'adapter_params' must be an object")
        else:
            from unigym.adapters import AdapterParamsError, \
                validate_spawn_params
            try:
                validate_spawn_params(self.adapter_params)
            except AdapterParamsError as e:
                errs.append(str(e))
        if self.label is not None:
            err = check_label(self.label)
            if err:
                errs.append(err)
        return errs

    @classmethod
    def from_dict(cls, d: dict) -> RunSpec:
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown RunSpec keys: {sorted(unknown)} "
                             f"(known: {sorted(known)})")
        return cls(
            config=d.get("config", ""),
            backend=d.get("backend"),
            passthrough=list(d.get("passthrough") or []),
            serve_camera=d.get("serve_camera"),
            env=dict(d.get("env") or {}),
            multi=bool(d.get("multi")),
            gpu=d.get("gpu"),
            paused=bool(d.get("paused")),
            direct=bool(d.get("direct")),
            label=(d.get("label") or None),
            adapter_params=(dict(d.get("adapter_params") or {})
                            if isinstance(d.get("adapter_params"), (dict,))
                            or d.get("adapter_params") is None
                            else cast(dict, d.get("adapter_params"))))


@dataclass
class RunHandle:
    """The spawned run (LAUNCHED-line fields; json-able for the API)."""

    pid: int
    backend: str
    log_path: str
    out_dir: str
    config: str
    container: str | None = None
    state_path: str | None = None

    def to_dict(self) -> dict:
        return {"pid": self.pid, "backend": self.backend,
                "log_path": self.log_path, "out_dir": self.out_dir,
                "config": self.config, "container": self.container,
                "state_path": self.state_path}


def start_run(spec: RunSpec, cfg: dict | None = None) -> RunHandle:
    """Spawn one run: setsid-detached, independent log, state json.

    The ONE builder (cspawn.build_spawn) synthesizes the command; the
    addition over the bare spawn is the HARD-set env (env_force, e.g. an
    allocated CUDA_VISIBLE_DEVICES).  No serve knobs -- every run
    is a pure SHM worker.
    """
    errs = spec.validate()
    if errs:
        raise ValueError("; ".join(errs))
    cfg = cfg or cspawn.load_cfg()
    config_path = Path(spec.config).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError(f"config not found: {config_path}")
    passthrough = list(spec.passthrough)
    if spec.adapter_params:
        # structured channel -> the collect_cli flag (JSON round-trip is
        # exact for the scalar / scalar-list value domain)
        passthrough += ["--adapter-params",
                        json.dumps(spec.adapter_params, sort_keys=True)]
    if spec.label:
        passthrough += ["--label", str(spec.label)]
    ss = cspawn.build_spawn(
        cfg, config_path, backend=spec.backend, passthrough=passthrough,
        serve_camera=spec.serve_camera, bg=True, multi=spec.multi,
        paused=spec.paused, direct=spec.direct)
    if spec.env:
        ss.env_force.update(dict(spec.env))
    proc = ss.spawn()
    return RunHandle(pid=proc.pid, backend=ss.backend,
                     log_path=str(ss.log_path), out_dir=str(ss.out_dir),
                     config=str(config_path), container=ss.container,
                     state_path=str(ss.state_path)
                     if ss.state_path else None)


def wait_once(log: str | Path, out_dir: str | Path,
              container: str | None = None, teardown: bool = True,
              ) -> dict:
    """Single-shot completion check -- the `unigym wait` loop primitive.

    The CLI keeps the loop/verdict formatting; this reports the facts:
    completion criteria (spawn_spec.is_complete, shared with the old
    server), container state, and -- when complete and requested -- the
    teardown action actually taken.
    """
    log, out_dir = Path(log), Path(out_dir)
    complete = cspawn.is_complete(log, out_dir)
    cstate = cspawn.container_state(container) if container else None
    action = None
    if complete and teardown and container:
        action = cspawn.teardown_container(container)
    return {"complete": complete, "container_state": cstate,
            "teardown": action, "log_tail": cspawn.log_tail(log)}
