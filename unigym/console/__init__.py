"""Runtime debug console : stepping + inspector + save_npz.

Package layout (logic lives here, the bridge only hosts
hooks; docs/contracts/console_contract.md is the contract twin):

* ``stepping.py``   StepGate state machine (pause / request_step / wait
  park loop) -- replaces the bridge's inline freeze loop.
* ``inspector.py``  Inspector: fixed queries (summary/obs/extra), the
  pass-through eval (direct eval/exec, full
  builtins -- module docstring for the rationale), the deterministic
  inline/npz formatting rule and the npz dump
  (data_out/<run>/inspect/req_<id>.npz).
* this module       the :class:`Console` facade + the console instruction
  schema parser (:func:`parse_console`, stdlib-only) + ``attach`` wiring.

Bridge touchpoints stay minimal: ``Bridge._install_console`` registers the
three handlers (step / inspect / save_npz) into the dispatch table and the
pause branch parks through the gate.  This module keeps numpy/torch out of
its import time (the bridge imports ``stepping`` from the stdlib-only CLI
path); the inspector is imported lazily inside the facade.
"""

from __future__ import annotations

from .stepping import MAX_FAST_FORWARD, ConsoleError

__all__ = [
    "Console",
    "ConsoleError",
    "attach",
    "parse_console",
    "CONSOLE_TYPES",
    "MAX_FAST_FORWARD",
]

CONSOLE_TYPES = ("step", "inspect", "eval", "save_npz")


def parse_console(obj) -> tuple[str, dict]:
    """Validate one console instruction -> ``(bridge_command, payload)``.

    Envelope keys are whitelisted per type (an unknown key is an explicit
    :class:`ConsoleError`, never silently ignored) -- same discipline as
    the stream schema.  ``eval`` maps onto the ``inspect`` bridge command
    (target="eval"); that keeps the SHM command table at three entries.

    Stdlib only: safe to call from the bridge HTTP thread before torch is
    guaranteed warm.
    """
    if not isinstance(obj, dict):
        raise ConsoleError(
            f"console instruction must be a JSON object, " f"got {type(obj).__name__}"
        )
    t = obj.get("type")
    if not isinstance(t, str) or t not in CONSOLE_TYPES:
        raise ConsoleError(f"unknown console type {t!r} (valid: {', '.join(CONSOLE_TYPES)})")

    if t == "step":
        extra = set(obj) - {"type", "n"}
        if extra:
            raise ConsoleError(
                f"step: unexpected key(s) {sorted(extra)} "
                f"(expected {{type, n?}}; the step contract "
                f"has no unit field)"
            )
        n = obj.get("n", 1)
        if isinstance(n, bool) or not isinstance(n, int):
            raise ConsoleError(f"step: 'n' must be an integer, got {n!r}")
        if not 1 <= n <= MAX_FAST_FORWARD:
            raise ConsoleError(f"step: 'n' must be in [1, {MAX_FAST_FORWARD}], got {n}")
        return "step", {"n": n}

    if t == "inspect":
        extra = set(obj) - {"type", "target", "key"}
        if extra:
            raise ConsoleError(
                f"inspect: unexpected key(s) {sorted(extra)} " f"(expected {{type, target, key?}})"
            )
        target = obj.get("target")
        if target not in ("summary", "obs", "extra"):
            raise ConsoleError(f"inspect: 'target' must be summary|obs|extra," f" got {target!r}")
        payload: dict = {"target": target}
        if target in ("obs", "extra"):
            key = obj.get("key")
            if not isinstance(key, str) or not key:
                raise ConsoleError(
                    f"inspect: 'key' (non-empty string) is required for " f"target={target!r}"
                )
            payload["key"] = key
        return "inspect", payload

    if t == "eval":
        extra = set(obj) - {"type", "expr"}
        if extra:
            raise ConsoleError(
                f"eval: unexpected key(s) {sorted(extra)} " f"(expected {{type, expr}})"
            )
        expr = obj.get("expr")
        if not isinstance(expr, str) or not expr.strip():
            raise ConsoleError("eval: 'expr' (non-empty string) is required")
        return "inspect", {"target": "eval", "expr": expr}

    # save_npz
    extra = set(obj) - {"type", "source", "keys", "path"}
    if extra:
        raise ConsoleError(
            f"save_npz: unexpected key(s) {sorted(extra)} "
            f"(expected {{type, source?, keys?, path?}})"
        )
    source = obj.get("source", "all")
    if source not in ("obs", "extra", "all"):
        raise ConsoleError(f"save_npz: 'source' must be obs|extra|all, got {source!r}")
    payload = {"source": source}
    keys = obj.get("keys")
    if keys is not None:
        if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
            raise ConsoleError("save_npz: 'keys' must be a non-empty list of strings")
        payload["keys"] = keys
    path = obj.get("path")
    if path is not None:
        if not isinstance(path, str) or not path.strip():
            raise ConsoleError("save_npz: 'path' must be a non-empty string")
        payload["path"] = path
    return "save_npz", payload


class Console:
    """Debug-console facade mounted on one live bridge.

    Args:
        bridge: the run's :class:`~unigym.runtime.bridge.Bridge`.
        env: EnvBackend (eval namespace + freeze-period render; optional --
            queries degrade explicitly when absent).
        adapter: Adapter (eval namespace; may be None).
        out_dir: npz dump root (default: the bridge's ``out_dir`` =
            ``data_out/<run>``; dumps land in ``<out_dir>/inspect/``).
    """

    def __init__(self, bridge, env=None, adapter=None, out_dir=None):
        self.bridge = bridge
        self.env = env
        self.adapter = adapter
        self.out_dir = out_dir if out_dir is not None else getattr(bridge, "out_dir", None)
        from .inspector import Inspector

        self.inspector = Inspector(
            env=env,
            adapter=adapter,
            out_dir=self.out_dir,
            status_fn=bridge.read_status,
            bundle_fn=lambda: getattr(bridge, "_bundle", None) or {},
        )
        self.attached = False

    def attach(self) -> Console:
        """Register the three handlers into the bridge dispatch table."""
        self.bridge._install_console(self)
        self.attached = True
        return self

    # -- bridge-dispatch handlers (engine thread; errors land in the ack) --- #
    def cmd_step(self, rid: int, params: dict) -> dict:
        if self.bridge.done:
            raise ConsoleError("run finished: cannot step a DONE run")
        n = self.bridge._gate.request_step(params.get("n", 1))
        return {
            "stepping": n,
            "paused": True,
            "hint": f"engine will advance {n} policy step(s) then " f"re-park (paused)",
        }

    def cmd_inspect(self, rid: int, params: dict) -> dict:
        return self.inspector.handle(rid, params)

    def cmd_save_npz(self, rid: int, params: dict) -> dict:
        return self.inspector.save_npz(rid, params)


def attach(bridge, env=None, adapter=None, out_dir=None) -> Console:
    """Mount a console onto a live bridge (collect_cli ``--serve`` wiring)."""
    return Console(bridge, env=env, adapter=adapter, out_dir=out_dir).attach()
