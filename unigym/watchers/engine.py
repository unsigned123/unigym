"""watchers.engine -- WatcherEngine: the run-side evaluation point.

Design docs/designs/watchers.md §2/§8.  Mounted INSIDE ``Bridge.tick``
before ``_gate.arrive()`` (a trigger that asks for pause parks the SAME
tick).  Every tick:

    for each enabled watch (budgeted):
        guard (tier-1, optional) -> skip when false
        result = predicate(ctx)          # template | expr | callable
        fire = trigger(result.fired)     # edge/level/once/cooldown (§8.5)
        if fire: execute actions         # pause/capture/dump/mark/notify

Fail-safe (§2.1): a raising predicate NEVER escapes into the engine
loop -- error event + strike; 3 consecutive strikes auto-disable the
watch (mirror of the park-loop degrade-once discipline).  Budget (§2.2):
tier-2 watches capped (default 8 active), ``every`` amortization,
optional tier-1 guard as the armed-window.

Persistence: ``<out_dir>/watchers.json`` is the truth (registry is a
view -- the thin-shell principle); events stream to events.jsonl (timeline).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import templates as _tpl
from .base import (
    CAPTURE,
    DIRECT,
    DUMP,
    Level,
    PAUSE,
    WatchContext,
    WatchError,
    WatchResult,
    now_wall,
    _CAPTURE_BUDGET,
    _DUMP_BUDGET,
)
from .spec import TIER2_CAP, WatchSpec, check_callable_path, compile_expr, exec_block, load_callable

__all__ = ["WatcherEngine", "WatchState"]

STRIKES_TO_DISABLE = 3


# --------------------------------------------------------------------------- #
# per-watch runtime state
# --------------------------------------------------------------------------- #
class WatchState:
    __slots__ = (
        "prev",
        "hold",
        "cooldown_until",
        "strikes",
        "fired",
        "disabled",
        "disabled_reason",
        "last_tick",
        "last_result",
        "instance",
        "setup_done",
        "last_degrade_tick",
    )

    def __init__(self):
        self.prev = False
        self.hold = 0
        self.cooldown_until = 0
        self.strikes = 0
        self.fired = 0
        self.disabled = False
        self.disabled_reason = ""
        self.last_tick: int | None = None
        self.last_result: WatchResult | None = None
        self.instance: Any | None = None  # callable watcher instance (§8.3)
        self.setup_done = False
        self.last_degrade_tick = -(10**9)


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #
class WatcherEngine:
    """Per-run watcher host (engine-thread only, like the console).

    Args:
        run_id: run id (events + store naming only).
        out_dir: run output dir (watchers.json, events.jsonl, captures).
        env / adapter: run objects for the WatchContext.
        event_log: the shared timeline writer (None -> drop events).
        bridge: owning Bridge (pause/capture/dump/direct actions); may
            be None in offline tests.
        specs: born-armed declarations (yaml entry point, §8.6).
    """

    def __init__(
        self, run_id: str, out_dir, env=None, adapter=None, event_log=None, bridge=None, specs=None
    ):
        self.run_id = run_id
        self.out_dir = Path(out_dir) if out_dir else None
        self.env = env
        self.adapter = adapter
        self.event_log = event_log
        self.bridge = bridge
        self._specs: dict[str, WatchSpec] = {}
        self._states: dict[str, WatchState] = {}
        self._expr_code: dict[str, tuple] = {}  # name -> (code, is_exec)
        self._guard_code: dict[str, tuple] = {}
        self.window_cache: dict = {}  # key_window deques
        self.last_overhead_us = 0.0
        self.captures = 0
        self.dumps = 0
        self._last_bundle: dict | None = None
        self._last_ctx: WatchContext | None = None
        if self.out_dir:
            (self.out_dir / "watch").mkdir(parents=True, exist_ok=True)
        for d in specs or []:
            self.add(d, persist=False)
        self._persist()
        _tpl._CURRENT_ENGINE = self  # window store hook

    # ------------------------------------------------------------------ #
    # declaration management (add/list/clear/update/test -- §8.6)
    # ------------------------------------------------------------------ #
    def _keys(self) -> list[str]:
        return self._last_ctx.keys() if self._last_ctx is not None else []

    def add(self, spec_dict: dict, persist: bool = True) -> dict:
        """Validate + compile + dry-run; persists and (re)arms."""
        spec = WatchSpec.from_dict(spec_dict)
        if spec.kind == "template":
            errors = _tpl.validate_template_params(spec.template_name, spec.params, self._keys())
            if errors:
                raise WatchError(
                    f"watch '{spec.name}': template '{spec.template}' "
                    f"invalid ({len(errors)} error(s)):\n" + "\n".join(f"  - {e}" for e in errors),
                    self._keys(),
                )
        elif spec.kind == "expr":
            self._expr_code[spec.name] = compile_expr(spec.expr)
        else:
            spec.callable_path = check_callable_path(spec.callable_path)
        if spec.guard:
            self._guard_code[spec.name] = compile_expr(spec.guard)
        # budget gate BEFORE arming (§2.2: tier-2 cap)
        tier = spec.effective_tier()
        if tier == 2:
            active = [
                n
                for n, s in self._specs.items()
                if s.effective_tier() == 2 and not self._states[n].disabled and n != spec.name
            ]
            if len(active) >= TIER2_CAP:
                raise WatchError(
                    f"watch '{spec.name}': tier-2 budget full "
                    f"({TIER2_CAP} active: {active}); disable or clear "
                    f"one, or lower --every/--tier"
                )
        if spec.every is None:
            spec.every = 5 if tier == 2 else 1
        # template-suggested trigger defaults (log/timeout), user wins
        if spec.kind == "template":
            for k, v in _tpl.get_template(spec.template_name)["suggest"].items():
                if k == "once":
                    spec.once = spec.once or bool(v)
                elif k == "mode" and spec.mode == "edge" and "mode" not in (spec_dict or {}):
                    spec.mode = v
                elif k == "hold" and spec.hold == 1 and "hold" not in (spec_dict or {}):
                    spec.hold = int(v)
                elif k == "cooldown" and "cooldown" not in (spec_dict or {}):
                    spec.cooldown = int(v)
        st = WatchState()
        if spec.kind == "callable":
            st.instance = self._instantiate(spec)
        result = None
        if self._last_ctx is not None:
            # validate-on-add (§2.2/§8.5): one trial evaluation; syntax and
            # missing-key errors reject HERE (candidates listed), runtime
            # failures after add are the fail-safe domain
            try:
                result = self._evaluate(spec, st, dry=True)
            except WatchError:
                raise
            except Exception as e:  # noqa: BLE001
                raise WatchError(
                    f"watch '{spec.name}': dry-run evaluation failed: "
                    f"{type(e).__name__}: {e} (available keys: "
                    f"{self._keys() or ['(no tick evaluated yet)']})",
                    self._keys(),
                ) from None
        self._specs[spec.name] = spec
        self._states[spec.name] = st
        doc = {
            "ok": True,
            "name": spec.name,
            "tier": tier,
            "every": spec.every,
            "dry_run": result.as_dict() if result is not None else None,
        }
        if persist:
            self._persist()
        return doc

    def clear(self, name: str | None = None) -> dict:
        if name in (None, "", "all", "__all__"):
            removed = sorted(self._specs)
            self._specs.clear()
            self._states.clear()
            self._expr_code.clear()
            self._guard_code.clear()
            self.window_cache.clear()
        else:
            if name not in self._specs:
                raise WatchError(f"no such watch: {name!r} (armed: " f"{sorted(self._specs)})")
            removed = [name]
            self._specs.pop(name)
            self._states.pop(name)
            self._expr_code.pop(name, None)
            self._guard_code.pop(name, None)
        self._persist()
        return {"ok": True, "removed": removed}

    def update(self, name: str, patch: dict) -> dict:
        if name not in self._specs:
            raise WatchError(f"no such watch: {name!r} (armed: {sorted(self._specs)})")
        spec = self._specs[name]
        d = spec.to_dict()
        allowed = {"on", "mode", "hold", "once", "cooldown", "every", "enabled", "note"}
        unknown = set(patch) - allowed
        if unknown:
            raise WatchError(
                f"watch '{name}': cannot update {sorted(unknown)} "
                f"(updateable: {sorted(allowed)}; use clear+add to "
                f"redefine the predicate)"
            )
        d.update({k: v for k, v in patch.items()})
        if patch.get("enabled") is False:
            self._states[name].disabled = True
            self._states[name].disabled_reason = "disabled via update"
        elif patch.get("enabled") is True:
            self._states[name].disabled = False
            self._states[name].disabled_reason = ""
            self._states[name].strikes = 0
        if patch.get("enabled") is not None:
            d.pop("enabled", None)  # runtime flag, not declarative
        # re-validate the patched declaration (fresh state = state reset)
        fresh = WatchSpec.from_dict(d)
        if fresh.kind == "expr":
            self._expr_code[name] = compile_expr(fresh.expr)
        if fresh.guard:
            self._guard_code[name] = compile_expr(fresh.guard)
        self._specs[name] = fresh
        st = self._states[name]
        st.prev, st.hold, st.cooldown_until = False, 0, 0
        self._persist()
        return {"ok": True, "name": name, "watchers": self.list()}

    def list(self) -> list[dict]:
        out = []
        for name in sorted(self._specs):
            spec, st = self._specs[name], self._states[name]
            what = (
                spec.expr
                if spec.kind == "expr"
                else (
                    spec.callable_path
                    if spec.kind == "callable"
                    else f"{spec.template}({spec.params})"
                )
            )
            out.append(
                {
                    "name": name,
                    "kind": spec.kind,
                    "predicate": what,
                    "tier": spec.effective_tier(),
                    "every": spec.every,
                    "on": list(spec.on),
                    "mode": spec.mode,
                    "hold": spec.hold,
                    "once": spec.once,
                    "cooldown": spec.cooldown,
                    "note": spec.note,
                    "state": ("disabled" if st.disabled else "armed"),
                    "fired": st.fired,
                    "strikes": st.strikes,
                    "last_tick": st.last_tick,
                    "last_message": (st.last_result.message if st.last_result is not None else ""),
                    "disabled_reason": st.disabled_reason,
                }
            )
        return out

    def test(self, spec_dict: dict) -> dict:
        """One dry-run evaluation of a not-necessarily-persisted spec."""
        spec = WatchSpec.from_dict(spec_dict)
        if spec.kind == "template":
            errors = _tpl.validate_template_params(spec.template_name, spec.params, self._keys())
            if errors:
                raise WatchError(
                    f"watch test: template '{spec.template}' invalid:\n"
                    + "\n".join(f"  - {e}" for e in errors),
                    self._keys(),
                )
        elif spec.kind == "expr":
            compile_expr(spec.expr)
        else:
            check_callable_path(spec.callable_path)
        st = WatchState()
        if spec.kind == "callable":
            st.instance = self._instantiate(spec)
        if self._last_ctx is None:
            return {
                "ok": True,
                "note": "no tick evaluated yet -- compile " "check only",
                "dry_run": None,
            }
        result = self._evaluate(spec, st, dry=True)
        return {"ok": True, "dry_run": result.as_dict()}

    # ------------------------------------------------------------------ #
    # bridge command entry (cmd 15 "watch")
    # ------------------------------------------------------------------ #
    def handle(self, rid: int, params: dict) -> dict:
        op = params.get("watch_op")
        if op == "add":
            return self.add(params.get("spec") or {})
        if op == "test":
            return self.test(params.get("spec") or {})
        if op == "list":
            return {"ok": True, "watchers": self.list()}
        if op == "clear":
            return self.clear(params.get("name"))
        if op == "update":
            name = params.get("name")
            if not isinstance(name, str) or not name:
                raise WatchError("watch update: 'name' is required")
            return self.update(name, params.get("patch") or {})
        raise WatchError(f"unknown watch op {op!r} (add|list|clear|update|test)")

    # ------------------------------------------------------------------ #
    # per-tick evaluation (the ONLY engine-loop touchpoint)
    # ------------------------------------------------------------------ #
    def on_tick(self, t: int, bundle: dict) -> None:
        env = bundle.get("env")
        self._last_bundle = bundle
        self._last_ctx = WatchContext(
            t=int(t),
            bundle=bundle,
            env=env if env is not None else self.env,
            adapter=self.adapter,
            sim_time=self._sim_time(bundle),
            episodes=self._episodes(),
        )
        t0 = time.perf_counter()
        for name in list(self._specs):
            spec, st = self._specs[name], self._states[name]
            if st.disabled:
                continue
            if int(t) % max(1, spec.every or 1) != 0:
                continue
            if spec.guard and not self._eval_guard(spec, st):
                continue
            try:
                result = self._evaluate(spec, st, dry=False)
            except Exception as e:  # noqa: BLE001
                self._fail_safe(name, spec, e, t)
                continue
            st.last_result = result
            st.last_tick = int(t)
            st.strikes = 0
            if self._trigger(spec, st, bool(result.fired), int(t)):
                self._fire(spec, st, result, int(t))
        self.last_overhead_us = (time.perf_counter() - t0) * 1e6

    def on_reset(self, env_ids) -> None:
        """Episode boundary (via Bridge.on_episode_end): callable
        on_reset hooks (§8.3); expr/template state is trigger-level."""
        ctx = self._last_ctx
        for name, spec in self._specs.items():
            inst = self._states[name].instance
            if inst is not None and hasattr(inst, "on_reset"):
                try:
                    inst.on_reset(ctx)
                except Exception as e:  # noqa: BLE001
                    self._fail_safe(name, spec, e, ctx.t if ctx else -1)

    def _evaluate(self, spec: WatchSpec, st: WatchState, dry: bool) -> WatchResult:
        ctx = self._last_ctx
        if ctx is None:
            raise WatchError("no tick evaluated yet")
        if spec.kind == "template":
            return _tpl.get_template(spec.template_name)["check"](ctx, spec.params)
        if spec.kind == "expr":
            code, is_exec = self._expr_code.get(spec.name) or compile_expr(spec.expr)
            ns = ctx.eval_namespace()
            if is_exec:
                value = exec_block(code, ns)
            else:
                value = eval(code, ns)  # noqa: S307
            if isinstance(value, WatchResult):
                return value
            return WatchResult.quiet(bool(value))
        inst = st.instance
        if inst is None:
            raise WatchError(f"callable watcher '{spec.name}' is not instantiated")
        if not st.setup_done and hasattr(inst, "setup"):
            inst.setup(ctx)
            st.setup_done = True  # dry-run counts
        res = inst.check(ctx)
        if not isinstance(res, WatchResult):
            raise TypeError(
                f"callable watcher '{spec.name}'.check returned "
                f"{type(res).__name__}, expected WatchResult"
            )
        return res

    def _eval_guard(self, spec: WatchSpec, st: WatchState) -> bool:
        try:
            code, is_exec = self._guard_code.get(spec.name) or compile_expr(spec.guard)
            ctx = self._last_ctx
            if ctx is None:
                return False
            ns = ctx.eval_namespace()
            value = exec_block(code, ns) if is_exec else eval(code, ns)
            return bool(value)
        except Exception:  # noqa: BLE001
            return False  # guard never fires

    # ------------------------------------------------------------------ #
    # trigger layer (§8.5 -- independent of the predicate)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _trigger(spec: WatchSpec, st: WatchState, fired: bool, t: int) -> bool:
        if not fired:
            st.prev = False
            st.hold = 0
            return False
        if spec.mode == "level":
            st.hold += 1
            if st.hold < spec.hold:
                return False
        else:  # edge
            if st.prev:
                return False
        st.prev = True
        if t < st.cooldown_until:
            return False
        st.cooldown_until = t + spec.cooldown
        return True

    def _fire(self, spec: WatchSpec, st: WatchState, result: WatchResult, t: int) -> None:
        st.fired += 1
        actions = result.actions or spec.on
        paused = False
        data = dict(result.data)
        data["message"] = result.message
        if PAUSE in actions and self.bridge is not None:
            self.bridge.pause_for_watcher()
            paused = True
        if CAPTURE in actions:
            data["capture"] = self._capture(spec, result, t)
        if DUMP in actions:
            data["dump"] = self._dump(spec, t)
        if DIRECT in actions and self.bridge is not None:
            data["direct"] = self._direct(spec)
        if st.fired == 1 and spec.once:
            st.disabled = True
            st.disabled_reason = "once: auto-disarmed (add/update rearms)"
        if self.event_log is not None:
            self.event_log.append(
                "watcher",
                spec.name,
                tick=t,
                env=(self._last_ctx.env_id if self._last_ctx is not None else None),
                level=result.level.value,
                data={**data, "actions": list(actions), "paused": paused, "fired_count": st.fired},
            )
        self._persist()

    def _fail_safe(self, name: str, spec: WatchSpec, err: Exception, t: int) -> None:
        st = self._states[name]
        reason = f"{type(err).__name__}: {err}"
        if isinstance(err, WatchError):
            # missing key / data-shaped degrade (§2.2): event but NO
            # strike -- born-armed packs may reference keys that appear
            # mid-run; rate-limit the event spam (1 / 100 ticks / watch)
            if t - st.last_degrade_tick >= 100:
                st.last_degrade_tick = t
                if self.event_log is not None:
                    self.event_log.append(
                        "watcher_degrade",
                        name,
                        tick=t,
                        level=Level.WARN.value,
                        data={"error": reason},
                    )
                print(f"[watcher] {name} degraded (no key yet?): {reason}", flush=True)
            return
        st.strikes += 1
        if st.strikes >= STRIKES_TO_DISABLE:
            st.disabled = True
            st.disabled_reason = (
                f"fail-safe: {STRIKES_TO_DISABLE} " f"consecutive failures (last: {reason})"
            )
        if self.event_log is not None:
            self.event_log.append(
                "watcher_error",
                name,
                tick=t,
                level=Level.ERROR.value,
                data={"error": reason, "strikes": st.strikes, "disabled": st.disabled},
            )
        print(
            f"[watcher] {name} evaluation failed "
            f"({st.strikes}/{STRIKES_TO_DISABLE}): {reason}"
            + (" -> DISABLED" if st.disabled else ""),
            flush=True,
        )
        self._persist()

    # ------------------------------------------------------------------ #
    # actions (all reuse existing pieces; budgeted, §2.5)
    # ------------------------------------------------------------------ #
    def _capture(self, spec: WatchSpec, result: WatchResult, t: int) -> dict:
        if self.out_dir is None:
            return {"skipped": "no output dir"}
        if self.captures >= _CAPTURE_BUDGET:
            return {
                "skipped": f"capture budget {_CAPTURE_BUDGET} "
                "exhausted (degraded to mark+notify)"
            }
        cm = getattr(self.bridge, "camera_manager", None) if self.bridge is not None else None
        if cm is None or not getattr(cm, "names", None):
            return {"skipped": "no cameras in this run"}
        from ..runtime.bridge import _encode_jpeg

        cam = getattr(self.bridge, "camera", None) or cm.names[0]
        frame = cm.get_view(cam, None)
        jpg = _encode_jpeg(frame)
        out = self.out_dir / "watch" / "capture"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"t{int(t):07d}_{spec.name}_{cam}.jpg"
        path.write_bytes(jpg)
        sidecar = path.with_suffix(path.suffix + ".json")
        sidecar.write_text(
            json.dumps(
                {
                    "source": "watcher",
                    "watcher": spec.name,
                    "level": result.level.value,
                    "message": result.message,
                    "data": result.data,
                    "tick": int(t),
                    "camera": cam,
                    "ts_wall": now_wall(),
                    "bytes": len(jpg),
                },
                default=str,
            )
        )
        self.captures += 1
        return {"path": str(path), "camera": cam}

    def _dump(self, spec: WatchSpec, t: int) -> dict:
        """One npz on the trace key face (obs/... extra/... + tick)."""
        if self.out_dir is None:
            return {"skipped": "no output dir"}
        if self.dumps >= _DUMP_BUDGET:
            return {"skipped": f"dump budget {_DUMP_BUDGET} exhausted"}
        import numpy as np
        from .base import _to_numpy

        ctx = self._last_ctx
        if ctx is None:
            return {"skipped": "no tick evaluated yet"}
        arrays: dict = {"tick": np.asarray(int(t))}
        for prefix, src in (("obs/", ctx.obs_cache), ("extra/", ctx.extra)):
            for k in sorted(src):
                try:
                    arrays[prefix + k] = _to_numpy(src[k])
                except Exception:  # noqa: BLE001
                    pass
        out = self.out_dir / "watch" / "dump"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"t{int(t):07d}_{spec.name}.npz"
        np.savez(path, **arrays)
        self.dumps += 1
        return {"path": str(path), "keys": sorted(arrays)}

    def _direct(self, spec: WatchSpec) -> dict:
        from ..runtime.bridge import CMD_DIRECT

        handler = getattr(self.bridge, "_handlers", {}) if self.bridge is not None else {}
        fn = handler.get(CMD_DIRECT)
        if fn is None:
            return {"skipped": "director not mounted (--direct)"}
        payload = dict(spec.params.get("direct") or {})
        payload.setdefault("op", spec.params.get("direct_op", "status"))
        try:
            res = fn(0, payload)
            return {"result": res}
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------------ #
    # status / persistence (file is truth)
    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        states = list(self._states.values())
        fired = sum(s.fired for s in states)
        disabled = [n for n, s in self._states.items() if s.disabled]
        last = None
        fired_names = [
            (self._states[n].last_tick or -1, n) for n in self._specs if self._states[n].fired
        ]
        if fired_names:
            _, name = max(fired_names)
            st = self._states[name]
            last = {
                "name": name,
                "tick": st.last_tick,
                "level": (st.last_result.level.value if st.last_result is not None else None),
                "message": (st.last_result.message if st.last_result is not None else ""),
            }
        return {
            "armed": len(self._specs) - len(disabled),
            "total": len(self._specs),
            "disabled": disabled,
            "fired_total": fired,
            "captures": self.captures,
            "dumps": self.dumps,
            "overhead_us": round(self.last_overhead_us, 1),
            "last": last,
            "badges": [f"watcher disabled: {n}" for n in disabled],
        }

    def status_brief(self) -> dict:
        """The per-tick status-block face (small; the notify surface --
        the ONLY watcher face cross-process readers see)."""
        st = self.status()
        return {
            "armed": st["armed"],
            "disabled": st["disabled"],
            "fired_total": st["fired_total"],
            "captures": st["captures"],
            "dumps": st["dumps"],
            "overhead_us": st["overhead_us"],
            "last": st["last"],
        }

    def _persist(self) -> None:
        if self.out_dir is None:
            return
        doc = {
            "schema": 1,
            "run_id": self.run_id,
            "updated": now_wall(),
            "watchers": [s.to_dict() for s in (self._specs[n] for n in sorted(self._specs))],
            "runtime": {
                n: {
                    "fired": self._states[n].fired,
                    "disabled": self._states[n].disabled,
                    "disabled_reason": self._states[n].disabled_reason,
                    "strikes": self._states[n].strikes,
                }
                for n in sorted(self._specs)
            },
        }
        path = self.out_dir / "watchers.json"
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, indent=1, default=str))
            tmp.replace(path)
        except OSError as e:  # best effort (additive)
            print(f"[watcher] persist failed: {e}", flush=True)

    # ------------------------------------------------------------------ #
    def _instantiate(self, spec: WatchSpec) -> Any:
        return load_callable(spec.callable_target)()

    def _episodes(self) -> int:
        return int(getattr(self.bridge, "_episodes", 0) or 0) if self.bridge is not None else 0

    @staticmethod
    def _sim_time(bundle: dict) -> float:
        tm = bundle.get("term_metrics")
        return float(tm.get("sim_time", 0.0)) if isinstance(tm, dict) else 0.0
