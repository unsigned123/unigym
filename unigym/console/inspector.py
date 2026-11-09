"""Inspector: fixed queries + pass-through eval + npz dump.

eval is DIRECT -- no restricted
namespace, no builtin whitelist.  The original sandbox-ish namespace was a
foot-guard, not a security sandbox, and in practice it rejected most real
diagnostics (assignments, imports, multi-line snippets, ``print``),
which cost far more than it protected.  Threat model: campus intranet +
token-gated write face + runs that live minutes -- there is nothing to
defend against, so eval/exec run with FULL builtins against the injected
run names (``np torch obs_cache extra bundle adapter env``); a trailing
expression's value is returned.  The inspector can therefore mutate run
state (assignments, method calls) -- that is a documented capability now,
not an accident; exceptions still never escape (ConsoleError -> ack).

Response formatting is a deterministic rule (explicit
thresholds, no fuzzy size judgment):

* inline summary ALWAYS: ``key / shape / dtype / mean / std / first8``
  (first 8 flattened values; mean/std over all elements, ``None`` when
  non-finite, floats rounded to 4 decimals for JSON-safety);
* ``numel <= 64``  -> append ``values``: the full flattened data inline;
* ``numel > 64``   -> dump ``data_out/<run>/inspect/req_<rid>.npz``
  (original key names) and return ``npz_path``;
* eval results: ndarray / tensor / dict-of-arrays -> the same rule;
  scalars / short strings / small scalar lists -> ``inline``.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import torch

from .stepping import ConsoleError

__all__ = ["Inspector", "INLINE_NUMEL", "PREVIEW_N", "MAX_STR",
           "EVAL_NAMESPACE"]

INLINE_NUMEL = 64       # numel <= 64 -> full inline values; else npz dump
PREVIEW_N = 8           # the always-present first-8 preview
MAX_STR = 200           # inline string / repr truncation

EVAL_NAMESPACE = ("np", "torch", "obs_cache", "extra", "bundle", "adapter",
                  "env")
"""Names injected into every eval (docs/contracts/console_contract.md §4) --
convenience injections on top of full builtins, NOT a restriction; the
tests lock this tuple; add names only together with a contract update."""


def _num(x) -> float | None:
    """JSON-safe float (non-finite -> None, rounded to 4 decimals)."""
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, 4)


def _jsonable(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return _jsonable(v.item())
    return v


class Inspector:
    """Runtime inspector (engine-thread query face).

    eval is DIRECT (module docstring): statements,
    assignments, imports and arbitrary attribute access all execute
    against the live run objects; the last expression's value is
    returned.  ``env``/``adapter`` may be None (``obs`` queries /
    ``obs_cache`` in eval degrade to an explicit ConsoleError / empty
    mapping).

    Args:
        env: EnvBackend or None.
        adapter: Adapter or None (eval namespace entry).
        out_dir: npz dump root (``<out_dir>/inspect/req_<rid>.npz``);
            ``None`` falls back to ``./data_out`` (documented default --
            real runs always pass the collect output dir).
        status_fn: zero-arg callable -> bridge status dict (summary).
        bundle_fn: zero-arg callable -> the current step bundle dict.
    """

    def __init__(self, env=None, adapter=None, out_dir=None,
                 status_fn=None, bundle_fn=None):
        self.env = env
        self.adapter = adapter
        self.out_dir = out_dir
        self._status_fn = status_fn
        self._bundle_fn = bundle_fn

    # ------------------------------------------------------------------ #
    # fixed query set
    # ------------------------------------------------------------------ #
    def handle(self, rid: int, params: dict) -> dict:
        """Dispatch one inspect payload ({target, key?} | {target: eval})."""
        target = params.get("target")
        if target == "eval":
            return self.eval_expr(params.get("expr"))
        if target == "summary":
            return self.summary()
        if target in ("obs", "extra"):
            key = params.get("key")
            if not isinstance(key, str):
                raise ConsoleError(
                    f"inspect: {target} key must be a string, got {key!r}")
            source = self._source(target)
            if key not in source:
                raise ConsoleError(
                    f"inspect: unknown {target} key {key!r} "
                    f"(available: {sorted(source)})")
            return self.format_array(key, source[key], rid=rid)
        raise ConsoleError(f"inspect: unknown target {target!r}")

    def summary(self) -> dict:
        """phase/step/episodes + obs/extra key inventory (read-only)."""
        out = {
            "num_envs": int(getattr(self.env, "num_envs", 0) or 0),
            "obs_keys": sorted(self._source("obs")),
            "extra_keys": sorted(self._source("extra")),
        }
        if self._status_fn is not None:
            try:
                st = self._status_fn()
            except Exception:            # noqa: BLE001 -- summary never fails
                st = {}
            for k in ("phase", "step", "episodes"):
                out[k] = st.get(k)
        return out

    def _bundle(self) -> dict:
        """Current bundle snapshot (empty when no provider is wired)."""
        return self._bundle_fn() if self._bundle_fn is not None else {}

    def _source(self, which: str) -> dict:
        if which == "obs":
            src = getattr(self.env, "obs_cache", None)
            return src if isinstance(src, dict) else {}
        bundle = self._bundle() or {}
        src = bundle.get("extra")
        return src if isinstance(src, dict) else {}

    # ------------------------------------------------------------------ #
    # formatting (the deterministic rule)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_numpy(val) -> np.ndarray:
        if isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy()
        return np.asarray(val)

    def format_array(self, key: str, val, rid: int | None = None) -> dict:
        """One array -> inline summary (+ values | npz_path) per the rule."""
        a = self._to_numpy(val)
        flat = a.reshape(-1)
        out: dict = {"key": key, "shape": [int(x) for x in a.shape],
                     "dtype": str(a.dtype)}
        if flat.size:
            f64 = flat.astype(np.float64)
            out["mean"] = _num(f64.mean())
            out["std"] = _num(f64.std())     # population std (ddof=0)
        out["first8"] = [_num(x) for x in flat[:PREVIEW_N]]
        if flat.size and flat.size <= INLINE_NUMEL:
            out["values"] = [_num(x) for x in flat]
        elif flat.size > INLINE_NUMEL:
            out["npz_path"] = str(self._dump_npz({key: a}, rid=rid))
        return out

    # ------------------------------------------------------------------ #
    # pass-through eval
    # ------------------------------------------------------------------ #
    def eval_expr(self, expr) -> dict:
        """Direct eval/exec of *expr* in the run namespace.

        Full builtins, arbitrary attribute access, imports, assignments,
        multi-line statements -- everything runs.  A trailing expression's
        value is captured and returned (AST: the last ``Expr`` statement
        is rewritten into an assignment to a reserved name).  Errors
        never escape: everything (including SyntaxError) is wrapped into
        :class:`ConsoleError` -> ack, the engine loop lives.
        """
        if not isinstance(expr, str) or not expr.strip():
            raise ConsoleError("eval: 'expr' (non-empty string) is required")
        ns = self._namespace()
        try:
            try:
                result = eval(compile(expr, "<console>", "eval"), ns)  # noqa: S307
            except SyntaxError:
                result = self._exec_block(expr, ns)
        except ConsoleError:
            raise
        except Exception as e:               # noqa: BLE001 -> ack, run lives
            raise ConsoleError(
                f"eval failed: {type(e).__name__}: {e}") from None
        return self.format_result(result)

    @staticmethod
    def _exec_block(expr: str, ns: dict):
        """Statement / multi-line path (direct exec).

        The trailing expression statement's value is returned so
        ``x = np.arange(4)`` then ``x.sum()``-style diagnostics are not
        swallowed as ``None``.  No trailing expression -> ``None``.
        """
        tree = ast.parse(expr, "<console>")
        ret = "__console_ret__"
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            tree.body[-1] = ast.Assign(
                targets=[ast.Name(id=ret, ctx=ast.Store())],
                value=tree.body[-1].value)
            ast.fix_missing_locations(tree)
        exec(compile(tree, "<console>", "exec"), ns)    # noqa: S102
        return ns.pop(ret, None)

    def _namespace(self) -> dict:
        bundle = self._bundle() or {}
        return {
            "np": np,
            "torch": torch,
            "obs_cache": self._source("obs"),
            "extra": self._source("extra"),
            "bundle": bundle,
            "adapter": self.adapter,
            "env": self.env,
        }

    def format_result(self, result) -> dict:
        """eval result -> inline / array rule (contract §formatting rules)."""
        if isinstance(result, (torch.Tensor, np.ndarray)):
            return self.format_array("result", result)
        if isinstance(result, dict):
            return {"dict": {str(k): self.format_result(v)
                             for k, v in result.items()}}
        if isinstance(result, (list, tuple)):
            vals = list(result)
            if len(vals) <= INLINE_NUMEL and all(
                    v is None or isinstance(v, (int, float, bool, str))
                    for v in vals):
                return {"inline": [_jsonable(v) for v in vals]}
            return {"inline": repr(result)[:MAX_STR]}
        if result is None or isinstance(result, (int, float, bool,
                                                 np.integer, np.floating,
                                                 np.bool_)):
            return {"inline": _jsonable(result)}
        if isinstance(result, str):
            return {"inline": result[:MAX_STR]}
        return {"inline": repr(result)[:MAX_STR]}

    # ------------------------------------------------------------------ #
    # npz dump
    # ------------------------------------------------------------------ #
    def save_npz(self, rid: int, params: dict) -> dict:
        """Dump obs/extra arrays to one npz (original key names)."""
        source = params.get("source", "all")
        keys = params.get("keys")
        sources: dict[str, dict] = {}
        if source in ("obs", "all"):
            sources["obs"] = self._source("obs")
        if source in ("extra", "all"):
            sources["extra"] = self._source("extra")
        arrays: dict[str, np.ndarray] = {}
        skipped: list[str] = []
        for src in sources.values():
            for k, v in src.items():
                if keys is not None and k not in keys:
                    continue
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    arrays[k] = self._to_numpy(v)
                else:
                    skipped.append(k)
        if keys is not None:
            available = sorted({k for src in sources.values()
                                for k in src})
            missing = sorted(set(keys) - set(arrays) - set(skipped))
            if missing:
                raise ConsoleError(
                    f"save_npz: unknown key(s) {missing} "
                    f"(available: {available})")
        if not arrays:
            raise ConsoleError(
                "save_npz: nothing to dump (no array keys matched)")
        target = self._dump_npz(arrays, rid, params.get("path"))
        return {"npz_path": str(target), "keys": sorted(arrays),
                "skipped": sorted(set(skipped))}

    def _dump_npz(self, arrays: dict, rid: int | None,
                  path: str | None = None) -> Path:
        target = self._resolve_path(path, rid)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, **arrays)
        return target

    def _resolve_path(self, path, rid: int | None) -> Path:
        root = Path(self.out_dir) if self.out_dir \
            else Path("data_out")
        if path:
            p = Path(path)
            target = p if p.is_absolute() else root / p
        else:
            target = root / "inspect" / f"req_{rid}.npz"
        if target.suffix != ".npz":          # np.savez would append .npz
            target = target.with_name(target.name + ".npz")
        return target
