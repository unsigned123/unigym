"""watchers.base -- typed data face for programmable observers.

Design: docs/designs/watchers.md §8.  Three pieces live
here:

* :class:`WatchResult` -- the behavior interface.  A watcher decides
  "what happened and what to suggest" (fired / level / message / data /
  actions); the framework decides "how to execute safely" (PAUSE ->
  StepGate, CAPTURE -> jpeg+sidecar, DUMP -> npz, MARK -> events.jsonl,
  NOTIFY -> status surface).  Static ``on:`` is the default action set;
  a non-empty ``WatchResult.actions`` overrides for that one fire.
* :class:`WatchContext` -- the information interface handed to every
  predicate evaluation.  Its KEY FACE is the trace contract key face
  (docs/contracts/trace_contract.md §2): ``obs/<key>``, ``extra/<key>``,
  ``action/<name>``, ``action/raw`` -- recorded keys, watcher-reference
  keys and candidate-listing keys are ONE table (§8.4).  Convenience
  accessors (pose/dist/contact/action) reuse exactly that face.
* :class:`WatchError` -- add-time rejection with candidate discovery
  (contacts-discovery UX: a bad key fails fast and lists what IS
  available).

Stdlib + lazy numpy/torch only; engine-thread discipline like the
console (exceptions never escape into the engine loop -- fail-safe in
engine.py).
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Level",
    "ACTIONS",
    "PAUSE",
    "CAPTURE",
    "DUMP",
    "MARK",
    "NOTIFY",
    "DIRECT",
    "WatchResult",
    "WatchContext",
    "WatchError",
    "DEFAULT_ON",
]


class Level(str, enum.Enum):
    """Trigger severity (WatchResult.level; events.jsonl carries it)."""

    INFO = "info"
    WARN = "warn"
    ERROR = "error"


# trigger action face (design §2.3 -- all reuse existing pieces)
PAUSE = "pause"  # StepGate.pause() -- parks the engine THIS tick
CAPTURE = "capture"  # broadcast-camera jpeg + sidecar (watcher/tick/value)
DUMP = "dump"  # obs+extra npz (save_npz key face) at <out>/watch/dump
MARK = "mark"  # events.jsonl watcher event (the timeline)
NOTIFY = "notify"  # status surface (armed/fired/last + webui badge)
DIRECT = "direct"  # P2: director op via the registered handler

ACTIONS = (PAUSE, CAPTURE, DUMP, MARK, NOTIFY, DIRECT)
DEFAULT_ON = (MARK, NOTIFY)

_CAPTURE_BUDGET = 64  # per run (disk-full history, design §2.5)
_DUMP_BUDGET = 16


class WatchError(ValueError):
    """A watch declaration was rejected (fail-fast + candidate keys).

    Raised by add/update/test before anything persists; the message
    lists the available trace-face keys when a key reference failed --
    same UX as contacts/discovery.py.
    """

    def __init__(self, message: str, discovered: list[str] | None = None):
        super().__init__(message)
        self.discovered = list(discovered or [])


@dataclass
class WatchResult:
    """What a watcher says for one evaluation (design §8.2).

    fired:    the boolean fact (feeds the trigger layer: edge/level/
              once/cooldown -- design §8.5).
    level:    INFO | WARN | ERROR (events + badges).
    message:  one human/agent-readable line.
    data:     json-safe dict (the evidence: values, env indexes...).
    actions:  non-empty -> overrides the spec's static ``on`` for THIS
              fire (declarative default + dynamic override).
    """

    fired: bool
    level: Level = Level.INFO
    message: str = ""
    data: dict = field(default_factory=dict)
    actions: tuple = ()

    @classmethod
    def info(cls, message: str, data: dict | None = None) -> WatchResult:
        return cls(fired=True, level=Level.INFO, message=message, data=data or {})

    @classmethod
    def warn(cls, message: str, data: dict | None = None) -> WatchResult:
        return cls(fired=True, level=Level.WARN, message=message, data=data or {})

    @classmethod
    def trigger(cls, message: str, actions: tuple = (), data: dict | None = None) -> WatchResult:
        return cls(
            fired=True, level=Level.WARN, message=message, data=data or {}, actions=tuple(actions)
        )

    @classmethod
    def quiet(cls, value: bool = False) -> WatchResult:
        """No-fire result (the plainFalse path of expr predicates)."""
        return cls(fired=bool(value))

    def as_dict(self) -> dict:
        return {
            "fired": self.fired,
            "level": self.level.value,
            "message": self.message,
            "data": self.data,
            "actions": list(self.actions),
        }


import operator as _op

_OP_TABLE = {
    "gt": _op.gt,
    "lt": _op.lt,
    "gte": _op.ge,
    "lte": _op.le,
    "eq": _op.eq,
    "abs_gt": _op.gt,
    "abs_lt": _op.lt,
    "abs_gte": _op.ge,
    "abs_lte": _op.le,
}
_REDUCES = ("index", "abs", "last", "min", "max", "abs_max", "norm", "mean")


def _row_scalar(row, name, dim, reduce):
    """One env-row -> float per the dim/reduce rule (hot path; raises
    WatchError with the key name on shape misuse)."""
    if dim is not None:
        if row.ndim == 0 or not (-row.shape[-1] <= int(dim) < row.shape[-1]):
            raise WatchError(
                f"{name}: dim {dim} out of range (trailing axis size "
                f"{row.shape[-1] if row.ndim else 0})"
            )
        row = row[int(dim)]
    if reduce == "index":
        if row.ndim != 0:
            raise WatchError(f"{name}: reduce=index needs dim= for a vector key")
        return float(row)
    if reduce == "abs":
        if row.ndim != 0:
            raise WatchError(f"{name}: reduce=abs needs dim= for a vector key")
        return abs(float(row))
    if reduce == "last":
        return float(row[-1])
    import numpy as np

    flat = row.reshape(-1).astype(np.float64)
    if flat.size == 0:
        return None
    if reduce == "min":
        return float(flat.min())
    if reduce == "max":
        return float(flat.max())
    if reduce == "abs_max":
        return float(np.abs(flat).max())
    if reduce == "norm":
        return float(np.sqrt((flat * flat).sum()))
    if reduce == "mean":
        return float(flat.mean())
    raise WatchError(f"unknown reduce {reduce!r} " f"(index|abs|last|min|max|abs_max|norm|mean)")


def _to_numpy(val):
    import numpy as np
    import torch

    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    return np.asarray(val)


@dataclass
class WatchContext:
    """One evaluation's information interface (design §8.4).

    Built per tick by the engine; handed to template checks, bound into
    the expr namespace and passed to callable ``check(ctx)``.  All
    accessors read the single data face (bundle + obs_cache); nothing
    here mutates run state.
    """

    t: int
    bundle: dict
    env: Any = None
    adapter: Any = None
    sim_time: float = 0.0
    episodes: int = 0
    env_id: int | None = None  # scalar-eval context (expr tier)
    _np_cache: dict = field(default_factory=dict)  # per-tick arr memo

    @property
    def step(self) -> int:
        return self.t

    @property
    def extra(self) -> dict:
        ex = self.bundle.get("extra")
        return ex if isinstance(ex, dict) else {}

    @property
    def obs_cache(self) -> dict:
        src = getattr(self.env, "obs_cache", None)
        return src if isinstance(src, dict) else {}

    @property
    def num_envs(self) -> int:
        return int(getattr(self.env, "num_envs", 0) or 0)

    # ------------------------------------------------------------------ #
    # the key face (trace contract §2 -- ONE table)
    # ------------------------------------------------------------------ #
    def keys(self) -> list[str]:
        """Available trace-face keys (candidate-discovery listing)."""
        out = [f"obs/{k}" for k in self.obs_cache]
        out += [f"extra/{k}" for k in self.extra]
        targets = self.bundle.get("targets")
        if isinstance(targets, dict):
            out += [f"action/{k}" for k in targets]
        if "raw" in self.bundle:
            out.append("action/raw")
        return sorted(set(out))

    def has_key(self, name: str) -> bool:
        return name in self.keys()

    def _arr(self, name: str):
        """Key -> numpy view, memoized per tick (many watchers read the
        same key; the tensor->numpy conversion is paid once)."""
        cache = getattr(self, "_np_cache", None)
        if cache is None:
            cache = self._np_cache = {}
        if name not in cache:
            cache[name] = _to_numpy(self.key(name))
        return cache[name]

    def key(self, name: str):
        """Resolve one trace-face key -> the underlying array value."""
        if name.startswith("obs/"):
            src = self.obs_cache
            k = name[4:]
        elif name.startswith("extra/"):
            src = self.extra
            k = name[6:]
        elif name == "action/raw":
            return self.bundle.get("raw")
        elif name.startswith("action/"):
            targets = self.bundle.get("targets")
            na = targets.get(name[7:]) if isinstance(targets, dict) else None
            if na is None:
                raise WatchError(f"unknown key {name!r} (available: {self.keys()})", self.keys())
            return getattr(na, "values", na)
        else:
            raise WatchError(
                f"key {name!r} must use the trace face prefix "
                f"obs/ | extra/ | action/ (available: {self.keys()})",
                self.keys(),
            )
        if k not in src:
            raise WatchError(f"unknown key {name!r} (available: {self.keys()})", self.keys())
        return src[k]

    # ------------------------------------------------------------------ #
    # scalar reduction (template tier)
    # ------------------------------------------------------------------ #
    def scalar(
        self, name: str, env: int | None = None, dim: int | None = None, reduce: str = "index"
    ) -> tuple[float | None, int | None]:
        """Trace-face key -> (value, env_index) per the reduction rule.

        env=None scans every env: returns (value, first env that makes
        the comparison interesting) -- for raw data that is simply env 0
        (deterministic); comparators that scan use :meth:`compare`.
        ``reduce``: index (pick ``dim``), abs (at ``dim``),
        min/max/abs_max/norm over the trailing axis, last (final col).
        """
        a = self._arr(name)
        env_i = 0 if env is None else int(env)
        if a.ndim == 0:
            return float(a), (None if env is None else env_i)
        if env_i >= a.shape[0]:
            raise WatchError(f"{name}: env {env_i} out of range (num_envs={a.shape[0]})")
        row = a[env_i]
        if dim is not None:
            # dim indexes the TRAILING axis (pos[..., 2] = z of an
            # (N, 3) key -- the natural caller semantics)
            if row.ndim == 0 or not (-row.shape[-1] <= int(dim) < row.shape[-1]):
                raise WatchError(
                    f"{name}: dim {dim} out of range (trailing axis size "
                    f"{row.shape[-1] if row.ndim else 0})"
                )
            row = row[int(dim)] if row.ndim == 1 else row[..., int(dim)]
        if reduce in ("index", "abs", "last"):
            if row.ndim != 0:
                if reduce == "last":
                    row = row[-1]
                else:
                    raise WatchError(
                        f"{name}: reduce={reduce} needs dim= for a vector " f"key (shape {a.shape})"
                    )
            v = abs(float(row)) if reduce == "abs" else float(row)
        else:
            import numpy as np

            flat = row.reshape(-1).astype(np.float64)
            if flat.size == 0:
                return None, env_i
            if reduce == "min":
                v = float(flat.min())
            elif reduce == "max":
                v = float(flat.max())
            elif reduce == "abs_max":
                v = float(np.abs(flat).max())
            elif reduce == "norm":
                v = float(np.sqrt((flat * flat).sum()))
            elif reduce == "mean":
                v = float(flat.mean())
            else:
                raise WatchError(
                    f"unknown reduce {reduce!r} " f"(index|abs|last|min|max|abs_max|norm|mean)"
                )
        return v, (None if env is None else env_i)

    def compare(
        self,
        name: str,
        op: str,
        value: float,
        env: int | None = None,
        dim: int | None = None,
        reduce: str = "index",
    ) -> tuple[bool, float | None, int | None]:
        """Comparator scan across envs (env=None) or one env.

        Returns (fired, value, env_index).  abs_* ops compare |value|.
        Hot path: called per (watcher, evaluated tick) -- the operator
        table and flat row extraction keep this ~1µs.
        """
        fn = _OP_TABLE.get(op)
        if fn is None:
            raise WatchError(
                f"unknown comparator {op!r} (gt|lt|gte|lte|eq|abs_gt|" "abs_lt|abs_gte|abs_lte)"
            )
        a = self._arr(name)
        if a.ndim == 0:
            idx = (0,) if env is None else (int(env),)
        elif env is None:
            idx = range(a.shape[0])
        else:
            env = int(env)
            if env >= a.shape[0]:
                raise WatchError(f"{name}: env {env} out of range " f"(num_envs={a.shape[0]})")
            idx = (env,)
        neg = op.startswith("abs_")
        for i in idx:
            v = _row_scalar(a[i] if a.ndim else a, name, dim, reduce)
            if v is None:
                continue
            x = abs(v) if neg else v
            if fn(x, value):
                return True, v, i
        return False, None, (0 if env is None else int(env))

    # ------------------------------------------------------------------ #
    # convenience accessors (reuse the same data face)
    # ------------------------------------------------------------------ #
    def pose(self, entity: str, env: int = 0):
        """World position (3,) float of 'base'|'pelvis'|'robot'|<prop>."""
        name = {
            "base": "extra/base_pos_w",
            "pelvis": "extra/base_pos_w",
            "robot": "extra/base_pos_w",
        }.get(entity, f"extra/object/{entity}/pos")
        import numpy as np

        return np.asarray(self._arr(name)[int(env)], dtype=float)

    def dist(self, a: str | list, b: str | list, env: int = 0) -> float:
        """Euclidean distance between two entities / points."""
        import numpy as np

        pa = self.pose(a, env) if isinstance(a, str) else _as_point(a)
        pb = self.pose(b, env) if isinstance(b, str) else _as_point(b)
        return float(np.linalg.norm(pa - pb))

    def contact(self, group: str, env: int | None = None) -> bool:
        """Contact-group truth (extra/contact/<group>; any env or one)."""
        a = self._arr(f"extra/contact/{group}")
        if env is None:
            return bool(a.any())
        return bool(a[int(env)])

    def action(self, entry: str, env: int = 0):
        """NamedAction values row (N, dim) -> (dim,) for one env."""
        return self._arr(f"action/{entry}")[int(env)]

    # ------------------------------------------------------------------ #
    # expr tier namespace (console eval pass-through semantics)
    # ------------------------------------------------------------------ #
    def eval_namespace(self) -> dict:
        """The documented expr binding (watchers_contract §4).

        Console-eval names (np torch obs_cache extra bundle adapter env)
        plus watcher conveniences (t/step/sim_time/episode/env_id and
        key/action/contact/pose/dist).  Full builtins -- same
        pass-through ruling as the inspector.
        """
        import numpy as np

        try:
            import torch
        except ImportError:  # pragma: no cover
            torch = None
        ns = {
            "np": np,
            "torch": torch,
            "obs_cache": self.obs_cache,
            "extra": self.extra,
            "bundle": self.bundle,
            "adapter": self.adapter,
            "env": self.env,
            "t": self.t,
            "step": self.t,
            "sim_time": self.sim_time,
            "episode": self.episodes,
            "env_id": self.env_id,
            "key": self.key,
            "action": self.action,
            "contact": self.contact,
            "pose": self.pose,
            "dist": self.dist,
            "WatchResult": WatchResult,  # expr may speak §8.2 too
        }
        return ns


def _as_point(p) -> Any:
    import numpy as np

    v = np.asarray(p, dtype=float).reshape(-1)
    if v.size != 3:
        raise WatchError(f"point must be [x,y,z], got {p!r}")
    return v


def now_wall() -> float:
    return time.time()


def finite(x) -> float | None:
    """JSON-safe float (None/non-finite -> None, 6 decimals)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, 6)
