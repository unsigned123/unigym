"""Benchmark task protocol (B1): live-first episode scorers.

A benchmark run evaluates completed episodes instead of (or alongside)
recording them.  The data path is:

    SyncRunner --callbacks--> EvaluationSession --EpisodeData + EvalContext-->
        Task.score --EpisodeResult--> Task.aggregate --> benchmark_report.json

Contract (docs/contracts/benchmark_contract.md):

* two-phase scoring: ``score(ep, ctx) -> EpisodeResult`` per episode and
  ``aggregate(results) -> dict`` at run end;
* ``ep`` is the episode history/timeline/metrics payload; ``ctx`` is the
  read-only, per-env live context created by ``EvaluationSession`` before reset.
  Tasks may read ``ctx.env.record()``, ``ctx.env.obs()`` and (when the
  backend exposes it) ``ctx.env.query``; they cannot mutate, step, reset or
  read another env.  The context is closed after all tasks for that env.
* ``Task(params, meta)`` binds all configuration at construction time;
* ``success`` (task-judged quality) is recorded separately from
  ``ended_by_termination`` (engine-observed early termination) -- early
  termination is NOT automatically a failure, each task decides;
* ``EpisodeData`` arrays are plain numpy (T, .) in *recorded key names*
  (the shard npz names, e.g. ``base_pos``/``gravity_b``), so pure tasks stay
  unit-testable with synthetic episodes;
* there is no offline re-scoring guarantee: reports are live run
  observations, not reconstructible artifacts.

This module remains importable without torch/isaaclab (the context type is a
type-checking-only import).
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .termination import EpisodeTermination, Termination

if TYPE_CHECKING:
    from unigym.evaluation import EvalContext
    from unigym.packs.runtime import PackRuntime


#: Key prefixes that stay GLOBAL in the recorded data (objects, contact
#: groups, cameras, sites, monitor namespaces).  A robot-state leaf nested
#: under one of these is still a global key.
GLOBAL_KEY_PREFIXES = (
    "object/",
    "contact/",
    "cam/",
    "site/",
    "contact_ops/",
    "action_target/",
    "nav/",
    "joints/",
)

#: Bare global keys (episode bookkeeping + adapter raw frames).
GLOBAL_BARE_KEYS = frozenset(
    {
        "time_in_episode",
        "episode_id",
        "action_raw",
        "in_recovery",
        "language",
    }
)

#: Per-articulation robot-state leaf names.  v103.5 records every robot
#: state column under ``articulation/<name>/<leaf>``; both the recorded
#: leaf names (``base_pos``) and the pre-rename backend names
#: (``base_pos_w``) resolve to the same namespace.
ROBOT_STATE_LEAVES = frozenset(
    {
        "joint_pos",
        "joint_vel",
        "applied_torque",
        "gravity_b",
        "base_pos",
        "base_pos_w",
        "base_quat_w",
        "base_lin_vel",
        "base_lin_vel_w",
        "base_lin_vel_b",
        "base_ang_vel",
        "base_ang_vel_w",
        "base_ang_vel_b",
        "command",
    }
)

#: Fields whose string values name a recorded key (used to detect a task's
#: robot-state reads at construction time).
_KEY_FIELD_NAMES = ("key", "key2", "state", "cmd", "pos", "quat")


def is_robot_state_key(key) -> bool:
    """True when *key* names per-articulation robot state.

    Already-namespaced keys (``articulation/<name>/...``) and keys under a
    global prefix are NOT robot-state requests -- they carry their own
    scope.
    """
    if not isinstance(key, str) or not key:
        return False
    if key.startswith("articulation/"):
        return False
    if any(key.startswith(p) for p in GLOBAL_KEY_PREFIXES):
        return False
    if key in GLOBAL_BARE_KEYS:
        return False
    return key.rsplit("/", 1)[-1] in ROBOT_STATE_LEAVES


def robot_state_key_uses(spec, _path: str = "") -> list[tuple[str, str]]:
    """``[(field_path, key)]`` of every robot-state read declared in *spec*.

    Recursively walks the metric spec (nested ``hold_streak: specs`` /
    ``keypoint_in_region`` included) looking at the recorded-key fields
    only.  Used for the construction-time articulation requirement; an
    empty list means the task reads global/object/contact keys only.
    """
    out: list[tuple[str, str]] = []
    if isinstance(spec, dict):
        for field, value in spec.items():
            path = f"{_path}.{field}" if _path else str(field)
            if field in _KEY_FIELD_NAMES and is_robot_state_key(value):
                out.append((path, value))
            elif isinstance(value, (dict, list, tuple)):
                out.extend(robot_state_key_uses(value, path))
    elif isinstance(spec, (list, tuple)):
        for i, value in enumerate(spec):
            out.extend(robot_state_key_uses(value, f"{_path}[{i}]"))
    return out


def resolve_episode_key(data: dict, key: str, articulation: str | None) -> str:
    """Recorded key for *key* under the task's explicit *articulation*.

    There is no primary binding and no
    "unique articulation" inference for robot-state keys.

    * a key already named ``articulation/<name>/...`` passes through; when
      *articulation* is given it must match the name;
    * global keys (``object/``, ``contact/``, ``cam/``, ``site/``,
      ``contact_ops/``, monitor namespaces, bookkeeping/raw frames) pass
      through unchanged;
    * a robot-state leaf (``base_pos``, ``gravity_b``, ``joint_pos``,
      ``base_lin_vel``, ``command``, ...) resolves to
      ``articulation/<articulation>/<key>`` and REQUIRES a non-None
      *articulation* -- a missing declaration fails fast here.

    ``data`` is accepted so callers pass the episode dict uniformly; the
    resolution itself is a pure key transformation (no data inspection).
    """
    del data  # resolution is a pure key transformation, never data-driven
    if not isinstance(key, str) or not key:
        raise ValueError(f"episode key must be a non-empty string, got {key!r}")
    if key.startswith("articulation/"):
        parts = key.split("/", 2)
        if len(parts) != 3 or not parts[1] or not parts[2]:
            raise ValueError(
                f"malformed namespaced episode key {key!r} (expected "
                "'articulation/<name>/<key>')"
            )
        if articulation is not None and parts[1] != articulation:
            raise ValueError(
                f"episode key {key!r} names articulation {parts[1]!r} but "
                f"the task declares articulation {articulation!r}"
            )
        return key
    if any(key.startswith(p) for p in GLOBAL_KEY_PREFIXES):
        return key
    if key in GLOBAL_BARE_KEYS:
        return key
    if is_robot_state_key(key):
        if articulation is None:
            raise ValueError(
                f"episode key {key!r} is per-articulation robot state; the "
                "task must declare an explicit `articulation: <name>` "
                "(no unique-articulation inference)"
            )
        return f"articulation/{articulation}/{key}"
    return key


@dataclass
class EpisodeData:
    """One completed episode, assembled pre-reset by EvaluationSession.

    Attributes:
        data: per-key numpy arrays with (T, .) leading time axis, keyed by
            the *recorded* key names (shard npz names: ``joint_pos``,
            ``base_pos``, ``gravity_b``, ``base_lin_vel``, ``command``, ...).
        env_id: backend env index the episode ran on.
        episode_id: backend per-env episode counter (increments on reset).
        length: number of policy steps buffered for this episode (T).
        duration_s: wall-clock duration of the episode in seconds (from
            ``extra["time_in_episode"]`` at the final step).
        ended_by_termination: True if the termination group fired on this
            env at the final step (engine bundle ``terminated[env_id]``).
            Purely observational -- see the ``success`` discussion above.
        policy_dt: policy period (s) of the run.
        sim_dt: simulation period (s) of the run.
        step_index: canonical int64 tick axis ``1..T`` (frame 0 is tick 1).
        time_s: authoritative per-frame time ``step_index * policy_dt``.
        reset_index: tick of the reset sample (canonical ``0``).
        reset_sample: selected pre-policy-step observation arrays captured by
            the reset chain (never recorder frame 0).
        reset_valid: initializer legality of the reset, or None when the
            initializer does not judge it.
        reset_reason: initializer-provided reason for an invalid reset.
        terminal_index: canonical tick of the primary termination.
        termination: the structured primary/all termination bundle.
        terminations: the full tuple of terminating reasons for the tick.
    """

    data: dict[str, np.ndarray]
    env_id: int
    episode_id: int
    length: int
    duration_s: float
    ended_by_termination: bool
    rewound: bool = False
    """True when a rewind forked the timeline while this episode was open
    (Round B B1): the buffer keeps ALL forked data (no truncation) and the
    episode is excluded from physics-legal claims — the ``directed``
    family (benchmark_contract §4)."""
    policy_dt: float = 0.0
    sim_dt: float = 0.0
    step_index: np.ndarray | None = None
    time_s: np.ndarray | None = None
    reset_index: int = 0
    reset_sample: Mapping[str, np.ndarray] | None = None
    reset_valid: bool | None = None
    reset_reason: str | None = None
    terminal_index: int | None = None
    termination: EpisodeTermination | None = None
    terminations: tuple[Termination, ...] = ()

    def __post_init__(self) -> None:
        length = int(self.length)
        if length < 0:
            raise ValueError(f"episode length must be >= 0, got {self.length!r}")
        if self.reset_index != 0:
            raise ValueError(f"reset_index is the canonical reset tick 0, got {self.reset_index!r}")
        if self.reset_valid is not None and not isinstance(self.reset_valid, bool):
            raise TypeError(f"reset_valid must be bool or None, got {self.reset_valid!r}")
        if self.reset_reason is not None and not isinstance(self.reset_reason, str):
            raise TypeError(f"reset_reason must be str or None, got {self.reset_reason!r}")
        index = self.step_index
        times = self.time_s
        if index is None and times is None and self.policy_dt > 0.0:
            index = np.arange(1, length + 1, dtype=np.int64)
            times = index.astype(np.float64) * float(self.policy_dt)
        if index is not None:
            index = np.asarray(index, dtype=np.int64).reshape(-1)
            if index.shape != (length,):
                raise ValueError(f"step_index shape {index.shape} != episode length ({length},)")
            if length and int(index[0]) != 1:
                raise ValueError(f"step_index must start at canonical tick 1, got {int(index[0])}")
        if times is not None:
            times = np.asarray(times, dtype=np.float64).reshape(-1)
            if times.shape != (length,):
                raise ValueError(f"time_s shape {times.shape} != episode length ({length},)")
            if index is not None and self.policy_dt > 0.0:
                expected = index.astype(np.float64) * float(self.policy_dt)
                if not np.allclose(times, expected, rtol=1e-9, atol=1e-12):
                    raise ValueError("time_s is not the canonical step_index * policy_dt mapping")
        object.__setattr__(self, "step_index", index)
        object.__setattr__(self, "time_s", times)
        terms = tuple(self.terminations)
        if self.termination is not None:
            if not isinstance(self.termination, EpisodeTermination):
                raise TypeError("termination must be an EpisodeTermination or None")
            if not terms:
                terms = self.termination.all
            if self.terminal_index is None:
                object.__setattr__(self, "terminal_index", self.termination.primary.step_index)
        if not all(isinstance(t, Termination) for t in terms):
            raise TypeError("terminations must contain only Termination objects")
        object.__setattr__(self, "terminations", terms)
        if self.terminal_index is not None:
            if not isinstance(self.terminal_index, int) or self.terminal_index < 0:
                raise ValueError(
                    f"terminal_index must be an int >= 0 or None (0 = reset "
                    f"tick), got {self.terminal_index!r}"
                )
            if length and self.terminal_index > length:
                raise ValueError(
                    f"terminal_index {self.terminal_index} is outside the "
                    f"episode-local axis 0..{length} (global run ticks are "
                    "not episode ticks)"
                )


@dataclass(frozen=True)
class EpisodeResult:
    """Per-episode scoring output of ``Task.score``.

    Attributes:
        metrics: flat dict of scalar metrics, snake_case names; units are
            documented in the task's docstring (never in the key).
        success: task-judged success for this episode, or None when the
            task does not define a success criterion.  ``success`` is NOT
            inferred from the engine termination outcome.
        termination: optional structured termination returned by the task;
            the sink validates and records it alongside the core facts.
        debug: optional non-aggregated extras (excluded from the report's
            metric statistics; tasks may leave it None).
    """

    metrics: dict[str, float]
    success: bool | None
    termination: Termination | None = None
    debug: dict | None = None


class Task(ABC):
    """Base class of all benchmark tasks.

    Class attributes:
        name: registry key (yaml ``benchmark.tasks[].name``); subclasses
            must override.
        requires_articulation: True for tasks that read per-articulation
            robot state (``base_pos`` / ``gravity_b`` / ``joint_pos`` /
            ``base_lin_vel`` / ``command`` / ...).  Those tasks MUST name
            their articulation in params -- there is no primary binding
            and no unique-articulation inference.

    Args:
        params: task parameters from the yaml section (already stripped of
            the ``name`` key).  Stored verbatim; ``self.p`` is a shorthand.
            ``articulation`` (a declared articulation name) is read from
            here and validated against ``meta["articulations"]``.
        meta: run-level context supplied by collect.py (``run_name``,
            ``seed``, ``num_envs``, ``policy_dt``, ``joint_names``,
            ``has_command``, ``articulations``).  Read-only by convention.
    """

    name: str = "task"
    #: subclasses reading per-articulation robot state set this True
    requires_articulation: bool = False

    def __init__(self, params: dict | None = None, meta: dict | None = None):
        self.params: dict = dict(params or {})
        self.meta: dict = dict(meta or {})
        self.p = self.params  # convenience shorthand
        self.articulation: str | None = self._bind_articulation()
        if self.requires_articulation and self.articulation is None:
            raise ValueError(
                f"benchmark task '{self.name}' reads per-articulation robot "
                "state and requires an explicit `articulation: <name>` "
                f"(available: {self.available_articulations() or 'none'})"
            )

    def available_articulations(self) -> tuple[str, ...]:
        """Articulation names declared for this run (``meta`` order)."""
        return tuple((self.meta.get("articulations") or {}).keys())

    def _bind_articulation(self) -> str | None:
        raw = self.p.get("articulation")
        if raw is None:
            return None
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(
                f"benchmark task '{self.name}': articulation must be a "
                f"non-empty string, got {raw!r}"
            )
        name = raw.strip()
        available = self.available_articulations()
        if available and name not in available:
            raise ValueError(
                f"benchmark task '{self.name}': articulation {name!r} is "
                f"not one of the run's articulations {list(available)}"
            )
        return name

    def resolve_key(self, data: dict, key: str) -> str:
        """Recorded key for *key* under this task's declared articulation."""
        return resolve_episode_key(data, key, self.articulation)

    @abstractmethod
    def score(self, ep: EpisodeData, ctx: EvalContext) -> EpisodeResult:
        """Score one episode from its history and its live per-env context.

        ``ctx.env`` is read-only and scoped to ``ep.env_id``; a task may not
        mutate/step/reset the environment or read another env.  Tasks that
        only need ``ep`` may ignore ``ctx``.
        """

    def score_detail(
        self,
        ep: EpisodeData,
        ctx: EvalContext,
        *,
        retain_frames: bool = False,
    ) -> tuple[EpisodeResult, dict, dict]:
        """``score`` plus optional formula leaves/frames.

        The default has no declaration-derived detail: it returns
        ``(score(ep, ctx), {}, {})``.  ``FormulaTask`` overrides it to attach
        predicate leaves and per-frame series.
        """
        del retain_frames
        return self.score(ep, ctx), {}, {}

    def aggregate(self, results: list[EpisodeResult]) -> dict:
        """Default run-level aggregation (overridable).

        For every metric name (first-seen order) reports
        ``{mean, std, min, max, p10, p90}`` (population std, linear
        percentile interpolation, plain floats).  ``success_rate`` is the
        mean over episodes with ``success is not None`` (None when no
        episode defines success).  ``n_episodes`` is the result count.

        An empty list aggregates to ``{"n_episodes": 0, "success_rate":
        None}``.
        """
        if not results:
            return {"n_episodes": 0, "success_rate": None}
        metric_names: list[str] = []
        for r in results:
            for k in r.metrics:
                if k not in metric_names:
                    metric_names.append(k)
        out: dict = {}
        for k in metric_names:
            vals = np.array([float(r.metrics[k]) for r in results if k in r.metrics], dtype=float)
            out[k] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "p10": float(np.percentile(vals, 10)),
                "p90": float(np.percentile(vals, 90)),
            }
        judged = [r.success for r in results if r.success is not None]
        out["success_rate"] = float(np.mean(judged)) if judged else None
        out["n_episodes"] = len(results)
        return out


TASK_REGISTRY: dict[str, type[Task]] = {}


def register(cls: type[Task]) -> type[Task]:
    """Class decorator: add a Task subclass to TASK_REGISTRY by ``name``."""
    if cls.name in TASK_REGISTRY:
        raise ValueError(f"benchmark task '{cls.name}' already registered")
    TASK_REGISTRY[cls.name] = cls
    return cls


_POSITIONAL_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def _require_live_task(task: Task) -> None:
    """Reject a task that does not implement the live scoring signatures.

    ``EvaluationSession`` isolates task exceptions, so a legacy ``score(ep)`` task
    would be silently skipped instead of rejected.  Validation happens here at
    build time: ``score(self, ep, ctx)`` and
    ``score_detail(self, ep, ctx, *, retain_frames=False)``.
    """
    name = getattr(task, "name", type(task).__name__)
    for method_name, expected in (
        ("score", "score(self, ep, ctx)"),
        ("score_detail", "score_detail(self, ep, ctx, *, retain_frames=False)"),
    ):
        method = getattr(task, method_name, None)
        if method is None or not callable(method):
            raise ValueError(
                f"benchmark task {name!r} ({type(task).__name__}) has no callable "
                f"{method_name}; expected {expected}"
            )
        signature = inspect.signature(method)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in _POSITIONAL_KINDS
        ]
        required_keyword = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and parameter.default is inspect.Parameter.empty
        ]
        variadic = any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        names = tuple(parameter.name for parameter in positional)
        if variadic or names != ("ep", "ctx") or required_keyword:
            raise ValueError(
                f"benchmark task {name!r} ({type(task).__name__}) must implement "
                f"{expected}; got {method_name}{signature} | hint: a pure task "
                "migrates with `def score(self, ep, ctx): del ctx`"
            )
        if method_name == "score_detail":
            retained = signature.parameters.get("retain_frames")
            overridden = getattr(type(task), "score_detail", None) is not Task.score_detail
            if overridden and retained is None:
                raise ValueError(
                    f"benchmark task {name!r} ({type(task).__name__}) must implement "
                    f"{expected}; got {method_name}{signature} (an overridden "
                    "score_detail must declare keyword-only `retain_frames` with "
                    "a default)"
                )
            if retained is not None and (
                retained.kind is not inspect.Parameter.KEYWORD_ONLY
                or retained.default is inspect.Parameter.empty
            ):
                raise ValueError(
                    f"benchmark task {name!r} ({type(task).__name__}) must implement "
                    f"{expected}; got {method_name}{signature} (retain_frames must "
                    "be keyword-only with a default)"
                )


def build_tasks(sections: list[dict], meta: dict, runtime: PackRuntime) -> list[Task]:
    """Instantiate tasks from yaml ``benchmark.tasks`` sections.

    Routing (B1.5): a section with ``class: formula`` builds a
    FormulaTask from the remaining keys (description/metrics/success -- see
    benchmarks/formula.py); its ``name`` is the report key only.  Sections
    without ``class`` resolve through the pack index (benchmarks/packs.py):
    a name in TASK_REGISTRY keeps the exact stable behavior (report key =
    the class ``name``); any other name resolves as ``<pack>/<task>`` or a
    bare name unique across all namespaces -- the instance report key
    becomes the canonical ``<pack>/<task>``.  Any other ``class`` value
    raises ValueError.

    ``runtime`` is the caller's :class:`~unigym.packs.runtime.PackRuntime`;
    pack discovery reads only its ``benchmarks`` catalog entries
    (benchmarks/packs.discover_packs).  ``<pack>/*`` wildcard sections
    expand to the whole pack (sorted).

    Every instantiated task (registry, formula or pack plugin) must implement
    the live ``score(self, ep, ctx)`` and
    ``score_detail(self, ep, ctx, *, retain_frames=False)`` signatures; a
    legacy task fails here instead of being silently skipped by the sink.

    Unknown task names raise ValueError (mirrors the adapter registry).
    Duplicate task names raise ValueError (EvaluationSession keys results by
    task name -- duplicates would silently merge).
    """
    from .packs import discover_packs  # lazy: packs.py imports this module

    index = discover_packs(runtime)
    sections = index.expand_wildcards(sections)
    tasks: list[Task] = []
    seen_names: set[str] = set()
    for section in sections:
        section = dict(section)
        name = section.pop("name", None)
        cls_key = section.pop("class", None)
        if cls_key is None:
            def_, _pack = index.resolve_task(name)
            task = def_.instantiate(section, meta)
            if def_.pack != "builtin":
                task.name = def_.canonical  # report key is namespaced
        elif cls_key == "formula":
            from .formula import FormulaTask  # lazy: avoids the import cycle

            task = FormulaTask(params=section, meta=meta)
            if name is not None:
                task.name = str(name)
        else:
            raise ValueError(
                f"benchmark task section {name!r}: unknown class "
                f"{cls_key!r} (only 'formula', a pack reference, or "
                f"omitted for the named registry {sorted(TASK_REGISTRY)})"
            )
        _require_live_task(task)
        if task.name in seen_names:
            raise ValueError(
                f"duplicate benchmark task name '{task.name}' (EvaluationSession "
                "keys results by task name; give formula tasks unique "
                "yaml names)"
            )
        seen_names.add(task.name)
        tasks.append(task)
    return tasks
