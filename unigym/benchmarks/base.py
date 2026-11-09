"""Benchmark task protocol (B1): pure episode judges, zero isaaclab.

A benchmark run evaluates completed episodes instead of (or alongside)
recording them.  The data path is:

    SyncRunner --callbacks--> BenchmarkSink --EpisodeData--> Task.score
        --EpisodeResult--> Task.aggregate --> benchmark_report.json

Contract (docs/contracts/benchmark_contract.md):

* two-phase scoring: ``score(ep) -> EpisodeResult`` per episode (pure
  function of the episode data -- tasks never touch the environment) and
  ``aggregate(results) -> dict`` at run end;
* ``Task(params, meta)`` binds all configuration at construction time so
  scoring itself is parameter-free;
* ``success`` (task-judged quality) is recorded separately from
  ``ended_by_termination`` (engine-observed early termination) -- early
  termination is NOT automatically a failure, each task decides;
* numpy-only: EpisodeData arrays are plain numpy (T, .) in *recorded key
  names* (the shard npz names, e.g. ``base_pos``/``gravity_b``), so tasks
  are unit-testable offline with synthetic episodes.

This module is intentionally importable without torch/isaaclab.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


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
        "foot_contact",
        "foot_contact_force_z",
        "foot_pos_z",
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
    """One completed episode, assembled pre-reset by BenchmarkSink.

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


@dataclass(frozen=True)
class EpisodeResult:
    """Per-episode scoring output of ``Task.score``.

    Attributes:
        metrics: flat dict of scalar metrics, snake_case names; units are
            documented in the task's docstring (never in the key).
        success: task-judged success for this episode, or None when the
            task does not define a success criterion.
        debug: optional non-aggregated extras (excluded from the report's
            metric statistics; tasks may leave it None).
    """

    metrics: dict[str, float]
    success: bool | None
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
    def score(self, ep: EpisodeData) -> EpisodeResult:
        """Score one episode; pure function, no side effects."""

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


def build_tasks(sections: list[dict], meta: dict, packs_cfg: list | None = None) -> list[Task]:
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

    ``packs_cfg`` is the run yaml's ``benchmark.packs`` list of pack
    directories (see packs.discover_packs for the full scan order).
    ``<pack>/*`` wildcard sections expand to the whole pack (sorted).

    Unknown task names raise ValueError (mirrors the adapter registry).
    Duplicate task names raise ValueError (BenchmarkSink keys results by
    task name -- duplicates would silently merge).
    """
    from .packs import discover_packs  # lazy: packs.py imports this module

    index = discover_packs(packs_cfg)
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
        if task.name in seen_names:
            raise ValueError(
                f"duplicate benchmark task name '{task.name}' (BenchmarkSink "
                "keys results by task name; give formula tasks unique "
                "yaml names)"
            )
        seen_names.add(task.name)
        tasks.append(task)
    return tasks
