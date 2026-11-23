"""EvaluationSession: the runner sink for benchmark evaluation.

The session is the public runner sink for benchmark mode (``mode: benchmark``
and ``both``).  It owns the evaluation lifecycle -- early stop, per-step
buffering, episode-end dispatch, rewind marks, the bridge metrics feed and the
per-task success counters -- and drives three injected collaborators:

* an :class:`~unigym.runner.evaluation.buffer.EpisodeBuffer` owns the
  open-episode history (per-env step slices, rewind marks);
* an :class:`~unigym.runner.evaluation.scoring.EpisodeEvaluator` assembles the
  ``EpisodeData`` and invokes every configured benchmark task;
* an :class:`~unigym.benchmarks.reporting.EpisodeReporter` owns run-level
  aggregation and artifact writing (``BenchmarkReporter`` in benchmark runs).

The session constructs none of them: benchmark wiring builds the concrete
collaborators and injects them, so the session stays reporter-agnostic.

Callback contract (docs/contracts/runner_contract.md §3):

* ``on_step(t, bundle)`` -- buffer per-env step slices (``_KEY_MAP`` key
  normalization identical to CollectSink), dispatch ``trigger='both'`` tasks
  against one vector-scoped context derived from the runner's boundary view
  (``bundle["view"].with_record(normalized_record)``; per-env CPU
  ``TickSample``s land on the buffered step) and maintain the early-stop flag
  (``completed episodes >= min_episodes``) plus step counters;
* ``on_episode_end(env_ids, bundle)`` -- BEFORE the reset chain: per env,
  take the buffered history and delegate to the ``EpisodeEvaluator`` (it
  assembles ``EpisodeData`` and builds the scoped context; ``ended_by_termination``
  comes from ``bundle["terminated"][env_id]`` -- the terminated mask covers ALL
  envs, so it must be indexed with the due id, never sliced by position),
  then hand the evaluation to the reporter and count the episode;
* ``on_step_end(t, bundle)`` -- light progress printing (every 500 steps);
* ``on_run_end()`` -- ``reporter.finish(...)`` (per-task ``aggregate`` ->
  report dict -> write artifacts) and return the report (the engine's ``run()``
  return value).  Incomplete (still-buffered) episodes are NOT scored:
  benchmarks judge completed episodes only.

Device/thread discipline matches CollectSink: episode buffers hold cpu
clones (``slice_step_for_env``).  ``EpisodeData`` stays numpy; the live
context is the only path to the engine, it is read-only, vector-scoped for
per-tick dispatch and per-env for episode scoring, and it is always derived
from the runner-owned ``bundle["view"]`` boundary view.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import torch

from unigym.benchmarks.base import Task, TickResult
from unigym.benchmarks.reporting import EpisodeReporter
from unigym.core.episode_result import EpisodeResultView
from unigym.evaluation import EvalContext, EvalViewError

from .buffer import TickSample

if TYPE_CHECKING:
    from unigym.runtime.rewind_providers import EvaluationSessionProvider

    from .buffer import EpisodeBuffer
    from .scoring import EpisodeEvaluator

__all__ = ["EvaluationSession"]

# extra key -> recorded episode_data key (identical to CollectSink's map;
# applied by SUFFIX so ``articulation/<name>/base_pos_w`` -> ``.../base_pos``)
_KEY_MAP = {
    "base_pos_w": "base_pos",
    "base_lin_vel_b": "base_lin_vel",
    "base_ang_vel_b": "base_ang_vel",
}


def _record_key(key: str) -> str:
    mapped = _KEY_MAP.get(key)
    if mapped is not None:
        return mapped
    head, sep, leaf = key.rpartition("/")
    if sep and leaf in _KEY_MAP:
        return f"{head}/{_KEY_MAP[leaf]}"
    return key


def _tick_samples(result: object, task_name: str, num_envs: int) -> list[TickSample]:
    """Validate one vector ``TickResult`` and slice it into per-env samples.

    A wrong return type, a non-tensor / non-mapping metric, a metric whose
    leading axis is not the env axis, or a non-bool / wrong-shape ``fired``
    raises with a teaching message; the caller disables the task.
    """
    if not isinstance(result, TickResult):
        raise TypeError(
            f"task '{task_name}' on_tick returned {type(result).__name__}, " "expected TickResult"
        )
    metrics = result.metrics
    if not hasattr(metrics, "items"):
        raise TypeError(
            f"task '{task_name}' TickResult.metrics must be a mapping, "
            f"got {type(metrics).__name__}"
        )
    per_env: list[dict[str, np.ndarray]] = [{} for _ in range(num_envs)]
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"task '{task_name}' tick metric key must be a non-empty string, got {key!r}"
            )
        if not torch.is_tensor(value):
            raise TypeError(
                f"task '{task_name}' tick metric {key!r} must be a tensor, "
                f"got {type(value).__name__}"
            )
        if value.ndim < 1 or value.shape[0] != num_envs:
            raise ValueError(
                f"task '{task_name}' tick metric {key!r} must have a leading "
                f"env axis ({num_envs}, ...), got shape {tuple(value.shape)}"
            )
        array = value.detach().to("cpu").numpy()
        for ei in range(num_envs):
            per_env[ei][key] = np.array(array[ei], copy=True)
    fired = result.fired
    fired_array: np.ndarray | None = None
    if fired is not None:
        if not torch.is_tensor(fired) or fired.dtype is not torch.bool:
            raise TypeError(
                f"task '{task_name}' TickResult.fired must be a bool tensor or "
                f"None, got {type(fired).__name__}"
            )
        if tuple(fired.shape) != (num_envs,):
            raise ValueError(
                f"task '{task_name}' TickResult.fired must have shape "
                f"({num_envs},), got {tuple(fired.shape)}"
            )
        fired_array = fired.detach().to("cpu").numpy()
    return [
        TickSample(
            metrics=per_env[ei],
            fired=None if fired_array is None else np.array(fired_array[ei], copy=True),
        )
        for ei in range(num_envs)
    ]


class EvaluationSession:
    """Drive the evaluation lifecycle over injected collaborators.

    Args:
        tasks: instantiated ``benchmarks.base.Task`` objects in declaration
            order (task identity and the metrics feed key order).
        buffer: the open-episode history owner.
        evaluator: the episode assembler and task invoker.
        reporter: the run-level aggregation and artifact writer.
        meta: optional run-level context (only ``num_envs`` is read here).
        num_envs: env-vector width; defaults to ``meta["num_envs"]`` or 1.
        min_episodes: early-stop threshold -- ``stop_when`` (passed to
            ``SyncRunner.run``) becomes True once this many episodes
            have completed.
        max_steps: configured step budget (informational; the engine's
            loop length is wired by collect.py).
        director: optional Director -- episodes it touched are
            labelled ``directed: true`` in the report (honest boundary:
            directed successes never mix into physics-legal claims,
            director_contract.md §7).

    The session holds no live env.  Per-tick tasks read the runner-owned
    boundary view (``bundle["view"]``); a bundle without that view fails fast.
    """

    def __init__(
        self,
        *,
        tasks: list[Task],
        buffer: EpisodeBuffer,
        evaluator: EpisodeEvaluator,
        reporter: EpisodeReporter,
        meta: dict | None = None,
        num_envs: int | None = None,
        min_episodes: int = 1,
        max_steps: int = 100000,
        director=None,
    ):
        self.tasks = list(tasks)
        self.meta = dict(meta or {})
        self.min_episodes = int(min_episodes)
        self.max_steps = int(max_steps)
        self.num_envs = int(self.meta.get("num_envs", 1) if num_envs is None else num_envs)
        # injected collaborators; the session constructs none of them
        self.evaluator = evaluator
        self.episode_buffer = buffer
        self.reporter = reporter
        # per-task successful-score counters (the reporter's result lists are
        # not part of the reporter protocol; the session owns the live feed)
        self._task_counts: dict[str, int] = {t.name: 0 for t in self.tasks}
        self.n_episodes = 0
        self.steps_total = 0
        self._stop = False
        self._t0 = time.time()
        self._metrics_pending: list[dict] = []  # bridge metrics feed
        self._director = director
        # tasks disabled for the RUN by a per-tick contract failure; process
        # truth (never restored by a rewind) and never per-env partial
        self._disabled_tasks: set[str] = set()

    # ------------------------------------------------------------------ #
    # rewind provider (structural discovery; name stays ``benchmark``)
    # ------------------------------------------------------------------ #
    @property
    def rewind_provider(self) -> EvaluationSessionProvider:
        """The canonical benchmark provider wrapper for this session."""
        from unigym.runtime.rewind_providers import EvaluationSessionProvider

        return EvaluationSessionProvider(self)

    # ------------------------------------------------------------------ #
    # early stop predicate (passed to SyncRunner.run(stop_when=...))
    # ------------------------------------------------------------------ #
    def episodes_done(self) -> bool:
        """Return True once ``n_episodes >= min_episodes`` (engine early stop)."""
        return self.n_episodes >= self.min_episodes

    # ------------------------------------------------------------------ #
    # per-step buffering (pre-reset data side, like CollectSink)
    # ------------------------------------------------------------------ #
    def on_step(self, t: int, bundle: dict) -> None:
        """Buffer one normalized per-env step and update the run counters."""
        record = bundle["record"]
        rec = {}
        for k, v in record.items():
            if k in ("should_reset", "reset_env_ids"):
                continue
            rec[_record_key(k)] = v
        # record the adapter's raw action frame too (the collect
        # sink already does) -- FormulaTask metrics may need adapter-carried
        # signals, e.g. the Psi0 height command inside the 36-dim row
        raw = bundle.get("raw")
        if raw is not None:
            rec["action_raw"] = raw
        eff_cmd = bundle["cmd"]
        if eff_cmd is not None:
            rec["command"] = eff_cmd
        self.episode_buffer.append_step(rec)
        self._dispatch_ticks(t, rec, bundle)
        self.steps_total = t + 1
        self._stop = self.episodes_done()
        self.reporter.note_progress(episodes_total=self.n_episodes, steps_total=self.steps_total)

    def _dispatch_ticks(self, t: int, rec: dict, bundle: dict) -> None:
        """Run active ``both`` tasks once on one vector-scoped context.

        Buffering happens before dispatch, so a failing task can never lose
        episode history.  The context is derived from the runner's boundary
        view (``bundle["view"].with_record(rec)``), so the normalized record
        keys reach ``on_tick``.  The derived child is closed in ``finally``;
        the shared boundary view is owned by the runner and is never closed
        here.  A missing view fails fast.  Any contract failure disables the
        offending task for the RUN (one log line) and leaves every other task
        untouched.
        """
        view = bundle.get("view")
        if view is None:
            raise EvalViewError(
                "benchmark tick dispatch requires bundle['view']; the runner "
                "publishes one read-only boundary view per tick"
            )
        active = [
            task
            for task in self.tasks
            if task.trigger == "both" and task.name not in self._disabled_tasks
        ]
        if not active:
            return
        ctx = EvalContext(env=view.with_record(rec), t=t)
        try:
            for task in active:
                try:
                    samples = _tick_samples(task.on_tick(ctx), task.name, self.num_envs)
                except Exception as exc:  # noqa: BLE001 -- isolate/disable task
                    self._disable_task(task.name, exc)
                    continue
                for ei, sample in enumerate(samples):
                    self.episode_buffer.append_tick(ei, task.name, sample)
        finally:
            ctx.close()

    def _disable_task(self, name: str, exc: Exception) -> None:
        """Disable one task for the run after a per-tick contract failure."""
        self._disabled_tasks.add(name)
        print(
            f"[benchmark] task '{name}' disabled for this run: on_tick "
            f"contract failure {exc!r}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # rewind marking (Round B B1; engine boundary notification)
    # ------------------------------------------------------------------ #
    def on_rewind(self, info: dict) -> None:
        """A rewind forked the timeline: mark every OPEN episode buffer.

        The engine calls this at the rewind boundary (before the restored
        timeline's first step); ``info`` is the rewind record
        ``{tick, restored_tick, snap_id, label?}``.  The records stay
        attached to the open buffers and ride the assembled episodes as
        the ``rewound`` flag + per-episode rewind rows (directed family,
        benchmark_contract §4).
        """
        self.episode_buffer.mark_rewound(info)
        print(
            f"[rewind] episode buffers marked rewound at tick "
            f"{info.get('tick')} (restored to {info.get('restored_tick')})",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # episode end (before-reset data side; reset chain is engine-driven)
    # ------------------------------------------------------------------ #
    def on_episode_end(self, env_ids, bundle: dict) -> None:
        """Evaluate and publish every due env before the reset chain."""
        # per-env read-only result views for the sampler hook: successful
        # tasks only, in task declaration order (a raising task is omitted
        # exactly as it is omitted from the report)
        episode_results: dict[int, tuple[EpisodeResultView, ...]] = {}
        for env_id in env_ids.tolist():
            ei = int(env_id)
            taken = self.episode_buffer.take(ei)
            self.n_episodes += 1
            episode_results[ei] = ()
            if not taken.steps:
                continue
            evaluation = self.evaluator.evaluate(
                taken, bundle, ei, skip_tasks=frozenset(self._disabled_tasks)
            )
            ep = evaluation.episode
            # directed labelling: the Director's per-env mark is
            # still live here (this session runs before the director's
            # episode-end snapshot in the fan-out)
            directed = bool(self._director is not None and self._director.is_episode_directed(ei))
            self.reporter.on_episode(evaluation, directed=directed)
            for score in evaluation.scores:
                self._task_counts[score.task] = self._task_counts.get(score.task, 0) + 1
            episode_results[ei] = evaluation.views
            # bridge metrics feed: PRIMARY task (first successful) supplies the
            # episode's reward/success slot (exact alignment with the env
            # order the engine hands to every sink, bridge included).
            primary = evaluation.scores[0].result if evaluation.scores else None
            self._metrics_pending.append(
                {
                    "reward": (primary.metrics.get("reward") if primary is not None else None),
                    "success": (primary.success if primary is not None else None),
                    "directed": directed,
                    "rewound": ep.rewound,
                }
            )
        # hand the per-env views to the engine, which forwards them to the
        # command sampler right after this callback (same boundary bundle)
        bundle["episode_results"] = episode_results
        self.reporter.note_progress(episodes_total=self.n_episodes, steps_total=self.steps_total)

    # ------------------------------------------------------------------ #
    # light progress (post-reset, mirrors CollectSink's logging rhythm)
    # ------------------------------------------------------------------ #
    def on_step_end(self, t: int, bundle: dict) -> None:
        """Print light run progress every 500 policy steps."""
        if (t + 1) % 500 == 0:
            print(
                f"[benchmark] step {t + 1}/{self.max_steps} | episodes "
                f"{self.n_episodes}/{self.min_episodes}",
                flush=True,
            )

    # ------------------------------------------------------------------ #
    # run end: aggregate + report json + summary dict
    # ------------------------------------------------------------------ #
    def on_run_end(self) -> dict:
        """Finish the run: aggregate, write artifacts and return the report."""
        return self.reporter.finish(
            episodes_total=self.n_episodes,
            steps_total=self.steps_total,
            wall_time_s=time.time() - self._t0,
        )

    # ------------------------------------------------------------------ #
    # bridge metrics feed (duck-typed source; runtime_contract §2)
    # ------------------------------------------------------------------ #
    def metrics_drain(self) -> list[dict]:
        """Drain per-episode ``{reward, success}`` scored since the last call.

        Order-aligned with the engine's env_ids; the bridge consumes this
        right after its own on_episode_end callback.
        """
        out, self._metrics_pending = self._metrics_pending, []
        return out

    def metrics_task(self) -> dict | None:
        """Return the primary task identity (canonical ``<pack>/<task>`` split)."""
        if not self.tasks:
            return None
        pack, _, name = self.tasks[0].name.rpartition("/")
        return {"pack": pack or None, "name": name or self.tasks[0].name}

    def metrics_progress(self) -> dict:
        """Return benchmark counters for the live view (episode-level)."""
        return {
            "episodes": self.n_episodes,
            "min_episodes": self.min_episodes,
            "by_task": dict(self._task_counts),
        }

    def metrics_report(self) -> dict | None:
        """Return the bounded live report (reporter-owned draft; no growing lists)."""
        if not self.tasks:
            return None
        return self.reporter.draft_report()
