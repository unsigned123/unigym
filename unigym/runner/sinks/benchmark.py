"""BenchmarkSink: episode-scoring consumer (B1), sibling of CollectSink.

Same callback mechanism and episode-buffering machinery as CollectSink
(``slice_step_for_env``/``assemble_episode`` from ``runner/episodes.py``),
but instead of writing shards it delegates each completed episode (pre-reset --
record-before-reset invariant) to an
``unigym.runner.evaluation.scoring.EpisodeEvaluator``, which assembles the
``EpisodeData`` and scores it with every configured ``benchmarks`` Task.  The
sink feeds the successful evaluations to a
``unigym.benchmarks.reporting.BenchmarkReporter``, which owns the run-level
aggregation and the ``benchmark_report.json`` / ``results.json`` /
``metrics.npz`` artifacts.

Callback contract (docs/contracts/runner_contract.md §3):

* ``on_step(t, bundle)`` -- buffer per-env step slices (``_KEY_MAP`` key
  normalization identical to CollectSink) and maintain the early-stop flag
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
context is the only path to the engine, and it is read-only and per-env.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from unigym.benchmarks.base import Task
from unigym.benchmarks.episode import EpisodeLimits, RunIdentity
from unigym.benchmarks.reporting import (
    METRICS_NPZ,
    REPORT_SCHEMA,
    RESULTS_JSON,
    RESULTS_SCHEMA,
    BenchmarkReporter,
)
from unigym.core.episode_result import EpisodeResultView

from ..evaluation.buffer import EpisodeBuffer
from ..evaluation.scoring import EpisodeEvaluator

if TYPE_CHECKING:
    from unigym.env.backends.base import EnvBackend

__all__ = [
    "BenchmarkSink",
    "METRICS_NPZ",
    "REPORT_SCHEMA",
    "RESULTS_JSON",
    "RESULTS_SCHEMA",
]

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


class BenchmarkSink:
    """Scores completed episodes with a set of benchmark Tasks.

    Args:
        tasks: instantiated ``benchmarks.base.Task`` objects.
        meta: run-level context dict handed to tasks and embedded in the
            report (``run_name``, ``seed``, ``policy_dt``,
            ``joint_names``, ``num_envs``).
        min_episodes: early-stop threshold -- ``stop_when`` (passed to
            ``SyncRunner.run``) becomes True once this many episodes
            have completed.
        max_steps: configured step budget (informational; the engine's
            loop length is wired by collect.py).
        report_path: where ``benchmark_report.json`` is written at run end.
        director: optional Director  -- episodes it touched are
            labelled ``directed: true`` in the report (honest boundary:
            directed successes never mix into physics-legal claims,
            director_contract.md §7).
        env: the live env backend, injected by ``_build_sinks``.  It is used
            only to scope the per-env evaluation context (device, backend
            label, articulations and a lazy read-only query factory); it is
            never read from ``bundle["env"]``.  ``None`` is the explicit
            record-only mode for direct unit wiring: the context has
            ``record``/``obs`` and no query.
    """

    def __init__(
        self,
        tasks: list[Task],
        meta: dict,
        min_episodes: int,
        max_steps: int,
        report_path: str,
        director=None,
        retain_frames: bool = False,
        limits: EpisodeLimits | None = None,
        run_identity: RunIdentity | None = None,
        env: EnvBackend | None = None,
    ):
        self.tasks = list(tasks)
        self.meta = dict(meta)
        self.min_episodes = int(min_episodes)
        self.max_steps = int(max_steps)
        self.num_envs = int(meta.get("num_envs", 1))
        # episode facts + task invocation are owned by the evaluator; the sink
        # only feeds evaluated episodes to the report owner
        self.evaluator = EpisodeEvaluator(
            tasks=self.tasks,
            meta=self.meta,
            env=env,
            limits=limits,
            retain_frames=bool(retain_frames),
        )
        # open-episode history (steps + rewind marks + tick samples) is owned
        # by the buffer; the sink only delegates append/take/mark
        self.episode_buffer = EpisodeBuffer(self.num_envs)
        # run-level aggregation and report/results/metrics writing
        self.reporter = BenchmarkReporter(
            tasks=self.tasks,
            meta=self.meta,
            report_path=str(report_path),
            run_identity=run_identity,
        )
        self.n_episodes = 0
        self.steps_total = 0
        self._stop = False
        self._t0 = time.time()
        self._metrics_pending: list[dict] = []  # bridge metrics feed
        self._director = director

    # ------------------------------------------------------------------ #
    # early stop predicate (passed to SyncRunner.run(stop_when=...))
    # ------------------------------------------------------------------ #
    def episodes_done(self) -> bool:
        """True once ``n_episodes >= min_episodes`` (engine early stop)."""
        return self.n_episodes >= self.min_episodes

    # ------------------------------------------------------------------ #
    # per-step buffering (pre-reset data side, like CollectSink)
    # ------------------------------------------------------------------ #
    def on_step(self, t: int, bundle: dict) -> None:
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
        self.steps_total = t + 1
        self._stop = self.episodes_done()
        self.reporter.note_progress(episodes_total=self.n_episodes, steps_total=self.steps_total)

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
        benchmark_contract §4)."""
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
            evaluation = self.evaluator.evaluate(taken, bundle, ei)
            ep = evaluation.episode
            # directed labelling: the Director's per-env mark is
            # still live here (this sink runs before the director's
            # episode-end snapshot in the fan-out)
            directed = bool(self._director is not None and self._director.is_episode_directed(ei))
            self.reporter.on_episode(evaluation, directed=directed)
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
        return self.reporter.finish(
            episodes_total=self.n_episodes,
            steps_total=self.steps_total,
            wall_time_s=time.time() - self._t0,
        )

    # ------------------------------------------------------------------ #
    # bridge metrics feed (duck-typed source; runtime_contract §2)
    # ------------------------------------------------------------------ #
    def metrics_drain(self) -> list[dict]:
        """Per-episode {reward, success} scored since the previous drain
        (order-aligned with the engine's env_ids; the bridge consumes this
        right after its own on_episode_end callback)."""
        out, self._metrics_pending = self._metrics_pending, []
        return out

    def metrics_task(self) -> dict | None:
        """Primary task identity (canonical ``<pack>/<task>`` split)."""
        if not self.tasks:
            return None
        pack, _, name = self.tasks[0].name.rpartition("/")
        return {"pack": pack or None, "name": name or self.tasks[0].name}

    def metrics_progress(self) -> dict:
        """Benchmark counters for the live view (episode-level)."""
        return {
            "episodes": self.n_episodes,
            "min_episodes": self.min_episodes,
            "by_task": {t.name: len(self.reporter.results[t.name]) for t in self.tasks},
        }

    def metrics_report(self) -> dict | None:
        """Bounded live report (reporter-owned draft; no growing lists)."""
        if not self.tasks:
            return None
        return self.reporter.draft_report()
