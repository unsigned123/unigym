"""BenchmarkSink: episode-scoring consumer (B1), sibling of CollectSink.

Same callback mechanism and episode-buffering machinery as CollectSink
(``slice_step_for_env``/``assemble_episode`` from ``runner/episodes.py``),
but instead of writing shards it assembles each completed episode into an
``EpisodeData`` (pre-reset -- record-before-reset invariant), scores it with
every configured ``benchmarks`` Task, and at run end aggregates the results
into ``benchmark_report.json``.

Callback contract (docs/contracts/runner_contract.md §3):

* ``on_step(t, bundle)`` -- buffer per-env step slices (``_KEY_MAP`` key
  normalization identical to CollectSink) and maintain the early-stop flag
  (``completed episodes >= min_episodes``) plus step counters;
* ``on_episode_end(env_ids, bundle)`` -- BEFORE the reset chain: per env,
  assemble ``EpisodeData`` (duration from ``record["time_in_episode"]``,
  ``ended_by_termination`` from ``bundle["terminated"][env_id]`` -- the
  terminated mask covers ALL envs, so it must be indexed with the due id,
  never sliced by position), score it with each task (one task raising is
  logged with the task name and skipped -- a broken task never poisons the
  run), count the episode;
* ``on_step_end(t, bundle)`` -- light progress printing (every 500 steps);
* ``on_run_end()`` -- per-task ``aggregate`` -> report dict -> write
  ``benchmark_report.json`` -> return the report (the engine's ``run()``
  return value).  Incomplete (still-buffered) episodes are NOT scored:
  benchmarks judge completed episodes only.

Device/thread discipline matches CollectSink: episode buffers hold cpu
clones (``slice_step_for_env``), task scoring is pure numpy.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from unigym.benchmarks.base import EpisodeData, EpisodeResult, Task
from unigym.benchmarks.episode import (
    EpisodeLimits,
    RunIdentity,
    timeline_time_s,
)
from unigym.benchmarks.termination import (
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationExtension,
    TerminationSource,
    code_counts,
    outcome_counts,
    source_counts,
)
from unigym.core.episode_result import EpisodeResultView

from ..episodes import assemble_episode, slice_step_for_env

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


REPORT_SCHEMA = "benchmark_report"
RESULTS_SCHEMA = "results"
METRICS_NPZ = "metrics.npz"
RESULTS_JSON = "results.json"


def _verdict(success, ended_by_termination: bool) -> str:
    """Results verdict (analyze_contract §6): the task is the judge --
    success verdicts win over the observational termination flag."""
    if success is True:
        return "success"
    if success is False:
        return "fail"
    return "fell" if ended_by_termination else "na"


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
    ):
        self.tasks = list(tasks)
        self.meta = dict(meta)
        self.limits = limits
        self.run_identity = run_identity
        self.min_episodes = int(min_episodes)
        self.max_steps = int(max_steps)
        self.report_path = str(report_path)
        # retain the per-frame metric sequences (metrics.npz,
        # benchmark_contract §results)
        self.retain_frames = bool(retain_frames)
        self.num_envs = int(meta.get("num_envs", 1))
        self.ep_bufs: list[list[dict]] = [[] for _ in range(self.num_envs)]
        self.episodes: list[EpisodeData] = []  # all scored EpisodeData
        self.results: dict[str, list] = {t.name: [] for t in self.tasks}
        # per-(episode, task) detail docs, results.json face
        self.episode_docs: list[dict] = []
        self.n_episodes = 0
        self.steps_total = 0
        self._stop = False
        self._t0 = time.time()
        self._metrics_pending: list[dict] = []  # bridge metrics feed
        self._director = director
        self.episode_directed: list[bool] = []  # parallel to episodes
        # per-episode structured termination docs (benchmark report)
        self._episode_terminations: list[dict] = []
        self._terminal_primary: list[Termination | None] = []
        # rewind marking (Round B B1): per-env rewind records for the OPEN
        # episode buffers (process-order interval intersection -- policy
        # tick numbers repeat across a fork, so the open-buffer rule is
        # the honest reading of "episode interval intersection"); parallel to ep_bufs
        self._ep_rewinds: list[list[dict]] = [[] for _ in range(self.num_envs)]

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
        for ei in range(self.num_envs):
            self.ep_bufs[ei].append(slice_step_for_env(rec, ei))
        self.steps_total = t + 1
        self._stop = self.episodes_done()

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
        for buf in self._ep_rewinds:
            buf.append(dict(info))
        print(
            f"[rewind] episode buffers marked rewound at tick "
            f"{info.get('tick')} (restored to {info.get('restored_tick')})",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # episode end (before-reset data side; reset chain is engine-driven)
    # ------------------------------------------------------------------ #
    def _episode_core_terminations(
        self, ei: int, length: int, bundle: dict, reset: dict
    ) -> list[Termination]:
        """Core facts for one episode: env terms, reset legality, deadline."""
        policy_dt = float(self.meta.get("policy_dt") or 0.0)
        time_s = length * policy_dt if policy_dt > 0.0 else None
        terms: list[Termination] = []
        bundle_terms = (bundle.get("terminations") or {}).get(int(ei))
        if bundle_terms is not None:
            terms.extend(bundle_terms.all)
        if reset.get("reset_valid") is False:
            extensions = {}
            reason = reset.get("reset_reason")
            if reason:
                extensions["initializer"] = TerminationExtension(
                    namespace="initializer", schema_version=1, data={"reason": str(reason)}
                )
            terms.append(
                Termination(
                    outcome=EpisodeOutcome.INVALID_RESET,
                    code="initializer.invalid_reset",
                    source=TerminationSource.INITIALIZER,
                    step_index=0,
                    time_s=0.0,
                    extensions=extensions,
                )
            )
        if self.limits is not None and policy_dt > 0.0:
            timeout_steps = self.limits.validate_against(policy_dt)
            # deadline fires EXACTLY at the declared tick; an overrun (length
            # > timeout_steps) did not end through the deadline authority and
            # must not be stamped as one.
            if timeout_steps is not None and length == timeout_steps:
                terms.append(
                    Termination(
                        outcome=EpisodeOutcome.DEADLINE,
                        code="runtime.task_deadline",
                        source=TerminationSource.RUNTIME,
                        step_index=length,
                        time_s=time_s,
                        metrics={"timeout_steps": float(timeout_steps)},
                    )
                )
        return terms

    def on_episode_end(self, env_ids, bundle: dict) -> None:
        record = bundle["record"]
        terminated = bundle["terminated"]  # (N,) mask over ALL envs
        reset_facts = bundle.get("reset_facts") or {}
        policy_dt = float(self.meta.get("policy_dt") or 0.0)
        sim_dt = float(self.meta.get("sim_dt") or 0.0)
        # per-env read-only result views for the sampler hook: successful
        # tasks only, in task declaration order (a raising task is omitted
        # exactly as it is omitted from the report)
        episode_results: dict[int, tuple[EpisodeResultView, ...]] = {}
        for ei in env_ids.tolist():
            steps = self.ep_bufs[ei]
            self.ep_bufs[ei] = []
            rewinds, self._ep_rewinds[ei] = self._ep_rewinds[ei], []
            self.n_episodes += 1
            episode_results[int(ei)] = ()
            if not steps:
                continue
            length = len(steps)
            reset = dict(reset_facts.get(int(ei)) or {})
            times = timeline_time_s(length, policy_dt) if policy_dt > 0.0 else None
            core_terms = self._episode_core_terminations(int(ei), length, bundle, reset)
            episode_termination = (
                EpisodeTermination.from_terminations(core_terms) if core_terms else None
            )
            ep = EpisodeData(
                data=assemble_episode(steps),
                env_id=int(ei),
                episode_id=int(record["episode_id"][ei]),
                length=length,
                duration_s=(
                    float(times[-1])
                    if times is not None and len(times)
                    else float(record["time_in_episode"][ei])
                ),
                # index by env id -- terminated is a full (N,) mask, NOT a
                # per-due vector even when len(env_ids) == num_envs
                ended_by_termination=bool(terminated[ei]),
                rewound=bool(rewinds),
                policy_dt=policy_dt,
                sim_dt=sim_dt,
                reset_index=0,
                reset_sample=reset.get("reset_sample"),
                reset_valid=reset.get("reset_valid"),
                reset_reason=reset.get("reset_reason"),
                termination=episode_termination,
                terminations=(() if episode_termination is None else episode_termination.all),
            )
            self.episodes.append(ep)
            # directed labelling: the Director's per-env mark is
            # still live here (this sink runs before the director's
            # episode-end snapshot in the fan-out)
            directed = bool(
                self._director is not None and self._director.is_episode_directed(int(ei))
            )
            self.episode_directed.append(directed)
            # bridge metrics feed: PRIMARY task (first) supplies the
            # episode's reward/success slot (exact alignment with the env
            # order the engine hands to every sink, bridge included)
            primary = None
            task_terms: list[Termination] = []
            views: list[EpisodeResultView] = []
            docs_start = len(self.episode_docs)
            for task in self.tasks:
                try:
                    detail = getattr(task, "score_detail", None)
                    if detail is not None:
                        result, leaves, frames = detail(ep, retain_frames=self.retain_frames)
                    else:
                        result, leaves, frames = task.score(ep), {}, {}
                    if not isinstance(result, EpisodeResult):
                        raise TypeError(
                            f"task '{task.name}' returned {type(result).__name__}, "
                            "expected EpisodeResult"
                        )
                    if result.termination is not None:
                        if not isinstance(result.termination, Termination):
                            raise TypeError(
                                f"task '{task.name}' returned a non-Termination "
                                f"termination ({type(result.termination).__name__})"
                            )
                        task_terms.append(result.termination)
                    self.results[task.name].append(result)
                    views.append(
                        EpisodeResultView(
                            task=task.name,
                            success=result.success,
                            metrics=result.metrics,
                        )
                    )
                    if primary is None:
                        primary = result
                    # results episode doc (engine's own
                    # verdicts -- analyze/probe render, never re-compute)
                    self.episode_docs.append(
                        {
                            "env": int(ei),
                            "episode_id": ep.episode_id,
                            "verdict": _verdict(result.success, ep.ended_by_termination),
                            "gate": task.name,
                            "predicates": dict(leaves),
                            "metrics": {k: float(v) for k, v in result.metrics.items()},
                            "rewound": ep.rewound,
                            **({"frames": frames} if frames else {}),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 -- isolate tasks
                    print(
                        f"[benchmark] task '{task.name}' failed on "
                        f"env{ei}/ep{ep.episode_id}: {exc!r} (skipped)",
                        flush=True,
                    )
            # task-declared terminations participate in core precedence;
            # extensions stay JSON-only in the per-episode document.
            episode_results[int(ei)] = tuple(views)
            all_terms = list(core_terms) + task_terms
            final_termination = (
                EpisodeTermination.from_terminations(all_terms) if all_terms else None
            )
            # Contract gate: EVERY recorded termination (including a task's
            # declared Termination) must live on the episode-local axis.  A
            # global run tick here would silently corrupt terminal_index and
            # the report; reject it instead of normalizing.
            if final_termination is not None:
                for term in final_termination.all:
                    step = term.step_index
                    if step is not None and not 0 <= step <= length:
                        raise ValueError(
                            "benchmark episode contract: termination "
                            f"{term.code!r} step_index={step} is outside the "
                            f"episode-local axis 0..{length} (global run ticks "
                            "are not episode ticks)"
                        )
            ep.termination = final_termination
            ep.terminations = () if final_termination is None else final_termination.all
            ep.terminal_index = (
                None if final_termination is None else final_termination.primary.step_index
            )
            termination_doc = None if final_termination is None else final_termination.to_json()
            for doc in self.episode_docs[docs_start:]:
                doc["termination"] = termination_doc
                doc["reset"] = {
                    "index": 0,
                    "valid": ep.reset_valid,
                    "reason": ep.reset_reason,
                    "sample_keys": ([] if ep.reset_sample is None else sorted(ep.reset_sample)),
                }
            self._episode_terminations.append(
                {
                    "env": int(ei),
                    "episode_id": ep.episode_id,
                    "primary": (
                        None if final_termination is None else final_termination.primary.to_json()
                    ),
                    "all": (
                        []
                        if final_termination is None
                        else [t.to_json() for t in final_termination.all]
                    ),
                }
            )
            self._terminal_primary.append(
                None if final_termination is None else final_termination.primary
            )
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
    def _report_body(self, wall: float | None) -> dict:
        """The report document (on_run_end shape; draft when wall is None)."""
        tasks_report = {}
        for task in self.tasks:
            results = self.results[task.name]
            agg = task.aggregate(results)
            tasks_report[task.name] = {
                "params": dict(task.params),
                "n_episodes": agg.get("n_episodes", len(results)),
                "metrics": {k: v for k, v in agg.items() if isinstance(v, dict)},
                "success_rate": agg.get("success_rate"),
            }
        run_identity = None if self.run_identity is None else self.run_identity.to_json()
        contacts = self.meta.get("contacts")
        return {
            "schema": REPORT_SCHEMA,
            "run_name": self.meta.get("run_name"),
            "meta": {
                "run": run_identity,
                "joint_names_count": len(self.meta.get("joint_names") or []),
                "policy_dt": self.meta.get("policy_dt"),
                "sim_dt": self.meta.get("sim_dt"),
                "seed": self.meta.get("seed"),
                **({"contacts": contacts} if contacts is not None else {}),
            },
            "tasks": tasks_report,
            "episodes_total": self.n_episodes,
            "steps_total": self.steps_total,
            # benchmark contract terminal facts (per-episode extensions preserved
            # in ``episode_terminations``; no cross-namespace aggregation)
            "terminal_outcome_counts": outcome_counts(self._terminal_primary),
            "terminal_code_counts": code_counts(self._terminal_primary),
            "terminal_source_counts": source_counts(self._terminal_primary),
            "episode_terminations": list(self._episode_terminations),
            # directed boundary (contract §7): per-episode flags +
            # the aggregate; directed successes are fixture validations,
            # never policy capability
            "directed": any(self.episode_directed),
            "episodes_directed": [
                {"episode_id": ep.episode_id, "env_id": ep.env_id, "directed": d}
                for ep, d in zip(self.episodes, self.episode_directed)
            ],
            # rewind boundary (Round B B1, benchmark_contract §4): episodes
            # that straddled a timeline fork keep ALL forked data and are
            # labelled -- rewound successes never mix into physics-legal
            # claims either
            "rewound": any(ep.rewound for ep in self.episodes),
            "episodes_rewound": [
                {"episode_id": ep.episode_id, "env_id": ep.env_id, "rewound": ep.rewound}
                for ep in self.episodes
            ],
            "wall_time_s": wall,
        }

    def on_run_end(self) -> dict:
        report = self._report_body(wall=time.time() - self._t0)
        parent = os.path.dirname(self.report_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        self._write_results()
        print(f"[benchmark] report -> {self.report_path}", flush=True)
        for name, entry in report["tasks"].items():
            print(
                f"[benchmark]   {name}: n={entry['n_episodes']} "
                f"success_rate={entry['success_rate']}",
                flush=True,
            )
        return report

    # ------------------------------------------------------------------ #
    # results.json / metrics.npz (benchmark_contract §results)
    # ------------------------------------------------------------------ #
    def results_document(self) -> dict:
        """The results document (engine verdicts; zero re-derivation)."""
        episodes = []
        for doc in self.episode_docs:
            episodes.append({k: v for k, v in doc.items() if k != "frames"})
        backend = self.meta.get("backend")
        return {
            "schema": RESULTS_SCHEMA,
            # complete run identity even for benchmark-only runs without
            # meta.json (no pack fills these fields manually).
            "run": (None if self.run_identity is None else self.run_identity.to_json()),
            "run_dir": os.path.dirname(self.report_path) or ".",
            "backend": backend,
            "episodes": episodes,
            "source": "engine",
            "terminal_outcome_counts": outcome_counts(self._terminal_primary),
            "terminal_code_counts": code_counts(self._terminal_primary),
            "terminal_source_counts": source_counts(self._terminal_primary),
        }

    def _write_results(self) -> None:
        parent = os.path.dirname(self.report_path)
        if not parent or not self.episode_docs:
            return  # nothing scored
        doc = self.results_document()
        with open(os.path.join(parent, RESULTS_JSON), "w") as f:
            json.dump(doc, f, indent=1, ensure_ascii=False)
        frames = {
            f"metric/e{i}/{mid}": fr
            for i, d in enumerate(self.episode_docs)
            for mid, fr in (d.get("frames") or {}).items()
        }
        if frames:
            np.savez(os.path.join(parent, METRICS_NPZ), **frames)
            print(
                f"[benchmark] results -> {RESULTS_JSON} + {METRICS_NPZ} "
                f"({len(frames)} frame series)",
                flush=True,
            )
        else:
            print(f"[benchmark] results -> {RESULTS_JSON}", flush=True)

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
            "by_task": {t.name: len(self.results[t.name]) for t in self.tasks},
        }

    def metrics_report(self) -> dict | None:
        """Draft report (same shape as on_run_end, wall_time_s=None).

        The per-episode ``episodes_directed`` / ``episodes_rewound``
        detail is OMITTED here
        (the final artifact keeps it): it grows one entry per episode,
        and the typed-query ack block clips at PAYLOAD_JSON (8KB) -- a
        ~130+-episode draft used to arrive unparseable, which the query
        face answers as ``{"report": null}`` mid-run.  The draft is the
        LIVE view's data source; its aggregate face is bounded by
        construction (per-task stats only).
        """
        if not self.tasks:
            return None
        body = self._report_body(wall=None)
        body.pop("episodes_directed", None)
        body.pop("episodes_rewound", None)
        body.pop("episode_terminations", None)
        return body
