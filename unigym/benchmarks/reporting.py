"""Benchmark reporting: reporter vocabulary and the benchmark reporter.

This module owns the typed handoff between the episode evaluator and the
run-level reporter (:class:`TaskScore`, :class:`EpisodeEvaluation`,
:class:`EpisodeReporter`) and the benchmark-specific implementation
(:class:`BenchmarkReporter`).  The evaluator produces one evaluation per
episode; the reporter owns run aggregation, the benchmark report schema and
the ``benchmark_report.json`` / ``results.json`` / ``metrics.npz`` artifacts.

The reporter never re-judges a task: it records the engine's verdicts and
delegates run-level reduction to :meth:`Task.aggregate`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from unigym.core.episode_result import EpisodeResultView
from unigym.core.episode_termination import (
    EpisodeTermination,
    Termination,
    code_counts,
    outcome_counts,
    source_counts,
)

from .base import EpisodeData, EpisodeResult, Task
from .episode import RunIdentity

__all__ = [
    "EpisodeEvaluation",
    "EpisodeReporter",
    "METRICS_NPZ",
    "REPORT_SCHEMA",
    "RESULTS_JSON",
    "RESULTS_SCHEMA",
    "BenchmarkReporter",
    "TaskScore",
]

REPORT_SCHEMA = "benchmark_report"
RESULTS_SCHEMA = "results"
METRICS_NPZ = "metrics.npz"
RESULTS_JSON = "results.json"


@dataclass(frozen=True, slots=True)
class TaskScore:
    """One successful task's episode result and declaration detail.

    Attributes:
        task: the task's configured report key.
        result: the task-judged ``EpisodeResult`` (never re-derived).
        leaves: the task's predicate leaves (empty for a plain task).
        frames: the task's retained per-frame metric series (empty when the
            task kept none).
    """

    task: str
    result: EpisodeResult
    leaves: Mapping[str, bool]
    frames: Mapping[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class EpisodeEvaluation:
    """One completed episode and the successful task scores for it.

    Attributes:
        episode: the assembled, fully merged ``EpisodeData`` (core and task
            terminations applied).
        scores: successful task scores in task declaration order; a task that
            raised or returned a wrong type is omitted.
        views: read-only ``EpisodeResultView`` for the sampler hook, aligned
            with ``scores``.
        termination: the final merged ``EpisodeTermination`` (core and task
            facts under one precedence), or ``None`` when nothing terminated.
    """

    episode: EpisodeData
    scores: tuple[TaskScore, ...]
    views: tuple[EpisodeResultView, ...]
    termination: EpisodeTermination | None


@runtime_checkable
class EpisodeReporter(Protocol):
    """Run-level reporter fed by the evaluation session.

    The reporter owns report/results schema and run aggregation; it never
    re-judges a task.  ``on_episode`` receives the final evaluation plus the
    director label observed at the same boundary.
    """

    def on_episode(self, evaluation: EpisodeEvaluation, *, directed: bool) -> None:
        """Record one completed episode evaluation and its director label."""
        ...

    def note_progress(self, *, episodes_total: int, steps_total: int) -> None:
        """Publish the live run counters for the bounded draft report."""
        ...

    def draft_report(self) -> dict:
        """Return the bounded live report document."""
        ...

    def finish(self, *, episodes_total: int, steps_total: int, wall_time_s: float) -> dict:
        """Finalize and return the run report document."""
        ...


def _verdict(success, ended_by_termination: bool) -> str:
    """Return the results verdict for one task result.

    The task is the judge (analyze_contract §6): success verdicts win over
    the observational termination flag.
    """
    if success is True:
        return "success"
    if success is False:
        return "fail"
    return "fell" if ended_by_termination else "na"


class BenchmarkReporter:
    """Own the benchmark report, results and metrics artifacts for one run.

    The reporter accumulates the per-episode evaluations it is given and
    writes the run artifact at :meth:`finish`.  It never scores a task and
    never re-derives a verdict.

    Args:
        tasks: the run's instantiated tasks in declaration order (aggregation
            and report key order).
        meta: run-level context embedded in the report (``run_name``,
            ``policy_dt``, ``sim_dt``, ``seed``, ``joint_names``,
            ``num_envs``, optional ``contacts``/``backend``).
        report_path: where ``benchmark_report.json`` is written; the results
            artifacts are written next to it.
        run_identity: optional complete run identity embedded in both
            artifacts.
    """

    def __init__(
        self,
        *,
        tasks: list[Task],
        meta: dict,
        report_path: str,
        run_identity: RunIdentity | None = None,
    ) -> None:
        self.tasks = list(tasks)
        self.meta = dict(meta)
        self.report_path = str(report_path)
        self.run_identity = run_identity
        # scored episodes (for the directed/rewound documents) and per-task
        # verdicts (run aggregation)
        self.episodes: list[EpisodeData] = []
        self.results: dict[str, list[EpisodeResult]] = {t.name: [] for t in self.tasks}
        # per-(episode, task) detail docs, results.json face
        self.episode_docs: list[dict] = []
        # directed/rewound documents; parallel to ``episodes`` where noted
        self.episode_directed: list[bool] = []
        # per-episode structured termination docs (benchmark report)
        self._episode_terminations: list[dict] = []
        self._terminal_primary: list[Termination | None] = []
        # live run counters for the bounded draft; the authoritative totals are
        # passed to ``finish`` at run end
        self._episodes_total = 0
        self._steps_total = 0

    # ------------------------------------------------------------------ #
    # live run progress (draft telemetry only)
    # ------------------------------------------------------------------ #
    def note_progress(self, *, episodes_total: int, steps_total: int) -> None:
        """Publish the run's live counters for :meth:`draft_report`.

        The final artifact uses the authoritative totals passed to
        :meth:`finish`; this feed keeps the bounded live draft aligned with
        the run that is currently in flight.
        """
        self._episodes_total = int(episodes_total)
        self._steps_total = int(steps_total)

    # ------------------------------------------------------------------ #
    # episode accumulation (one pass; no post-hoc patching)
    # ------------------------------------------------------------------ #
    def on_episode(self, evaluation: EpisodeEvaluation, *, directed: bool) -> None:
        """Record one completed evaluation and build its result documents."""
        ep = evaluation.episode
        final_termination = evaluation.termination
        termination_doc = None if final_termination is None else final_termination.to_json()
        reset_doc = {
            "index": 0,
            "valid": ep.reset_valid,
            "reason": ep.reset_reason,
            "sample_keys": ([] if ep.reset_sample is None else sorted(ep.reset_sample)),
        }
        self.episodes.append(ep)
        self.episode_directed.append(bool(directed))
        # results episode docs (engine's own verdicts -- analyze/probe render,
        # never re-compute); declaration order, successful tasks only.
        for score in evaluation.scores:
            self.results[score.task].append(score.result)
            self.episode_docs.append(
                {
                    "env": ep.env_id,
                    "episode_id": ep.episode_id,
                    "verdict": _verdict(score.result.success, ep.ended_by_termination),
                    "gate": score.task,
                    "predicates": dict(score.leaves),
                    "metrics": {k: float(v) for k, v in score.result.metrics.items()},
                    "rewound": ep.rewound,
                    **({"frames": score.frames} if score.frames else {}),
                    "termination": termination_doc,
                    "reset": reset_doc,
                }
            )
        self._episode_terminations.append(
            {
                "env": ep.env_id,
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

    # ------------------------------------------------------------------ #
    # run report / draft
    # ------------------------------------------------------------------ #
    def _report_body(self, wall: float | None, *, episodes_total: int, steps_total: int) -> dict:
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
            "episodes_total": episodes_total,
            "steps_total": steps_total,
            # benchmark contract terminal facts (per-episode extensions
            # preserved in ``episode_terminations``; no cross-namespace
            # aggregation)
            "terminal_outcome_counts": outcome_counts(self._terminal_primary),
            "terminal_code_counts": code_counts(self._terminal_primary),
            "terminal_source_counts": source_counts(self._terminal_primary),
            "episode_terminations": list(self._episode_terminations),
            # directed boundary (contract §7): per-episode flags + the
            # aggregate; directed successes are fixture validations, never
            # policy capability
            "directed": any(self.episode_directed),
            "episodes_directed": [
                {"episode_id": ep.episode_id, "env_id": ep.env_id, "directed": d}
                for ep, d in zip(self.episodes, self.episode_directed)
            ],
            # rewind boundary (benchmark_contract §4): episodes that straddled
            # a timeline fork keep ALL forked data and are labelled -- rewound
            # successes never mix into physics-legal claims either
            "rewound": any(ep.rewound for ep in self.episodes),
            "episodes_rewound": [
                {"episode_id": ep.episode_id, "env_id": ep.env_id, "rewound": ep.rewound}
                for ep in self.episodes
            ],
            "wall_time_s": wall,
        }

    def draft_report(self) -> dict:
        """Draft report (same shape as on_run_end, wall_time_s=None).

        The per-episode ``episodes_directed`` / ``episodes_rewound`` /
        ``episode_terminations`` detail is OMITTED here (the final artifact
        keeps it): it grows one entry per episode, and the typed-query ack
        block clips at PAYLOAD_JSON (8KB) -- a ~130+-episode draft used to
        arrive unparseable, which the query face answers as ``{"report":
        null}`` mid-run.  The draft is the LIVE view's data source; its
        aggregate face is bounded by construction (per-task stats only).
        """
        body = self._report_body(
            wall=None,
            episodes_total=self._episodes_total,
            steps_total=self._steps_total,
        )
        body.pop("episodes_directed", None)
        body.pop("episodes_rewound", None)
        body.pop("episode_terminations", None)
        return body

    def finish(self, *, episodes_total: int, steps_total: int, wall_time_s: float) -> dict:
        """Aggregate, write report/results/metrics and return the report dict."""
        report = self._report_body(
            wall=wall_time_s,
            episodes_total=episodes_total,
            steps_total=steps_total,
        )
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
