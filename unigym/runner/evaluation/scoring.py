"""Episode evaluation: assemble one completed episode and score its tasks.

``EpisodeEvaluator`` owns the benchmark episode-end decision surface: it builds
the :class:`~unigym.benchmarks.base.EpisodeData` from the buffered history and
the runner bundle (data, ids, timeline, reset facts and core terminations),
creates one read-only per-env :class:`~unigym.evaluation.EvalContext`, invokes
every task in declaration order with exception isolation, merges the core and
task terminations under benchmark precedence, and validates the episode-local
axis.  It returns an :class:`~unigym.benchmarks.reporting.EpisodeEvaluation`;
it writes no report and keeps no run-level accumulator.

The evaluator is benchmark-layer: it depends on ``unigym.benchmarks`` and
``unigym.evaluation`` but not on a reporter implementation or the runner sinks.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from unigym.benchmarks.base import EpisodeData, EpisodeResult, Task, TickTrace
from unigym.benchmarks.episode import EpisodeLimits, timeline_time_s
from unigym.benchmarks.reporting import EpisodeEvaluation, TaskScore
from unigym.benchmarks.termination import (
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationExtension,
    TerminationSource,
)
from unigym.core.episode_result import EpisodeResultView
from unigym.evaluation import EnvView, EvalContext
from unigym.query.facade import Query

from ..episodes import assemble_episode
from .buffer import TakenEpisode

if TYPE_CHECKING:
    from unigym.env.backends.base import EnvBackend
    from unigym.env.obs import ObsSnapshot

__all__ = ["EpisodeEvaluator", "build_env_view"]


def build_env_view(
    env: EnvBackend | None,
    *,
    record: Mapping[str, object],
    obs: ObsSnapshot | None,
    env_id: int | None,
    num_envs: int,
) -> EnvView:
    """Build a read-only evaluation view at one runner boundary.

    Production wiring injects ``env``: the view carries that backend's
    device/label/articulations and a lazy read-only query factory.  With
    ``env is None`` (direct unit wiring) the view is explicitly
    record/obs-only and ``query`` is unavailable.
    """
    if env is None:
        return EnvView(
            record=record,
            obs=obs,
            env_id=env_id,
            num_envs=num_envs,
            device="cpu",
            backend="record-only",
        )

    def _query_factory() -> Query:
        query = getattr(env, "query", None)
        if not isinstance(query, Query):
            raise NotImplementedError(f"backend {type(env).__name__!r} has no runtime query")
        return query

    query_factory = (
        _query_factory if inspect.getattr_static(env, "query", None) is not None else None
    )
    return EnvView(
        record=record,
        obs=obs,
        env_id=env_id,
        num_envs=num_envs,
        device=getattr(env, "device", "cpu"),
        backend=str(getattr(env, "_backend_label", None) or type(env).__name__),
        query_factory=query_factory,
        articulations=getattr(env, "articulations", None) or {},
    )


class EpisodeEvaluator:
    """Score one completed episode and return its merged evaluation.

    Args:
        tasks: instantiated ``benchmarks.base.Task`` objects in declaration
            order.
        meta: run-level context (``policy_dt``, ``sim_dt``, ``num_envs``).
        env: the live env backend used only to scope the per-env evaluation
            context; ``None`` is the explicit record/obs-only mode.
        limits: declared episode deadline, or ``None``.
        retain_frames: whether ``score_detail`` retains per-frame metric
            series.
    """

    def __init__(
        self,
        *,
        tasks: list[Task],
        meta: dict,
        env: EnvBackend | None = None,
        limits: EpisodeLimits | None = None,
        retain_frames: bool = False,
    ):
        self.tasks = list(tasks)
        self.meta = dict(meta)
        self._env = env
        self.limits = limits
        self.retain_frames = bool(retain_frames)
        self.num_envs = int(self.meta.get("num_envs", 1))

    def evaluate(
        self,
        taken: TakenEpisode,
        bundle: dict,
        env_id: int,
        *,
        skip_tasks: frozenset[str] = frozenset(),
    ) -> EpisodeEvaluation:
        """Build and score one episode for *env_id*.

        ``taken`` is the buffer's removed history for one env; ``bundle`` is
        the live pre-reset episode-end bundle.  Tasks named in ``skip_tasks``
        are omitted from scoring without affecting the episode facts.
        """
        ei = int(env_id)
        steps = taken.steps
        rewinds = taken.rewinds
        record = bundle["record"]
        terminated = bundle["terminated"]
        reset_facts = bundle.get("reset_facts") or {}
        reset = dict(reset_facts.get(ei) or {})
        policy_dt = float(self.meta.get("policy_dt") or 0.0)
        sim_dt = float(self.meta.get("sim_dt") or 0.0)

        length = len(steps)
        times = timeline_time_s(length, policy_dt) if policy_dt > 0.0 else None
        core_terms = self._episode_core_terminations(ei, length, bundle, reset)
        episode_termination = (
            EpisodeTermination.from_terminations(core_terms) if core_terms else None
        )
        ticks, tick_failed = self._assemble_ticks(taken, ei, skip_tasks)
        ep = EpisodeData(
            data=assemble_episode(steps),
            env_id=ei,
            episode_id=int(record["episode_id"][ei]),
            length=length,
            duration_s=(
                float(times[-1])
                if times is not None and len(times)
                else float(record["time_in_episode"][ei])
            ),
            # index by env id -- terminated is a full (N,) mask, NOT a
            # per-due vector even when the due set equals every env
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
            ticks=ticks,
        )
        scores, views, task_terms = self._score_tasks(
            ep, self._eval_context(ei, bundle), ei, skip_tasks | tick_failed
        )
        # task-declared terminations participate in core precedence;
        # extensions stay JSON-only in the per-episode document.
        all_terms = list(core_terms) + task_terms
        final_termination = EpisodeTermination.from_terminations(all_terms) if all_terms else None
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
        return EpisodeEvaluation(
            episode=ep,
            scores=tuple(scores),
            views=tuple(views),
            termination=final_termination,
        )

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

    def _assemble_ticks(
        self, taken: TakenEpisode, ei: int, skip_tasks: frozenset[str]
    ) -> tuple[Mapping[str, TickTrace] | None, frozenset[str]]:
        """Assemble per-task tick traces for the non-skipped ``both`` tasks.

        Returns ``(ticks, failed)`` where ``ticks`` is ``None`` when the run
        has no ``both`` task at all and ``failed`` names tasks whose buffered
        trace is structurally inconsistent (unstable key set, missing sample,
        wrong shape).  A failed task is omitted from this episode's scores and
        views exactly like a raising task, consistently for every env.
        """
        configured = [task for task in self.tasks if getattr(task, "trigger", "episode") == "both"]
        if not configured:
            return None, frozenset()
        traces: dict[str, TickTrace] = {}
        failed: set[str] = set()
        for task in configured:
            if task.name in skip_tasks:
                continue
            try:
                traces[task.name] = _trace_from_steps(task.name, taken.ticks, len(taken.steps))
            except _TickTraceError as exc:
                failed.add(task.name)
                print(
                    f"[benchmark] task '{task.name}' tick trace invalid on "
                    f"env{ei}: {exc} (skipped)",
                    flush=True,
                )
        return traces, frozenset(failed)

    def _eval_context(self, ei: int, bundle: dict) -> EvalContext:
        """Build the read-only per-env context for one due env."""
        return EvalContext(
            env=build_env_view(
                self._env,
                record=bundle["record"],
                obs=bundle.get("obs"),
                env_id=ei,
                num_envs=self.num_envs,
            )
        )

    def _score_tasks(
        self,
        ep: EpisodeData,
        eval_ctx: EvalContext,
        ei: int,
        skip_tasks: frozenset[str],
    ) -> tuple[list[TaskScore], list[EpisodeResultView], list[Termination]]:
        """Score every task in declaration order, isolating task failures."""
        scores: list[TaskScore] = []
        task_terms: list[Termination] = []
        views: list[EpisodeResultView] = []
        try:
            for task in self.tasks:
                if task.name in skip_tasks:
                    continue
                try:
                    result, leaves, frames = task.score_detail(
                        ep, eval_ctx, retain_frames=self.retain_frames
                    )
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
                    scores.append(
                        TaskScore(
                            task=task.name,
                            result=result,
                            leaves=leaves,
                            frames=frames,
                        )
                    )
                    views.append(
                        EpisodeResultView(
                            task=task.name,
                            success=result.success,
                            metrics=result.metrics,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 -- isolate tasks
                    print(
                        f"[benchmark] task '{task.name}' failed on "
                        f"env{ei}/ep{ep.episode_id}: {exc!r} (skipped)",
                        flush=True,
                    )
        finally:
            eval_ctx.close()
        return scores, views, task_terms


class _TickTraceError(ValueError):
    """A buffered tick trace violates the stable key set / shape contract."""


def _trace_from_steps(task_name: str, tick_steps: list[dict], length: int) -> TickTrace:
    """Stack one task's per-step samples into a ``(T, ...)`` trace.

    The metric key set and the presence of ``fired`` must be stable across
    every buffered step; any drift, missing sample or step-count mismatch
    raises :class:`_TickTraceError` so the caller can fail the task.
    """
    if len(tick_steps) != length:
        raise _TickTraceError(f"buffered tick steps {len(tick_steps)} != episode length {length}")
    if not tick_steps:
        return TickTrace()
    first = tick_steps[0].get(task_name)
    if first is None:
        raise _TickTraceError("missing tick sample on the first step")
    keys = tuple(first.metrics)
    for key in keys:
        if not isinstance(key, str) or not key:
            raise _TickTraceError(f"invalid metric key {key!r}")
    fired_present = first.fired is not None
    values: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    fired_values: list[np.ndarray] = []
    for index, step in enumerate(tick_steps):
        sample = step.get(task_name)
        if sample is None:
            raise _TickTraceError(f"missing tick sample on step {index}")
        if set(sample.metrics) != set(keys):
            raise _TickTraceError(
                f"metric key set drift on step {index}: "
                f"{sorted(sample.metrics)} != {sorted(keys)}"
            )
        if (sample.fired is not None) != fired_present:
            raise _TickTraceError(f"fired presence drift on step {index}")
        for key in keys:
            values[key].append(np.asarray(sample.metrics[key]))
        if fired_present:
            fired_values.append(np.asarray(sample.fired))
    metrics: dict[str, np.ndarray] = {}
    for key, series in values.items():
        try:
            metrics[key] = np.stack(series)
        except ValueError as exc:
            raise _TickTraceError(
                f"metric {key!r} has inconsistent per-step shapes: {exc}"
            ) from exc
    fired = None if not fired_present else np.asarray(fired_values, dtype=bool)
    return TickTrace(metrics=metrics, fired=fired)
