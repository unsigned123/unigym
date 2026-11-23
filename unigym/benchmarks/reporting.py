"""Benchmark reporting vocabulary shared by the evaluator and reporters.

This module owns the typed handoff between the episode evaluator and a future
run-level reporter: :class:`TaskScore` is one successful task's result plus its
declaration detail, :class:`EpisodeEvaluation` is the full per-episode verdict,
and :class:`EpisodeReporter` is the run-level sink protocol.

Only the types and the protocol live here.  The concrete benchmark reporter
(JSON schema, aggregation, file writing) is a separate implementation so the
evaluator never owns report policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from unigym.core.episode_result import EpisodeResultView

from .base import EpisodeData, EpisodeResult
from .termination import EpisodeTermination

__all__ = ["EpisodeEvaluation", "EpisodeReporter", "TaskScore"]


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

    def draft_report(self) -> dict:
        """Return the bounded live report document."""
        ...

    def finish(self, *, episodes_total: int, steps_total: int, wall_time_s: float) -> dict:
        """Finalize and return the run report document."""
        ...
