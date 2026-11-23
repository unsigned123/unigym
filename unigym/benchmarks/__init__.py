"""Benchmark tasks: live-first episode scoring/registry.

Tasks receive the episode history plus a read-only per-env ``EvalContext`` and
implement ``score(self, ep, ctx)``; an overriding task also implements
``score_detail(self, ep, ctx, *, retain_frames=False)``.  ``build_tasks``
validates those signatures at build time, so a legacy ``score(ep)`` task fails
loudly instead of being silently skipped by the evaluation session.

A task opts into per-policy-tick detection with ``trigger: both`` and an
overridden ``on_tick(self, ctx)``; its ``TickTrace`` history reaches ``score``
through ``EpisodeData.ticks`` and never appears in the report.  ``TickResult``
and ``TickTrace`` are the intentional public tick types; the raw per-step
samples stay internal to ``unigym.runner.evaluation``.  ``EpisodeData`` remains
the episode history/timeline/metrics payload.  A report is a live run
observation: there is no guarantee that a stored episode can be scored again
from history alone.  The module stays importable without torch/isaaclab (the
context type is a type-checking-only import).  Concrete tasks live in
``tasks.py`` and self-register via ``@register`` on import.

Ownership: ``EpisodeBuffer`` (runner.evaluation) owns open-episode history,
``EpisodeEvaluator`` (runner.evaluation) assembles ``EpisodeData`` and invokes
``Task.score``, ``EvaluationSession`` (runner.evaluation) owns the runner sink
lifecycle, and ``BenchmarkReporter`` (``unigym.benchmarks.reporting``) owns the
run-level report/results schema.  There is no ``BenchmarkSink``.
"""

from __future__ import annotations

from unigym.core.episode_termination import (
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationExtension,
    TerminationSource,
)

from . import termination as termination
from .base import (
    TASK_REGISTRY,
    EpisodeData,
    EpisodeResult,
    Task,
    TickResult,
    TickTrace,
    build_tasks,
    register,
)
from .episode import (
    EpisodeLimits,
    RobotIdentity,
    RunIdentity,
    timeline_step_index,
    timeline_time_s,
)

__all__ = [
    "TASK_REGISTRY",
    "EpisodeData",
    "EpisodeLimits",
    "EpisodeOutcome",
    "EpisodeResult",
    "EpisodeTermination",
    "RobotIdentity",
    "RunIdentity",
    "Task",
    "Termination",
    "TerminationExtension",
    "TerminationSource",
    "TickResult",
    "TickTrace",
    "build_tasks",
    "register",
    "timeline_step_index",
    "timeline_time_s",
]

# import side-effect: populate TASK_REGISTRY with the first-party tasks
from . import tasks as _tasks  # noqa: E402,F401  (stand, velocity_tracking)
