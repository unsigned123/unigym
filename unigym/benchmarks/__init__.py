"""Benchmark tasks: live-first episode scoring/registry (B1).

Tasks receive the episode history plus a read-only per-env ``EvalContext``.
The module stays importable without torch/isaaclab (the context type is a
type-checking-only import).  Concrete tasks live in ``tasks.py`` and
self-register via ``@register`` on import.
"""

from __future__ import annotations

from .base import (
    TASK_REGISTRY,
    EpisodeData,
    EpisodeResult,
    Task,
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
from .termination import (
    EpisodeOutcome,
    EpisodeTermination,
    Termination,
    TerminationExtension,
    TerminationSource,
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
    "build_tasks",
    "register",
    "timeline_step_index",
    "timeline_time_s",
]

# import side-effect: populate TASK_REGISTRY with the first-party tasks
from . import tasks as _tasks  # noqa: E402,F401  (stand, velocity_tracking)
