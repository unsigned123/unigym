"""Benchmark tasks: episode scoring/registry (B1).

numpy-only, zero isaaclab -- offline unit-testable.  Concrete tasks live in
``tasks.py`` and self-register via ``@register`` on import.
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

__all__ = [
    "TASK_REGISTRY",
    "EpisodeData",
    "EpisodeResult",
    "Task",
    "build_tasks",
    "register",
]

# import side-effect: populate TASK_REGISTRY with the first-party tasks
from . import tasks as _tasks  # noqa: E402,F401  (stand, velocity_tracking)
