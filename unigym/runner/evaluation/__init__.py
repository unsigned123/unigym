"""Evaluation pipeline building blocks for runner sinks.

The package owns the pieces that turn the runner callback stream into scored
episodes.  This slice exposes the pure episode history buffer; task
evaluation and report ownership arrive in later slices.
"""

from __future__ import annotations

from .buffer import EpisodeBuffer, TakenEpisode, TickSample

__all__ = ["EpisodeBuffer", "TakenEpisode", "TickSample"]
