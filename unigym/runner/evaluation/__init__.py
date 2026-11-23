"""Evaluation pipeline building blocks for runner sinks.

This package exposes the pure episode history buffer.  Task evaluation and the
public sink live in their own modules (``scoring`` and ``session``); importing
``unigym.runner.evaluation`` does not pull the benchmark reporting layer.
"""

from __future__ import annotations

from .buffer import EpisodeBuffer, TakenEpisode, TickSample

__all__ = ["EpisodeBuffer", "TakenEpisode", "TickSample"]
