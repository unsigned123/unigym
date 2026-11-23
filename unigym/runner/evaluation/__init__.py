"""Evaluation pipeline building blocks for runner sinks.

The package-level public exports are intentionally buffer-only:
``EpisodeBuffer``, ``TakenEpisode`` and ``TickSample``.  The buffer imports no
benchmark type, so importing ``unigym.runner.evaluation`` does not pull the
benchmark or reporting layer.

The other collaborators are imported by their owning module path, not
re-exported here: ``unigym.runner.evaluation.session.EvaluationSession`` (the
public runner sink), ``unigym.runner.evaluation.scoring.EpisodeEvaluator`` and
``unigym.benchmarks.reporting.EpisodeReporter`` / ``BenchmarkReporter``.
"""

from __future__ import annotations

from .buffer import EpisodeBuffer, TakenEpisode, TickSample

__all__ = ["EpisodeBuffer", "TakenEpisode", "TickSample"]
