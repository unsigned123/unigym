"""Sinks: consumers of the SyncRunner callback stream.

The benchmark sink is ``unigym.runner.evaluation.session.EvaluationSession``;
it lives with its buffer/evaluator rather than under this package and is not
re-exported here.  These exports stay the collect/fan-out composition helpers.
"""

from __future__ import annotations

from .collect import CollectSink
from .fanout import FanOutSink

__all__ = ["CollectSink", "FanOutSink"]
