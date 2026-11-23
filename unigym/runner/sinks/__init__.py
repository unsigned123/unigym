"""Sinks: consumers of the SyncRunner callback stream."""

from __future__ import annotations

from .collect import CollectSink
from .fanout import FanOutSink

__all__ = ["CollectSink", "FanOutSink"]
