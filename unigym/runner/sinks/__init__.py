"""Sinks: consumers of the SyncRunner callback stream.

These exports are the core collect/fan-out composition helpers.  A custom
sink selected through ``sinks:`` is built by its own factory, not re-exported
here.
"""

from __future__ import annotations

from .collect import CollectSink
from .fanout import FanOutSink

__all__ = ["CollectSink", "FanOutSink"]
