"""Freeze face (v056 freeze paradigm): skill halt -> world freeze.

The paradigm (design ``docs/designs/v056-freeze-paradigm.md``): the
agent thinks while the world stands still -- world time advances only
through explicit grants (``drive`` a skill / ``step`` a batch /
``time.resume``), and every skill halt is a freeze point.

This module is the RUNTIME half of the mount: it hands the adapter the
bridge's engine-thread park hook so the halt path can freeze the world
directly.  The path is deliberately IN-PROCESS (``bridge.pause_for_halt``,
a flag write + gate park), not ``DaemonClient.pause()``: the halt fires
on the engine thread inside ``adapter.act()`` -- an HTTP pause from
there would block the very thread that must write the command ack (a
guaranteed timeout-stall per drive), while the direct hook freezes at
the halt tick's own step boundary with zero latency and works on
daemonless runs too.

Capability-gated: adapters without ``set_freeze_hook`` (every non-agent
adapter) mount nothing -- zero behavior change.
"""
from __future__ import annotations

__all__ = ["attach"]


def attach(bridge, adapter) -> None:
    """Mount the halt-freeze hook on a live bridge (collect_cli wiring).

    No-op for adapters that do not declare the face (explicit duck-typed
    capability, same discipline as the rewind face's ``rewindable``).
    """
    hook = getattr(adapter, "set_freeze_hook", None)
    if callable(hook):
        hook(bridge.pause_for_halt)
