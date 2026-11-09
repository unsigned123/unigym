"""watchers -- programmable run observers.

Layout (mirror of unigym/console/ -- logic lives here, the bridge only
hosts the hook; docs/contracts/watchers_contract.md is the contract
twin):

* ``base.py``       WatchContext / WatchResult / Level / WatchError
                    (§8.2/§8.4; key face = trace contract key face).
* ``templates.py``  built-in predicate templates (tier 1, first-class
                    per §8.1; registry = contract §5).
* ``engine.py``     WatcherEngine (per-tick evaluation before the step
                    gate, trigger layer, fail-safe, actions, store).
* ``timeline.py``   EventLog + TimelineSink (§3 events.jsonl).
* this module       yaml loading (born-armed entry, §8.6) + the
                    :func:`attach` wiring used by collect_cli.

Bridge touchpoints stay minimal: one attribute (``bridge._watchers``),
one tick hook call, one command registration (cmd 15 "watch") -- the
attach function below does all three.
"""

from __future__ import annotations

__all__ = [
    "WatcherEngine",
    "WatchSpec",
    "WatchError",
    "WatchResult",
    "WatchContext",
    "Level",
    "EventLog",
    "TimelineSink",
    "read_tail",
    "load_watch_yaml",
    "attach",
    "TEMPLATE_REGISTRY",
]

from .base import Level, WatchContext, WatchError, WatchResult
from .engine import WatcherEngine
from .spec import WatchSpec
from .templates import TEMPLATE_REGISTRY
from .timeline import EventLog, TimelineSink, read_tail


def load_watch_yaml(path) -> list[dict]:
    """``--watch watchers.yaml`` -> spec dicts (born-armed, §8.6).

    Accepts ``{"watchers": [...]}`` or a bare list; the file is recorded
    verbatim in the run manifest for reproducibility.
    """
    import yaml

    p = __import__("pathlib").Path(path)
    if not p.is_file():
        raise WatchError(f"--watch file not found: {p}")
    try:
        doc = yaml.safe_load(p.read_text())
    except (OSError, ValueError) as e:
        raise WatchError(f"--watch file unreadable: {e}") from None
    if isinstance(doc, dict):
        doc = doc.get("watchers")
    if not isinstance(doc, list) or not doc:
        raise WatchError(
            f"--watch {p}: expected a non-empty 'watchers:' list, got " f"{type(doc).__name__}"
        )
    return doc


def attach(bridge, env=None, adapter=None, out_dir=None, event_log=None, specs=None):
    """Mount the WatcherEngine on a live bridge (collect_cli wiring).

    * ``bridge._watchers`` -- the per-tick hook target (evaluated in
      ``Bridge.tick`` BEFORE ``_gate.arrive()``: trigger -> pause parks
      the same tick).
    * command "watch" (cmd 15) -> ``engine.handle`` (add/list/clear/
      update/test from the CLI through the daemon).
    * ``Bridge.on_episode_end`` -> callable ``on_reset`` hooks (§8.3).
    """
    engine = WatcherEngine(
        run_id=bridge.run_id,
        out_dir=out_dir,
        env=env,
        adapter=adapter,
        event_log=event_log,
        bridge=bridge,
        specs=specs,
    )
    bridge._watchers = engine
    bridge.register_command("watch", engine.handle)
    n = len(engine._specs)
    print(
        f"[watcher] engine mounted: {n} watcher(s) born-armed " f"({sorted(engine._specs) or '-'})",
        flush=True,
    )
    return engine
