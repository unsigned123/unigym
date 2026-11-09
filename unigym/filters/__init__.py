"""Episode filter registry + chain builder (numpy-only, unit-testable).

Yaml usage::

    filters:
      - name: base_contact
        params: {max_fallen_ratio: 0.1, height_threshold: 0.15}
      - name: velocity
        params: {min_abs_vx: 0.1}

Filters are applied in list order; the first ``False`` discards the episode.
"""
from __future__ import annotations

import inspect

from .base import EpisodeFilter
from .base_contact import FallenFilter, TerminationFilter
from .velocity import VelocityRangeFilter

FILTER_REGISTRY = {
    "base_contact": FallenFilter,
    "fallen": FallenFilter,            # alias
    "termination": TerminationFilter,  # generic (terminations-based)
    "velocity": VelocityRangeFilter,
}


def build_filter_chain(filters_cfg: list | None,
                       articulations=None) -> list[EpisodeFilter]:
    """Build an ordered list of filters from the yaml ``filters`` section.

    *articulations* is the run's ordered policy-binding names.  When exactly
    one binding exists, filters that accept an ``articulation`` param get it
    injected (the cardinality convenience, not a primary).  Returns an
    empty list when *filters_cfg* is None or empty (no filtering).
    """
    if not filters_cfg:
        return []
    single = None
    if articulations is not None and len(list(articulations)) == 1:
        single = list(articulations)[0]
    chain: list[EpisodeFilter] = []
    for entry in filters_cfg:
        name = entry["name"]
        if name not in FILTER_REGISTRY:
            raise ValueError(
                f"unknown filter '{name}'; available: {sorted(FILTER_REGISTRY)}")
        params = dict(entry.get("params", {}))
        cls = FILTER_REGISTRY[name]
        if single is not None and "articulation" not in params \
                and "articulation" in inspect.signature(cls.__init__).parameters:
            params["articulation"] = single
        chain.append(cls(**params))
    return chain


def run_filter_chain(chain: list[EpisodeFilter],
                     episode_data: dict) -> tuple[bool, str | None]:
    """Run all filters in order.

    Returns (keep, rejected_by).  *rejected_by* is the name of the first
    filter that returned False, or None if all passed.
    """
    for f in chain:
        if not f(episode_data):
            return False, f.name
    return True, None


__all__ = [
    "EpisodeFilter",
    "FallenFilter",
    "TerminationFilter",
    "VelocityRangeFilter",
    "FILTER_REGISTRY",
    "build_filter_chain",
    "run_filter_chain",
]
