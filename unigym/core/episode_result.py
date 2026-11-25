"""Read-only episode result view handed to command samplers.

The runner's episode boundary exposes completed-episode results published by a
sink to a :class:`~unigym.commands.base.CommandSampler`, so user-owned
multi-stage command logic can react to the previous episode.  This module owns
that view: a pure, frozen snapshot of one result with no task object, no
episode history and no numpy arrays.  It deliberately lives in ``unigym.core``
so the command layer never depends on a sink implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class EpisodeResultView:
    """One result for one completed episode.

    Attributes:
        task: the result's configured name (``<pack>/<name>`` or bare).
        success: the judged verdict, or ``None`` when no success criterion is
            defined.  Never inferred from the engine termination.
        metrics: the result's flat scalar metrics, copied into a read-only
            mapping at construction.
    """

    task: str
    success: bool | None
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType({str(k): float(v) for k, v in self.metrics.items()}),
        )
