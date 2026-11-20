"""Read-only benchmark result view handed to command samplers.

The runner's episode boundary exposes completed-episode benchmark results to a
:class:`~unigym.commands.base.CommandSampler` so user-owned multi-stage command
logic can react to the previous episode.  This module owns that view: a pure,
frozen snapshot of one ``Task`` outcome with no task object, no ``EpisodeData``
and no numpy arrays.  It deliberately lives in ``unigym.core`` so the command
layer never depends on the benchmark layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class EpisodeResultView:
    """One task's result for one completed episode.

    Attributes:
        task: the task's configured name (``<pack>/<task>`` or bare).
        success: the task-judged verdict, or ``None`` when the task defines no
            success criterion.  Never inferred from the engine termination.
        metrics: the task's flat scalar metrics, copied into a read-only
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
