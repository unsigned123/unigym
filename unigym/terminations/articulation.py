"""Explicit per-articulation termination base.

A term that judges exactly ONE articulation's state declares that ownership
here: ``articulation`` is a required param, with no single-binding injection
and no flat single-robot fallback.  ``Termination`` itself stays a pure
condition interface with no articulation knowledge.
"""

from __future__ import annotations

from typing import Mapping

from .base import Termination, TerminationState


class ArticulationTermination(Termination):
    """A condition over exactly one articulation's state.

    params:
        articulation: REQUIRED articulation name; the term reads
            ``articulation/<name>/...`` from :class:`TerminationState`.
    """

    requires_articulation = True

    def __init__(self, params: Mapping | None = None, name: str | None = None):
        super().__init__(params, name)
        articulation = self.params.get("articulation")
        if not articulation:
            raise ValueError(
                f"{type(self).__name__} requires explicit 'articulation' "
                f"(no single-binding injection)"
            )
        self.articulation = str(articulation)

    def source(self, state: TerminationState) -> Mapping:
        """This term's articulation state map (``articulation/<name>/...``)."""
        return state.articulation(self.articulation, term=type(self).__name__)
