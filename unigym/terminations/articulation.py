"""Explicit per-articulation termination base.

A term that judges exactly ONE articulation's state declares that ownership
here: ``articulation`` is a required param, with no single-binding injection
and no flat single-robot fallback.  ``Termination`` itself stays a pure
condition interface with no articulation knowledge.
"""

from __future__ import annotations

from typing import Mapping

from unigym.evaluation import EvalContext

from .base import Termination


class ArticulationTermination(Termination):
    """A condition over exactly one articulation's live state.

    params:
        articulation: REQUIRED articulation name; the term reads
            ``articulation/<name>/...`` from the evaluation record.
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

    def source(self, ctx: EvalContext) -> Mapping[str, object]:
        """This term's articulation record leaves (``articulation/<name>/...``)."""
        return ctx.env.articulation(self.articulation)
