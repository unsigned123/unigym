"""Live termination check interface.

A :class:`Termination` judges *per-sample* whether a termination condition
fires, given a read-only :class:`~unigym.evaluation.EvalContext` built at the
live post-step boundary.  Implementations read only the context's record,
articulation views, joint limits and optional query; they never mutate the
environment.  Termination is live-only: there is no recorded/offline state
path and no reconstruction from stored episode data.

Every articulation is a first-class citizen.  Per-articulation terms read
``ctx.env.articulation(name)``; there is no flat single-robot state and no
primary binding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    import torch

    from unigym.evaluation import EvalContext


def body_tilt_exceeded(projected_gravity_b: torch.Tensor, cos_threshold: float) -> torch.Tensor:
    """Body-tilt term shared by fall/base_contact terminations.

    ``projected_gravity_b[..., 2] == -cos(tilt)`` (upright == -1), so

        tilt > tilt_threshold  <=>  gravity_b_z > -cos(tilt_threshold)
    """
    return projected_gravity_b[..., 2] > -cos_threshold


class Termination(ABC):
    """Single live termination-condition checker.

    A pure condition over :class:`~unigym.evaluation.EvalContext`; it carries
    no articulation knowledge (see :class:`ArticulationTermination` for terms
    that judge exactly one articulation).

    Args:
        params: the yaml ``params`` section for this termination.
        name: optional yaml-level name (defaults to the class name); used
            as the key in :meth:`TerminationGroup.check_detail` output.
    """

    def __init__(self, params: Mapping | None = None, name: str | None = None):
        self.params = dict(params or {})
        self._cfg_name = name

    @abstractmethod
    def check(self, ctx: EvalContext) -> torch.Tensor:
        """Return a ``(num_envs,)`` bool mask for the live post-step state."""

    @property
    def name(self) -> str:
        return self._cfg_name or type(self).__name__


def parse_body_names(body_names) -> list[str]:
    """Normalize ``body_names`` config (str ``"base|head"`` or list) -> list."""
    if body_names is None:
        return []
    if isinstance(body_names, str):
        return [s.strip() for s in body_names.split("|") if s.strip()]
    return [str(s) for s in body_names]
