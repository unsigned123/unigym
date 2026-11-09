"""Termination registry + builders (numpy/torch agnostic, unit-testable).

Yaml usage (top-level ``terminations`` section, optional; entries are
OR-combined — any term firing marks the env "terminated" for
logging/filtering; nothing actually resets in collection mode)::

    terminations:
      - name: fall
        articulation: g1            # REQUIRED for per-articulation terms
        params: {height_threshold: 0.15, tilt_threshold: 0.8}
      - name: base_contact
        articulation: g1
        params: {body_names: "base", force_threshold: 5.0}
      - name: joint_limits
        articulation: g1
        params: {margin: 0.05}

Explicit articulation ownership: every per-articulation term carries an
explicit ``articulation`` (top-level or under ``params``).  There is no
single-binding injection: a missing ``articulation`` fails fast at
construction, whether the run has one binding or several.
"""
from __future__ import annotations

from .articulation import ArticulationTermination
from .base import Termination, TerminationState
from .composite import InvertedTermination, TerminationGroup
from .fall import (
    COS_DEFAULT_TILT,
    DEFAULT_HEIGHT_THRESHOLD,
    DEFAULT_TILT_THRESHOLD,
    FallTermination,
)
from .base_contact import BaseContactTermination
from .joint_limits import JointLimitTermination

TERMINATION_REGISTRY = {
    "fall": FallTermination,
    "base_contact": BaseContactTermination,
    "joint_limits": JointLimitTermination,
}


def build_terminations(cfg: list | None) -> list[Termination]:
    """Build a list of terminations from the yaml ``terminations`` section.

    Per-articulation terms must carry an explicit ``articulation`` (top-level
    or under ``params``); the term's own base class enforces it.  ``None``/
    empty cfg -> empty list.
    """
    terms: list[Termination] = []
    if not cfg:
        return terms
    for entry in cfg:
        name = entry.get("name")
        if name not in TERMINATION_REGISTRY:
            raise ValueError(
                f"unknown termination '{name}'; "
                f"available: {sorted(TERMINATION_REGISTRY)}")
        params = dict(entry.get("params", {}))
        explicit = entry.get("articulation")
        if explicit is not None:
            params["articulation"] = explicit
        term = TERMINATION_REGISTRY[name](params, name=name)
        if entry.get("invert", False):  # NOT(term) health conditions
            term = InvertedTermination(term, name=name)
        terms.append(term)
    return terms


def build_termination_group(cfg: list | None, mode: str = "or"
                            ) -> TerminationGroup:
    """Build an OR (default) TerminationGroup from the yaml section.

    ``None``/empty cfg -> empty group that never fires (no default fall
    group is synthesised).
    """
    return TerminationGroup(build_terminations(cfg), mode=mode)


__all__ = [
    "Termination",
    "ArticulationTermination",
    "TerminationState",
    "TerminationGroup",
    "InvertedTermination",
    "FallTermination",
    "BaseContactTermination",
    "JointLimitTermination",
    "TERMINATION_REGISTRY",
    "DEFAULT_HEIGHT_THRESHOLD",
    "DEFAULT_TILT_THRESHOLD",
    "COS_DEFAULT_TILT",
    "build_terminations",
    "build_termination_group",
]
