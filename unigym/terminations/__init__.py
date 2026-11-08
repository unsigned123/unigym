"""Termination registry + builders (numpy/torch agnostic, unit-testable).

Yaml usage (top-level ``terminations`` section, optional; entries are
OR-combined — any term firing marks the env "terminated" for
logging/filtering; nothing actually resets in collection mode)::

    terminations:
      - name: fall
        articulation: g1            # OPTIONAL iff exactly one policy binding
        params: {height_threshold: 0.15, tilt_threshold: 0.8}
      - name: base_contact
        articulation: g1
        params: {body_names: "base", force_threshold: 5.0}
      - name: joint_limits
        articulation: g1
        params: {margin: 0.05}

v103.6 cardinality rule: every per-articulation term carries an explicit
``articulation``.  When a term omits it and the run proves exactly ONE policy
binding, the run layer injects that name (a cardinality convenience, not a
primary).  With several bindings an omitted ``articulation`` is a fail-fast.
"""
from __future__ import annotations

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


def _entry_articulation(entry: dict, name, articulations) -> str | None:
    """Resolve the term's articulation under the v103.6 cardinality rule.

    Explicit ``articulation`` (top-level or under ``params``) wins.  When it
    is omitted and *articulations* names exactly one policy binding, inject
    that name; with several bindings, fail-fast naming the legal set.
    """
    explicit = entry.get("articulation")
    if explicit is None:
        explicit = dict(entry.get("params") or {}).get("articulation")
    if explicit is not None:
        return str(explicit)
    if articulations is None:
        return None  # let the term's own requirement raise
    names = list(articulations)
    if len(names) == 1:
        return str(names[0])
    raise ValueError(
        f"termination {name!r} omits 'articulation' but the run has "
        f"{len(names)} policy articulations; declare one of "
        f"{sorted(names)} explicitly (v103.6: no primary binding)")


def build_terminations(cfg: list | None,
                       articulations=None) -> list[Termination]:
    """Build a list of terminations from the yaml ``terminations`` section.

    *articulations* is the run's ordered policy-binding names (or ``None``
    when unknown); see :func:`_entry_articulation` for the injection rule.
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
        articulation = _entry_articulation(entry, name, articulations)
        if articulation is not None:
            params["articulation"] = articulation
        term = TERMINATION_REGISTRY[name](params, name=name)
        if entry.get("invert", False):  # P11: NOT(term) health conditions
            term = InvertedTermination(term, name=name)
        terms.append(term)
    return terms


def build_termination_group(cfg: list | None, mode: str = "or",
                            articulations=None) -> TerminationGroup:
    """Build an OR (default) TerminationGroup from the yaml section.

    ``None``/empty cfg -> empty group that never fires (pure collection
    mode stays unaffected; callers fall back to a per-binding fall group for
    progress-log semantics, see :func:`default_fall_group`).
    """
    return TerminationGroup(
        build_terminations(cfg, articulations=articulations), mode=mode)


def default_fall_group(
        articulation: str,
        height_threshold: float = DEFAULT_HEIGHT_THRESHOLD,
        tilt_threshold: float | None = DEFAULT_TILT_THRESHOLD,
) -> TerminationGroup:
    """The inline height+tilt rule for ONE articulation (the P9-fix semantics).

    Used when no ``terminations`` section is configured.  *articulation* is
    required (v103.6): a per-policy-binding builder calls this once per
    binding; there is no flat single-robot fallback.
    """
    return TerminationGroup([FallTermination({
        "articulation": articulation,
        "height_threshold": height_threshold,
        "tilt_threshold": tilt_threshold,
    })])


__all__ = [
    "Termination",
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
    "default_fall_group",
]
