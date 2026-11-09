"""Termination check interface (numpy/torch agnostic, unit-testable).

A :class:`Termination` judges *per-sample* whether a termination condition
fires, given a :class:`TerminationState` snapshot extracted from the env's
per-step ``extra`` dict (live, torch) or an episode's recorded arrays
(offline, numpy).  Implementations must rely on operator overloading only
(``< | ~ .sum(-1)``) so the same code path works for both backends.

Every articulation is a first-class citizen.  :class:`TerminationState`
carries ONLY the namespaced ``articulations`` mapping (``articulation/<name>/``
keys), global body-contact data, per-articulation joint limits, and a batch
reference -- there is no flat single-robot state and no primary binding.

NOTE (design): in data-collection mode nothing actually resets when a
termination fires — IsaaclabEnv episodes end by duration only.  The
termination system is a *detection* layer consumed by progress logging
(collect.py), summary stats (logger), and episode filtering (filters/).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, cast


def all_false(x):
    """All-False bool mask shaped like ``x``'s leading dim (numpy or torch).

    ``x[..., 0] > inf`` is False for any finite/NaN value and keeps the
    backend type of ``x`` (torch tensors stay torch, numpy stays numpy).
    """
    return x[..., 0] > float("inf")


def body_tilt_exceeded(projected_gravity_b, cos_threshold: float):
    """Body-tilt term shared by fall/base_contact terminations.

    ``projected_gravity_b[..., 2] == -cos(tilt)`` (upright == -1), so

        tilt > tilt_threshold  <=>  gravity_b_z > -cos(tilt_threshold)

    Works on (N, 3) and (T, 3) arrays alike (numpy or torch).
    """
    return projected_gravity_b[..., 2] > -cos_threshold


#: recorded episode leaves -> live-extra leaf spelling (``collect.py`` maps
#: ``base_pos_w`` -> ``base_pos`` when writing; undo it here so a termination
#: reads one canonical leaf regardless of live/offline provenance).
_RECORD_ALIASES = {"base_pos": "base_pos_w"}


def _articulation_states(values: dict,
                         renames: Mapping[str, str] | None = None) -> dict:
    """Group ``articulation/<name>/<key>`` entries by articulation.

    The per-articulation inner keys are the canonical live names
    (``base_pos_w`` / ``gravity_b`` / ``foot_contact`` / ``joint_pos`` ...);
    *renames* maps recorded-only leaf spellings onto them.
    """
    out: dict = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.startswith("articulation/"):
            continue
        parts = key.split("/", 2)
        if len(parts) != 3 or not parts[1] or not parts[2]:
            continue
        leaf = parts[2]
        if renames:
            leaf = renames.get(leaf, leaf)
        out.setdefault(parts[1], {})[leaf] = value
    return out


@dataclass
class TerminationState:
    """Per-step state needed by termination checks.

    Populated live from an env's ``extra`` (torch, ``(N, ...)``) via
    :meth:`from_extra`, or offline from an episode's recorded arrays
    (numpy, ``(T, ...)``) via :meth:`from_episode_data`.  Every robot-state
    field lives under ``articulations[name]`` (no flat primary).
    Optional fields may be missing; individual terminations decide how to
    degrade (skip the term or never fire).
    """

    articulations: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict)                     # {name: {key: (N, ...)}}
    body_contact_forces: Any | None = None        # (N, num_bodies, 3)
    body_names: Sequence[str] | None = None       # index -> body name (forces)
    joint_pos_limits: Mapping[str, tuple] | None = None  # {name: (lo, hi)}
    batch: object | None = None                   # (N,)/(T,) per-sample ref

    # ------------------------------------------------------------------ #
    # constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_extra(cls, extra: dict, joint_pos_limits=None,
                   body_names: Sequence[str] | None = None
                   ) -> TerminationState:
        """Build from an env's per-step ``extra`` dict (torch tensors).

        Robot state is read from ``articulation/<name>/...`` keys only; the
        global ``body_contact_forces`` / ``body_names`` and
        ``time_in_episode`` (the batch reference) are honoured when present.
        """
        return cls(
            articulations=_articulation_states(extra, _RECORD_ALIASES),
            body_contact_forces=extra.get("body_contact_forces"),
            body_names=(extra.get("body_names") if "body_names" in extra
                        else body_names),
            joint_pos_limits=joint_pos_limits,
            batch=extra.get("time_in_episode"),
        )

    @classmethod
    def from_episode_data(cls, episode_data: dict) -> TerminationState:
        """Build from an assembled episode dict (numpy arrays, leading T).

        Recorded state keys are namespaced (``articulation/<name>/...``) and
        mapped onto the canonical live leaf names (``base_pos`` ->
        ``base_pos_w``).
        """
        supplied = episode_data.get("articulations")
        if isinstance(supplied, Mapping):
            articulations = {
                str(name): {_RECORD_ALIASES.get(k, k): v
                            for k, v in (inner or {}).items()}
                for name, inner in supplied.items()}
        else:
            articulations = _articulation_states(episode_data,
                                                 _RECORD_ALIASES)
        return cls(
            articulations=cast(
                "Mapping[str, Mapping[str, object]]", articulations),
            body_contact_forces=episode_data.get("body_contact_forces"),
            body_names=episode_data.get("body_names"),
            joint_pos_limits=None,
            batch=episode_data.get("time_in_episode"),
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def articulation(self, name, *, term: str | None = None) -> Mapping:
        """The named articulation's state map, or a fail-fast error."""
        source = (self.articulations or {}).get(name)
        if source is None:
            label = f"{term}: " if term else ""
            raise ValueError(
                f"{label}state has no articulation {name!r} keys; "
                f"available: {sorted(self.articulations or {})}")
        return source

    def reference(self):
        """Any ``(N,)``-leading array: the batch ref, else any articulation."""
        if self.batch is not None:
            return self.batch
        for inner in (self.articulations or {}).values():
            for key in ("base_pos_w", "joint_pos", "foot_contact", "gravity_b"):
                value = inner.get(key)
                if value is not None:
                    return value
        return None


class Termination(ABC):
    """Single termination-condition checker.

    A pure condition over :class:`TerminationState`; it carries no
    articulation knowledge (see :class:`ArticulationTermination` for terms
    that judge exactly one articulation).

    Args:
        params: the yaml ``params`` section for this termination.
        name: optional yaml-level name (defaults to the class name); used
            as the key in :meth:`TerminationGroup.check_detail` output.
    """

    #: set by subclasses that need per-articulation joint limits; lets
    #: collect.py fetch limits from the env only when required
    requires_joint_limits = False

    def __init__(self, params: Mapping | None = None,
                 name: str | None = None):
        self.params = dict(params or {})
        self._cfg_name = name

    @abstractmethod
    def check(self, state: TerminationState) -> Any:
        """Return an (N,) bool mask: True = the condition fires for that env."""

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
