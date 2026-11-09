"""Fallen filter: discard episodes with excessive fall indicators.

The height/tilt fall logic now lives in the unified
termination system (:mod:`unigym.terminations`).  :class:`FallenFilter`
keeps its yaml interface unchanged (``max_fallen_ratio`` /
``height_threshold`` / ``check_tilt`` / ``tilt_threshold`` /
``check_all_feet_airborne``) but internally builds a
:class:`~unigym.terminations.TerminationGroup` with a
:class:`~unigym.terminations.FallTermination` and runs it over the
episode's recorded arrays.  The "all four feet airborne" term has no
termination counterpart (it is an episode-quality signal, not a state
fault) and stays as a filter-local OR.

Recorded state is read from ``articulation/<name>/...`` only.  A
filter may name its ``articulation`` explicitly; otherwise it resolves the
unique recorded articulation (a cardinality fact) and fails fast when the
episode carries several.
"""
from __future__ import annotations

from unigym.terminations import (
    DEFAULT_HEIGHT_THRESHOLD,
    DEFAULT_TILT_THRESHOLD,
    FallTermination,
    TerminationGroup,
    TerminationState,
    build_termination_group,
)

from .base import EpisodeFilter


def _single_articulation(state: TerminationState, who: str) -> str:
    """The unique recorded articulation, or a fail-fast naming the set."""
    names = sorted(state.articulations or {})
    if len(names) == 1:
        return names[0]
    raise ValueError(
        f"{who}: needs exactly one articulation in the episode data, found "
        f"{names}; pass an explicit 'articulation' (v103.6: no primary)")


class FallenFilter(EpisodeFilter):
    """Discard episodes with excessive fall indicators.

    A step counts as "fallen" when ANY of:

    * ``base_z < height_threshold`` (low crouch / lying flat),
    * body tilt from vertical > ``tilt_threshold`` (side/back/face fall;
      requires ``gravity_b`` in episode data, skipped otherwise),
    * all four feet airborne (``check_all_feet_airborne``).

    Args:
        max_fallen_ratio: if the fraction of fallen steps exceeds this
            value, the episode is discarded.
        height_threshold: base-z below which a step counts as "fallen" (m).
        check_all_feet_airborne: if True, also count steps where all four
            feet are airborne (foot_contact all False) as fallen steps.
        check_tilt: if True (default), count steps with body tilt >
            ``tilt_threshold`` as fallen (uses ``gravity_b`` when present).
        tilt_threshold: body-tilt-from-vertical threshold in rad
            (default 0.8 rad ≈ 46 deg; normal locomotion stays < ~30 deg).
        articulation: explicit articulation to judge; omit to use the unique
            recorded articulation (fail-fast when several).
    """

    def __init__(self, max_fallen_ratio: float = 0.1,
                 height_threshold: float = DEFAULT_HEIGHT_THRESHOLD,
                 check_all_feet_airborne: bool = True,
                 check_tilt: bool = True,
                 tilt_threshold: float | None = DEFAULT_TILT_THRESHOLD,
                 articulation: str | None = None):
        self.max_fallen_ratio = float(max_fallen_ratio)
        self.height_threshold = float(height_threshold)
        self.check_all_feet_airborne = bool(check_all_feet_airborne)
        self.check_tilt = bool(check_tilt)
        self.tilt_threshold = (None if tilt_threshold is None
                               else float(tilt_threshold))
        self.articulation = articulation
        self.group = (None if articulation is None
                      else self._build_group(articulation))

    def _build_group(self, articulation: str) -> TerminationGroup:
        # unified termination backend: height OR tilt (tilt term disabled
        # when check_tilt=False, mirroring the reference semantics exactly)
        return TerminationGroup([FallTermination({
            "articulation": articulation,
            "height_threshold": self.height_threshold,
            "tilt_threshold": (self.tilt_threshold if self.check_tilt
                               else None),
        })])

    def __call__(self, episode_data: dict) -> bool:
        state = TerminationState.from_episode_data(episode_data)
        name = self.articulation or _single_articulation(state, "FallenFilter")
        if self.group is None:
            self.group = self._build_group(name)
        fallen = self.group.check(state)

        if self.check_all_feet_airborne:
            foot_contact = state.articulation(name).get("foot_contact")
            if foot_contact is not None:
                # (T, 4) bool: True when all feet off ground (works for
                # numpy episodes and torch batches alike)
                all_air = ~(foot_contact.sum(-1) > 0)
                fallen = fallen | all_air

        ratio = float(fallen.mean())
        return ratio <= self.max_fallen_ratio


def _bind_cfg_articulation(cfg: list, articulation: str) -> list:
    """Bind the filter's explicit articulation into sub-term configs.

    The termination builder no longer injects an articulation; this filter
    owns its one explicit articulation, so sub-terms that omit one inherit it.
    """
    bound = []
    for entry in cfg:
        item = dict(entry)
        params = dict(item.get("params") or {})
        if (item.get("articulation") is None
                and params.get("articulation") is None):
            params["articulation"] = articulation
        item["params"] = params
        bound.append(item)
    return bound


class TerminationFilter(EpisodeFilter):
    """Generic episode filter on top of the termination system.

    Discards episodes where the combined termination mask is True for more
    than ``max_terminated_ratio`` of the steps.  Accepts the same
    ``terminations`` config format as the top-level yaml section::

        - name: termination
          params:
            max_terminated_ratio: 0.05
            articulation: g1            # OPTIONAL iff one recorded binding
            terminations:
              - name: fall
                params: {height_threshold: 0.15, tilt_threshold: 0.8}
    """

    def __init__(self, max_terminated_ratio: float = 0.1,
                 terminations: list | None = None,
                 articulation: str | None = None):
        self.max_terminated_ratio = float(max_terminated_ratio)
        self._cfg = list(terminations or [])
        self.articulation = articulation
        self.group = (build_termination_group(
            _bind_cfg_articulation(self._cfg, articulation))
            if articulation is not None else None)

    def __call__(self, episode_data: dict) -> bool:
        state = TerminationState.from_episode_data(episode_data)
        if self.group is None:
            name = self.articulation or _single_articulation(
                state, "TerminationFilter")
            self.group = build_termination_group(
                _bind_cfg_articulation(self._cfg, name))
        if len(self.group) == 0:
            return True  # nothing configured -> keep everything
        ratio = float(self.group.check(state).mean())
        return ratio <= self.max_terminated_ratio
