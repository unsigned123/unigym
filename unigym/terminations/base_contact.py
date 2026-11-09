"""Base-contact termination: non-foot body part bearing significant force.

Preferred signal: per-body net contact forces (``state.body_contact_forces``
+ ``state.body_names``, global); fires when the net-force magnitude on any
listed body exceeds ``force_threshold``.

LIMITATION (current env): IsaaclabEnv's contact sensor only covers the feet
(``prim_path="{ENV_REGEX_NS}/Robot/.*_foot"``, see isaaclab2/env.py), so
body-level forces are unavailable live.  In that case this termination
degrades to a geometry approximation

    contact ≈ (base_z < height_threshold) AND (tilt > tilt_threshold)

i.e. the base is low enough to touch the ground AND the body is tilted
(the conjunction keeps the approximation a strict subset of
``FallTermination``, so adding it to a fall-configured OR group does not
change the group's firing set).  When the env later gains a full-body
contact sensor, feeding ``extra["body_contact_forces"]`` and
``extra["body_names"]`` switches this termination to the exact force rule
with no code change.

v103.6: the geometry approximation judges exactly one articulation
(``articulation`` is a required param); the global force signal is unchanged.
"""
from __future__ import annotations

import math

from .articulation import ArticulationTermination
from .base import (
    TerminationState,
    all_false,
    body_tilt_exceeded,
    parse_body_names,
)
from .fall import DEFAULT_HEIGHT_THRESHOLD, DEFAULT_TILT_THRESHOLD


class BaseContactTermination(ArticulationTermination):
    """Fire when a non-foot body (base, head, ...) carries ground contact.

    params:
        articulation: REQUIRED articulation name for the geometry fallback
            (v103.6, no flat fallback).
        body_names: str ``"base|head"`` or list of body names to monitor.
        force_threshold: net contact-force magnitude threshold in N
            (default 5.0; only used with real body forces).
        height_threshold: approximation low-height threshold (default 0.15).
        tilt_threshold: approximation body-tilt threshold in rad
            (default 0.8).
    """

    def __init__(self, params: dict | None = None, name: str | None = None):
        super().__init__(params, name)
        self.body_names = parse_body_names(self.params.get("body_names"))
        self.force_threshold = float(self.params.get("force_threshold", 5.0))
        self.height_threshold = float(
            self.params.get("height_threshold", DEFAULT_HEIGHT_THRESHOLD))
        tilt = self.params.get("tilt_threshold", DEFAULT_TILT_THRESHOLD)
        self.tilt_threshold = None if tilt is None else float(tilt)
        self._cos_tilt = (None if self.tilt_threshold is None
                          else math.cos(self.tilt_threshold))

    def check(self, state: TerminationState):
        forces, names = state.body_contact_forces, state.body_names
        if forces is not None and names:
            idx = [i for i, n in enumerate(list(names))
                   if n in self.body_names]
            if idx:
                f = forces[..., idx, :]                     # (N, k, 3)
                mag = (f * f).sum(-1) ** 0.5                # (N, k)
                return (mag > self.force_threshold).sum(-1) > 0
            # requested bodies absent from the sensor -> no signal; fall
            # through to the geometry approximation rather than never firing
        return self._check_approx(state)

    def _check_approx(self, state: TerminationState):
        """Height+tilt approximation (subset of FallTermination's OR)."""
        source = self.source(state)
        base_pos_w = source.get("base_pos_w")
        gravity_b = source.get("gravity_b")
        if base_pos_w is None or gravity_b is None or self._cos_tilt is None:
            ref = base_pos_w if base_pos_w is not None else gravity_b
            if ref is None:
                raise ValueError(
                    f"BaseContactTermination requires {self.articulation!r} "
                    f"base_pos_w or gravity_b in state")
            return all_false(ref)
        low = base_pos_w[..., 2] < self.height_threshold
        tilted = body_tilt_exceeded(gravity_b, self._cos_tilt)
        return low & tilted
