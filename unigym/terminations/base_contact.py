"""Base-contact termination: non-foot body part bearing significant force.

Preferred signal: per-body net contact forces from the live record
(``body_contact_forces`` + ``body_names``, global); fires when the net-force
magnitude on any listed body exceeds ``force_threshold``.

LIMITATION (current env): IsaaclabEnv's contact sensor only covers the feet
(``prim_path="{ENV_REGEX_NS}/Robot/.*_foot"``, see isaaclab2/env.py), so
body-level forces are unavailable live.  In that case this termination
degrades to a geometry approximation

    contact ≈ (base_z < height_threshold) AND (tilt > tilt_threshold)

i.e. the base is low enough to touch the ground AND the body is tilted
(the conjunction keeps the approximation a strict subset of
``FallTermination``, so adding it to a fall-configured OR group does not
change the group's firing set).  When the env later gains a full-body
contact sensor, feeding ``record["body_contact_forces"]`` and
``record["body_names"]`` switches this termination to the exact force rule
with no code change.

The geometry approximation judges exactly one articulation
(``articulation`` is a required param); the global force signal is unchanged.
"""

from __future__ import annotations

import math

import torch

from unigym.evaluation import EvalContext

from .articulation import ArticulationTermination
from .base import body_tilt_exceeded, parse_body_names
from .fall import DEFAULT_HEIGHT_THRESHOLD, DEFAULT_TILT_THRESHOLD


class BaseContactTermination(ArticulationTermination):
    """Fire when a non-foot body (base, head, ...) carries ground contact.

    params:
        articulation: REQUIRED articulation name for the geometry fallback
            (no flat fallback).
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
        self.height_threshold = float(self.params.get("height_threshold", DEFAULT_HEIGHT_THRESHOLD))
        tilt = self.params.get("tilt_threshold", DEFAULT_TILT_THRESHOLD)
        self.tilt_threshold = None if tilt is None else float(tilt)
        self._cos_tilt = None if self.tilt_threshold is None else math.cos(self.tilt_threshold)

    def check(self, ctx: EvalContext) -> torch.Tensor:
        record = ctx.env.record()
        forces = record.get("body_contact_forces")
        names = record.get("body_names")
        if isinstance(forces, torch.Tensor) and names:
            idx = [i for i, n in enumerate(list(names)) if n in self.body_names]
            if idx:
                selected = forces[..., idx, :]  # (N, k, 3)
                mag = (selected * selected).sum(-1) ** 0.5  # (N, k)
                return (mag > self.force_threshold).sum(-1) > 0
            # requested bodies absent from the sensor -> no signal; fall
            # through to the geometry approximation rather than never firing
        return self._check_approx(ctx)

    def _check_approx(self, ctx: EvalContext) -> torch.Tensor:
        """Height+tilt approximation (subset of FallTermination's OR)."""
        source = self.source(ctx)
        base_pos_w = source.get("base_pos_w")
        gravity_b = source.get("gravity_b")
        has_base = isinstance(base_pos_w, torch.Tensor)
        has_gravity = isinstance(gravity_b, torch.Tensor)
        if not has_base or not has_gravity or self._cos_tilt is None:
            ref = base_pos_w if has_base else gravity_b
            if not isinstance(ref, torch.Tensor):
                raise ValueError(
                    f"BaseContactTermination requires {self.articulation!r} "
                    f"base_pos_w or gravity_b in the evaluation record"
                )
            return torch.zeros(ref.shape[0], dtype=torch.bool, device=ref.device)
        low = base_pos_w[..., 2] < self.height_threshold
        tilted = body_tilt_exceeded(gravity_b, self._cos_tilt)
        return low & tilted
