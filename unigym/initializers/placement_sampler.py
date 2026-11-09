"""Pure deterministic placement sampling over explicit rectangular geometry."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from unigym.initializers.placement import (
    PlacementRejection,
    PlacementResult,
    PlacementRule,
    order_placement_rules,
)
from unigym.core.region import RegionDefinition

__all__ = [
    "PlacementExhaustion",
    "PlacementSample",
    "PlacementSamplingError",
    "sample_placements",
]


class PlacementSamplingError(ValueError):
    """Raised when a declared placement domain cannot contain its footprint."""


@dataclass(frozen=True)
class PlacementExhaustion:
    """Structured result for a rule that used its full candidate budget."""

    object_name: str
    region_key: str
    attempts: int
    rejections: tuple[PlacementRejection, ...]


@dataclass(frozen=True)
class PlacementSample:
    """Accepted placements, rejected candidates, and exhausted rules."""

    accepted: tuple[PlacementResult, ...]
    rejections: tuple[PlacementRejection, ...]
    exhausted: tuple[PlacementExhaustion, ...]


CandidatePredicate = Callable[[PlacementRule, tuple[float, float, float], float, float], object]


def _rot2(angle: float, xy) -> tuple:
    """Rotate an XY pair by ``angle`` (region frame -> world)."""
    c, s = math.cos(angle), math.sin(angle)
    return (c * xy[0] - s * xy[1], s * xy[0] + c * xy[1])


def _bounds(value, label: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Validate finite local XYZ bounds represented by minimum and maximum."""
    try:
        low, high = value
        low, high = tuple(float(v) for v in low), tuple(float(v) for v in high)
    except (TypeError, ValueError) as exc:
        raise PlacementSamplingError(f"{label} must be a pair of XYZ bounds") from exc
    if (len(low) != 3 or len(high) != 3
            or any(not math.isfinite(v) for v in (*low, *high))
            or any(a >= b for a, b in zip(low, high))):
        raise PlacementSamplingError(f"{label} must contain finite increasing XYZ bounds")
    return low, high


def _rectangle(region: RegionDefinition, relation: str):
    """Return the region frame: center, extents, yaw, and z limits.

    This face extends the v076 world-aligned boundary to
    yaw-only frames (the official sampler works in the fixture frame:
    ``reference_rot`` carries the fixture yaw and the ranges ride it).
    A non-yaw rotation stays refused.
    """
    if region.kind != ("support" if relation == "on" else "interior"):
        raise PlacementSamplingError(
            f"region {region.name!r} kind {region.kind!r} cannot satisfy {relation!r}")
    quat = np.asarray(region.quat, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise PlacementSamplingError(
            f"region {region.name!r}: quat must be non-zero")
    quat = quat / norm
    if np.allclose(quat, (1.0, 0.0, 0.0, 0.0), atol=1e-9):
        yaw = 0.0
    elif abs(quat[1]) < 1e-9 and abs(quat[2]) < 1e-9:
        yaw = math.remainder(2.0 * math.atan2(float(quat[3]),
                                              float(quat[0])), 2.0 * math.pi)
    else:
        raise PlacementSamplingError(
            "non-yaw region frames are unsupported; provide "
            "world-aligned or yaw-only geometry")
    x, y, z = region.pos
    z_limits = ((z, z + region.height_m) if relation == "in" else (z, z))
    return (x, y), region.width_m, region.depth_m, yaw, z_limits


def _footprint(low, high, yaw: float):
    """Return yaw-rotated XY corners relative to the object's origin."""
    c, s = math.cos(yaw), math.sin(yaw)
    return tuple((c * x - s * y, s * x + c * y)
                 for x in (low[0], high[0]) for y in (low[1], high[1]))


def _yaw_envelope(low, high, lo: float, hi: float):
    """The footprint's XY extent over the yaw interval (exact criticals).

    Feasibility is checked for every yaw in the declared interval. The
    extrema of each rotated corner occur at interval endpoints or a
    derivative zero, so include those critical angles analytically.
    """
    angles = [lo, hi]
    for x0, y0 in ((low[0], low[1]), (low[0], high[1]),
                   (high[0], low[1]), (high[0], high[1])):
        for base in (math.atan2(-y0, x0), math.atan2(x0, y0)):
            for k in range(math.floor((lo - base) / math.pi) - 1,
                           math.ceil((hi - base) / math.pi) + 2):
                angle = base + k * math.pi
                if lo <= angle <= hi:
                    angles.append(angle)
    points = [point for angle in angles
              for point in _footprint(low, high, angle)]
    return (min(p[0] for p in points), max(p[0] for p in points),
            min(p[1] for p in points), max(p[1] for p in points))


def _draw_axis(rng, anchor, domain_lo, domain_hi):
    """One axis' position: the anchor fraction or a uniform draw.

    A degenerate axis draws nothing (the v076 stream contract:
    consumption only for real choices).
    """
    if anchor is not None:
        return domain_lo + anchor * (domain_hi - domain_lo)
    if domain_lo != domain_hi:
        return float(rng.uniform(domain_lo, domain_hi))
    return domain_lo


def sample_placements(
    rules: tuple[PlacementRule, ...],
    regions: Mapping[str, RegionDefinition],
    collision_bounds: Mapping[str, object],
    candidate_predicate: CandidatePredicate,
    rng: np.random.Generator,
) -> PlacementSample:
    """Sample placements without backend access or live-state mutation.

    Bounds are object-origin-local min/max XYZ tuples. Region geometry
    may be world-aligned or yaw-only (sampling runs
    in the region's frame, the official ``reference_rot`` semantics --
    the rule's yaw range is region-relative). The predicate receives
    (rule, origin position, yaw radians, minimum clearance metres) and
    returns True or a rejection diagnostic mapping with optional
    reason/conflicting_object/distance.  An anchored axis
    (``anchor_percent`` entry set) replaces that axis' uniform draw
    with the official percent position on the free domain and consumes
    no RNG for it; the other axis, the yaw draw and the predicate
    probing are unchanged.
    """
    if not callable(candidate_predicate):
        raise TypeError("candidate_predicate must be callable")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    ordered = order_placement_rules(rules)
    prepared = []
    for rule in ordered:
        try:
            region = regions[rule.region_key]
            bounds = collision_bounds[rule.object_name]
        except KeyError as exc:
            raise PlacementSamplingError(f"missing explicit geometry for {exc.args[0]!r}") from exc
        if not isinstance(region, RegionDefinition):
            raise PlacementSamplingError(f"region {rule.region_key!r} is not a RegionDefinition")
        low, high = _bounds(bounds, f"collision bounds for {rule.object_name!r}")
        center, width, depth, yaw_region, z_limits = _rectangle(region, rule.relation)
        # the object's footprint rotates by (world yaw - region yaw) in
        # the region frame; the critical-angle scan runs on the shifted
        # interval so the envelope stays exact under the frame change
        min_dx, max_dx, min_dy, max_dy = _yaw_envelope(
            low, high, rule.yaw_range[0] - yaw_region, rule.yaw_range[1] - yaw_region
        )
        domain = (
            -width / 2 + rule.boundary_margin - min_dx,
            width / 2 - rule.boundary_margin - max_dx,
            -depth / 2 + rule.boundary_margin - min_dy,
            depth / 2 - rule.boundary_margin - max_dy,
        )
        if domain[0] > domain[1] or domain[2] > domain[3]:
            raise PlacementSamplingError(
                f"{rule.object_name!r} footprint leaves no feasible domain in {rule.region_key!r}"
            )
        if rule.relation == "in" and (low[2] < 0 or high[2] > z_limits[1] - z_limits[0]):
            raise PlacementSamplingError(
                f"{rule.object_name!r} vertical bounds do not fit interior {rule.region_key!r}"
            )
        anchors = rule.anchor_percent or (None, None)
        prepared.append((rule, low, center, yaw_region, domain, anchors, z_limits[0]))

    accepted, rejections, exhausted = [], [], []
    for rule, low, center, yaw_region, domain, anchors, support_z in prepared:
        rule_rejections = []
        for attempt in range(1, rule.max_attempts + 1):
            yaw = (
                float(rng.uniform(*rule.yaw_range))
                if rule.yaw_range[0] != rule.yaw_range[1]
                else rule.yaw_range[0]
            )
            lx = _draw_axis(rng, anchors[0], domain[0], domain[1])
            ly = _draw_axis(rng, anchors[1], domain[2], domain[3])
            offset = _rot2(yaw_region, (lx, ly))
            position = (center[0] + offset[0], center[1] + offset[1], support_z - low[2])
            outcome = candidate_predicate(rule, position, yaw_region + yaw, rule.min_clearance)
            if outcome is True:
                accepted.append(PlacementResult(rule.object_name, position, yaw_region + yaw))
                break
            diagnostic = outcome if isinstance(outcome, Mapping) else {}
            rejection = PlacementRejection(
                rule.object_name,
                rule.region_key,
                attempt,
                str(diagnostic.get("reason", "candidate_rejected")),
                diagnostic.get("conflicting_object"),
                diagnostic.get("distance"),
            )
            rule_rejections.append(rejection)
            rejections.append(rejection)
        else:
            exhausted.append(
                PlacementExhaustion(
                    rule.object_name, rule.region_key, rule.max_attempts, tuple(rule_rejections)
                )
            )
    return PlacementSample(tuple(accepted), tuple(rejections), tuple(exhausted))
