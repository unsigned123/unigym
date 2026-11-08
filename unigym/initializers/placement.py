"""Backend-neutral placement declarations and query outcomes."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "PlacementRule", "PlacementResult", "PlacementRejection",
    "PlacementCapabilityError", "order_placement_rules", "rules_from_params",
]


class PlacementCapabilityError(RuntimeError):
    """A required backend placement query is unavailable."""


@dataclass(frozen=True)
class PlacementRule:
    """Closed declaration for placing one already-declared object.

    The region face names either one region (``region_key``) or an
    ordered candidate list (``region_keys``); a list rule resolves to
    exactly one region per reset through the initializer's RNG
    (docs/designs/v078-variant-and-axis-completion.md §8, the official
    ``sample_reset_region`` selection face).  ``anchor_percent`` is the
    official ``obj_x_percent``/``obj_y_percent`` anchor: ``[x, y]``
    fractions of the region's free placement extent in [0, 1] (0.5 =
    centered), or ``None`` for uniform sampling.  Each axis may
    alternatively be ``None`` (v078 F4: the official ``pos`` pairs mix
    anchored and free axes -- ``[null, 1.0]`` anchors y and leaves x
    uniform), so the anchor face is per-axis nullable.
    """

    object_name: str
    relation: Literal["on", "in"]
    region_key: str = ""
    yaw_range: tuple[float, float] = (0.0, 0.0)
    boundary_margin: float = 0.0
    min_clearance: float = 0.0
    order: int = 0
    max_attempts: int = 1
    region_keys: tuple[str, ...] = ()
    anchor_percent: tuple[float | None, float | None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.object_name, str) or not self.object_name.strip():
            raise ValueError("object_name must be a non-empty string")
        if self.relation not in ("on", "in"):
            raise ValueError("relation must be 'on' or 'in'")
        self._validate_region_face()
        self._validate_anchor_percent()
        if (not isinstance(self.yaw_range, tuple) or len(self.yaw_range) != 2
                or any(not isinstance(v, (int, float)) or not math.isfinite(v)
                       for v in self.yaw_range)
                or self.yaw_range[0] > self.yaw_range[1]):
            raise ValueError("yaw_range must be an ordered numeric pair")
        for name in ("boundary_margin", "min_clearance"):
            value = getattr(self, name)
            if (not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < 0):
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.order, int):
            raise ValueError("order must be an integer")
        if not isinstance(self.max_attempts, int) or self.max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")

    def _validate_region_face(self) -> None:
        """Exactly one of the single-key / candidate-list region faces."""
        if self.region_key and self.region_keys:
            raise ValueError(
                "declare exactly one of region_key (single region) or "
                "region_keys (ordered candidate list)")
        if not self.region_key and not self.region_keys:
            raise ValueError(
                "declare one region face: region_key (a non-empty string) "
                "or region_keys (a non-empty string list)")
        if self.region_key and (not isinstance(self.region_key, str)
                                or not self.region_key.strip()):
            raise ValueError("region_key must be a non-empty string")
        if self.region_keys:
            if (not isinstance(self.region_keys, tuple)
                    or any(not isinstance(key, str) or not key.strip()
                           for key in self.region_keys)
                    or len(set(self.region_keys)) != len(self.region_keys)):
                raise ValueError(
                    "region_keys must be an empty tuple or a non-empty tuple "
                    "of unique non-empty strings")

    def _validate_anchor_percent(self) -> None:
        """The official obj_x/y_percent face: null or a nullable pair."""
        if self.anchor_percent is None:
            return
        if (not isinstance(self.anchor_percent, tuple)
                or len(self.anchor_percent) != 2
                or all(value is None for value in self.anchor_percent)
                or any(not (value is None
                            or (not isinstance(value, bool)
                                and isinstance(value, (int, float))
                                and math.isfinite(value)
                                and 0.0 <= value <= 1.0))
                       for value in self.anchor_percent)):
            raise ValueError(
                "anchor_percent must be None or an [x, y] pair whose "
                "entries are fractions in [0, 1] or None (per-axis "
                "uniform)")

    @property
    def region_choices(self) -> tuple[str, ...]:
        """The rule's region candidates in declared order."""
        return ((self.region_key,) if self.region_key else self.region_keys)


@dataclass(frozen=True)
class PlacementResult:
    """Accepted placement expressed in world coordinates."""

    object_name: str
    position: tuple[float, float, float]
    yaw: float


@dataclass(frozen=True)
class PlacementRejection:
    """Rejected candidate and stable diagnostic classification."""

    object_name: str
    region_key: str
    attempt: int
    reason: str
    conflicting_object: str | None = None
    distance: float | None = None


def order_placement_rules(rules: tuple[PlacementRule, ...]) -> tuple[PlacementRule, ...]:
    """Return rules in declared order, preserving input order for ties."""
    if not isinstance(rules, tuple) or any(not isinstance(rule, PlacementRule)
                                           for rule in rules):
        raise ValueError("rules must be a tuple of PlacementRule")
    return tuple(rule for _, rule in sorted(
        enumerate(rules), key=lambda item: (item[1].order, item[0])))


def rules_from_params(value) -> tuple[PlacementRule, ...]:
    """Normalize the deliberately small, config-facing rule vocabulary.

    ``region_key`` (single string) and ``regions`` (non-empty string
    list) are the two region faces; declaring both is an error.
    ``anchor_percent`` accepts ``None`` (uniform) or an ``[x, y]`` pair
    whose entries are fractions in [0, 1] or ``None`` (per-axis
    uniform, the F4 official ``pos`` face); ``yaw_range_deg`` stays the
    degree spelling of ``yaw_range``.
    """
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("initializer.params.rules must be a non-empty list")
    rules = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"placement rule {index} must be a mapping")
        data = dict(raw)
        if "yaw_range_deg" in data:
            data["yaw_range"] = tuple(math.radians(v) for v in data.pop("yaw_range_deg"))
        if "regions" in data:
            if "region_key" in data:
                raise ValueError(
                    f"placement rule {index}: declare either region_key or "
                    "regions, not both")
            data["region_keys"] = tuple(data.pop("regions"))
        if (data.get("anchor_percent") is not None
                and not isinstance(data["anchor_percent"], tuple)):
            data["anchor_percent"] = tuple(data["anchor_percent"])
        rules.append(PlacementRule(**data))
    return tuple(rules)
