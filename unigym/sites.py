"""Neutral asset-level site poses.

A :class:`SiteSpec` is authoring data owned by a robot profile or a generated
prop asset; it is not a backend entity.  A :class:`SiteEntry` adds the runtime
owner and kind for the observation inventory, so a selected site name resolves
to exactly one asset-level declaration.

The spec carries a body-local position and orientation.  The quaternion is
normalized once at construction with an overflow-safe norm; a zero,
non-finite or non-normalizable orientation fails instead of silently falling
back to identity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

__all__ = ["SiteEntry", "SiteSpec"]


def _require_name(value: object, what: str) -> str:
    """Validate one non-empty, whitespace-free site/owner name."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    if value != value.strip() or any(character.isspace() for character in value):
        raise ValueError(f"{what} must not contain whitespace, got {value!r}")
    return value


def _finite_entries(value: object, length: int, what: str) -> tuple[float, ...]:
    """Validate one finite numeric sequence of a fixed length."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{what} must be {length} finite numbers, got {value!r}")
    # The isinstance check above establishes a sequence; the cast only
    # replaces the builtin container's unknown item type.
    entries = cast(Sequence[object], value)
    numbers: list[float] = []
    for entry in entries:
        if (
            isinstance(entry, bool)
            or not isinstance(entry, (int, float))
            or not math.isfinite(entry)
        ):
            raise ValueError(f"{what} must be {length} finite numbers, got {value!r}")
        numbers.append(float(entry))
    if len(numbers) != length:
        raise ValueError(f"{what} must be {length} finite numbers, got {value!r}")
    return tuple(numbers)


def _require_vec3(value: object, what: str) -> tuple[float, float, float]:
    """Validate one finite 3-vector and return canonical floats."""
    x, y, z = _finite_entries(value, 3, what)
    return (x, y, z)


def _require_quat_wxyz(value: object, what: str) -> tuple[float, float, float, float]:
    """Validate, normalize and return one finite non-zero wxyz quaternion."""
    w, x, y, z = _finite_entries(value, 4, what)
    norm = math.hypot(w, x, y, z)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{what} must have a finite non-zero norm, got {value!r}")
    return (w / norm, x / norm, y / norm, z / norm)


@dataclass(frozen=True, slots=True)
class SiteSpec:
    """One asset-owned site pose (body-local, metres, wxyz quaternion).

    Args:
        name: run-wide observation name; non-empty and whitespace-free.
        body: owner-local body name (canonical profile id for an
            articulation, payload body name for a generated prop asset).
        pos: body-local position, metres.
        quat_wxyz: body-local orientation as a normalized wxyz quaternion.
    """

    name: str
    body: str
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_name(self.name, "SiteSpec name"))
        object.__setattr__(self, "body", _require_name(self.body, "SiteSpec body"))
        object.__setattr__(self, "pos", _require_vec3(self.pos, f"site {self.name!r} pos"))
        object.__setattr__(
            self,
            "quat_wxyz",
            _require_quat_wxyz(self.quat_wxyz, f"site {self.name!r} quat_wxyz"),
        )


@dataclass(frozen=True, slots=True)
class SiteEntry:
    """One site inventory row: the owning asset and its neutral spec."""

    owner: str
    kind: Literal["articulation", "prop"]
    spec: SiteSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner", _require_name(self.owner, "SiteEntry owner"))
        if not isinstance(self.spec, SiteSpec):
            raise ValueError(f"SiteEntry spec must be a SiteSpec, got {type(self.spec).__name__}")
        if self.kind not in ("articulation", "prop"):
            raise ValueError(
                f"SiteEntry {self.spec.name!r} kind must be 'articulation' or "
                f"'prop', got {self.kind!r}"
            )
