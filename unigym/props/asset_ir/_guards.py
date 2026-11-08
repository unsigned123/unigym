"""Scalar/vector validation guards shared by the IR dataclasses.

Every guard fails fast with a ``ValueError`` carrying the owner label
(``"joint 'lid'"`` style) so a provider sees which payload row broke;
returning the normalized value keeps the frozen dataclasses' fields
canonical (tuples of floats, strict bools).
"""
from __future__ import annotations

import math


def _require_vec(owner: str, name: str, value, length: int) -> tuple:
    """Validate a fixed-length finite float vector; returns a tuple."""
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in value
        )
    ):
        raise ValueError(f"{owner}: {name} must be {length} finite numbers, got {value!r}")
    return tuple(float(v) for v in value)


def _require_quat(owner: str, value) -> tuple:
    """Validate a non-zero (w, x, y, z) quaternion."""
    quat = _require_vec(owner, "quat", value, 4)
    if sum(v * v for v in quat) < 1e-24:
        raise ValueError(f"{owner}: quat must be non-zero")
    return quat


def _require_range(owner: str, value) -> tuple:
    """Validate a finite [lo, hi] range with lo <= hi."""
    bounds = _require_vec(owner, "range", value, 2)
    if bounds[0] > bounds[1]:
        raise ValueError(f"{owner}: range {list(bounds)} has lo > hi")
    return bounds


def _require_nonneg(owner: str, name: str, value) -> float:
    """Validate one finite non-negative scalar."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{owner}: {name} must be a finite number >= 0, got {value!r}")
    return float(value)


def _require_finite(owner: str, name: str, value) -> float:
    """Validate one finite scalar of any sign (the MJCF float face)."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{owner}: {name} must be a finite number, got {value!r}")
    return float(value)


def _require_rgba(owner: str, value) -> tuple:
    """Validate one (r, g, b, a) color with components in [0, 1]."""
    rgba = _require_vec(owner, "rgba", value, 4)
    if any(v < 0.0 or v > 1.0 for v in rgba):
        raise ValueError(f"{owner}: rgba components must be within [0, 1], got {value!r}")
    return rgba


def _require_bool(owner: str, name: str, value) -> bool:
    """Validate a strict boolean (no int coercion)."""
    if not isinstance(value, bool):
        raise ValueError(f"{owner}: {name} must be a bool, got {value!r}")
    return value


def _require_unique(seen: dict[str, set], kind: str, name: str) -> None:
    """Record one element name; fail-fast on a duplicate within the asset."""
    if name in seen[kind]:
        raise ValueError(f"duplicate {kind} name {name!r} in the asset definition")
    seen[kind].add(name)
