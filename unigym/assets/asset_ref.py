"""The one neutral asset reference type.

An :class:`AssetRef` carries the same asset in the backend families the
framework knows (``mjcf`` / ``urdf`` / ``usd``).  It is pure data: no
resolution, no filesystem access and no backend import.  Format selection
and resource resolution belong to the composition/backend boundary.

The module imports only the standard library plus :mod:`unigym._typing`, so
``import unigym.assets.asset_ref`` stays free of ``yaml``, scenes, robots and
every backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from unigym._typing import JsonValue

__all__ = ["ASSET_FORMATS", "AssetRef", "AssetRefError"]

ASSET_FORMATS: tuple[str, ...] = ("mjcf", "urdf", "usd")
"""The closed asset-format vocabulary (stable declaration order)."""


class AssetRefError(ValueError):
    """A malformed asset reference (fail-fast, no alias)."""


def _optional_str(value: JsonValue, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AssetRefError(f"{what} must be a string, got {value!r}")
    return value


@dataclass(frozen=True)
class AssetRef:
    """One articulation asset, per backend family (may leave all None)."""

    mjcf: str | None = None
    urdf: str | None = None
    usd: str | None = None

    def __post_init__(self) -> None:
        for key in ASSET_FORMATS:
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, str(value))

    def to_dict(self) -> dict[str, str]:
        """Non-empty asset fields (stable key order)."""
        out: dict[str, str] = {}
        for key in ASSET_FORMATS:
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    @classmethod
    def from_mapping(
        cls,
        value: AssetRef | Mapping[str, JsonValue],
        what: str = "model",
    ) -> AssetRef:
        """Normalize an ``AssetRef`` or ``{mjcf|urdf|usd: <path>}`` mapping."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AssetRefError(
                f"{what} must be a mapping {{mjcf|urdf|usd: <path>}}, " f"got {value!r}"
            )
        unknown = sorted(set(value) - set(ASSET_FORMATS))
        if unknown:
            raise AssetRefError(
                f"{what} declares unknown keys {unknown}; allowed keys are "
                f"{list(ASSET_FORMATS)}"
            )
        return cls(
            mjcf=_optional_str(value.get("mjcf"), f"{what}.mjcf"),
            urdf=_optional_str(value.get("urdf"), f"{what}.urdf"),
            usd=_optional_str(value.get("usd"), f"{what}.usd"),
        )
