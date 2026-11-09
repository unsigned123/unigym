"""Native terrain declaration for the closed scene IR (S1b-iii).

A :class:`Terrain` is a pure-data height field: the scene layer (②)
carries the shape of the ground, never an engine element.  The MuJoCo
composer renders it as one ``hfield`` geom (or one triangulated mesh on
engines without the hfield API); the Isaac Lab mapping is S1b-iv.

``heights`` is a ``(ny, nx)`` float32 array of absolute world-z
elevations.  ``base_z`` is the depth of the solid slab under the lowest
elevation point (the MJCF hfield ``base_z``); it is NOT an elevation
offset -- add a constant to ``heights`` to lift the terrain.  ``size`` is
the ``(sx, sy)`` metric extent of the rectangle and ``resolution`` the
``(nx, ny)`` sample count.

NumPy is the only dependency.  ``flat`` returns ``None``: "no terrain" is
the absence of the neutral member, not a zero field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np

from unigym._typing import JsonValue

from .scene_ir import require_neutral_name

__all__ = ["Terrain"]

_TERRAIN_KIND = "heightfield"

#: the array-like input a height map may arrive as.
HeightsInput: TypeAlias = np.ndarray | Sequence[Sequence[float]]


def _empty_meta() -> dict[str, JsonValue]:
    """An empty terrain metadata table (typed dataclass default)."""
    return {}


def _finite(value: object, what: str) -> float:
    """Validate one finite real scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"terrain: {what} must be a finite number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"terrain: {what} must be finite, got {value!r}")
    return value


def _pair(
    values: Sequence[object],
    what: str,
    *,
    positive: bool,
) -> tuple[float, float]:
    """Validate a finite 2-vector (optionally strictly positive)."""
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"terrain: {what} must be a 2-vector of finite numbers, got {values!r}")
    if len(values) != 2 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in values
    ):
        raise ValueError(f"terrain: {what} must be a 2-vector of finite numbers, got {values!r}")
    nums = tuple(float(v) for v in values if isinstance(v, (int, float)))
    pair = (nums[0], nums[1])
    if positive and (pair[0] <= 0 or pair[1] <= 0):
        raise ValueError(f"terrain: {what} components must be > 0, got {values!r}")
    return pair


def _resolution(values: Sequence[object]) -> tuple[int, int]:
    """Validate an integer ``(nx, ny)`` resolution with both axes >= 2."""
    if not isinstance(values, (list, tuple)):
        raise ValueError(
            f"terrain: resolution must be a 2-vector of integers >= 2, " f"got {values!r}"
        )
    if len(values) != 2 or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in values
    ):
        raise ValueError(
            f"terrain: resolution must be a 2-vector of integers >= 2, " f"got {values!r}"
        )
    nums = tuple(int(v) for v in values if isinstance(v, int) and not isinstance(v, bool))
    return (nums[0], nums[1])


def _grid(
    nx: int,
    ny: int,
    sx: float,
    sy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinates of an ``(ny, nx)`` grid over the rectangle."""
    xs = (np.arange(nx, dtype=np.float64) + 0.5) * (sx / nx) - sx / 2.0
    ys = (np.arange(ny, dtype=np.float64) + 0.5) * (sy / ny) - sy / 2.0
    return xs, ys


@dataclass(frozen=True)
class Terrain:
    """One height-field terrain (absolute elevations, metres)."""

    kind: str = _TERRAIN_KIND
    size: tuple[float, float] = (0.0, 0.0)
    resolution: tuple[int, int] = (2, 2)
    heights: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=np.float32))
    base_z: float = 0.0
    meta: Mapping[str, JsonValue] = field(default_factory=_empty_meta)

    def __post_init__(self):
        """Validate the kind, extent, resolution, height map and metadata."""
        if self.kind != _TERRAIN_KIND:
            raise ValueError(f"terrain: kind {self.kind!r} must be {_TERRAIN_KIND!r}")
        object.__setattr__(self, "size", _pair(self.size, "size", positive=True))
        object.__setattr__(self, "resolution", _resolution(self.resolution))
        heights = np.array(self.heights, dtype=np.float32, copy=True)
        if heights.ndim != 2:
            raise ValueError(
                f"terrain: heights must be a (ny, nx) array, got shape " f"{heights.shape}"
            )
        nx, ny = self.resolution
        if heights.shape != (ny, nx):
            raise ValueError(
                f"terrain: heights shape {heights.shape} does not match "
                f"resolution (nx={nx}, ny={ny}) -> (ny, nx)={(ny, nx)}"
            )
        if not np.isfinite(heights).all():
            raise ValueError("terrain: heights must all be finite")
        object.__setattr__(self, "heights", heights)
        base_z = _finite(self.base_z, "base_z")
        if base_z < 0:
            raise ValueError(f"terrain: base_z must be >= 0, got {base_z!r}")
        object.__setattr__(self, "base_z", base_z)
        if not isinstance(self.meta, Mapping):
            raise ValueError(f"terrain: meta must be a mapping, got " f"{type(self.meta).__name__}")
        meta = dict(self.meta)
        raw_name = meta.get("name", "terrain")
        if not isinstance(raw_name, str):
            raise ValueError(
                f"terrain name {raw_name!r} must be a lowercase snake_case "
                f"identifier (path-like spellings are rejected)"
            )
        meta["name"] = require_neutral_name("terrain", raw_name)
        object.__setattr__(self, "meta", meta)

    @property
    def name(self) -> str:
        """The neutral terrain name (geom/hfield asset name)."""
        name = self.meta["name"]
        assert isinstance(name, str)
        return name

    @property
    def zmin(self) -> float:
        """Lowest absolute elevation in the height map."""
        return float(self.heights.min())

    @property
    def zmax(self) -> float:
        """Highest absolute elevation in the height map."""
        return float(self.heights.max())

    def dict(self) -> dict[str, JsonValue]:
        """A JSON-friendly debugging view (heights as nested lists)."""
        return {
            "kind": self.kind,
            "name": self.name,
            "size": list(self.size),
            "resolution": list(self.resolution),
            "heights": self.heights.tolist(),
            "base_z": self.base_z,
            "meta": dict(self.meta),
        }

    def __repr__(self) -> str:
        """Compact debug representation (never the full height map)."""
        nx, ny = self.resolution
        return (
            f"Terrain(name={self.name!r}, size={self.size}, "
            f"resolution=({nx}, {ny}), z=[{self.zmin:.4g}, "
            f"{self.zmax:.4g}], base_z={self.base_z:.4g})"
        )

    # ------------------------------------------------------------------ #
    # constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def flat(cls, size: tuple[float, float] = (0.0, 0.0)) -> None:
        """The absent terrain: ``None`` (``World.terrain.flat`` clears)."""
        return None

    @classmethod
    def heightmap(
        cls,
        heights: HeightsInput,
        size: tuple[float, float] | None = None,
        base_z: float = 0.0,
        name: str = "terrain",
        meta: Mapping[str, JsonValue] | None = None,
    ) -> Terrain:
        """Build a terrain from an explicit ``(ny, nx)`` elevation matrix."""
        array = np.asarray(heights, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(
                f"terrain: heights must be a (ny, nx) array, got shape " f"{array.shape}"
            )
        ny, nx = array.shape
        if size is None:
            size = (float(nx), float(ny))
        combined: dict[str, JsonValue] = dict(meta) if meta is not None else {}
        combined["name"] = name
        return cls(
            kind=_TERRAIN_KIND,
            size=size,
            resolution=(nx, ny),
            heights=array,
            base_z=base_z,
            meta=combined,
        )

    @classmethod
    def stairs(
        cls,
        steps: int,
        rise: float,
        run: float,
        width: float,
        base_z: float = 0.0,
        margin: float | None = None,
        resolution: tuple[int, int] | None = None,
        name: str = "terrain",
        meta: Mapping[str, JsonValue] | None = None,
    ) -> Terrain:
        """A staircase climbing +x: ``steps`` bands of ``rise`` by ``run``.

        Flat margins (default one ``run`` wide) sit at the bottom and top
        so a drop test has a landing on both sides.  Heights start at 0.
        """
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError(f"terrain stairs: steps must be an int >= 1, " f"got {steps!r}")
        rise = _finite(rise, "stairs rise")
        run = _finite(run, "stairs run")
        width = _finite(width, "stairs width")
        if rise <= 0 or run <= 0 or width <= 0:
            raise ValueError(
                f"terrain stairs: rise/run/width must be > 0, got " f"({rise}, {run}, {width})"
            )
        margin = run if margin is None else _finite(margin, "stairs margin")
        if margin < 0:
            raise ValueError(f"terrain stairs: margin must be >= 0, got " f"{margin!r}")
        sx = steps * run + 2.0 * margin
        sy = width
        nx, ny = resolution or (max(2, steps * 4 + 8), 8)
        xs, _ys = _grid(nx, ny, sx, sy)
        band = np.floor((xs - (-sx / 2.0 + margin)) / run) + 1.0
        levels = np.clip(band, 0.0, float(steps)) * rise
        heights = np.repeat(levels[None, :], ny, axis=0)
        extra: dict[str, JsonValue] = dict(meta) if meta is not None else {}
        extra.update(
            {
                "seed": None,
                "kind": "stairs",
                "steps": steps,
                "rise": rise,
                "run": run,
                "width": width,
                "margin": margin,
                "name": name,
            }
        )
        return cls.heightmap(heights, size=(sx, sy), base_z=base_z, name=name, meta=extra)

    @classmethod
    def slope(
        cls,
        angle_deg: float,
        size: tuple[float, float] = (4.0, 4.0),
        base_z: float = 0.0,
        resolution: tuple[int, int] | None = None,
        name: str = "terrain",
        meta: Mapping[str, JsonValue] | None = None,
    ) -> Terrain:
        """A linear ramp along +x whose rise matches ``angle_deg``."""
        angle = _finite(angle_deg, "slope angle_deg")
        if not -90.0 < angle < 90.0:
            raise ValueError(f"terrain slope: angle_deg must be in (-90, 90), got {angle!r}")
        sx, sy = _pair(size, "slope size", positive=True)
        nx, ny = resolution or (max(2, int(sx * 8)), max(2, int(sy * 8)))
        xs = np.linspace(-sx / 2.0, sx / 2.0, nx)
        levels = (xs + sx / 2.0) * math.tan(math.radians(angle))
        heights = np.repeat(levels[None, :], ny, axis=0)
        if not np.any(heights):
            raise ValueError("terrain slope: angle_deg produced an all-zero height map")
        extra: dict[str, JsonValue] = dict(meta) if meta is not None else {}
        extra.update({"seed": None, "kind": "slope", "angle_deg": angle, "name": name})
        return cls.heightmap(heights, size=(sx, sy), base_z=base_z, name=name, meta=extra)
