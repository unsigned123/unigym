"""Neutral region declarations shared by scene and asset definitions.

Standard library only: the neutral-name guard is inlined here so this
module never imports the props (asset-IR) layer.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

#: engine element names a neutral declaration may never be NAMED after
#: (the mechanical anti-leak rule; the same closed set the scene IR uses).
_ENGINE_ELEMENT_NAMES = frozenset(
    {
        "body",
        "geom",
        "joint",
        "site",
        "freejoint",
        "frame",
        "include",
        "asset",
        "default",
        "worldbody",
        "world",
        "actuator",
        "motor",
        "position",
        "velocity",
        "intvelocity",
        "general",
        "damper",
        "muscle",
        "adhesion",
        "contact",
        "pair",
        "equality",
        "tendon",
        "sensor",
        "keyframe",
        "numeric",
        "text",
        "tuple",
        "flex",
        "skin",
        "hfield",
        "texture",
        "material",
        "mesh",
        "skybox",
    }
)

_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")


def _name(kind: str, value: str) -> str:
    """Apply the shared neutral element-name guard without importing asset IR."""
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise ValueError(f"{kind} name {value!r} must be a lowercase snake_case identifier")
    if value in _ENGINE_ELEMENT_NAMES:
        raise ValueError(
            f"{kind} name {value!r} is an engine element name -- providers "
            f"speak the neutral IR vocabulary only (design v072 §12.4-1)")
    return value


def _vector(name: str, value, length: int) -> tuple:
    """Validate and normalize a finite vector without backend dependencies."""
    if (not isinstance(value, (list, tuple)) or len(value) != length
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in value)):
        raise ValueError(f"region: {name} must be {length} finite numbers")
    return tuple(float(v) for v in value)


@dataclass(frozen=True)
class RegionDefinition:
    """A named owner-local support, interior, or opening region.

    Dimensions are full extents in metres: support uses width/depth, interior
    uses width/depth/height, and opening uses width/height plus depth_m. Poses
    are right-handed owner-local metres and wxyz quaternions. The owner and
    optional body/site references are local to the containing scene instance
    or asset definition.
    """

    name: str
    owner: str
    kind: str
    pos: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)
    width_m: float = 0.0
    depth_m: float = 0.0
    height_m: float | None = None
    body: str | None = None
    site: str | None = None

    def __post_init__(self):
        """Validate names, kind-specific dimensions, and the local pose."""
        object.__setattr__(self, "name", _name("region", self.name))
        object.__setattr__(self, "owner", _name("region owner", self.owner))
        if self.kind not in ("support", "interior", "opening"):
            raise ValueError(f"region {self.name!r}: unsupported kind {self.kind!r}")
        object.__setattr__(self, "pos", _vector("pos", self.pos, 3))
        quat = _vector("quat", self.quat, 4)
        if sum(v * v for v in quat) < 1e-24:
            raise ValueError(f"region {self.name!r}: quat must be non-zero")
        object.__setattr__(self, "quat", quat)
        for field in ("width_m", "depth_m"):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"region {self.name!r}: {field} must be finite and > 0")
            object.__setattr__(self, field, float(value))
        if self.kind == "interior":
            if (isinstance(self.height_m, bool)
                    or not isinstance(self.height_m, (int, float))
                    or not math.isfinite(self.height_m) or self.height_m <= 0):
                raise ValueError(f"region {self.name!r}: interior height_m must be finite and > 0")
            object.__setattr__(self, "height_m", float(self.height_m))
        elif self.height_m is not None:
            raise ValueError(f"region {self.name!r}: height_m is only valid for interior")
        for field in ("body", "site"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _name(f"region {field}", value))


__all__ = ["RegionDefinition"]
