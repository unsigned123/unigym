"""Dynamic primitive props: cylinder / capsule (S2c).

The content/props path owns dynamic free bodies (a freejoint plus
mass/inertia).  Sphere and box already have handlers with that behaviour
(:class:`~unigym.props.ball.BallProp` and
:class:`~unigym.props.container.BoxProp`); this module adds the missing
cylinder and capsule kinds with the SAME declaration vocabulary:

    objects:
      types:
        - name: peg
          type: cylinder
          radius: 0.03
          half_height: 0.05          # local z, matching the scene primitives
          spawn: {mass: 0.2, friction: 1.0, rgba: [0.8, 0.6, 0.2, 1.0]}
          placement: {mode: fixed, x_range: [0.4, 0.4],
                      y_range: [0.0, 0.0], z_range: [0.3, 0.3]}

Dimensions are the neutral scene-primitive ones: ``radius`` and
``half_height`` (``cylinder``'s half-height; for ``capsule`` the
half-length of its cylindrical section, so its resting centre height is
``radius + half_height``).  ``spawn.mass`` and ``spawn.density`` are
mutually exclusive; mass wins as an explicit declaration, density is
multiplied by the analytic volume, and an absent pair falls back to
``DEFAULT_MASS``.  The backend always receives the resolved MASS: mujoco
derives inertia from the geom (engine-computed), isaaclab applies
``MassPropertiesCfg`` (engine-computed from the spawner shape).

The box and sphere kinds keep their existing handlers (``CuboidCfg`` on
isaaclab for the single-box kind, generated URDF for container/table, and
``SphereCfg`` respectively) so their long-standing configs and state keys
are unchanged; they share :func:`resolve_mass` for the density vocabulary.
"""
from __future__ import annotations

import math


from .base import (
    PropSpec,
)

#: default total mass per primitive kind (kg); overridden by spawn.mass or
#: spawn.density.  sphere/box mirror their existing handlers' defaults.
DEFAULT_MASS = {"sphere": 0.1, "cylinder": 0.1, "capsule": 0.1, "box": 0.1}


def _positive(value, what: str) -> float:
    """Validate one finite strictly-positive scalar."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{what} must be a finite number > 0, got {value!r}")
    return float(value)


def primitive_volume(kind: str, spec: PropSpec) -> float:
    """Analytic volume (m^3) of one primitive declaration.

    Raises:
        ValueError: Unknown kind or a missing/invalid dimension.
    """
    if kind == "sphere":
        radius = _dimension(spec, "radius")
        return 4.0 / 3.0 * math.pi * radius ** 3
    if kind == "cylinder":
        radius = _dimension(spec, "radius")
        half = _dimension(spec, "half_height")
        return math.pi * radius ** 2 * (2.0 * half)
    if kind == "capsule":
        radius = _dimension(spec, "radius")
        half = _dimension(spec, "half_height")
        return (math.pi * radius ** 2 * (2.0 * half)
                + 4.0 / 3.0 * math.pi * radius ** 3)
    if kind == "box":
        if (not isinstance(spec.size, (list, tuple)) or len(spec.size) != 3):
            raise ValueError(
                f"prop {spec.name!r} (type 'box'): size must be "
                f"[sx, sy, sz], got {spec.size!r}")
        sides = [_positive(v, f"prop {spec.name!r} size component")
                 for v in spec.size]
        return sides[0] * sides[1] * sides[2]
    raise ValueError(
        f"prop {spec.name!r}: no primitive volume for kind {kind!r} "
        f"(known: sphere | cylinder | capsule | box)")


def _dimension(spec: PropSpec, name: str) -> float:
    """Read a radius/half_height from the spec, then spawn, then fail."""
    value = getattr(spec, name, None)
    if value is None:
        value = dict(spec.spawn or {}).get(name)
    if value is None:
        raise ValueError(
            f"primitive prop {spec.name!r} (type {spec.type!r}): "
            f"{name} is required")
    return _positive(value, f"prop {spec.name!r} {name}")


def resolve_mass(spec: PropSpec, default: float | None = None) -> float:
    """Resolve ``spawn.mass`` / ``spawn.density`` to one explicit mass.

    ``mass`` and ``density`` are mutually exclusive.  A declared mass is
    passed through verbatim (the engine derives inertia from the shape);
    a density is multiplied by the analytic volume.  When neither is
    declared the kind's default is used, never silently zero.
    """
    spawn = dict(spec.spawn or {})
    mass = spawn.get("mass")
    density = spawn.get("density")
    if mass is not None and density is not None:
        raise ValueError(
            f"prop {spec.name!r}: spawn.mass and spawn.density are "
            f"mutually exclusive (declare one)")
    if mass is not None:
        return _positive(mass, f"prop {spec.name!r} spawn.mass")
    if density is not None:
        density = _positive(density, f"prop {spec.name!r} spawn.density")
        return density * primitive_volume(spec.type, spec)
    if default is None:
        default = DEFAULT_MASS.get(spec.type)
    if default is None:
        raise ValueError(
            f"prop {spec.name!r}: no default mass for kind {spec.type!r}")
    return _positive(default, f"prop {spec.name!r} default mass")


def material_colors(spawn: dict) -> tuple[float, float, float]:
    """The cosmetic colour from ``spawn.color`` / ``spawn.rgba``."""
    raw = spawn.get("color", spawn.get("rgba", (0.8, 0.6, 0.2)))
    return tuple(float(v) for v in list(raw)[:3])


def validate_primitive_spec(spec: PropSpec) -> None:
    """Fail-fast validation for the cylinder / capsule kinds."""
    if spec.type not in ("cylinder", "capsule"):
        return
    spec.radius = _dimension(spec, "radius")
    spec.half_height = _dimension(spec, "half_height")
    resolve_mass(spec)   # validates mass/density exclusivity + positivity


def __getattr__(name: str):
    """Lazy compat re-export of the isaaclab primitive handler (G4)."""
    if name in ("PrimitiveProp", "_build_primitive_spawn"):
        from importlib import import_module

        handler = import_module(
            "unigym.backends.isaaclab2.props.handlers.primitive")
        return getattr(handler, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
