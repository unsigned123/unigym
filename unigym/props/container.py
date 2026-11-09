"""Box-family props: container / table / box (L1).

Three registered kinds (props yaml ``type:``), one handler:

  - ``table``     : single slab box (desktop), origin at the outer bottom
                    face center;
  - ``box``       : single solid cube, origin at the bottom face center --
                    the classic pick object (any hand-contact criterion is
                    a declared contact group, contacts/base.py);
  - ``container`` : open-top box = bottom slab + 4 side walls; the yaml
                    declares the INNER cavity as
                    ``aabb: {center: [x,y,z], half_extents: [hx,hy,hz]}``
                    (relative to the prop origin) -- the FormulaTask
                    ``point_in_box_fraction`` parameter source.

Geometry is generated, not authored: a PropSpec carries ``size`` (outer
envelope ``[sx, sy, sz]``) and (for container) the declared ``aabb``;
``box_layout()`` turns them into axis-aligned box primitives
``[(center(3), half(3))]`` which feed either a minimal URDF (single link,
multiple visual+collision boxes, generated into the repo-local
content-addressed cache for isaaclab's ``UrdfFileCfg``) or the MjSpec body
builder (props mujoco backend) -- "equivalent combination", one source of
geometry.  The isaaclab spawner uses a native ``CuboidCfg`` for the
single-box ``box`` kind (no generated URDF); the multi-box
``container``/``table`` kinds keep the generated URDF.  Box assets are
deliberately plain (no meshes/textures); color is optional and cosmetic.

The handler subclasses :class:`GenericProp` (root-pose tracking, free-joint
dynamic rigid body) and adds: aabb validation, aabb metadata exposure
(``aabb_center``/``aabb_half``), and world-frame ``get_world_pose`` for the
``object/*`` record-key layer (props/object_state).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from pathlib import Path

import numpy as np

from .base import (
    PropSpec,
)

# kind -> default total mass (kg); overridable via spawn: {mass: ...}
_DEFAULT_MASS = {"container": 1.0, "table": 15.0, "box": 0.1}
_DEFAULT_COLOR = {
    "container": (0.45, 0.30, 0.15),
    "table": (0.55, 0.55, 0.58),
    "box": (0.90, 0.75, 0.10),
}


# --------------------------------------------------------------------------- #
# spec validation (fail-fast, kind-specific)
# --------------------------------------------------------------------------- #
def _positive_size(spec: PropSpec) -> np.ndarray:
    size = np.asarray(spec.size, dtype=float) if spec.size is not None else None
    if size is None or size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError(
            f"prop {spec.name!r} (type {spec.type!r}): size must be a "
            f"[sx, sy, sz] list of positive numbers (outer envelope, m), "
            f"got {spec.size!r}"
        )
    return size


def _declared_aabb(spec: PropSpec) -> tuple[np.ndarray, np.ndarray]:
    aabb = spec.aabb or {}
    try:
        center = np.asarray(aabb["center"], dtype=float)
        half = np.asarray(aabb["half_extents"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"prop {spec.name!r} (type {spec.type!r}): aabb must be a "
            f"{{center: [x,y,z], half_extents: [hx,hy,hz]}} mapping relative "
            f"to the prop origin, got {spec.aabb!r}"
        ) from exc
    if center.shape != (3,) or half.shape != (3,):
        raise ValueError(
            f"prop {spec.name!r}: aabb center/half_extents must have exactly "
            f"3 numbers each, got {center.shape} / {half.shape}"
        )
    if not np.isfinite(center).all() or not np.isfinite(half).all():
        raise ValueError(f"prop {spec.name!r}: aabb values must be finite")
    if (half < 0).any():
        raise ValueError(
            f"prop {spec.name!r}: aabb half_extents must be non-negative, " f"got {half!r}"
        )
    return center, half


def validate_box_spec(spec: PropSpec) -> None:
    """Kind-specific PropSpec validation (called by the handler ctor)."""
    if spec.type not in _DEFAULT_MASS:
        return
    size = _positive_size(spec)
    center, half = _declared_aabb(spec)
    lo, hi = center - half, center + half
    # origin convention: bottom face center -> z in [0, sz]; x/y centered
    if lo[2] < -1e-9 or hi[2] > size[2] + 1e-9:
        raise ValueError(
            f"prop {spec.name!r}: aabb z-range [{lo[2]}, {hi[2]}] escapes "
            f"the outer envelope z [0, {size[2]}] (origin is the bottom "
            f"face center)"
        )
    if spec.type == "container" and lo[2] <= 1e-9:
        raise ValueError(
            f"prop {spec.name!r}: container cavity floor must sit ABOVE the "
            f"origin (a bottom-less container cannot hold objects); declare "
            f"aabb center z > half z (bottom slab thickness = "
            f"{center[2] - half[2]:.4f} m)"
        )
    for ax, s in ((0, size[0]), (1, size[1])):
        if abs(center[ax]) > s / 2 + 1e-9 or half[ax] > s / 2 + 1e-9:
            raise ValueError(
                f"prop {spec.name!r}: aabb escapes the outer envelope on "
                f"axis {ax} (half {half[ax]} > envelope half {s / 2})"
            )


# --------------------------------------------------------------------------- #
# geometry: one source for both backends
# --------------------------------------------------------------------------- #
def box_layout(spec: PropSpec) -> list[tuple[np.ndarray, np.ndarray]]:
    """Axis-aligned box primitives ``[(center(3), half(3))]`` for a spec.

    ``table``/``box``: a single box spanning the whole envelope.  The origin
    is the bottom-face center, so the box center sits at ``z = sz/2``.
    ``container``: the bottom slab (full footprint, up to the cavity floor)
    plus up to 4 side walls spanning ``z`` from the cavity floor to the top
    rim; a wall whose declared thickness is ~0 is skipped.
    ``usd`` : the mujoco box/box_texture tiers substitute the
    declared ``size`` envelope for the USD asset -- a single box, same as
    ``table``/``box`` (requires no aabb declaration).
    """
    size = _positive_size(spec)
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    if spec.type in ("table", "box", "usd"):
        boxes.append((np.array([0.0, 0.0, size[2] / 2]), size / 2))
        return boxes

    center, half = _declared_aabb(spec)
    sx, sy, sz = size
    z_floor = max(center[2] - half[2], 0.0)  # cavity floor height
    if z_floor > 1e-9:  # bottom slab
        boxes.append((np.array([0.0, 0.0, z_floor / 2]), np.array([sx / 2, sy / 2, z_floor / 2])))
    z_top = min(center[2] + half[2] + 0.0, sz)  # walls span to the top rim
    wall_h = (z_top - z_floor) / 2
    z_c = (z_top + z_floor) / 2
    if wall_h <= 1e-9:
        return boxes
    # side walls: each spans from the cavity edge (c + s*h) to the envelope
    # edge (s*size/2); ~zero-thickness walls are skipped
    for sign in (+1.0, -1.0):
        t = abs(size[0] / 2 - sign * center[0] - half[0])
        if t > 1e-9:
            cx = (center[0] + sign * half[0] + sign * size[0] / 2) / 2
            boxes.append((np.array([cx, 0.0, z_c]), np.array([t / 2, sy / 2, wall_h])))
    for sign in (+1.0, -1.0):
        t = abs(size[1] / 2 - sign * center[1] - half[1])
        if t > 1e-9:
            cy = (center[1] + sign * half[1] + sign * size[1] / 2) / 2
            boxes.append((np.array([0.0, cy, z_c]), np.array([sx / 2, t / 2, wall_h])))
    return boxes


def _box_urdf_xml(spec: PropSpec, boxes, mass: float, color) -> str:
    """Minimal single-link URDF: one inertial + N visual/collision boxes."""
    size = np.asarray(spec.size, dtype=float)
    # envelope-approximate diagonal inertia (props are not precision bodies)
    ixx = mass / 12.0 * (size[1] ** 2 + size[2] ** 2)
    iyy = mass / 12.0 * (size[0] ** 2 + size[2] ** 2)
    izz = mass / 12.0 * (size[0] ** 2 + size[1] ** 2)
    rgba = " ".join(f"{c:.4f}" for c in color) + " 1.0"
    parts = [
        '<?xml version="1.0"?>',
        f"<!-- generated box-family prop ({spec.type}, name={spec.name}); "
        "origin at the outer bottom-face center -->",
        '<robot name="box_prop">',
        '  <link name="base_link">',
        f'    <inertial><origin xyz="0 0 {size[2] / 2:.6f}" rpy="0 0 0"/>'
        f'<mass value="{mass:.6f}"/><inertia '
        f'ixx="{ixx:.8f}" iyy="{iyy:.8f}" izz="{izz:.8f}" '
        'ixy="0" ixz="0" iyz="0"/></inertial>',
    ]
    for i, (c, h) in enumerate(boxes):
        cxml = " ".join(f"{v:.6f}" for v in c)
        sxml = " ".join(f"{2 * v:.6f}" for v in h)
        parts += [
            f'    <visual><origin xyz="{cxml}" rpy="0 0 0"/><geometry>'
            f'<box size="{sxml}"/></geometry>'
            f'<material name="mat"><color rgba="{rgba}"/></material></visual>',
            f'    <collision><origin xyz="{cxml}" rpy="0 0 0"/><geometry>'
            f'<box size="{sxml}"/></geometry></collision>',
        ]
    parts += ["  </link>", "</robot>", ""]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# generated-URDF cache (content addressed, repo-local)
# --------------------------------------------------------------------------- #
_URDF_CACHE_ENV = "UNIGYM_URDF_CACHE"
"""Override the cache root (tests + isolated runs); default is repo-local."""


def urdf_cache_dir() -> Path:
    """The generated-URDF cache root (``UNIGYM_URDF_CACHE`` or repo-local)."""
    override = os.environ.get(_URDF_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2] / ".cache" / "unigym" / "urdf"


def _urdf_cache_path(xml: bytes) -> Path:
    """Content-addressed cache path: ``sha256(xml) + ".urdf"``.

    The hash covers the whole rendered XML, including the generated comment
    that carries ``type`` and ``name``.  Two props with identical geometry
    but different names therefore get two files; this is accepted so the
    cached file remains self-describing.  The number of files is bounded by
    the number of distinct prop declarations, not by spawn count -- the
    original leak (one new temp file per process spawn) is gone.
    """
    return urdf_cache_dir() / f"{hashlib.sha256(xml).hexdigest()}.urdf"


def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a same-directory temp + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def purge_urdf_cache(max_bytes: int | None = None, include_legacy: bool = False) -> dict:
    """Delete cached URDFs (all, or oldest-first until ``max_bytes``).

    Args:
        max_bytes: Keep the cache at or below this size, removing the oldest
            files first.  ``None`` removes every cached URDF.
        include_legacy: Also remove legacy ``/tmp/unigym_*_*.urdf`` temp
            files written by the pre-cache spawn path.

    Returns:
        A summary mapping with the cache dir, removed count, freed bytes and
        the legacy-file count.
    """
    root = urdf_cache_dir()
    files = sorted(root.glob("*.urdf")) if root.is_dir() else []
    sized = [(path.stat().st_mtime, path.stat().st_size, path) for path in files]
    total = sum(size for _, size, _ in sized)
    if max_bytes is None:
        doomed = sized
    else:
        doomed = []
        for entry in sorted(sized):
            if total <= max_bytes:
                break
            doomed.append(entry)
            total -= entry[1]
    removed = 0
    freed = 0
    for _, size, path in doomed:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        removed += 1
        freed += size
    legacy = _purge_legacy_urdfs() if include_legacy else 0
    return {"dir": str(root), "removed": removed, "freed_bytes": freed, "legacy_removed": legacy}


def _legacy_urdf_paths() -> list[Path]:
    """Legacy temp-URDF files from the pre-cache spawn path."""
    return sorted(Path(tempfile.gettempdir()).glob("unigym_*_*.urdf"))


def _purge_legacy_urdfs() -> int:
    """Remove legacy ``/tmp/unigym_*_*.urdf`` files; return the count."""
    removed = 0
    for path in _legacy_urdf_paths():
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        removed += 1
    return removed


def generate_box_urdf(spec: PropSpec) -> str:
    """Return the content-addressed cache path for a generated box URDF.

    Used by the multi-box ``container``/``table`` kinds.  The ``box`` kind
    takes the native isaaclab ``CuboidCfg`` route and never reaches here.
    Identical XML reuses one file; the default root is
    ``<repo>/.cache/unigym/urdf`` (gitignored) and is visible inside the
    isaaclab container because the repo mounts read-write at
    ``/workspace/unigym``.  ``UNIGYM_URDF_CACHE`` overrides the root.
    """
    from .primitive import resolve_mass

    mass = resolve_mass(spec, default=_DEFAULT_MASS[spec.type])
    color = tuple(float(v) for v in dict(spec.spawn).get("color", _DEFAULT_COLOR[spec.type]))
    xml = _box_urdf_xml(spec, box_layout(spec), mass, color).encode("utf-8")
    path = _urdf_cache_path(xml)
    if not path.exists():
        _write_atomic(path, xml)
    return str(path)


def __getattr__(name: str):
    """Lazy compat re-export of the isaaclab BoxProp handler (G4)."""
    if name in ("BoxProp", "_build_cuboid_spawn", "_box_origin_offset"):
        from .base import _compat_class

        if name == "BoxProp":
            return _compat_class("BoxProp")
        from importlib import import_module

        box = import_module("unigym.backends.isaaclab2.props.handlers.box")
        return getattr(box, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
