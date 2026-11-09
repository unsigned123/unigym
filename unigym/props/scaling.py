"""The official RoboCasa fixture scaling law as neutral build-time arithmetic.

Transcription source (frozen RoboCasa v1.0 ``8f3c96ec``, robosuite 1.5.2):

* ``robocasa/models/fixtures/fixture.py`` ``set_scale_from_size`` (the
  registry-size entry: per-axis target/native ratios, the None-axis
  native fallback, the ``max_size`` clamp);
* ``robocasa/models/objects/objects.py`` ``MujocoXMLObjectRobocasa.set_scale``
  (the operative per-element law for every ``Fixture`` -- the robocasa
  override wins over the robosuite generic via the ``Fixture`` MRO);
* ``robosuite/utils/mjcf_utils.py`` ``normalize_scale_array`` (shape
  normalization the override re-implements);
* ``robosuite/utils/transform_utils.py`` ``convert_quat`` + ``quat2mat``
  (the float32 rotation path feeding the rotation-aware geom sizes).

The mass/inertia half of the law is ABSENCE: neither scaling entry point
touches ``<inertial>``, ``mass`` or ``density`` anywhere (verified by
source inspection); compiled masses/inertias are the MuJoCo compiler's
density-times-volume derivation over the scaled geometry.  The scaled
provider payload therefore needs no inertia arithmetic -- it feeds the
same scaled geometry to the same compiler, and the mechanism goldens
pin the compiled equality exactly (docs/audits/v078-scale-face.md).

Registry size semantics (``[null, 0.47, null]``): a None entry keeps the
native axis via the official fallback order (x falls back to z then y;
y to x then z; z to x then y -- the official ``or`` chain, which is
equivalent to explicit None checks on the fail-closed positive domain
this module accepts).  Everything fail-closed: non-finite or
non-positive sizes/scales/natives raise ``ValueError`` where the
official path would silently produce degenerate geometry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .asset_ir import (
    AssetDefinition,
    Body,
    Geom,
    Joint,
    MeshDef,
    Site,
    asset_roots,
    iter_bodies,
)
from unigym.core.region import RegionDefinition

__all__ = [
    "parse_size_spec",
    "normalize_scale",
    "native_extents",
    "scale_from_registry_size",
    "resolve_build_scale",
    "scale_asset_definition",
]

#: robosuite.utils.transform_utils.EPS (the quat2mat degenerate-norm gate)
_EPS = np.finfo(float).eps * 4.0

#: the official region measurement keys, in lookup order
# (robocasa/models/fixtures/fixture.py ``width``/``depth``/``height``)
_REGION_KEYS = ("main", "bbox")


def _positive(value: Any, what: str) -> float:
    """One finite strictly-positive float (the fail-closed domain)."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{what} must be a finite positive number, got {value!r}")
    return float(value)


def parse_size_spec(value: Any) -> tuple:
    """Validate one registry size triple -> ``(float | None, ...)``.

    Each entry is ``None`` (native fallback axis) or a finite positive
    metre value; anything else is refused.  Returns a fresh tuple of
    ``float | None`` entries.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"size override must be a 3-entry sequence, got {value!r}")
    return tuple(
        None if entry is None else _positive(entry, f"size[{i}]") for i, entry in enumerate(value)
    )


def normalize_scale(value: Any) -> tuple:
    """The official scale normalization (scalar / 1-entry / 3-entry forms).

    Transcribed from ``MujocoXMLObjectRobocasa.set_scale`` (float/int ->
    uniform triple, one-entry -> uniform triple, three entries verbatim,
    anything else refused); positivity is this module's fail-closed
    addition on top.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        entries = [value]
    elif isinstance(value, (list, tuple)):
        entries = list(value)
    else:
        raise ValueError(f"scale must be a number or a sequence, got {value!r}")
    if len(entries) == 1:
        entries = [entries[0]] * 3
    if len(entries) != 3:
        raise ValueError(f"scale must have 1 or 3 entries, got {value!r}")
    return tuple(_positive(entry, f"scale[{i}]") for i, entry in enumerate(entries))


def _region_geoms(asset: AssetDefinition) -> dict:
    """The payload's region marker geoms keyed by their registry key.

    Mirrors the official region discovery (the ``reg_<key>`` name
    convention after the fixture naming prefix; our localized names
    carry ``_reg_<key>`` as the trailing segment).
    """
    found: dict[str, Geom] = {}
    for root in asset_roots(asset):
        for body in iter_bodies(root):
            for geom in body.geoms:
                if geom.name and "_reg_" in geom.name:
                    found[geom.name.rsplit("_reg_", 1)[1]] = geom
    return found


def native_extents(asset: AssetDefinition) -> tuple:
    """The payload's native (width, depth, height) from its region markers.

    The official ``cur_size``: the ``main`` region marker (else
    ``bbox``) read as full extents -- twice the half-sizes, with the
    official cylinder region convention ``[radius, radius, height]``.
    Fails closed when the payload carries no measurable region marker
    (the official constructor would crash on the same input).
    """
    regions = _region_geoms(asset)
    marker = None
    for key in _REGION_KEYS:
        marker = regions.get(key)
        if marker is not None:
            break
    if marker is None:
        raise ValueError(
            "payload carries no reg_main/reg_bbox region marker to measure; "
            f"(region keys found: {sorted(regions)})"
        )
    if marker.type == "cylinder":
        if len(marker.size) != 2:
            raise ValueError(
                f"region geom {marker.name!r}: cylinder size must be "
                f"(radius, half-height), got {marker.size!r}"
            )
        return (2.0 * marker.size[0], 2.0 * marker.size[0], 2.0 * marker.size[1])
    if len(marker.size) != 3:
        raise ValueError(
            f"region geom {marker.name!r}: expected a 3-entry half-size, " f"got {marker.size!r}"
        )
    return tuple(2.0 * half for half in marker.size)


def scale_from_registry_size(size: Any, native: tuple, max_size: Any = None) -> tuple:
    """The official ``set_scale_from_size`` arithmetic -> per-axis scale.

    ``size`` is a registry triple (None = native axis); ``native`` is
    :func:`native_extents` output; ``max_size`` is the optional
    registry clamp triple (the official downward-adjustment pass).
    Per-axis ratios for the specified axes, then the official native
    fallback for the None axes (x <- z <- y; y <- x <- z; z <- x <- y),
    then the official ``max_size`` clamp.  Fail-closed on non-positive
    natives/ratios (the official path would divide by or produce
    degenerate factors silently).
    """
    spec = parse_size_spec(size)
    clamp = None if max_size is None else parse_size_spec(max_size)
    cur = tuple(_positive(entry, f"native extents[{i}]") for i, entry in enumerate(native))
    scale: list = [
        spec[0] / cur[0] if spec[0] is not None else None,
        spec[1] / cur[1] if spec[1] is not None else None,
        spec[2] / cur[2] if spec[2] is not None else None,
    ]
    # the official fallback chain (fixture.py: the `or` ladder), written
    # as explicit None checks -- equivalent on the positive domain
    if scale[0] is None:
        scale[0] = scale[2] if scale[2] is not None else scale[1]
    if scale[1] is None:
        scale[1] = scale[0] if scale[0] is not None else scale[2]
    if scale[2] is None:
        scale[2] = scale[0] if scale[0] is not None else scale[1]
    if clamp is not None:
        adjustment = 1.0
        for i in range(3):
            if clamp[i] is None:
                continue
            adjustment = min(adjustment, clamp[i] / (cur[i] * scale[i]))
        scale = [entry * adjustment for entry in scale]
    return tuple(_positive(entry, f"scale[{i}]") for i, entry in enumerate(scale))


def resolve_build_scale(params: Mapping[str, Any], asset: AssetDefinition) -> tuple | None:
    """The build-time scale from the provider ``size``/``scale`` params.

    ``size`` (official registry semantics) resolves through
    :func:`native_extents` + :func:`scale_from_registry_size`;
    ``scale`` is the direct per-axis multiplier form.  The two params
    are mutually exclusive; neither present -> ``None`` (identity).
    """
    size = params.get("size")
    scale = params.get("scale")
    if size is not None and scale is not None:
        raise ValueError("size and scale params are mutually exclusive")
    if size is not None:
        return scale_from_registry_size(size, native_extents(asset))
    if scale is not None:
        return normalize_scale(scale)
    return None


def _quat2mat(quat: tuple) -> np.ndarray:
    """The official rotation path, transcribed numba-semantics-exact.

    The IR stores the MJCF (w, x, y, z) order; the official chain
    (``convert_quat`` to xyzw, then ``quat2mat``'s ``[3, 0, 1, 2]``
    reindex) lands back on (w, x, y, z), so the reordering folds away.
    The official ``quat2mat`` runs under the numba decorator
    (robosuite.utils.numba.jit_decorator, ENABLE_NUMBA), whose measured
    semantics this port reproduces bit-exactly:

    * the quaternion casts to float32 and the norm accumulates in
      float32 (``np.dot``);
    * the ``sqrt(2/n)`` factor divides in float64;
    * the in-place ``q *= factor`` rounds per element through float64
      back to float32;
    * ``np.outer`` products stay float32;
    * the diagonal entries evaluate in float64 (exact on the float32
      products), the off-diagonals subtract/add in float32 first and
      promote to float64 after.

    The result is a float64 matrix whose entries are float32-representable
    (diagonals) or float32-rounded (off-diagonals) -- the rotation the
    official geom-size law consumes.
    """
    q = np.asarray(quat, dtype=np.float64).astype(np.float32)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(3)
    factor = math.sqrt(2.0 / float(n))
    q = (q.astype(np.float64) * factor).astype(np.float32)
    q2 = np.outer(q, q)
    return np.array(
        [
            [
                1.0 - float(q2[2, 2]) - float(q2[3, 3]),
                float(q2[1, 2] - q2[3, 0]),
                float(q2[1, 3] + q2[2, 0]),
            ],
            [
                float(q2[1, 2] + q2[3, 0]),
                1.0 - float(q2[1, 1]) - float(q2[3, 3]),
                float(q2[2, 3] - q2[1, 0]),
            ],
            [
                float(q2[1, 3] - q2[2, 0]),
                float(q2[2, 3] + q2[1, 0]),
                1.0 - float(q2[1, 1]) - float(q2[2, 2]),
            ],
        ]
    )


def _scale_geom_size(size: tuple, quat: tuple, scale: tuple) -> tuple:
    """The official rotation-aware geom size law.

    Three-entry sizes (boxes/ellipsoids) scale per-axis in the world
    frame and come back through the inverse rotation (absolute value,
    per the official transcription); two-entry cylinders carry the
    radius on both in-plane axes, scale there, and keep the radius
    plus the z entry; anything else is refused exactly as the official
    path refuses it.  Identity quats (every authored row today) reduce
    to plain per-axis multiplication, bit-exactly.
    """
    rot = _quat2mat(quat)
    values = np.asarray(size, dtype=np.float64)
    if len(values) == 3:
        scaled = np.abs(np.linalg.inv(rot) @ ((rot @ values) * scale))
        return tuple(float(v) for v in scaled)
    if len(values) == 2:
        world = rot @ np.concatenate((values[0:1], values))
        scaled = np.abs(np.linalg.inv(rot) @ (world * scale))[1:3]
        return tuple(float(v) for v in scaled)
    raise ValueError(
        f"geom size must have 2 or 3 entries to scale (official refusal), " f"got {size!r}"
    )


def _scale_geom(geom: Geom, scale: tuple) -> Geom:
    """One geom: per-axis position, rotation-aware size, untouched mass facts.

    The official path moves ``pos`` and ``size`` only -- ``mass`` /
    ``density`` / ``rgba`` / contact hints are not its business, and the
    mesh-referencing form carries no size vector at all (the official
    ``size is not None`` guard).
    """
    updates: dict = {}
    if geom.pos != (0.0, 0.0, 0.0):
        updates["pos"] = tuple(float(p * s) for p, s in zip(geom.pos, scale))
    if geom.size:
        if geom.type == "mesh":
            raise ValueError(f"geom {geom.name!r}: mesh geoms carry no size vector")
        updates["size"] = _scale_geom_size(geom.size, geom.quat, scale)
    if not updates:
        return geom
    return Geom(
        geom.name,
        geom.type,
        updates.get("size", geom.size),
        pos=updates.get("pos", geom.pos),
        quat=geom.quat,
        material=geom.material,
        mass=geom.mass,
        density=geom.density,
        rgba=geom.rgba,
        group=geom.group,
        collidable=geom.collidable,
        mesh=geom.mesh,
        hints=geom.hints,
    )


def _scale_joint(joint: Joint, scale: tuple) -> Joint:
    """One joint: per-axis position; slide ranges by the minimum factor.

    The official slide branch scales the range endpoints by
    ``min(scale)`` regardless of the joint axis (the single-nonzero-axis
    index it computes is dead), refuses a zero axis, and silently skips
    multi-axis slides.  An unlimited joint (``range=None``, the
    declared unlimited face) carries no nominal range in the IR, and
    the compiled unlimited model has no range to scale.
    """
    updates: dict = {}
    if joint.pos != (0.0, 0.0, 0.0):
        updates["pos"] = tuple(float(p * s) for p, s in zip(joint.pos, scale))
    if joint.type == "slide" and joint.range is not None:
        axis = np.asarray(joint.axis, dtype=np.float64)
        nonzero = np.where(axis != 0)[0]
        if len(nonzero) == 0:
            raise ValueError(f"joint {joint.name!r}: axis must be non-zero for slide joints")
        if len(nonzero) == 1:
            factor = min(scale)
            updates["range"] = tuple(float(v * factor) for v in joint.range)
    if not updates:
        return joint
    return Joint(
        joint.name,
        joint.type,
        axis=joint.axis,
        pos=updates.get("pos", joint.pos),
        range=updates.get("range", joint.range),
        damping=joint.damping,
        armature=joint.armature,
        frictionloss=joint.frictionloss,
        springdamper=joint.springdamper,
        stiffness=joint.stiffness,
    )


def _scale_site(site: Site, scale: tuple) -> Site:
    """One site: per-axis position; the official size-branch law.

    Scalar sizes (the sphere radius form) take the global mean;
    two-entry vectors take the in-plane mean plus the z factor;
    three-entry vectors scale per-axis.
    """
    updates: dict = {}
    if site.pos != (0.0, 0.0, 0.0):
        updates["pos"] = tuple(float(p * s) for p, s in zip(site.pos, scale))
    if site.size is not None:
        updates["size"] = site.size * float(np.mean(scale))
    if site.size_vector is not None:
        values = np.asarray(site.size_vector, dtype=np.float64)
        if len(values) == 3:
            updates["size_vector"] = tuple(float(v * s) for v, s in zip(values, scale))
        elif len(values) == 2:
            updates["size_vector"] = (
                float(values[0] * float(np.mean(scale[:2]))),
                float(values[1] * scale[2]),
            )
        else:
            updates["size_vector"] = (float(values[0] * float(np.mean(scale))),)
    if not updates:
        return site
    return Site(
        site.name,
        pos=updates.get("pos", site.pos),
        size=updates.get("size", site.size),
        rgba=site.rgba,
        type=site.type,
        size_vector=updates.get("size_vector", site.size_vector),
    )


def _scale_body(body: Body, scale: tuple) -> Body:
    """One body subtree: per-axis frame positions, scaled elements.

    Body quats, gravity compensation and explicit inertials are not the
    official scaling's business (it never touches them) -- the
    compiler's mass derivation over the scaled geometry is the whole
    inertia law.
    """
    return Body(
        body.name,
        pos=tuple(float(p * s) for p, s in zip(body.pos, scale)),
        quat=body.quat,
        geoms=tuple(_scale_geom(geom, scale) for geom in body.geoms),
        joints=tuple(_scale_joint(joint, scale) for joint in body.joints),
        sites=tuple(_scale_site(site, scale) for site in body.sites),
        children=tuple(_scale_body(child, scale) for child in body.children),
        gravcomp=body.gravcomp,
        inertial=body.inertial,
    )


def _scale_mesh(mesh: MeshDef, scale: tuple) -> MeshDef:
    """One mesh declaration: the official elementwise ``scale *=`` law."""
    if mesh.scale is None:
        return MeshDef(mesh.name, mesh.file, scale, mesh.refquat)
    return MeshDef(
        mesh.name, mesh.file, tuple(float(v * s) for v, s in zip(mesh.scale, scale)), mesh.refquat
    )


def _scale_region(region: RegionDefinition, scale: tuple) -> RegionDefinition:
    """One region row: per-axis pose and full-extent factors."""
    return RegionDefinition(
        region.name,
        region.owner,
        region.kind,
        pos=tuple(float(p * s) for p, s in zip(region.pos, scale)),
        quat=region.quat,
        width_m=region.width_m * scale[0],
        depth_m=region.depth_m * scale[1],
        height_m=None if region.height_m is None else region.height_m * scale[2],
        body=region.body,
        site=region.site,
    )


def scale_asset_definition(asset: AssetDefinition, scale: tuple) -> AssetDefinition:
    """Apply one per-axis scale over the whole payload (the official law).

    Meshes, the body tree (geoms/joints/sites recursively) and the
    owned region rows scale; textures, materials, solver hints and the
    prop identity do not.  ``scale`` is a validated 3-tuple (see
    :func:`normalize_scale`).
    """
    scale = normalize_scale(scale)
    if asset.root is not None:
        root = _scale_body(asset.root, scale)
    else:
        root = None
    return AssetDefinition(
        name=asset.name,
        root=root,
        forest=tuple(_scale_body(entry, scale) for entry in asset.forest),
        meshes=tuple(_scale_mesh(mesh, scale) for mesh in asset.meshes),
        textures=asset.textures,
        materials=asset.materials,
        regions=tuple(_scale_region(region, scale) for region in asset.regions),
        solver_hints=asset.solver_hints,
    )
