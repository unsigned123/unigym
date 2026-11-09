"""The AssetDefinition aggregate: one payload = body tree + declarations.

``AssetDefinition`` is the stage-2 result every provider returns (the
renderer's input contract).  Construction validates the envelope
(anchored root / forest), the declaration tables and the region rows;
:func:`validate_asset_definition` adds the cheap cross-element
duplicate-name walk re-runnable at every consumer boundary.  The tree
walkers here are the shared iteration faces over the root(s).
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ._declarations import MaterialDef, MeshDef, TextureDef
from ._elements import Body, SolverHints, iter_bodies
from ._guards import _require_unique

if TYPE_CHECKING:
    from unigym.model import ModelIR


def _fvec2(values) -> tuple[float, float]:
    """A numeric sequence as a fixed 2-vector (``range``/``texrepeat``)."""
    return (float(values[0]), float(values[1]))


def _fvec3(values) -> tuple[float, float, float]:
    """A numeric sequence as a fixed 3-vector (``pos``/``axis``/``size``)."""
    return (float(values[0]), float(values[1]), float(values[2]))


def _fvec4(values) -> tuple[float, float, float, float]:
    """A numeric sequence as a fixed 4-vector (``quat``/``rgba``)."""
    return (float(values[0]), float(values[1]),
            float(values[2]), float(values[3]))


def joint_names(asset: AssetDefinition) -> list[str]:
    """All generated joint names in tree order (resolve-time coverage face)."""
    return [joint.name for body in iter_asset_bodies(asset) for joint in body.joints]


def asset_roots(asset: AssetDefinition) -> tuple:
    """The payload's root bodies: the anchored root or the forest.

    The FOREST form (v075 P1: the verbatim tabletop kitchen bake)
    carries several WORLDBODY-LEVEL roots -- the attach machinery lands
    each under the attach frame, so a composed scene reproduces the
    source scene's sibling layout with no synthetic wrapper body.
    """
    if asset.forest:
        return asset.forest
    return (asset.root,)


def iter_asset_bodies(asset: AssetDefinition) -> Iterator[Body]:
    """Yield every payload body across the root(s), depth-first."""
    for root in asset_roots(asset):
        yield from iter_bodies(root)


def validate_asset_definition(asset: AssetDefinition) -> None:
    """Tree-level validation (duplicate element names across the payload).

    Construction-time guards run in every ``__post_init__`` and can never
    be bypassed (frozen dataclasses); this walk adds the cross-element
    checks and is cheap enough to re-run at every consumer boundary.
    """
    seen: dict[str, set] = {"body": set(), "geom": set(), "joint": set(), "site": set()}
    for body in iter_asset_bodies(asset):
        _require_unique(seen, "body", body.name)
        for geom in body.geoms:
            if geom.name is not None:
                _require_unique(seen, "geom", geom.name)
        for joint in body.joints:
            _require_unique(seen, "joint", joint.name)
        for site in body.sites:
            _require_unique(seen, "site", site.name)


@dataclass(frozen=True)
class AssetDefinition:
    """One generated prop asset: a body tree + declarations + solver hints.

    ``name`` is the prop identity the payload belongs to (the renderer
    fail-fasts on a mismatch with the resolved PropSpec name).
    ``meshes`` / ``textures`` / ``materials`` are the declaration tables
    (meshes S2.2, appearance S2.1): the renderer adds them to the child
    spec before the body tree, so the attach pass namespaces them (and
    every logical reference to them) together with the tree.

    The payload is an ANCHORED single root (``root``) or a FOREST of
    worldbody-level roots (``forest``, v075 P1: the verbatim tabletop
    kitchen bake) -- exactly one of the two.  The prop-runtime attach
    faces consume the anchored form only; the forest rides the scene
    composer's verbatim face (:mod:`unigym.backends.shared.mjfamily.scene.scene_render`).
    """

    name: str
    root: Body | None = None
    forest: tuple = ()
    meshes: tuple = ()
    textures: tuple = ()
    materials: tuple = ()
    regions: tuple = ()
    solver_hints: SolverHints = field(default_factory=SolverHints)

    def __post_init__(self):
        """Validate the envelope, the declaration tables and the tree."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("asset definition name must be a non-empty string")
        if (self.root is None) == (not self.forest):
            raise ValueError("asset definition: exactly one of root / forest must be " "declared")
        if self.root is not None and not isinstance(self.root, Body):
            raise ValueError("asset definition root must be a Body")
        if self.forest:
            if not isinstance(self.forest, tuple):
                raise ValueError("asset definition forest must be a tuple of Body")
            for entry in self.forest:
                if not isinstance(entry, Body):
                    raise ValueError(
                        "asset definition forest entries must be Body, "
                        f"got {type(entry).__name__}"
                    )
        for f, etype in (("meshes", MeshDef), ("textures", TextureDef), ("materials", MaterialDef)):
            entries = getattr(self, f)
            if not isinstance(entries, tuple):
                raise ValueError(f"asset definition {f} must be a tuple of " f"{etype.__name__}")
            for entry in entries:
                if not isinstance(entry, etype):
                    raise ValueError(
                        f"asset definition {f} entries must be "
                        f"{etype.__name__}, got {type(entry).__name__}"
                    )
        from unigym.core.region import RegionDefinition

        if not isinstance(self.regions, tuple) or any(
            not isinstance(region, RegionDefinition) for region in self.regions
        ):
            raise ValueError("asset definition regions must be a tuple of RegionDefinition")
        if not isinstance(self.solver_hints, SolverHints):
            raise ValueError("asset definition solver_hints must be a SolverHints")
        seen_mesh = set()
        seen_mat = set()
        seen_tex = set()
        for mesh in self.meshes:
            if mesh.name in seen_mesh:
                raise ValueError(f"duplicate mesh name {mesh.name!r} in the asset " f"definition")
            seen_mesh.add(mesh.name)
        for tex in self.textures:
            if tex.name in seen_tex:
                raise ValueError(f"duplicate texture name {tex.name!r} in the asset " f"definition")
            seen_tex.add(tex.name)
        for mat in self.materials:
            if mat.name in seen_mat:
                raise ValueError(
                    f"duplicate material name {mat.name!r} in the asset " f"definition"
                )
            seen_mat.add(mat.name)
            if mat.texture is not None and mat.texture.name not in seen_tex:
                raise ValueError(
                    f"material {mat.name!r}: texture reference "
                    f"{mat.texture.name!r} is not declared in the payload "
                    f"textures table"
                )
        for body in iter_asset_bodies(self):
            for geom in body.geoms:
                if geom.mesh is not None and geom.mesh not in seen_mesh:
                    raise ValueError(
                        f"geom {geom.name!r}: mesh name "
                        f"{geom.mesh!r} is not declared in the payload "
                        f"meshes table"
                    )
        validate_asset_definition(self)

    def to_model_ir(self, name: str | None = None) -> ModelIR:
        """Expand this asset payload into the unified :class:`ModelIR`.

        The result is declaration-only (``compiled=False``): an asset
        :class:`Inertial` sets ``Body.explicitinertial`` but is NOT
        projected into the compiled-only ``mass`` / ``inertia`` /
        ``ipos`` / ``iquat`` fields.  Textures and meshes have no base
        ``ModelIR`` table (they are backend extensions), so they are
        intentionally dropped; a material's texture reference survives
        on ``Material.textures``.  ``collidable`` maps to
        ``contype/conaffinity`` = 1/1 (collidable) or 0/0 (visual twin).
        """
        from unigym.model import (
            Body as ModelBody,
            Geom as ModelGeom,
            Joint as ModelJoint,
            Material as ModelMaterial,
            ModelIR,
            Site as ModelSite,
            Table,
        )

        bodies = []
        joints = []
        geoms = []
        sites = []

        def walk(body, parent_name, parent_path):
            path = parent_path + (body.name,)
            bodies.append(
                ModelBody(
                    name=body.name,
                    parent=parent_name,
                    path=path,
                    qualname="/".join(path) or body.name,
                    pos=_fvec3(body.pos),
                    quat=_fvec4(body.quat),
                    explicitinertial=body.inertial is not None,
                )
            )
            for joint in body.joints:
                joints.append(
                    ModelJoint(
                        name=joint.name,
                        type=joint.type,
                        body=body.name,
                        axis=_fvec3(joint.axis),
                        range=(
                            _fvec2(joint.range)
                            if joint.range is not None
                            else None
                        ),
                        limited=joint.range is not None,
                        pos=_fvec3(joint.pos),
                        damping=float(joint.damping),
                        armature=float(joint.armature),
                        path=path + (joint.name,),
                        qualname=f"{body.name}/{joint.name}",
                    )
                )
            for index, geom in enumerate(body.geoms):
                geom_name = geom.name or f"{body.name}__geom_{index}"
                size = (
                    _fvec3(geom.size)
                    if geom.size
                    else (0.0, 0.0, 0.0)
                )
                collidable = bool(geom.collidable)
                geoms.append(
                    ModelGeom(
                        name=geom_name,
                        type=geom.type,
                        body=body.name,
                        pos=_fvec3(geom.pos),
                        quat=_fvec4(geom.quat),
                        size=size,
                        material=geom.material.name if geom.material else None,
                        group=geom.group,
                        contype=1 if collidable else 0,
                        conaffinity=1 if collidable else 0,
                        density=(
                            float(geom.density)
                            if geom.density is not None
                            else None
                        ),
                        path=path + (geom_name,),
                        qualname=f"{body.name}/{geom_name}",
                    )
                )
            for site in body.sites:
                if site.size_vector is not None:
                    size = _fvec3(site.size_vector)
                elif site.size is not None:
                    size = (float(site.size), float(site.size),
                            float(site.size))
                else:
                    size = (0.0, 0.0, 0.0)
                sites.append(
                    ModelSite(
                        name=site.name,
                        type=site.type or "sphere",
                        body=body.name,
                        pos=_fvec3(site.pos),
                        size=size,
                        path=path + (site.name,),
                        qualname=f"{body.name}/{site.name}",
                    )
                )
            for child in body.children:
                walk(child, body.name, path)

        for root in asset_roots(self):
            walk(root, None, ())

        materials = [
            ModelMaterial(
                name=mat.name,
                rgba=(
                    _fvec4(mat.rgba)
                    if mat.rgba is not None
                    else (1.0, 1.0, 1.0, 1.0)
                ),
                textures=(mat.texture.name,) if mat.texture is not None else (),
                reflectance=mat.reflectance,
                specular=mat.specular,
                shininess=mat.shininess,
                texrepeat=(
                    _fvec2(mat.texrepeat)
                    if mat.texrepeat is not None
                    else (1.0, 1.0)
                ),
                texuniform=(
                    bool(mat.texuniform)
                    if mat.texuniform is not None
                    else False
                ),
            )
            for mat in self.materials
        ]

        model = ModelIR(
            name=name or self.name,
            backend=None,
            compiled=False,
            bodies=Table(bodies),
            joints=Table(joints),
            geoms=Table(geoms),
            sites=Table(sites),
            materials=Table(materials),
            meta={
                "source": "asset",
                "asset": self.name,
            },
        )
        model.validate()
        return model
