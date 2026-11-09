"""mujoco scene composer: neutral scene IR -> MjSpec (design v075 §3).

The L1 mechanism half of the scenes package: a neutral
:class:`~unigym.scenes.scene_ir.Scene` is BUILT into a single MjSpec
here -- ONE loop over the instances, no engine-special scene branch:

* PRIMITIVE GENERATORS (the closed core registry) return a neutral
  :class:`~unigym.scenes.primitives.ShapePayload`; this module is the
  ONLY place that translates it into engine elements (OUTER envelope ->
  half extents, ``layer`` hint -> the engine's visual/collision
  numbering, logical material -> the declared material table);
* ASSET INSTANCES resolve through the EXISTING props path --
  :func:`unigym.props.templates.resolve_prop_template` (the template's
  own schema gate) feeds the generated asset payload into the UNCHANGED
  :func:`unigym.backends.shared.mjfamily.scene.asset_render.add_ir_body` /
  ``add_ir_free_body`` attach machinery, so a composed subtree is
  element-for-element what the prop pipeline would attach on its own;
  the instance pose lands on the attach frame (the subtree keeps its
  local geometry verbatim);
* SCENE OPTIONS are this module's SOLE interpretation face (the
  asset-IR ``solver_hints`` ruling): the neutral ``Scene.engine_hints``
  mapping transfers onto ``spec.option`` (enum fields resolve by member
  name), and :class:`~unigym.scenes.primitives.ContactSpec` reuses the
  shared contact-kwarg interpreter.

Generated assets never touch disk (built + attached in memory, like
props).  Runtime integration is NOT part of this module: the composer
returns the compiled-ready spec (the P1 tabletop decomposition wires it
into the env's model-load seam).
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from types import MappingProxyType
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing-only (mujoco optional runtime)
    import mujoco

from unigym.core.asset_root import resolve_asset_path
from unigym.core.region import RegionDefinition
from unigym.scenes.generators import (
    build_payload,
    generator_schema,
    registered_generators,
)
from unigym.scenes.primitives import (
    BoxShape,
    CapsuleShape,
    CylinderShape,
    PlaneShape,
    SphereShape,
)
from unigym.scenes.scene_ir import Scene, validate_scene

_SCENE_MODEL = (
    '<mujoco model="unigym_scene_{name}">\n'
    '  <compiler angle="radian"/>\n'
    "  <worldbody/>\n"
    "</mujoco>\n"
)


@dataclass(frozen=True)
class RegionAttachment:
    """Read-only composed region declaration with fully qualified references."""

    name: str
    kind: str
    owner_body: str
    body: str | None
    site: str | None
    pos: tuple
    quat: tuple
    width_m: float
    depth_m: float
    height_m: float | None


@dataclass
class ComposedScene:
    """A spec-delegating result with an immutable region metadata table."""

    spec: mujoco.MjSpec
    regions: Mapping[str, object]
    terrain: object = None

    def __getattr__(self, name):
        """Delegate reads to the contained compiled-ready spec."""
        return getattr(self.spec, name)

    def compile(self):
        """Compile the contained MuJoCo specification."""
        return self.spec.compile()


@dataclass(frozen=True)
class BoundRegion:
    """A region resolved against model body/site ids, with live world pose."""

    definition: RegionAttachment
    owner_body_id: int
    body_id: int | None
    site_id: int | None

    def world_pose(self, model, data) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return world position and wxyz orientation from the current owner pose."""
        import mujoco
        import numpy as np

        owner_pos = data.xpos[self.owner_body_id]
        owner_mat = data.xmat[self.owner_body_id].reshape(3, 3)
        local_pos = np.asarray(self.definition.pos)
        pos = owner_pos + owner_mat @ local_pos
        owner_quat = np.empty(4)
        mujoco.mju_mat2Quat(owner_quat, owner_mat.reshape(-1))
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, owner_quat, np.asarray(self.definition.quat))
        return tuple(float(v) for v in pos), tuple(float(v) for v in quat)


def bind_regions(model, data, regions) -> object:
    """Validate composed body/site references and return immutable bindings."""
    import mujoco

    bound = {}
    for name, region in regions.items():
        owner_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     region.owner_body)
        if owner_id < 0:
            raise ValueError(f"region {name!r}: dangling owner {region.owner_body!r}")
        body_id = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     region.body) if region.body else None)
        site_id = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                     region.site) if region.site else None)
        if body_id is not None and body_id < 0:
            raise ValueError(f"region {name!r}: dangling body {region.body!r}")
        if site_id is not None and site_id < 0:
            raise ValueError(f"region {name!r}: dangling site {region.site!r}")
        if body_id is not None and model.body_parentid[body_id] != owner_id:
            raise ValueError(
                f"region {name!r}: body {region.body!r} is not an immediate child of its owner")
        if site_id is not None and model.site_bodyid[site_id] != owner_id:
            raise ValueError(
                f"region {name!r}: site {region.site!r} is not attached to its owner")
        bound[name] = BoundRegion(region, owner_id, body_id, site_id)
    return MappingProxyType(bound)


def _bound_region_definition(bound, model, data) -> RegionDefinition:
    """Convert composed attachment metadata into a world-frame placement region.

    The carrier is built without re-running the asset-IR declaration
    guards: its name is the region's own vocabulary name (the composed
    ``<instance>.<region>`` key is the resolver dict's address) and its
    owner is the COMPILED body name (prefixed scene namespace), which
    the neutral-name guard does not spell.  Both layers were validated
    at declaration and at bind time; the placement consumers address
    regions through the resolver mapping.
    """
    pos, quat = bound.world_pose(model, data)
    region = RegionDefinition.__new__(RegionDefinition)
    object.__setattr__(region, "name", bound.definition.name.rsplit(".", 1)[-1])
    object.__setattr__(region, "owner", bound.definition.owner_body)
    object.__setattr__(region, "kind", bound.definition.kind)
    object.__setattr__(region, "pos", pos)
    object.__setattr__(region, "quat", quat)
    object.__setattr__(region, "width_m", bound.definition.width_m)
    object.__setattr__(region, "depth_m", bound.definition.depth_m)
    object.__setattr__(region, "height_m", bound.definition.height_m)
    object.__setattr__(region, "body", None)
    object.__setattr__(region, "site", None)
    return region


def bind_placement_regions(model, declared_regions) -> dict:
    """Resolve declared scene regions into world-frame placement regions.

    The env construction seam over :func:`bind_regions`: a fresh private
    MjData is forwarded once so every owner body pose is the compiled
    model's initial state, and each bound attachment becomes a
    :class:`~unigym.core.region.RegionDefinition` carrying its
    world-frame pos/quat at that moment (the reset-placement consumers
    read the resolved table verbatim).
    """
    import mujoco

    region_data = mujoco.MjData(model)
    mujoco.mj_forward(model, region_data)
    bound = bind_regions(model, region_data, declared_regions)
    return {
        name: _bound_region_definition(region, model, region_data)
        for name, region in bound.items()
    }


class _InstanceHandler:
    """Minimal asset-renderer handler shim for one scene asset instance.

    The scene face has no PropManager: the resolved template entry IS
    the declaration.  ``spec.joints`` stays EMPTY (a scene instance has
    no config-override face -- the instance params are the
    declaration); actuated templates are REFUSED (scene instances are
    passive assemblies; actuation is prop-runtime vocabulary, and the
    shared renderer's servo face is out of the scene axis).
    """

    def __init__(self, name: str, entry: dict):
        joints = dict(entry.get("joints") or {})
        actuated = sorted(j for j, p in joints.items()
                          if p.get("mode") == "actuated")
        if actuated:
            raise ValueError(
                f"scene instance {name!r}: actuated joints {actuated} are "
                f"prop-runtime vocabulary -- scene instances are passive "
                f"assemblies")
        self.spec = SimpleNamespace(name=name, joints={})
        self.joints = joints
        self._asset_ranges: dict = {}


_FLOOR_GEOM_NAMES = frozenset(
    {"floor", "ground", "ground_plane", "groundplane"}
)
"""Base-model ground plane names the terrain replaces (by geom name)."""

_HFIELD_BASE_EPS = 1e-3
"""Smallest positive hfield base depth (MJCF rejects a non-positive one)."""


def _terrain_shape(terrain):
    """Return ``(numpy, heights, zmin, elevation, normalized)`` for *terrain*."""
    import numpy as np

    heights = np.asarray(terrain.heights, dtype=np.float64)
    zmin = float(heights.min())
    elevation = float(heights.max() - heights.min())
    normalized = ((heights - zmin) / elevation if elevation > 0.0
                  else np.zeros_like(heights))
    return np, heights, zmin, elevation, normalized


def _add_terrain_hfield(spec, terrain) -> dict:
    """Render *terrain* as one MJCF hfield geom."""
    import mujoco

    _np, _heights, zmin, elevation, normalized = _terrain_shape(terrain)
    nx, ny = terrain.resolution
    sx, sy = terrain.size
    name = terrain.name
    base_z = max(float(terrain.base_z), _HFIELD_BASE_EPS)
    spec.add_hfield(
        name=name,
        size=[sx / 2.0, sy / 2.0, elevation, base_z],
        nrow=ny,
        ncol=nx,
        userdata=normalized.ravel().tolist(),
    )
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=name,
        pos=[0.0, 0.0, zmin],
        group=0,
    )
    return {"kind": "hfield", "name": name}


def _add_terrain_mesh(spec, terrain) -> dict:
    """Render *terrain* as one triangulated mesh (engines without hfield)."""
    import mujoco

    np, heights, _zmin, _elevation, _normalized = _terrain_shape(terrain)
    nx, ny = terrain.resolution
    sx, sy = terrain.size
    name = terrain.name
    xs = (np.arange(nx) * (sx / (nx - 1))) - sx / 2.0
    ys = (np.arange(ny) * (sy / (ny - 1))) - sy / 2.0
    verts = []
    for iy in range(ny):
        for ix in range(nx):
            verts.extend([float(xs[ix]), float(ys[iy]),
                          float(heights[iy, ix])])
    faces = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a = iy * nx + ix
            b = a + 1
            c = a + nx
            d = c + 1
            faces.extend([a, b, c, b, d, c])
    spec.add_mesh(name=name, uservert=verts, userface=faces)
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=name,
        group=0,
    )
    return {"kind": "mesh", "name": name}


def _add_terrain(spec, terrain) -> dict:
    """Render the neutral terrain as ONE collidable geom + its metadata.

    A zero-elevation map selects the mesh path: MuJoCo normalizes hfield
    elevation data by its own range, so a constant field degenerates to a
    zero-thickness surface with no usable collision solid.
    """
    _np, _heights, _zmin, elevation, _normalized = _terrain_shape(terrain)
    if hasattr(spec, "add_hfield") and elevation > 0.0:
        meta = _add_terrain_hfield(spec, terrain)
    else:
        meta = _add_terrain_mesh(spec, terrain)
    meta["replace_floor"] = bool(terrain.meta.get("replace_floor", True))
    return meta


def _remove_base_floor(spec) -> None:
    """Delete the base model's floor/ground plane geoms (by name)."""
    import mujoco

    for geom in list(getattr(spec.worldbody, "geoms", ()) or ()):
        if (geom.name in _FLOOR_GEOM_NAMES
                and geom.type == mujoco.mjtGeom.mjGEOM_PLANE):
            spec.delete(geom)


def compose_scene(scene):
    """Build one neutral scene into a fresh MjSpec (pre-compile).

    Accepts the closed IR :class:`~unigym.scenes.scene_ir.Scene` only.
    ONE loop over the instances: registered primitive generators render
    their neutral payload, everything else resolves through the props
    path.  The caller owns ``compile()`` (and the env model-load seam).

    Args:
        scene: A :class:`~unigym.scenes.scene_ir.Scene`.

    Returns:
        The compiled-ready :class:`ComposedScene`.

    Raises:
        ValueError: The value is not a Scene, or any declaration fails
            its validation gate.
    """
    import mujoco

    if not isinstance(scene, Scene):
        raise ValueError(f"scene expected, got {type(scene).__name__}")
    validate_scene(scene)
    spec = mujoco.MjSpec.from_string(_SCENE_MODEL.format(name=scene.name))
    _apply_options(mujoco, spec, scene.engine_hints)
    _add_scene_materials(mujoco, spec, scene.materials)
    shape_generators = set(registered_generators())
    terrain = (_add_terrain(spec, scene.terrain)
               if scene.terrain is not None else None)
    asset_regions = {}
    for instance in scene.instances:
        if instance.generator in shape_generators:
            schema = generator_schema(instance.generator)
            params = dict(instance.params)
            # the legacy converter carries no per-instance name (its
            # instance id IS the declaration name); supply it here so the
            # payload keeps the historical slab/entity naming.  A
            # primitive that declares pos/quat takes the instance pose as
            # its default (the converter stores a box shell's placement on
            # the pose, not in params); an explicit param still wins.
            params.setdefault("name", instance.id)
            if "pos" in schema:
                params.setdefault("pos", tuple(instance.pose.xyz))
            if "quat" in schema:
                params.setdefault("quat", tuple(instance.pose.quat_wxyz))
            _add_shape_payload(
                spec, instance, build_payload(instance.generator, params)
            )
        else:
            asset_regions.update(_add_prop_instance(spec, scene.name, instance))
    regions = _scene_regions(scene)
    for key, region in asset_regions.items():
        if key in regions:
            raise ValueError(f"duplicate composed region name {key!r}")
        regions[key] = region
    return ComposedScene(spec, MappingProxyType(regions), terrain)


def _scene_regions(scene: Scene) -> dict:
    """Resolve the engine region attachments using the attach convention.

    The neutral :class:`~unigym.scenes.scene_ir.Region` carries only the
    address; the attachment table (kind/pose/body/site/dimensions) rides
    ``Scene.engine_hints["regions"]``.  Each row names its owning
    instance, whose verbatim mounting policy decides the composed body
    prefix exactly as the prop path does.
    """
    instances = {instance.id: instance for instance in scene.instances}
    regions = {}
    for region in scene.engine_hints.get("regions", ()):
        instance = instances[region["owner"]]
        verbatim = bool(instance.params.get("verbatim", False))
        prefix = "" if verbatim else f"Prop_{instance.id}_"
        owner = (f"{prefix}{region['body']}" if region["body"] else
                 region["owner"] if verbatim else
                 f"Prop_{instance.id}_{instance.id}")
        name = f"{region['owner']}.{region['name']}"
        if name in regions:
            raise ValueError(f"duplicate composed region name {name!r}")
        regions[name] = RegionAttachment(
            name, region["kind"], owner,
            f"{prefix}{region['body']}" if region["body"] else None,
            f"{prefix}{region['site']}" if region["site"] else None,
            region["pos"], region["quat"], region["width_m"],
            region["depth_m"], region["height_m"])
    return regions


def _apply_options(mujoco, spec, hints) -> None:
    """Transfer the neutral engine hints onto ``spec.option``.

    Enum fields resolve by member name (the ``mj_opt`` knob convention
    of the mujoco env backend); a silent field keeps the engine default.
    """
    opt = spec.option
    if hints.get("timestep") is not None:
        opt.timestep = float(hints["timestep"])
    if hints.get("gravity") is not None:
        opt.gravity = [float(v) for v in hints["gravity"]]
    if hints.get("integrator") is not None:
        opt.integrator = int(getattr(mujoco.mjtIntegrator, hints["integrator"]))
    if hints.get("solver") is not None:
        opt.solver = int(getattr(mujoco.mjtSolver, hints["solver"]))
    if hints.get("cone") is not None:
        opt.cone = int(getattr(mujoco.mjtCone, hints["cone"]))
    if hints.get("iterations") is not None:
        opt.iterations = int(hints["iterations"])
    if hints.get("impratio") is not None:
        opt.impratio = float(hints["impratio"])
    if hints.get("noslip_iterations") is not None:
        opt.noslip_iterations = int(hints["noslip_iterations"])


def _add_scene_materials(mujoco, spec, materials) -> None:
    """Declare the scene's textures + materials (declaration order).

    The material texture reference is declared at the MJCF ``texture=``
    slot through the shared full-width binding (the asset-IR renderer's
    ``material_textures`` discipline); texture ``file`` strings resolve
    through the shared repo-first -> pack ladder at spec-build time.

    WHY: the closed IR :class:`~unigym.scenes.scene_ir.Material` carries
    an opaque binding table (``Material.slots``) instead of typed
    fields.  Typed material fields land with the Phase-4 C-1 scaffold
    card (the build-pipeline DTO card), so this renderer keeps reading
    the converter-carried slots table until then.
    """
    from .asset_render import material_textures

    kind_enum = {
        "2d": mujoco.mjtTexture.mjTEXTURE_2D,
        "cube": mujoco.mjtTexture.mjTEXTURE_CUBE,
    }
    for material in materials:
        slots = material.slots
        spec.add_texture(name=material.name, type=kind_enum[slots["kind"]],
                         file=resolve_asset_path(slots["texture_file"]))
    for material in materials:
        slots = material.slots
        kwargs = {}
        if slots.get("rgba") is not None:
            kwargs["rgba"] = [float(v) for v in slots["rgba"]]
        if slots.get("shininess") is not None:
            kwargs["shininess"] = float(slots["shininess"])
        if slots.get("reflectance") is not None:
            kwargs["reflectance"] = float(slots["reflectance"])
        if slots.get("texuniform") is not None:
            kwargs["texuniform"] = bool(slots["texuniform"])
        if slots.get("texrepeat") is not None:
            kwargs["texrepeat"] = [float(v) for v in slots["texrepeat"]]
        kwargs["textures"] = material_textures(["", material.name])
        spec.add_material(name=material.name, **kwargs)


def _box_size(shape) -> list:
    """Box OUTER envelope -> mujoco half extents (unchanged mapping)."""
    return [float(v) / 2.0 for v in shape.size]


def _sphere_size(shape) -> list:
    """Sphere radius -> mujoco ``[radius, 0, 0]``."""
    return [float(shape.radius), 0.0, 0.0]


def _axial_size(shape) -> list:
    """Cylinder/capsule ``[radius, half_height]`` -> mujoco size slot."""
    return [float(shape.radius), float(shape.half_height), 0.0]


def _plane_size(shape) -> list:
    """Plane ``[sx, sy]`` half extents -> grid size (spacing 0.05)."""
    return [float(shape.size[0]), float(shape.size[1]), 0.05]


#: closed neutral shape class -> (mujoco geom enum member, size mapping).
_SHAPE_GEOM = MappingProxyType({
    BoxShape: ("mjGEOM_BOX", _box_size),
    SphereShape: ("mjGEOM_SPHERE", _sphere_size),
    CylinderShape: ("mjGEOM_CYLINDER", _axial_size),
    CapsuleShape: ("mjGEOM_CAPSULE", _axial_size),
    PlaneShape: ("mjGEOM_PLANE", _plane_size),
})


def _add_shape_payload(spec, instance, payload) -> None:
    """Render one neutral shape payload under a single instance anchor.

    This is the ONLY payload -> engine translation: one anchor named
    after the instance, one engine geom per shape (box / sphere /
    cylinder / capsule / plane through :data:`_SHAPE_GEOM`), the logical
    material resolved against the declared material table, and the
    neutral ``layer`` hint mapped onto the engine's official visual
    class (1) or collision class (0).  A present
    :class:`~unigym.scenes.primitives.ContactSpec` reuses the shared
    contact-kwarg interpreter.

    Args:
        spec: The compiled-ready MjSpec being composed.
        instance: The IR instance anchoring the payload.
        payload: The generator's neutral
            :class:`~unigym.scenes.primitives.ShapePayload`.

    Raises:
        TypeError: A payload entry is not one of the closed neutral
            shape classes.
    """
    import mujoco

    from .asset_render import _contact_kwargs

    body = spec.worldbody.add_body(name=instance.id)
    for shape in payload.shapes:
        entry = _SHAPE_GEOM.get(type(shape))
        if entry is None:
            raise TypeError(
                f"scene instance {instance.id!r}: unknown shape class "
                f"{type(shape).__name__}; the closed payload vocabulary is "
                f"{sorted(cls.__name__ for cls in _SHAPE_GEOM)}")
        geom_name, size_of = entry
        kwargs = {
            "name": shape.name,
            "type": getattr(mujoco.mjtGeom, geom_name),
            "size": size_of(shape),
            "pos": [float(v) for v in shape.pos],
            "quat": [float(v) for v in shape.quat_wxyz],
            "material": shape.material,
            "group": 1 if shape.layer == "visual" else 0,
        }
        if shape.contact is not None:
            kwargs.update(_contact_kwargs(shape.contact))
        body.add_geom(**kwargs)


def _add_prop_instance(spec, scene_name, instance) -> dict:
    """Resolve + attach one non-primitive generator instance at its pose.

    The EXISTING props path does the assembly (resolve -> IR payload ->
    child spec -> prefix attach); this function owns only the refusal
    boundaries (payload kind), the attach-frame pose, and the VERBATIM
    face (prefix-free attach for baked scene assemblies).  The reserved
    ``verbatim`` param key is popped here (it rides the instance payload
    because the closed IR has no such field).

    Args:
        spec: The compiled-ready MjSpec being composed.
        scene_name: The owning scene name (provenance in errors).
        instance: The IR instance to resolve and attach.

    Returns:
        The provider region attachments resolved into the composed name
        namespace.
    """
    from .asset_render import add_ir_body, add_ir_free_body
    from unigym.props.templates import resolve_prop_template

    params = dict(instance.params)
    verbatim = bool(params.pop("verbatim", False))
    where = f"scene:{scene_name}/instances/{instance.id}"
    entry, asset = resolve_prop_template(
        instance.generator, instance.id, params, where=where)
    if verbatim:
        _add_verbatim_instance(spec, where, instance, entry, asset)
        return _asset_regions(instance, asset, "")
    if entry.get("type") == "free_articulation":
        add_ir_free_body(spec, _InstanceHandler(instance.id, entry), asset)
    elif entry.get("type") == "articulation":
        add_ir_body(spec, _InstanceHandler(instance.id, entry), asset)
    else:
        raise ValueError(
            f"{where}: generator {instance.generator!r} produced prop type "
            f"{entry.get('type')!r} -- scene instances assemble generated "
            f"articulation payloads (rigid dressing is the primitive "
            f"payload vocabulary)")
    frame_name = f"Prop_{instance.id}_attach"
    frames = [f for f in _all_frames(spec) if f.name == frame_name]
    if len(frames) != 1:
        raise ValueError(
            f"{where}: expected exactly one attach frame {frame_name!r}, "
            f"found {len(frames)}")
    frames[0].pos = [float(v) for v in instance.pose.xyz]
    frames[0].quat = [float(v) for v in instance.pose.quat_wxyz]
    return _asset_regions(instance, asset, f"Prop_{instance.id}_")


def _asset_regions(instance, asset, prefix: str) -> dict:
    """Resolve provider region references into the composed name namespace."""
    if asset is None:
        return {}
    rows = {}
    for region in asset.regions:
        name = f"{instance.id}.{region.name}"
        if name in rows:
            raise ValueError(f"duplicate composed region name {name!r}")
        # a provider region declaring a body anchors THERE (the pos is
        # local to that body's frame), so the body is the owner and the
        # duplicate ref would trip bind_regions' parent-of-owner check
        # against itself
        owner = region.body or region.owner
        rows[name] = RegionAttachment(
            name, region.kind, f"{prefix}{owner}",
            None,
            f"{prefix}{region.site}" if region.site else None,
            region.pos, region.quat, region.width_m, region.depth_m,
            region.height_m)
    return rows


def _add_verbatim_instance(spec, where, instance, entry, asset) -> None:
    """Attach one instance WITHOUT the Prop_ prefix (v075 P1).

    The baked-assembly face: the payload's element names are the scene's
    own (the retired live scene's constructor namespace, cited by name
    from the placement/task tables), so the child spec attaches with an
    empty prefix and every root lands at the attach frame verbatim (a
    FOREST payload renders each worldbody-level root; the frame itself
    dissolves at compile).  The same refusal boundaries as the prefixed
    face apply: actuated joints are prop-runtime vocabulary, and only
    articulation payloads assemble here.
    """
    from .asset_render import _build_child

    if entry.get("type") != "articulation":
        raise ValueError(
            f"{where}: verbatim instances assemble anchored articulation "
            f"payloads, got {entry.get('type')!r}")
    handler = _InstanceHandler(instance.id, entry)
    child, _joints = _build_child(handler, asset)
    frame = spec.worldbody.add_frame(name=f"Scene_{instance.id}_attach")
    spec.attach(child, prefix="", suffix="", frame=frame)
    frame.pos = [float(v) for v in instance.pose.xyz]
    frame.quat = [float(v) for v in instance.pose.quat_wxyz]


def attach_composed(spec, composed, frame: str) -> Mapping[str, object]:
    """Join an ALREADY composed scene INTO a base spec at *frame*.

    The env model-load seam's core (v075 P1): the base spec is the
    robot world-part XML (it owns the world-level option/size/visual
    settings -- attach does not transfer child options); the composed
    scene spec's content (primitive payload anchors, prop instances,
    verbatim assemblies) attaches at the worldbody with no prefix and no
    synthetic wrapper.

    Args:
        spec: The base spec receiving the composed content.
        composed: A product of :func:`compose_scene`.
        frame: The attach frame name (the caller's scene-naming rule).

    Returns:
        The composed region table.
    """
    frame_obj = spec.worldbody.add_frame(name=frame)
    spec.attach(composed.spec, prefix="", suffix="", frame=frame_obj)
    terrain = getattr(composed, "terrain", None)
    if terrain is not None and terrain.get("replace_floor", True):
        _remove_base_floor(spec)
    return composed.regions


def attach_scene_definition(spec, scene: Scene) -> object:
    """Compose + attach one scene INTO a base spec (compatibility wrapper).

    The historical seam keeps its exact bytes: compose then attach at the
    conventional ``Scene_<name>_attach`` frame.  New callers use the
    scene-engine seam (:func:`attach_composed`); this wrapper stays for
    callers that only have the neutral scene.
    """
    return attach_composed(spec, compose_scene(scene),
                           f"Scene_{scene.name}_attach")


def _all_frames(spec) -> list:
    """The spec's frames.

    The spec-level list is authoritative; the worldbody-level list is
    the fallback for engines without it.
    """
    frames = list(getattr(spec, "frames", None) or [])
    if not frames:
        frames = list(getattr(spec.worldbody, "frames", None) or [])
    return frames
