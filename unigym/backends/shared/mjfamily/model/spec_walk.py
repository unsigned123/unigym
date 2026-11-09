"""Pre-compile MuJoCo ``MjSpec`` walkers for the neutral model IR.

Everything here reads the declared spec only: compiled addresses, mass and
inertia stay ``None`` because the compiler owns those values.
"""

from __future__ import annotations

import math
import os
from enum import Enum

from unigym.model.ir import (
    Actuator,
    Body,
    Camera,
    EqSlot,
    Geom,
    GeomContact,
    GeomGroup,
    Joint,
    Material,
    MjExt,
    ModelIR,
    Site,
    Texture,
)
from unigym.model.table import Table

_JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
_TEXTURE_TYPES = {0: "2d", 1: "cube", 2: "skybox"}
_BUILTIN_TYPES = {0: None, 1: "gradient", 2: "checker", 3: "flat"}
_INTEGRATOR_TYPES = {
    0: "euler",
    1: "rk4",
    2: "implicit",
    3: "implicitfast",
}
_SOLVER_TYPES = {0: "pgs", 1: "cg", 2: "newton"}
_CONE_TYPES = {0: "pyramidal", 1: "elliptic"}
_OPTION_ENUM_TYPES = {
    "integrator": _INTEGRATOR_TYPES,
    "solver": _SOLVER_TYPES,
    "cone": _CONE_TYPES,
}
_GEOM_TYPES = {
    0: "plane",
    1: "hfield",
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh",
    8: "sdf",
}
_TRN_TYPES = {
    0: "joint",
    1: "jointinparent",
    2: "slidercrank",
    3: "tendon",
    4: "site",
    5: "body",
    6: "so3",
    1000: "undefined",
}
_CAM_MODES = {
    0: "fixed",
    1: "track",
    2: "trackcom",
    3: "targetbody",
    4: "targetbodycom",
}
_EQ_TYPES = {
    0: "connect",
    1: "weld",
    2: "joint",
    3: "tendon",
    4: "flex",
    5: "flexvert",
    6: "flexstrain",
    7: "distance",
}


def _safe_name(kind, index, name):
    """Return the native name, or a deterministic synthetic one if unnamed."""
    if isinstance(name, str) and name:
        return name
    return f"<unnamed:{kind}:{index}>"


def _vec(values) -> tuple[float, ...]:
    """Return one numeric array as a tuple of floats."""
    return tuple(float(value) for value in values)


def _vec2(values) -> tuple[float, float]:
    """Return a numeric array as a fixed 2-vector (``range``/``texrepeat``)."""
    return (float(values[0]), float(values[1]))


def _vec3(values) -> tuple[float, float, float]:
    """Return a numeric array as a fixed 3-vector (``pos``/``axis``/``size``)."""
    return (float(values[0]), float(values[1]), float(values[2]))


def _vec4(values) -> tuple[float, float, float, float]:
    """Return a numeric array as a fixed 4-vector (``quat``/``rgba``)."""
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _finite(value):
    """Return a finite float, else ``None`` (never fake a value)."""
    number = float(value)
    return number if math.isfinite(number) else None


def _basename(value):
    """Return a file reference's basename, or ``None`` when empty."""
    text = str(value or "").replace("\\", "/")
    return os.path.basename(text) or None


def _optional_int(value, minimum=None):
    """Return an int when it is at least *minimum*, else ``None``."""
    number = int(value)
    if minimum is not None and number < minimum:
        return None
    return number


def _texture_row(texture, index):
    """Build one :class:`Texture` row from a spec texture."""
    return Texture(
        name=_safe_name("texture", index, texture.name),
        type=_TEXTURE_TYPES.get(int(texture.type), str(int(texture.type))),
        builtin=_BUILTIN_TYPES.get(int(texture.builtin)),
        file=_basename(getattr(texture, "file", "")),
        width=_optional_int(getattr(texture, "width", 0), minimum=1),
        height=_optional_int(getattr(texture, "height", 0), minimum=1),
        nchannel=_optional_int(getattr(texture, "nchannel", 0), minimum=1),
    )


def _geom_contact_row(geom, name):
    """Build one :class:`GeomContact` row from a spec geom."""
    friction = getattr(geom, "friction", None)
    solref = getattr(geom, "solref", None)
    solimp = getattr(geom, "solimp", None)
    priority = getattr(geom, "priority", None)
    condim = getattr(geom, "condim", None)
    return GeomContact(
        name=name,
        friction=_vec(friction) if friction is not None else None,
        solref=_vec(solref) if solref is not None else None,
        solimp=_vec(solimp) if solimp is not None else None,
        margin=_finite(getattr(geom, "margin", float("nan"))),
        gap=_finite(getattr(geom, "gap", float("nan"))),
        priority=int(priority) if priority is not None else None,
        condim=int(condim) if condim is not None else None,
    )


def _ref_name(value):
    """Return a string name from a string or reference object."""
    if isinstance(value, str):
        return value
    return str(getattr(value, "name", "") or "")


def _body_info(spec, bodies=None):
    """Return a retained body index: names, paths and the wrapper records.

    Pre-compile ``MjsBody.id`` is ``-1`` for most bodies, so the walk keys
    on the Python object identity of one retained ``spec.bodies`` snapshot
    (unstable across re-materialized pybind wrappers) and assigns synthetic
    names by list order for the unnamed bodies MJCF allows.
    """
    if bodies is None:
        bodies = list(spec.bodies)
    names = {}
    for index, body in enumerate(bodies):
        names[id(body)] = body.name or _safe_name("body", index, "")
    world = spec.worldbody
    paths = {}

    def path(body):
        key = id(body)
        if key in paths:
            return paths[key]
        if body is world:
            result = ()
        else:
            result = path(body.parent) + (names[key],)
        paths[key] = result
        return result

    for body in bodies:
        path(body)
    records = [(body, names[id(body)], paths[id(body)]) for body in bodies]
    by_name = {name: (name, value) for _, name, value in records}
    return {
        "records": records,
        "by_name": by_name,
        "world": world,
        "world_name": names.get(id(world), records[0][1] if records else "world"),
    }


def _body_ref(body, info):
    """Return ``(name, path)`` for one body object."""
    if body is info.get("world"):
        return info["world_name"], ()
    name = body.name
    if name and name in info["by_name"]:
        return info["by_name"][name]
    for candidate, candidate_name, candidate_path in info["records"]:
        if candidate == body:
            return candidate_name, candidate_path
    raise KeyError(f"body {name!r} is not in the retained spec body list")


def _body_row(body, info, world):
    """Build one :class:`Body` row from a spec body."""
    name, path = _body_ref(body, info)
    parent = None if body == world else _body_ref(body.parent, info)[0]
    return Body(
        name=name,
        parent=parent,
        path=path,
        qualname="/".join(path) or name,
        pos=_vec3(body.pos),
        quat=_vec4(body.quat),
        mocap=bool(body.mocap),
        explicitinertial=bool(body.explicitinertial),
    )


def _actuator_kind(name, target, trntype):
    """Return ``(kind, owner)`` for one actuator declaration."""
    if name.startswith("Prop_") or target.startswith("Prop_"):
        base = name if name.startswith("Prop_") else target
        parts = base.split("_")
        owner = "_".join(parts[:2]) if len(parts) >= 2 else base
        return "prop", owner
    if trntype in ("joint", "jointinparent"):
        return "robot", None
    return "other", None


def _actuator_row(actuator, index):
    """Build one :class:`Actuator` row (declared fields only)."""
    trntype = _TRN_TYPES.get(int(actuator.trntype), str(int(actuator.trntype)))
    target = _ref_name(actuator.target)
    name = _safe_name("actuator", index, actuator.name)
    kind, owner = _actuator_kind(name, target, trntype)
    ctrlrange = _vec2(actuator.ctrlrange) if int(actuator.ctrllimited) else None
    return Actuator(
        name=name,
        target=target,
        trntype=trntype,
        kind=kind,
        owner=owner,
        ctrlrange=ctrlrange,
        gear=_vec(actuator.gear),
        gainprm=_vec(actuator.gainprm),
        biasprm=_vec(actuator.biasprm),
        qualname=name,
    )


def _actuator_rows(spec):
    """Return every actuator row plus the ``joint target -> actuator`` map."""
    rows = []
    by_target = {}
    for index, actuator in enumerate(spec.actuators):
        row = _actuator_row(actuator, index)
        rows.append(row)
        if row.trntype in ("joint", "jointinparent") and row.target:
            by_target[row.target] = row.name
    return rows, by_target


def _joint_row(joint, index, info, actuator_by_target):
    """Build one :class:`Joint` row from a spec joint."""
    body_name, body_path = _body_ref(joint.parent, info)
    name = _safe_name("joint", index, joint.name)
    raw_range = _vec2(joint.range)
    limited = int(joint.limited) != 0 and raw_range != (0.0, 0.0)
    return Joint(
        name=name,
        type=_JOINT_TYPES.get(int(joint.type), str(int(joint.type))),
        body=body_name,
        axis=_vec3(joint.axis),
        range=raw_range if limited else None,
        limited=limited,
        pos=_vec3(joint.pos),
        damping=float(joint.damping[0]),
        armature=float(joint.armature),
        actuator=actuator_by_target.get(joint.name),
        path=body_path + (name,),
        qualname=f"{body_path[-1]}/{name}" if body_path else name,
    )


def _geom_row(geom, index, info):
    """Build one :class:`Geom` row from a spec geom."""
    name = _safe_name("geom", index, geom.name)
    body_name, body_path = _body_ref(geom.parent, info)
    material = geom.material if isinstance(geom.material, str) else None
    return Geom(
        name=name,
        type=_GEOM_TYPES.get(int(geom.type), str(int(geom.type))),
        body=body_name,
        pos=_vec3(geom.pos),
        quat=_vec4(geom.quat),
        size=_vec3(geom.size),
        material=material or None,
        group=int(geom.group),
        contype=int(geom.contype),
        conaffinity=int(geom.conaffinity),
        density=_finite(geom.density),
        mass=_finite(geom.mass),
        path=body_path + (name,),
        qualname=f"{body_name}/{name}",
    )


def _site_row(site, index, info):
    """Build one :class:`Site` row from a spec site."""
    name = _safe_name("site", index, site.name)
    body_name, body_path = _body_ref(site.parent, info)
    material = site.material if isinstance(site.material, str) else None
    return Site(
        name=name,
        type=_GEOM_TYPES.get(int(site.type), str(int(site.type))),
        body=body_name,
        pos=_vec3(site.pos),
        quat=_vec4(site.quat),
        size=_vec3(site.size),
        material=material or None,
        group=int(site.group),
        path=body_path + (name,),
        qualname=f"{body_name}/{name}",
    )


def _camera_row(camera, index, info):
    """Build one :class:`Camera` row from a spec camera."""
    name = _safe_name("camera", index, camera.name)
    body_name, body_path = _body_ref(camera.parent, info)
    target = _ref_name(camera.targetbody) or None
    return Camera(
        name=name,
        body=body_name,
        pos=_vec3(camera.pos),
        quat=_vec4(camera.quat),
        fovy=float(camera.fovy),
        mode=_CAM_MODES.get(int(camera.mode), str(int(camera.mode))),
        targetbody=target,
        path=body_path + (name,),
        qualname=f"{body_name}/{name}",
    )


def _material_row(material, index):
    """Build one :class:`Material` row from a spec material."""
    return Material(
        name=_safe_name("material", index, material.name),
        rgba=_vec4(material.rgba),
        textures=tuple(str(item) for item in material.textures),
        reflectance=_finite(material.reflectance),
        roughness=_finite(material.roughness),
        metallic=_finite(material.metallic),
        emission=_finite(material.emission),
        specular=_finite(material.specular),
        shininess=_finite(material.shininess),
        texrepeat=_vec2(material.texrepeat),
        texuniform=bool(material.texuniform),
    )


def _scalar(value):
    """Convert one option value into a plain scalar or tuple."""
    if isinstance(value, Enum):
        return int(value.value)
    if isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        try:
            return _scalar(value.item())
        except (ValueError, IndexError):
            pass
    try:
        return tuple(_scalar(item) for item in value)
    except TypeError:
        return str(value)


def _option_dict(option):
    """Return every public option knob as plain values.

    ``integrator``/``solver``/``cone`` are exported as their canonical
    names (not raw enum ints), and ``solref``/``solimp`` are exposed under
    those unprefixed keys in addition to MuJoCo's ``o_*`` spellings.
    """
    out = {}
    for key in dir(option):
        if key.startswith("_"):
            continue
        value = getattr(option, key)
        if callable(value):
            continue
        scalar = _scalar(value)
        names = _OPTION_ENUM_TYPES.get(key)
        if names is not None and isinstance(scalar, int):
            scalar = names.get(scalar, scalar)
        out[key] = scalar
    if "o_solref" in out:
        out["solref"] = out["o_solref"]
    if "o_solimp" in out:
        out["solimp"] = out["o_solimp"]
    return out


def _eq_row(equality, index):
    """Build one :class:`EqSlot` row from a spec equality."""
    return EqSlot(
        name=_safe_name("eq", index, equality.name),
        type=_EQ_TYPES.get(int(equality.type), str(int(equality.type))),
        obj1=_ref_name(equality.name1) or None,
        obj2=_ref_name(equality.name2) or None,
        active=bool(equality.active),
    )


def _ext(spec, body_rows, geom_rows):
    """Build the ``mj`` namespace extension from the declared spec."""
    groups = {}
    for geom in geom_rows:
        groups.setdefault(geom.group, []).append(geom.name)
    geom_groups = [
        GeomGroup(name=f"group{group}", group=group, geoms=tuple(names))
        for group, names in sorted(groups.items())
    ]
    eq_slots = [_eq_row(equality, index) for index, equality in enumerate(spec.equalities)]
    mocap = [row for row in body_rows if row.mocap]
    textures = [_texture_row(texture, index) for index, texture in enumerate(spec.textures)]
    geom_contact = [_geom_contact_row(geom, row.name) for geom, row in zip(spec.geoms, geom_rows)]
    return MjExt(
        opt=_option_dict(spec.option),
        geom_groups=Table(geom_groups),
        eq_slots=Table(eq_slots),
        mocap_bodies=Table(mocap),
        textures=Table(textures),
        geom_contact=Table(geom_contact),
    )


def build_from_spec(spec, backend="mujoco"):
    """Walk an uncompiled ``MjSpec`` and return the declared model IR."""
    bodies = list(spec.bodies)
    info = _body_info(spec, bodies)
    body_rows = [_body_row(body, info, spec.worldbody) for body in bodies]
    actuator_rows, actuator_by_target = _actuator_rows(spec)
    joint_rows = [
        _joint_row(joint, index, info, actuator_by_target)
        for index, joint in enumerate(spec.joints)
    ]
    geom_rows = [_geom_row(geom, index, info) for index, geom in enumerate(spec.geoms)]
    site_rows = [_site_row(site, index, info) for index, site in enumerate(spec.sites)]
    camera_rows = [_camera_row(camera, index, info) for index, camera in enumerate(spec.cameras)]
    material_rows = [_material_row(item, index) for index, item in enumerate(spec.materials)]
    scene = _ref_name(getattr(spec, "modelname", "")) or "model"
    ir = ModelIR(
        name=scene,
        backend=backend,
        compiled=False,
        bodies=Table(body_rows),
        joints=Table(joint_rows),
        geoms=Table(geom_rows),
        sites=Table(site_rows),
        cameras=Table(camera_rows),
        actuators=Table(actuator_rows),
        materials=Table(material_rows),
        meta={
            "source": getattr(spec, "modelfiledir", None),
            "scene": scene,
            "builder": spec,
        },
        ext={"mj": _ext(spec, body_rows, geom_rows)},
    )
    return ir.validate()
